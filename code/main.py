"""
main.py
-------
Entry point for the Buy or Wait? financial decision agent.

Usage
-----
    # Full run — all 250 requests → output.csv in repo root
    python code/main.py

    # Validate on the 25 solved examples first
    python code/main.py --sample

    # Quick smoke test (first N requests)
    python code/main.py --limit 5

    # Tune concurrency (default: 5)
    python code/main.py --concurrency 8

    # Custom output path
    python code/main.py --output /path/to/output.csv

Environment
-----------
    A local Ollama server with qwen2.5:7b must be available.
    OLLAMA_HOST is optional and defaults to http://127.0.0.1:11434.

Output
------
    output.csv                        — written to the repository root
    code/evaluation/usage_report.md — LLM token and cost summary

Architecture
------------
    For each request row, process_request() runs the full pipeline:

        1. parse_messages()          — extract salary amendments, cancellations
        2. fill_blank_event_amounts() — local Tesseract OCR on linked PNG documents
        3. build_financial_state()   — 90-day balance simulation
        4. get_request_payment_options() — fetch available plans
        5. make_decision()           — deterministic candidate generation and baseline explanation
        6. apply_local_qwen_decisions() — constrained local-Qwen candidate selection/explanation

    All requests are processed concurrently using asyncio with a bounded
    semaphore (--concurrency flag) to bound concurrent request processing and local OCR/model work.
    Results are sorted by request_id before writing to maintain a deterministic
    output order.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Ensure sibling modules are importable when run as `python code/main.py`
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv(Path(__file__).parent.parent / ".env")

from data_loader import (
    load_requests,
    load_sample_requests,
    get_profile,
    get_request_payment_options,
    get_user_events,
    get_request_messages,
    get_event_images,
)
from financial_state import build_financial_state
from image_reader import fill_blank_event_amounts, extract_image_text
from message_parser import parse_messages
from decision_engine import make_decision, DecisionResult
from usage_tracker import tracker
from prompts import DECISION_SYSTEM, DECISION_USER_TEMPLATE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT: Path = Path(__file__).parent.parent
OUTPUT_PATH: Path = REPO_ROOT / "output.csv"

OUTPUT_COLUMNS: List[str] = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]


# ---------------------------------------------------------------------------
# Per-request processing
# ---------------------------------------------------------------------------

async def process_request(
    row: pd.Series,
    semaphore: asyncio.Semaphore,
) -> DecisionResult:
    """
    Run the full Buy or Wait? pipeline for one request row.

    Stages:
      1. Parse messages for financial amendments (salary, cancellations, etc.)
      2. Extract amounts from linked PNG images (local Tesseract OCR)
      3. Build the 90-day financial state with amendments applied
      4. Fetch available payment options
      5. Generate plans and produce the final decision

    Any unhandled exception produces a safe not_recommended fallback row
    so the overall run never aborts mid-stream.
    """
    async with semaphore:
        request_id = str(row["request_id"])
        user_id    = str(row["user_id"])
        request_date: date = row["request_date"]

        try:
            # -- Profile check ------------------------------------------
            profile = get_profile(user_id)
            if profile is None:
                return _error_result(request_id, f"No profile found for {user_id}")

            # -- Stage 1: Message parsing --------------------------------
            amendments = await parse_messages(user_id, request_id, request_date)

            # -- Stage 2: Image extraction for blank event amounts -------
            events_df = get_user_events(user_id)
            image_amounts = await fill_blank_event_amounts(
                user_id, request_id, events_df
            )
            # Merge extracted image amounts into amendments
            for eid, amt in image_amounts.items():
                amendments["amount_overrides"][eid] = amt

            # -- Stage 3: Build financial state --------------------------
            # Convert amendments into the format expected by build_financial_state
            event_overrides: dict = {}
            for eid in amendments.get("cancelled_events", set()):
                event_overrides[eid] = {"cancelled": True}
            for eid, amt in amendments.get("amount_overrides", {}).items():
                event_overrides.setdefault(eid, {})["amount"] = amt
            for eid in amendments.get("pending_credits", set()):
                event_overrides.setdefault(eid, {})["is_pending_credit"] = True
            for eid in amendments.get("suppressed_credits", set()):
                event_overrides.setdefault(eid, {})["is_pending_credit"] = True
            if amendments.get("salary_terminated"):
                event_overrides["salary_terminated"] = True
            if amendments.get("suppress_projected_income"):
                event_overrides["suppress_projected_income"] = True
            if amendments.get("salary_override"):
                event_overrides["_salary_override"] = {
                    "salary_override": amendments["salary_override"],
                    "salary_date":     amendments.get("salary_date"),
                }

            financial_state = build_financial_state(
                user_id=user_id,
                request_date=request_date,
                amended_events=event_overrides,
            )

            # -- Stage 4: Payment options --------------------------------
            payment_options = get_request_payment_options(request_id)

            # -- Stage 5: Decision ---------------------------------------
            return await make_decision(
                request=row,
                financial_state=financial_state,
                payment_options_df=payment_options,
                profile=profile,
            )

        except Exception as exc:
            print(
                f"  ERROR {request_id}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return _error_result(request_id, str(exc))


def _error_result(request_id: str, reason: str) -> DecisionResult:
    """Safe fallback row when processing fails entirely."""
    return DecisionResult(
        request_id=request_id,
        amount_safe_to_pay=0.0,
        affordability_status="not_affordable",
        recommended_payment_method="not_recommended",
        payment_plan="none",
        earliest_date_for_full_payment="",
        spending_changes_needed="none",
        decision_explanation=f"Processing error: {reason[:120]}",
    )


# ---------------------------------------------------------------------------
# Run orchestrator
# ---------------------------------------------------------------------------

async def run(
    requests_df: pd.DataFrame,
    concurrency: int = 5,
) -> List[DecisionResult]:
    """
    Process all requests concurrently with a bounded semaphore.

    Results are accumulated as each coroutine completes, then sorted
    by request_id for deterministic output ordering.
    """
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [process_request(row, semaphore) for _, row in requests_df.iterrows()]

    results: List[DecisionResult] = []
    with tqdm(total=len(tasks), desc="Processing", unit="req") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)

    results.sort(key=lambda r: r.request_id)
    return results


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_output(results: List[DecisionResult], output_path: Path) -> None:
    """Write results to a CSV file with the exact required column order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "request_id":                   r.request_id,
                "amount_safe_to_pay":            round(r.amount_safe_to_pay, 2),
                "affordability_status":          r.affordability_status,
                "recommended_payment_method":    r.recommended_payment_method,
                "payment_plan":                  r.payment_plan,
                "earliest_date_for_full_payment": r.earliest_date_for_full_payment,
                "spending_changes_needed":       r.spending_changes_needed,
                "decision_explanation":          r.decision_explanation,
            })
    print(f"✓ Predictions written → {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point

def _decision_prompt(request: pd.Series, result: DecisionResult, profile: pd.Series) -> str:
    messages = get_request_messages(str(request["user_id"]), str(request["request_id"]))
    evidence = []
    if not messages.empty:
        evidence.extend(str(x) for x in messages.get("message_text", []).tolist())
    events = get_user_events(str(request["user_id"]))
    for _, event in events[events["amount"].isna()].iterrows():
        links = get_event_images(str(event["event_id"]))
        for _, link in links.iterrows():
            text = extract_image_text(str(link["image_id"]))
            if text:
                evidence.append(f"image {link['image_id']}: {text[:1200]}")
    context = dict(result.llm_context)
    context.update({
        "request_date": str(request["request_date"]),
        "desired_completion_date": str(request["desired_completion_date"]),
        "requested_amount": round(float(request["requested_amount"]), 2),
        "request_text": str(request.get("request_text", ""))[:1000],
    })
    return DECISION_USER_TEMPLATE.format(
        request=json.dumps(context, default=str),
        financial_context=json.dumps(result.llm_context, default=str),
        evidence=json.dumps(evidence[:12], ensure_ascii=False),
        candidates=json.dumps(result.candidate_options, default=str),
    )


def apply_local_qwen_decisions(requests_df: pd.DataFrame, results: List[DecisionResult]) -> None:
    from llm_providers import ProviderUnavailable, generate_qwen_decisions
    by_id = {result.request_id: result for result in results}
    prompts = {}
    for _, request in requests_df.iterrows():
        result = by_id.get(str(request["request_id"]))
        profile = get_profile(str(request["user_id"]))
        if result is not None and profile is not None and result.candidate_options:
            prompts[result.request_id] = _decision_prompt(request, result, profile)
    try:
        raw = generate_qwen_decisions(prompts, DECISION_SYSTEM)
    except ProviderUnavailable as exc:
        raise RuntimeError(f"Local Qwen model is required but unavailable: {exc}") from exc
    print("Provider: local Qwen via Ollama", file=sys.stderr)
    for request_id, text_value in raw.items():
        result = by_id.get(request_id)
        if result is None:
            continue
        try:
            parsed = json.loads(text_value.strip().removeprefix("```json").removesuffix("```").strip())
            candidate_id = str(parsed.get("candidate_id", ""))
            candidate = next((x for x in result.candidate_options if x["candidate_id"] == candidate_id), None)
            explanation = str(parsed.get("explanation", "")).strip()
            if candidate is None or not explanation:
                continue
            # Only copy fields from Python-validated candidates.
            result.affordability_status = candidate["affordability_status"]
            result.recommended_payment_method = candidate["method"]
            result.payment_plan = candidate["payment_plan"]
            result.earliest_date_for_full_payment = candidate["earliest_date_for_full_payment"]
            result.spending_changes_needed = candidate.get("spending_changes_needed", "none")
            result.decision_explanation = explanation[:500]
        except (ValueError, TypeError, AttributeError):
            continue

# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Buy or Wait? — AI financial decision agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run on dataset/sample_requests.csv (25 solved examples) instead of requests.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N requests",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        metavar="N",
        help="Maximum number of requests processed in parallel",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Override the output CSV path (default: <repo_root>/output.csv)",
    )
    args = parser.parse_args()

    # Load requests
    if args.sample:
        requests_df = load_sample_requests()
        print(f"Mode     : sample validation ({len(requests_df)} rows)")
    else:
        requests_df = load_requests()
        print(f"Mode     : full evaluation ({len(requests_df)} rows)")

    if args.limit:
        requests_df = requests_df.head(args.limit)
        print(f"Limited  : first {args.limit} requests")

    output_path = Path(args.output) if args.output else OUTPUT_PATH
    print(f"Output   : {output_path}")
    print(f"Concur.  : {args.concurrency}")
    print()

    # Run pipeline
    results = asyncio.run(run(requests_df, concurrency=args.concurrency))

    apply_local_qwen_decisions(requests_df, results)


    # Write predictions
    write_output(results, output_path)

    # Write usage report
    tracker.write_report(request_count=len(requests_df))
    print("✓ Usage report → code/evaluation/usage_report.md")

    # Summary
    s = tracker.summary()
    if s:
        print(
            f"\nLLM summary: {s['total_calls']} calls | "
            f"{s['total_tokens']:,} tokens | "
            f"${s['total_cost_usd']:.4f} USD"
        )


if __name__ == "__main__":
    main()
