"""
usage_tracker.py
================
Thread-safe token-usage accumulator for a local Qwen/Ollama call made during
a full dataset run.

Usage
-----
    from usage_tracker import tracker   # module-level singleton

    # After each local model response:
    tracker.record(model="qwen2.5:7b", purpose="candidate_adjudication",
                   request_id="request_42", input_tokens=320, output_tokens=64)

    # At the end of main.py:
    tracker.write_report()   # writes evaluation/usage_report.md

Report format
-------------
The generated usage_report.md satisfies the challenge submission requirement:
    - model provider and name
    - total model calls
    - total input and output tokens
    - total and average tokens per request
    - estimated total and per-request cost (USD)

Cost reference
--------------
    Qwen runs locally through Ollama. Provider-billed token cost is zero; local
    hardware/electricity cost is intentionally not estimated.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# Path is relative to this file (code/evaluation/usage_report.md)
REPORT_PATH: Path = Path(__file__).parent / "evaluation" / "usage_report.md"

# ---------------------------------------------------------------------------
# Cost table  (USD per 1 000 tokens)
# ---------------------------------------------------------------------------
# Local Ollama execution has no provider-billed token cost.
COST_PER_1K: Dict[str, Dict[str, float]] = {
    "qwen2.5:7b": {"input": 0.0, "output": 0.0},
}
DEFAULT_COST: Dict[str, float] = {"input": 0.0000, "output": 0.0000}


@dataclass
class CallRecord:
    """One recorded LLM API call."""
    model: str
    purpose: str        # "image_extraction" | "explanation" | "message_parse"
    request_id: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        """Estimated USD cost based on the static COST_PER_1K table."""
        rates = COST_PER_1K.get(self.model, DEFAULT_COST)
        return (
            self.input_tokens  / 1000.0 * rates["input"]
            + self.output_tokens / 1000.0 * rates["output"]
        )


class UsageTracker:
    """
    Thread-safe accumulator of LLM call records.

    One global instance (tracker) is imported by the local provider and main.py.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: List[CallRecord] = []

    def record(
        self,
        model: str,
        purpose: str,
        request_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Append one call record.  Safe to call from multiple threads."""
        with self._lock:
            self._records.append(CallRecord(
                model=model,
                purpose=purpose,
                request_id=request_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ))

    def summary(self) -> Dict:
        """Return a summary dictionary suitable for report generation."""
        with self._lock:
            records = list(self._records)

        if not records:
            return {}

        total_calls    = len(records)
        total_input    = sum(r.input_tokens for r in records)
        total_output   = sum(r.output_tokens for r in records)
        total_tokens   = total_input + total_output
        total_cost     = sum(r.cost_usd for r in records)

        per_model: Dict[str, Dict] = {}
        for r in records:
            m = per_model.setdefault(r.model, {
                "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                "purposes": set(),
            })
            m["calls"]         += 1
            m["input_tokens"]  += r.input_tokens
            m["output_tokens"] += r.output_tokens
            m["cost_usd"]      += r.cost_usd
            m["purposes"].add(r.purpose)

        unique_requests = len({r.request_id for r in records})
        avg_tokens = total_tokens / max(unique_requests, 1)
        avg_cost   = total_cost   / max(unique_requests, 1)

        return {
            "total_calls":            total_calls,
            "total_input_tokens":     total_input,
            "total_output_tokens":    total_output,
            "total_tokens":           total_tokens,
            "total_cost_usd":         total_cost,
            "unique_requests":        unique_requests,
            "avg_tokens_per_request": avg_tokens,
            "avg_cost_per_request":   avg_cost,
            "per_model":              per_model,
        }

    def write_report(self, request_count: int = 0) -> None:
        """
        Write evaluation/usage_report.md with the full-run summary.
        Creates the directory if it does not exist.
        This file is required in the code.zip submission.
        """
        s = self.summary()
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

        if not s:
            raise RuntimeError("Local Qwen produced no usage records; final output was not written.")

        lines = [
            "# Token Usage Report — Buy or Wait? Full Dataset Run",
            "",
            "## Model",
            "",
            "| Provider | Model | Tasks |",
            "|----------|-------|-------|",
        ]
        for model, m in s["per_model"].items():
            purposes = ", ".join(sorted(m["purposes"]))
            provider = "Ollama (local Qwen)"
            lines.append(f"| {provider} | `{model}` | {purposes} |")

        lines += [
            "",
            "## Overall Totals",
            "",
            f"- Total LLM calls        : {s['total_calls']}",
            f"- Requests processed     : {request_count}",
            f"- Total input tokens     : {s['total_input_tokens']:,}",
            f"- Total output tokens    : {s['total_output_tokens']:,}",
            f"- Total tokens           : {s['total_tokens']:,}",
            f"- Avg tokens / request   : {s['total_tokens'] / max(request_count, 1):.1f}",
            f"- Estimated total cost   : ${s['total_cost_usd']:.4f} USD",
            f"- Avg cost / request     : ${s['avg_cost_per_request']:.5f} USD",
            "",
            "## Per-Model Breakdown",
            "",
        ]
        for model, m in s["per_model"].items():
            lines += [
                f"### `{model}`",
                f"- Calls          : {m['calls']}",
                f"- Input tokens   : {m['input_tokens']:,}",
                f"- Output tokens  : {m['output_tokens']:,}",
                f"- Est. cost      : ${m['cost_usd']:.4f} USD",
                "",
            ]

        lines += [
            "---",
            "",
            "_All decisions are adjudicated by the required local Qwen model._",
            "_Local Ollama execution incurs no provider-billed token cost._",
        ]

        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
tracker = UsageTracker()
