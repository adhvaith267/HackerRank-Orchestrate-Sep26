"""
decision_engine.py
==================
Assembles the final output row for one request by:

1. Computing amount_safe_to_pay (binary-search on 90-day balance).
2. Finding spending changes (stop / reduce_to) when baseline is insufficient.
3. Generating candidate payment plans (plan_generator).
4. Selecting the best plan using the 6-rule tiebreaker.
5. Generating a concise decision_explanation via local Qwen.
6. Producing a DecisionResult with all required output fields.

Output fields produced
----------------------
    request_id, amount_safe_to_pay, affordability_status,
    recommended_payment_method, payment_plan,
    earliest_date_for_full_payment, spending_changes_needed,
    decision_explanation

LLM usage
---------
    Model : local qwen2.5:7b through Ollama
    Purpose: select a Python-validated candidate and generate its explanation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from data_loader import parse_pipe_list
from financial_state import FinancialState, FORECAST_DAYS
from plan_generator import PaymentPlan, generate_plans


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class DecisionResult:
    """One row of the final output.csv."""
    request_id: str
    amount_safe_to_pay: float
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str
    # Internal-only data used by the required local-Qwen adjudication stage.
    candidate_options: List[Dict[str, Any]] = field(default_factory=list)
    llm_context: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Spending-change logic
# ---------------------------------------------------------------------------

def find_spending_changes(
    financial_state: FinancialState,
    requested_amount: float,
    request_date: date,
    profile: pd.Series,
) -> Tuple[List[str], List[Tuple[str, float]], FinancialState]:
    """
    Search for up to 3 flexible spending changes that would make
    requested_amount safe to pay on request_date.

    Only events in categories the user is willing to stop or reduce are
    considered.  Protected categories are never touched.

    Returns
    -------
    (stop_event_ids, reduce_events, new_financial_state)
        stop_event_ids  : list of representative event_ids to stop
        reduce_events   : list of (event_id, new_amount) to reduce
        new_financial_state : state after applying the changes
    """
    willing_to_stop = set(parse_pipe_list(
        profile.get("expense_categories_user_is_willing_to_stop", "")
    ))
    willing_to_reduce = set(parse_pipe_list(
        profile.get("expense_categories_user_is_willing_to_reduce", "")
    ))
    protected = set(parse_pipe_list(
        profile.get("expense_categories_to_protect", "")
    ))

    # Collect candidate recurring debits eligible for change
    candidates: List[Tuple[str, Any]] = []  # ("stop"|"reduce", RecurringPattern)
    for pattern in financial_state.recurring_patterns:
        if pattern.direction != "debit":
            continue
        if pattern.category in protected:
            continue
        if pattern.flexibility == "stoppable" and pattern.category in willing_to_stop:
            candidates.append(("stop", pattern))
        elif pattern.flexibility in ("flexible",) and pattern.category in willing_to_reduce:
            candidates.append(("reduce", pattern))

    # Sort largest-saving first
    candidates.sort(key=lambda x: x[1].avg_amount, reverse=True)

    stop_ids: List[str] = []
    reduce_events: List[Tuple[str, float]] = []
    current_state = financial_state
    changes_used = 0

    for action, pattern in candidates:
        if changes_used >= 3:
            break
        # Already safe?
        if (
            current_state.balance_after_payment(requested_amount, request_date)
            >= current_state.minimum_balance
        ):
            break

        # Representative real event_id (last historical occurrence)
        rep_id = pattern.event_ids[-1]

        if action == "stop":
            # Collect all projected synthetic IDs for this pattern category
            candidate_ids = _projected_ids_for_pattern(current_state, pattern)
            trial_stops = stop_ids + [pattern.event_ids[-1]]   # add representative only
            trial_state = current_state.with_spending_change(trial_stops, reduce_events)
            before = current_state.balance_after_payment(requested_amount, request_date)
            after  = trial_state.balance_after_payment(requested_amount, request_date)
            if after > before:
                stop_ids = trial_stops
                current_state = trial_state
                changes_used += 1

        elif action == "reduce":
            min_allowed = pattern.minimum_allowed_amount
            if min_allowed is None:
                min_allowed = round(pattern.avg_amount * 0.5, 2)
            trial_reduces = reduce_events + [(rep_id, min_allowed)]
            trial_state = current_state.with_spending_change(stop_ids, trial_reduces)
            before = current_state.balance_after_payment(requested_amount, request_date)
            after  = trial_state.balance_after_payment(requested_amount, request_date)
            if after > before:
                reduce_events = trial_reduces
                current_state = trial_state
                changes_used += 1

    return stop_ids, reduce_events, current_state


def _projected_ids_for_pattern(fs: FinancialState, pattern) -> List[str]:
    """
    Collect all future synthetic event_ids that belong to a recurring pattern
    category (used as targets for with_spending_change).
    """
    ids = []
    seen: set = set()
    for d, entries in fs.daily_flows.items():
        if d < fs.request_date:
            continue
        for entry in entries:
            if entry.category == pattern.category and entry.is_recurring:
                if entry.event_id not in seen:
                    seen.add(entry.event_id)
                    ids.append(entry.event_id)
    if not ids and pattern.event_ids:
        ids = [pattern.event_ids[-1]]
    return ids


def format_spending_changes(
    stop_ids: List[str],
    reduce_events: List[Tuple[str, float]],
) -> str:
    """
    Format spending changes as "stop:event_id|reduce_to:event_id:amount".
    Returns "none" when no changes are needed.
    Only real (non-synthetic) event_ids are reported.
    """
    if not stop_ids and not reduce_events:
        return "none"

    def _real_id(eid: str) -> str:
        """Convert a synthetic projected ID to its source event ID."""
        if eid.startswith(("recur_", "category_", "salary_")):
            return eid.split("_", 1)[1].rsplit("_", 1)[0]
        return eid

    parts: List[str] = []
    for eid in stop_ids:
        parts.append(f"stop:{_real_id(eid)}")
    for eid, amt in reduce_events:
        parts.append(f"reduce_to:{_real_id(eid)}:{round(amt, 2)}")
    return "|".join(parts[:3])


# ---------------------------------------------------------------------------
# Main decision function
# ---------------------------------------------------------------------------

async def make_decision(
    request: pd.Series,
    financial_state: FinancialState,
    payment_options_df: pd.DataFrame,
    profile: pd.Series,
) -> DecisionResult:
    """
    Produce a complete DecisionResult for one request row.

    Steps:
    1. Compute amount_safe_to_pay.
    2. Find spending changes if needed.
    3. Generate all candidate plans.
    4. Select best plan.
    5. Assemble output fields.
    6. Build the deterministic baseline explanation before required local-Qwen adjudication.
    """
    request_id = str(request["request_id"])
    request_date: date = request["request_date"]
    requested_amount = float(request["requested_amount"])
    desired_completion_date: date = request["desired_completion_date"]
    allows_partial = bool(request.get("allows_partial_payment", False))
    request_text = str(request.get("request_text", ""))
    home_currency = str(profile["home_currency"])

    # ------------------------------------------------------------------ #
    # 1. Baseline safe amount                                              #
    # ------------------------------------------------------------------ #
    amount_safe = financial_state.amount_safe_to_pay(requested_amount, request_date)
    amount_safe = round(min(max(amount_safe, 0.0), requested_amount), 2)

    # ------------------------------------------------------------------ #
    # 2. Spending changes (only if baseline isn't already fully safe)      #
    # ------------------------------------------------------------------ #
    can_pay_full_now = (
        financial_state.balance_after_payment(requested_amount, request_date)
        >= financial_state.minimum_balance
    )
    if not can_pay_full_now:
        stop_ids, reduce_events, state_with_changes = find_spending_changes(
            financial_state, requested_amount, request_date, profile
        )
        has_changes = bool(stop_ids or reduce_events)
    else:
        stop_ids, reduce_events, state_with_changes = [], [], financial_state
        has_changes = False

    # ------------------------------------------------------------------ #
    # 3. Generate candidate plans                                          #
    # ------------------------------------------------------------------ #
    plans, earliest_full = generate_plans(
        request_id=request_id,
        user_id=str(request["user_id"]),
        request_date=request_date,
        requested_amount=requested_amount,
        desired_completion_date=desired_completion_date,
        allows_partial_payment=allows_partial,
        payment_options_df=payment_options_df,
        profile=profile,
        financial_state=financial_state,
        financial_state_with_changes=state_with_changes if has_changes else None,
        spending_changes_available=(stop_ids, reduce_events) if has_changes else None,
    )

    # ------------------------------------------------------------------ #
    # 4. Select best plan                                                  #
    # ------------------------------------------------------------------ #
    best_plan: Optional[PaymentPlan] = None
    if plans:
        # Prefer plans that complete by deadline; fall back to best overall
        deadline_plans = [p for p in plans if p.completes_by_deadline]
        best_plan = deadline_plans[0] if deadline_plans else plans[0]

    # ------------------------------------------------------------------ #
    # 5. Assemble output fields                                            #
    # ------------------------------------------------------------------ #
    if best_plan is None:
        method = "not_recommended"
        affordability = "not_affordable"
        payment_plan_str = "none"
        spending_str = "none"
        earliest_str = ""
    else:
        method = best_plan.method
        affordability = best_plan.affordability_status
        payment_plan_str = best_plan.payment_plan_str
        spending_str = (
            format_spending_changes(stop_ids, reduce_events)
            if best_plan.requires_spending_changes
            else "none"
        )
        # For affordable_now, earliest_date equals request_date
        if affordability == "affordable_now":
            earliest_str = request_date.strftime("%Y-%m-%d")
        elif earliest_full:
            earliest_str = earliest_full.strftime("%Y-%m-%d")
        else:
            earliest_str = ""

    # ------------------------------------------------------------------ #
    # 6. Decision explanation (local Qwen candidate adjudication)              #
    # ------------------------------------------------------------------ #
    candidate_options = []
    for index, plan in enumerate(plans):
        candidate_options.append({
            "candidate_id": plan.payment_option_id or f"{plan.method}_{index}",
            "method": plan.method,
            "affordability_status": plan.affordability_status,
            "payment_plan": plan.payment_plan_str,
            "earliest_date_for_full_payment": plan.earliest_date_str,
            "spending_changes_needed": (
                format_spending_changes(stop_ids, reduce_events)
                if plan.requires_spending_changes else "none"
            ),
            "requires_spending_changes": plan.requires_spending_changes,
            "total_payable": round(plan.total_payable, 2),
            "num_payments": plan.num_payments,
            "completes_by_deadline": plan.completes_by_deadline,
        })

    explanation = _deterministic_explanation(
        method=method,
        affordability=affordability,
        requested_amount=requested_amount,
        amount_safe=amount_safe,
        earliest_str=earliest_str,
        currency=home_currency,
        minimum_balance=float(profile["minimum_balance_to_keep"]),
        spending_str=spending_str,
    )

    return DecisionResult(
        request_id=request_id,
        amount_safe_to_pay=amount_safe,
        affordability_status=affordability,
        recommended_payment_method=method,
        payment_plan=payment_plan_str,
        earliest_date_for_full_payment=earliest_str,
        spending_changes_needed=spending_str,
        decision_explanation=explanation,
        candidate_options=candidate_options,
        llm_context={
            "current_balance": round(float(profile["current_available_balance"]), 2),
            "minimum_balance": round(float(profile["minimum_balance_to_keep"]), 2),
            "home_currency": home_currency,
            "payment_preferences": str(profile.get("payment_methods_user_will_consider", "")),
        },
    )


# ---------------------------------------------------------------------------
# Deterministic fallback explanation
# ---------------------------------------------------------------------------

def _deterministic_explanation(
    method: str,
    affordability: str,
    requested_amount: float,
    amount_safe: float,
    earliest_str: str,
    currency: str,
    minimum_balance: float,
    spending_str: str,
) -> str:
    """
    Generate the deterministic baseline explanation passed to the local-Qwen adjudicator.
    """
    amt   = f"{currency} {requested_amount:,.2f}"
    safe  = f"{currency} {amount_safe:,.2f}"
    minb  = f"{currency} {minimum_balance:,.2f}"

    if method == "full_payment" and affordability == "affordable_now":
        return (
            f"Pay {amt} today. The {minb} minimum balance remains protected "
            "throughout the 90-day forecast."
        )
    if method == "full_payment" and affordability == "affordable_with_plan":
        return (
            f"Apply the spending change ({spending_str}), then pay {amt} today. "
            f"The {minb} minimum is protected."
        )
    if method == "installments":
        return (
            f"Use the installment plan to pay {amt} by the deadline. "
            f"The {minb} minimum balance is protected at every payment."
        )
    if method == "partial_payment":
        return (
            f"Pay {safe} today and the remainder on {earliest_str}. "
            f"This completes {amt} while keeping the {minb} minimum protected."
        )
    if method == "wait":
        return (
            f"Wait until {earliest_str} to pay {amt} in full. "
            f"Paying earlier would breach the {minb} minimum balance."
        )
    # not_recommended
    return (
        f"None of the available payment options keeps the {minb} minimum "
        "balance protected within the forecast period."
    )
