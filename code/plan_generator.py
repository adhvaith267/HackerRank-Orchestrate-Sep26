"""
plan_generator.py
=================
Enumerates, validates, and ranks every candidate payment plan for a request.

Plans considered
----------------
1. full_payment
   Pay requested_amount in full on request_date.

2. installments
   Follow an exact schedule from request_payment_options.csv.
   One PaymentPlan is created per installment option that passes validation.

3. partial_payment
   Pay amount_safe_to_pay today, remainder on earliest_full_payment_date.
   Only valid when:
     • allows_partial_payment = True
     • user accepts partial_payment in payment_methods_user_will_consider
     • 0 < amount_safe_to_pay < requested_amount
     • remainder date ≤ desired_completion_date
   Two payments that sum to requested_amount exactly.

4. wait
   Pay requested_amount in full on earliest_full_payment_date.
   Only recommended when earliest_full_payment_date ≤ desired_completion_date
   and the user accepts full_payment.

Validation rules
----------------
- Method must be in payment_methods_user_will_consider.
- Installment duration must not exceed max_installment_months.
- Last installment date must be ≤ desired_completion_date.
- Every payment date within the 90-day window must individually pass
  the 90-day balance safety check (balance ≥ minimum_balance_to_keep).

Tiebreaker ranking (lower = better)
------------------------------------
Per the challenge spec (§ "Choosing Between Safe Plans"):
1. Completes request by desired_completion_date
2. Requires no spending changes
3. Minimises total amount paid
4. Starts earlier
5. Fewer payments
6. Lowest payment_option_id (lexicographic)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Optional, Tuple

import pandas as pd

from data_loader import parse_pipe_list
from financial_state import FinancialState, FORECAST_DAYS


# ---------------------------------------------------------------------------
# PaymentPlan dataclass
# ---------------------------------------------------------------------------

@dataclass
class PaymentPlan:
    """
    A fully specified payment recommendation.

    payment_schedule  : list of (date, amount) tuples in chronological order
    payment_option_id : source option id (or synthetic key for non-option plans)
    """
    method: str                               # full_payment | partial_payment | installments | wait
    affordability_status: str                 # affordable_now | affordable_with_plan | affordable_later
    payment_schedule: List[Tuple[date, float]] = field(default_factory=list)
    earliest_full_payment_date: Optional[date] = None
    payment_option_id: str = ""               # used as final tiebreaker
    total_payable: float = 0.0
    num_payments: int = 0
    completes_by_deadline: bool = False
    requires_spending_changes: bool = False

    # ------------------------------------------------------------------ #
    # Convenience properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def payment_plan_str(self) -> str:
        """
        Format: "YYYY-MM-DD:amount|YYYY-MM-DD:amount"
        Amounts are rounded to 2 decimal places.
        Returns "none" when the schedule is empty.
        """
        if not self.payment_schedule:
            return "none"
        return "|".join(
            f"{d.strftime('%Y-%m-%d')}:{_format_payment_amount(a)}"
            for d, a in self.payment_schedule
        )

    @property
    def earliest_date_str(self) -> str:
        """YYYY-MM-DD string or empty string."""
        return (
            self.earliest_full_payment_date.strftime("%Y-%m-%d")
            if self.earliest_full_payment_date
            else ""
        )

    @property
    def start_date(self) -> Optional[date]:
        return self.payment_schedule[0][0] if self.payment_schedule else None

    def rank_key(self) -> tuple:
        """
        Lower value = better plan.  Sorting by this key applies all 6 tiebreakers.
        """
        return (
            0 if self.completes_by_deadline else 1,
            0 if not self.requires_spending_changes else 1,
            round(self.total_payable, 2),
            self.start_date or date(9999, 1, 1),
            self.num_payments,
            self.payment_option_id or "zzz",
        )



def _format_payment_amount(amount: float) -> str:
    """Format plan amounts without noisy .0 while preserving cents."""
    value = round(float(amount), 2)
    return str(int(value)) if value.is_integer() else f"{value:.2f}"


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_plans(
    request_id: str,
    user_id: str,
    request_date: date,
    requested_amount: float,
    desired_completion_date: date,
    allows_partial_payment: bool,
    payment_options_df: pd.DataFrame,
    profile: pd.Series,
    financial_state: FinancialState,
    financial_state_with_changes: Optional[FinancialState] = None,
    spending_changes_available: Optional[Tuple] = None,
) -> Tuple[List[PaymentPlan], Optional[date]]:
    """
    Generate all valid payment plans for a request and return them ranked.

    Parameters
    ----------
    financial_state              : baseline state (no spending changes)
    financial_state_with_changes : state after applying spending changes
                                   (None if no beneficial changes were found)
    spending_changes_available   : (stop_ids, reduce_events) tuple used to
                                   mark plans that require spending changes

    Returns
    -------
    (plans, earliest_full_payment_date)
        plans : list of PaymentPlan sorted by rank_key() ascending (best first)
        earliest_full_payment_date : date | None (financial capacity, independent
                                     of method preference)
    """
    home_currency = str(profile["home_currency"])
    methods_ok = set(parse_pipe_list(
        profile.get("payment_methods_user_will_consider", "")
    ))
    max_months_raw = profile.get("max_installment_months")
    max_months: Optional[int] = (
        int(max_months_raw) if pd.notna(max_months_raw) and max_months_raw else None
    )

    end_date = request_date + timedelta(days=FORECAST_DAYS)
    plans: List[PaymentPlan] = []

    # ------------------------------------------------------------------
    # Compute earliest_full_payment_date (financial capacity measure)
    # Independent of payment method preferences.
    # ------------------------------------------------------------------
    earliest_full = financial_state.earliest_full_payment_date(
        requested_amount,
        from_date=request_date,
        to_date=min(desired_completion_date, end_date),
    )
    if earliest_full is None:
        # Extend beyond desired_completion_date within 90-day window
        earliest_full = financial_state.earliest_full_payment_date(
            requested_amount,
            from_date=request_date,
            to_date=end_date,
        )

    # ------------------------------------------------------------------
    # 1. full_payment
    # ------------------------------------------------------------------
    if "full_payment" in methods_ok:
        can_pay_now = (
            financial_state.balance_after_payment(requested_amount, request_date)
            >= financial_state.minimum_balance
        )
        if can_pay_now:
            plans.append(PaymentPlan(
                method="full_payment",
                affordability_status="affordable_now",
                payment_schedule=[(request_date, requested_amount)],
                earliest_full_payment_date=request_date,
                total_payable=requested_amount,
                num_payments=1,
                completes_by_deadline=(request_date <= desired_completion_date),
                requires_spending_changes=False,
                payment_option_id="aaa_full",   # sort before installments
            ))
        elif financial_state_with_changes is not None:
            # Try with spending changes
            can_with_changes = (
                financial_state_with_changes.balance_after_payment(
                    requested_amount, request_date
                )
                >= financial_state_with_changes.minimum_balance
            )
            if can_with_changes:
                plans.append(PaymentPlan(
                    method="full_payment",
                    affordability_status="affordable_with_plan",
                    payment_schedule=[(request_date, requested_amount)],
                    earliest_full_payment_date=request_date,
                    total_payable=requested_amount,
                    num_payments=1,
                    completes_by_deadline=(request_date <= desired_completion_date),
                    requires_spending_changes=True,
                    payment_option_id="aaa_full",
                ))

    # ------------------------------------------------------------------
    # 2. installments
    # ------------------------------------------------------------------
    if "installments" in methods_ok:
        installment_opts = payment_options_df[
            payment_options_df["payment_method"] == "installments"
        ]
        for _, opt in installment_opts.iterrows():
            # Try baseline state
            plan = _try_installment_plan(
                opt=opt,
                request_date=request_date,
                desired_completion_date=desired_completion_date,
                financial_state=financial_state,
                max_months=max_months,
                end_date=end_date,
            )
            if plan:
                plan.earliest_full_payment_date = earliest_full
                plans.append(plan)
            elif financial_state_with_changes is not None:
                # Try with spending changes
                plan_ch = _try_installment_plan(
                    opt=opt,
                    request_date=request_date,
                    desired_completion_date=desired_completion_date,
                    financial_state=financial_state_with_changes,
                    max_months=max_months,
                    end_date=end_date,
                )
                if plan_ch:
                    plan_ch.requires_spending_changes = True
                    plan_ch.affordability_status = "affordable_with_plan"
                    plan_ch.earliest_full_payment_date = earliest_full
                    plans.append(plan_ch)

    # ------------------------------------------------------------------
    # 3. partial_payment
    # ------------------------------------------------------------------
    if "partial_payment" in methods_ok and allows_partial_payment:
        amount_safe = financial_state.amount_safe_to_pay(requested_amount, request_date)
        if 0.0 < amount_safe < requested_amount:
            remainder = round(requested_amount - amount_safe, 2)
            # Find earliest date remainder can be paid (after paying amount_safe today)
            rem_date = _find_remainder_date(
                financial_state=financial_state,
                first_payment=amount_safe,
                first_date=request_date,
                remainder=remainder,
                from_date=request_date + timedelta(days=1),
                to_date=min(desired_completion_date, end_date),
            )
            if rem_date and rem_date <= desired_completion_date:
                plans.append(PaymentPlan(
                    method="partial_payment",
                    affordability_status="affordable_with_plan",
                    payment_schedule=[
                        (request_date, amount_safe),
                        (rem_date, remainder),
                    ],
                    earliest_full_payment_date=earliest_full,
                    total_payable=requested_amount,   # no financing fee
                    num_payments=2,
                    completes_by_deadline=(rem_date <= desired_completion_date),
                    requires_spending_changes=False,
                    payment_option_id="bbb_partial",
                ))

    # ------------------------------------------------------------------
    # 4. wait
    # ------------------------------------------------------------------
    if "full_payment" in methods_ok and earliest_full and earliest_full > request_date:
        plans.append(PaymentPlan(
            method="wait",
            affordability_status="affordable_later",
            payment_schedule=[(earliest_full, requested_amount)],
            earliest_full_payment_date=earliest_full,
            total_payable=requested_amount,
            num_payments=1,
            completes_by_deadline=(earliest_full <= desired_completion_date),
            requires_spending_changes=False,
            payment_option_id="ccc_wait",
        ))

    # Sort: best plan first
    plans.sort(key=lambda p: p.rank_key())
    return plans, earliest_full


# ---------------------------------------------------------------------------
# Installment plan validation
# ---------------------------------------------------------------------------

def _try_installment_plan(
    opt: pd.Series,
    request_date: date,
    desired_completion_date: date,
    financial_state: FinancialState,
    max_months: Optional[int],
    end_date: date,
) -> Optional[PaymentPlan]:
    """
    Validate one installment option against the user's constraints and
    the 90-day balance safety check.

    Returns a PaymentPlan on success, None on any constraint violation.
    """
    n = int(opt["number_of_payments"])
    first_date: date = opt["first_payment_date"]
    freq = int(opt["payment_frequency_days"]) if pd.notna(opt["payment_frequency_days"]) else 30
    per_payment = float(opt["payment_amount"])
    total = float(opt["total_payable_amount"])
    option_id = str(opt["payment_option_id"])

    # Constraint: max_installment_months
    if max_months is not None:
        total_months = ((n - 1) * freq) / 30.0
        if total_months > max_months:
            return None

    # Build exact schedule
    schedule: List[Tuple[date, float]] = []
    d = first_date
    for _ in range(n):
        schedule.append((d, per_payment))
        d += timedelta(days=freq)

    # Last payment must be on or before desired_completion_date
    if schedule[-1][0] > desired_completion_date:
        return None

    # Validate the complete schedule on the actual timeline. Earlier
    # installments remain debited, but intervening income is preserved.
    if _schedule_min_balance(financial_state, schedule, end_date) < financial_state.minimum_balance:
        return None

    last_date = schedule[-1][0]
    return PaymentPlan(
        method="installments",
        affordability_status="affordable_with_plan",
        payment_schedule=schedule,
        total_payable=total,
        num_payments=n,
        completes_by_deadline=(last_date <= desired_completion_date),
        requires_spending_changes=False,
        payment_option_id=option_id,
    )


def _schedule_min_balance(financial_state: FinancialState, schedule: List[Tuple[date, float]], end_date: date) -> float:
    """Minimum balance after applying every scheduled payment cumulatively."""
    payments = {}
    for pay_date, pay_amount in schedule:
        if financial_state.request_date <= pay_date <= end_date:
            payments[pay_date] = payments.get(pay_date, 0.0) + float(pay_amount)
    running = 0.0
    minimum = float("inf")
    for offset in range(FORECAST_DAYS + 1):
        d = financial_state.request_date + timedelta(days=offset)
        running += payments.get(d, 0.0)
        minimum = min(minimum, financial_state.daily_balance.get(d, financial_state.starting_balance) - running)
    return minimum


# ---------------------------------------------------------------------------
# Partial payment helpers
# ---------------------------------------------------------------------------

def _find_remainder_date(
    financial_state: FinancialState,
    first_payment: float,
    first_date: date,
    remainder: float,
    from_date: date,
    to_date: date,
) -> Optional[date]:
    """
    Find the earliest date ≥ from_date where, having already committed
    first_payment on first_date, the remainder can also be paid safely.
    """
    end_cap = financial_state.request_date + timedelta(days=FORECAST_DAYS)
    cap = min(to_date, end_cap)
    d = from_date
    while d <= cap:
        if _two_payment_min_balance(
            financial_state, first_payment, first_date, remainder, d
        ) >= financial_state.minimum_balance:
            return d
        d += timedelta(days=1)
    return None


def _two_payment_min_balance(
    fs: FinancialState,
    amt1: float,
    date1: date,
    amt2: float,
    date2: date,
) -> float:
    """
    Compute the minimum projected balance over the 90-day window when
    two separate debits (amt1 on date1, amt2 on date2) are applied.
    """
    min_bal = float("inf")
    extra = 0.0
    for offset in range(FORECAST_DAYS + 1):
        d = fs.request_date + timedelta(days=offset)
        if d == date1:
            extra -= amt1
        if d == date2:
            extra -= amt2
        projected = fs.daily_balance.get(d, fs.starting_balance) + extra
        min_bal = min(min_bal, projected)
    return min_bal
