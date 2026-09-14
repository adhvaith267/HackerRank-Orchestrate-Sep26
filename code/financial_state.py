"""
financial_state.py
==================
Reconstructs a user's projected daily cash balance for a 90-day window
starting from request_date and exposes safe-payment queries.

Pipeline
--------
1. Detect strictly periodic recurring expense patterns from settled history
   (minimum 2 occurrences at a consistent interval → project forward).
2. Detect recurring salary/income credit patterns similarly.
3. Add concrete future events (status = pending | scheduled):
   - Debits are reserved immediately.
   - Credits are counted ONLY if event_type is salary or income
     (pending credits such as refunds or bonuses are not counted until settled).
4. For each category NOT already fully covered by description-level recurring
   patterns, compute the 3-month average monthly spend and inject a monthly
   lump-sum projection.  This is critical for high-frequency variable-interval
   categories such as groceries, transport, and dining where individual
   transactions have different descriptions and irregular timing.
5. Project detected strictly-periodic recurring debit patterns forward.
6. Project salary forward using detected patterns, then monthly fallback
   from a single scheduled salary.
7. Apply any message / image amendments (salary overrides, cancellations,
   corrected amounts, pending-credit suppressions).

Key public interface
--------------------
    fs = build_financial_state(user_id, request_date, amended_events)

    fs.amount_safe_to_pay(requested_amount, on_date)
        → float  (binary-search result; never below 0 or above requested_amount)

    fs.earliest_full_payment_date(amount, from_date, to_date)
        → date | None

    fs.with_spending_change(stop_event_ids, reduce_events)
        → new FinancialState

Safety rule
-----------
Balance must NEVER fall below minimum_balance_to_keep at any point in the
90-day window, including on the day of every projected recurring expense.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data_loader import (
    convert_to_home_currency,
    get_user_events,
    get_profile,
    parse_pipe_list,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORECAST_DAYS: int = 90
"""Number of days to project the balance forward from request_date."""

MIN_RECURRENCE_COUNT: int = 2
ESSENTIAL_CATEGORIES = {"groceries", "utilities", "transport", "healthcare", "housing", "rent", "childcare", "debt"}
"""Minimum historical occurrences required to classify an event as recurring."""

INTERVAL_TOLERANCE: int = 5
"""Allowed deviation (days) from average interval to still be called consistent."""

PATTERN_RECENCY_DAYS: int = 60
"""Only project patterns whose last historical occurrence is within this many
days of request_date.  Older patterns may have lapsed and should not be
projected forward into the forecast window."""

MONTHLY_AVG_LOOKBACK_MONTHS: int = 3
"""How many months of settled history to average for category projections (kept
for MonthlyCategory computation used in spending-change suggestions)."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DailyEntry:
    """
    A single signed cash-flow entry on one calendar date.

    amount is in the user's home_currency:
        positive  = inflow  (salary credit, refund that has settled)
        negative  = outflow (expense, debt payment, subscription)
    """
    date: date
    amount: float
    event_id: str
    description: str
    is_recurring: bool      # True when projected by recurrence detection
    category: str
    flexibility: str        # "fixed" | "flexible" | "stoppable"


@dataclass
class RecurringPattern:
    """
    A detected regular expense or income stream.

    Produced by _detect_recurring_patterns() / _detect_salary_patterns().
    Used to generate projected DailyEntry objects for the forecast window.
    """
    description: str
    category: str
    direction: str              # "debit" | "credit"
    avg_amount: float           # average absolute amount across history
    interval_days: int          # snapped to nearest standard period (7/14/30/31…)
    last_date: date             # last confirmed occurrence date
    event_ids: List[str]        # all historical event_id values for this pattern
    flexibility: str            # from the most recent event row
    minimum_allowed_amount: Optional[float]  # minimum if user reduces it
    currency: str               # original currency of the events
    home_currency: str          # user's home currency for conversion


@dataclass
class MonthlyCategory:
    """
    A projection for a variable-interval category.
    Used when description-level recurring patterns undercount a category.
    """
    category: str
    monthly_amount: float       # 3-month avg monthly spend in home currency
    avg_interval_days: int      # average inter-event interval (snapped to 7/14/30/60)
    last_date: date             # last known expense date in category
    event_ids: List[str]        # representative event IDs for spending changes
    flexibility: str            # from the most common flexibility in category
    min_allowed_amount: Optional[float]


@dataclass
class FinancialState:
    """
    90-day projected daily balance for one user as of request_date.

    Build via build_financial_state(); do not construct directly.
    """
    user_id: str
    request_date: date
    home_currency: str
    starting_balance: float
    minimum_balance: float

    # date → list of signed DailyEntry objects (positive = inflow)
    daily_flows: Dict[date, List[DailyEntry]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # date → cumulative balance at end of that day
    daily_balance: Dict[date, float] = field(default_factory=dict)

    # Minimum baseline balance from each forecast date onward.
    suffix_min_balance: Dict[date, float] = field(default_factory=dict)

    # All detected recurring patterns (used for spending-change suggestions)
    recurring_patterns: List[RecurringPattern] = field(default_factory=list)

    # Monthly category projections (for spending changes)
    monthly_categories: List[MonthlyCategory] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Balance series                                                       #
    # ------------------------------------------------------------------ #

    def compute_balance_series(self) -> None:
        """
        Walk forward day by day from request_date, accumulating daily_flows
        to produce daily_balance.  Must be called after modifying daily_flows.
        """
        bal = self.starting_balance
        for offset in range(FORECAST_DAYS + 1):
            d = self.request_date + timedelta(days=offset)
            for entry in self.daily_flows.get(d, []):
                bal += entry.amount
            self.daily_balance[d] = bal

        running_min = float("inf")
        for offset in range(FORECAST_DAYS, -1, -1):
            d = self.request_date + timedelta(days=offset)
            running_min = min(running_min, self.daily_balance[d])
            self.suffix_min_balance[d] = running_min

    def min_balance_in_window(self, from_date: date, to_date: date) -> float:
        """Return the minimum projected balance over [from_date, to_date]."""
        values = [
            self.daily_balance[d]
            for d in self.daily_balance
            if from_date <= d <= to_date
        ]
        return min(values) if values else self.starting_balance

    # ------------------------------------------------------------------ #
    # Safe-payment queries                                                 #
    # ------------------------------------------------------------------ #

    def balance_after_payment(self, amount: float, on_date: date) -> float:
        """
        Return the minimum projected balance over the full 90-day window
        if we make an additional one-time debit of *amount* on *on_date*.

        The debit is applied on on_date and reduces every subsequent day's
        balance by the same amount (it is a point-in-time outflow).
        """
        if on_date <= self.request_date:
            return self.suffix_min_balance.get(self.request_date, self.starting_balance) - amount
        end_date = self.request_date + timedelta(days=FORECAST_DAYS)
        if on_date > end_date:
            return self.suffix_min_balance.get(self.request_date, self.starting_balance)
        before = [self.daily_balance[d] for d in self.daily_balance if d < on_date]
        before_min = min(before) if before else float("inf")
        return min(before_min, self.suffix_min_balance[on_date] - amount)

    def amount_safe_to_pay(self, requested_amount: float, on_date: date) -> float:
        """
        Compute the largest X directly from precomputed suffix minima in [0, requested_amount] such that
        paying X on on_date keeps the minimum projected balance >= minimum_balance
        throughout the full 90-day window.

        Returns 0.0 if even paying nothing keeps the balance below minimum
        (e.g. existing pending debits already breach it).
        Returns requested_amount if the full amount is safe.
        Result is rounded to 2 decimal places.
        """
        if self.balance_after_payment(0.0, on_date) < self.minimum_balance:
            return 0.0
        if on_date <= self.request_date:
            slack = self.suffix_min_balance[self.request_date] - self.minimum_balance
        else:
            end_date = self.request_date + timedelta(days=FORECAST_DAYS)
            if on_date > end_date:
                slack = self.suffix_min_balance[self.request_date] - self.minimum_balance
            else:
                before = [self.daily_balance[d] for d in self.daily_balance if d < on_date]
                slack = min(min(before) if before else float("inf"), self.suffix_min_balance[on_date]) - self.minimum_balance
        return round(min(max(slack, 0.0), requested_amount), 2)

    def earliest_full_payment_date(
        self,
        amount: float,
        from_date: date,
        to_date: date,
    ) -> Optional[date]:
        """
        Return the first date d in [from_date, to_date] where paying *amount*
        on d would keep the projected balance >= minimum_balance.
        Returns None if no such date exists within the window.
        """
        end_cap = self.request_date + timedelta(days=FORECAST_DAYS)
        cap = min(to_date, end_cap)
        d = from_date
        while d <= cap:
            if self.balance_after_payment(amount, d) >= self.minimum_balance:
                # A confirmed credit becomes available on its supplied
                # settlement date; do not add an unsupported extra-day delay.
                return d
            d += timedelta(days=1)
        return None

    # ------------------------------------------------------------------ #
    # Spending-change simulation                                           #
    # ------------------------------------------------------------------ #

    def with_spending_change(
        self,
        stop_event_ids: List[str],
        reduce_events: List[Tuple[str, float]],   # (event_id, new_amount)
    ) -> "FinancialState":
        """
        Return a NEW FinancialState with specified spending changes applied
        to all future (>= request_date) projected flows.

        stop_event_ids:  recurring events to remove entirely from the forecast.
        reduce_events:   recurring events to replace with a lower fixed amount.

        Historical flows (before request_date) are never modified.
        The returned state has compute_balance_series() already called.
        """
        stop_set = set(stop_event_ids)
        reduce_map = dict(reduce_events)

        def matches(target: str, event_id: str) -> bool:
            if event_id == target:
                return True
            for prefix in ("recur_", "category_", "salary_"):
                if event_id.startswith(prefix + target + "_"):
                    return True
            return False

        new_state = FinancialState(
            user_id=self.user_id,
            request_date=self.request_date,
            home_currency=self.home_currency,
            starting_balance=self.starting_balance,
            minimum_balance=self.minimum_balance,
            recurring_patterns=self.recurring_patterns,
            monthly_categories=self.monthly_categories,
        )

        for d, entries in self.daily_flows.items():
            new_entries: List[DailyEntry] = []
            for e in entries:
                if d >= self.request_date and any(matches(target, e.event_id) for target in stop_set):
                    continue  # stopped
                if d >= self.request_date:
                    matching_reduce = next((target for target in reduce_map if matches(target, e.event_id)), None)
                else:
                    matching_reduce = None
                if matching_reduce is not None:
                    sign = -1.0 if e.amount < 0 else 1.0
                    new_entries.append(DailyEntry(
                        date=e.date,
                        amount=sign * reduce_map[matching_reduce],
                        event_id=e.event_id,
                        description=e.description,
                        is_recurring=e.is_recurring,
                        category=e.category,
                        flexibility=e.flexibility,
                    ))
                else:
                    new_entries.append(e)
            new_state.daily_flows[d] = new_entries

        new_state.compute_balance_series()
        return new_state


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_financial_state(
    user_id: str,
    request_date: date,
    amended_events: Optional[Dict] = None,
) -> FinancialState:
    """
    Build a FinancialState for *user_id* evaluated as of *request_date*.

    Parameters
    ----------
    user_id : str
        Must exist in financial_profiles.csv.
    request_date : date
        The date on which the request is being evaluated.
    amended_events : dict, optional
        Overrides from message_parser / image_reader, keyed by event_id:
            {
                "event_id": {
                    "amount": float,        # corrected amount
                    "cancelled": bool,      # remove from forecast
                    "is_pending_credit": bool,  # suppress even if scheduled
                },
                "_salary_override": {
                    "salary_override": float,
                    "salary_date":     str | None,
                },
            }

    Returns
    -------
    FinancialState
        With daily_balance fully populated for [request_date, request_date+90].
    """
    profile = get_profile(user_id)
    if profile is None:
        raise ValueError(f"No financial profile found for user_id='{user_id}'")

    home_currency = str(profile["home_currency"])
    starting_balance = float(profile["current_available_balance"])
    minimum_balance = float(profile["minimum_balance_to_keep"])
    amended_events = amended_events or {}
    end_date = request_date + timedelta(days=FORECAST_DAYS)

    # Pre-compute income-suppression flags so they are available to all steps.
    salary_terminated: bool = bool(amended_events.get("salary_terminated", False))
    suppress_projected_income: bool = bool(
        salary_terminated or amended_events.get("suppress_projected_income", False)
    )

    events_df = get_user_events(user_id)

    state = FinancialState(
        user_id=user_id,
        request_date=request_date,
        home_currency=home_currency,
        starting_balance=starting_balance,
        minimum_balance=minimum_balance,
    )

    # ------------------------------------------------------------------
    # Step 1 – Detect recurring debit patterns from settled history
    # ------------------------------------------------------------------
    state.recurring_patterns = _detect_recurring_patterns(
        events_df, request_date, home_currency
    )

    # ------------------------------------------------------------------
    # Step 2 – Detect recurring salary/income credit patterns
    # ------------------------------------------------------------------
    if salary_terminated:
        salary_patterns = []
    else:
        # Unconfirmed bonuses, prizes, refunds, and commissions must not
        # suppress a separately confirmed recurring salary stream.
        salary_patterns = _detect_salary_patterns(events_df, request_date, home_currency)

    # ------------------------------------------------------------------
    # Step 3 – Apply salary override from messages (if any)
    # ------------------------------------------------------------------
    salary_override_amount: Optional[float] = None
    salary_override_date: Optional[date] = None
    sal_ovr = amended_events.get("_salary_override", {})
    if sal_ovr.get("salary_override"):
        salary_override_amount = float(sal_ovr["salary_override"])
        if sal_ovr.get("salary_date"):
            try:
                from datetime import datetime
                salary_override_date = datetime.strptime(
                    str(sal_ovr["salary_date"]), "%Y-%m-%d"
                ).date()
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Step 4 – Add concrete future events (pending / scheduled)
    # ------------------------------------------------------------------
    future_events = events_df[
        events_df["status"].isin(["pending", "scheduled"])
        & events_df["settlement_date"].notna()
        & (events_df["settlement_date"] > request_date)
        & (events_df["settlement_date"] <= end_date)
    ].copy()

    confirmed_salary_dates: set = set()  # track so we don't double-project salary

    for _, row in future_events.iterrows():
        event_id = str(row["event_id"])
        override = amended_events.get(event_id, {})

        # Skip if cancelled or suppressed by message parser
        if override.get("cancelled") or override.get("is_pending_credit"):
            continue
        if str(row.get("status", "")) in ("cancelled", "failed"):
            continue

        # Resolve amount (use image/message override if original is blank)
        raw_amount = row["amount"]
        if pd.isna(raw_amount):
            raw_amount = override.get("amount")
            if raw_amount is None:
                continue  # truly unknown amount — skip
        else:
            raw_amount = override.get("amount", raw_amount)

        currency = str(row.get("currency", home_currency))
        settle_date: date = row["settlement_date"]
        converted = convert_to_home_currency(raw_amount, currency, home_currency, settle_date)

        direction = str(row.get("direction", "debit"))
        event_type = str(row.get("event_type", ""))

        # Pending credits: only count confirmed salary / income
        if direction == "credit" and event_type not in ("salary", "income"):
            continue  # e.g. pending refunds, bonuses — not safe to count

        signed = -converted if direction == "debit" else converted

        # Apply salary override if this is the salary event
        if direction == "credit" and event_type in ("salary", "income"):
            if salary_override_amount is not None:
                signed = salary_override_amount
            if salary_override_date is not None:
                settle_date = salary_override_date
            confirmed_salary_dates.add(settle_date)

        state.daily_flows[settle_date].append(DailyEntry(
            date=settle_date,
            amount=signed,
            event_id=event_id,
            description=str(row.get("description", "")),
            is_recurring=False,
            category=str(row.get("category", "")),
            flexibility=str(row.get("flexibility", "fixed")),
        ))

    # ------------------------------------------------------------------
    # Step 5 – Project recurring STRICT-INTERVAL debit patterns
    # ------------------------------------------------------------------
    # Track which categories are "fully covered" by strict-interval patterns
    # so we don't double-count them in the monthly-average projection.
    strictly_covered_categories: set = set()

    for pattern in state.recurring_patterns:
        for occ_date in _project_occurrences(
            pattern.last_date, pattern.interval_days, request_date, end_date
        ):
            converted = convert_to_home_currency(
                pattern.avg_amount, pattern.currency, home_currency, occ_date
            )
            signed = -converted if pattern.direction == "debit" else converted
            synth_id = f"recur_{pattern.event_ids[-1]}_{occ_date}"
            state.daily_flows[occ_date].append(DailyEntry(
                date=occ_date,
                amount=signed,
                event_id=synth_id,
                description=pattern.description,
                is_recurring=True,
                category=pattern.category,
                flexibility=pattern.flexibility,
            ))
        strictly_covered_categories.add(pattern.category)

    # ------------------------------------------------------------------
    # Step 6 – Conservative variable-category projection
    # ------------------------------------------------------------------
    monthly_cats = _compute_monthly_category_averages(
        events_df=events_df,
        request_date=request_date,
        home_currency=home_currency,
        lookback_months=MONTHLY_AVG_LOOKBACK_MONTHS,
    )
    state.monthly_categories = monthly_cats
    for cat in monthly_cats:
        if cat.category in strictly_covered_categories:
            continue
        per_event = cat.monthly_amount * cat.avg_interval_days / 30.0
        _inject_category_spend(
            state=state,
            category=cat.category,
            interval_days=cat.avg_interval_days,
            per_interval_amount=per_event,
            base_event_id=cat.event_ids[-1],
            request_date=request_date,
            end_date=end_date,
            flexibility=cat.flexibility,
            description=f"Projected {cat.category} spending",
            prefix="category",
            last_category_date=cat.last_date,
        )

    # ------------------------------------------------------------------
    # Step 7 – Project salary forward
    # ------------------------------------------------------------------
    if not salary_terminated:
        for sal_pattern in salary_patterns:
            for occ_date in _project_occurrences(
                sal_pattern.last_date, sal_pattern.interval_days, request_date, end_date
            ):
                if occ_date in confirmed_salary_dates:
                    continue
                if any(abs((occ_date - d).days) <= 5 for d in confirmed_salary_dates):
                    continue
                amt = salary_override_amount if salary_override_amount else sal_pattern.avg_amount
                converted = convert_to_home_currency(
                    amt, sal_pattern.currency, home_currency, occ_date
                )
                synth_id = f"salary_{sal_pattern.event_ids[-1]}_{occ_date}"
                state.daily_flows[occ_date].append(DailyEntry(
                    date=occ_date,
                    amount=converted,
                    event_id=synth_id,
                    description=sal_pattern.description,
                    is_recurring=True,
                    category=sal_pattern.category,
                    flexibility="fixed",
                ))

        # Fallback: single future salary, project monthly
        if not salary_patterns and confirmed_salary_dates:
            base_date = sorted(confirmed_salary_dates)[0]
            # Find the salary amount from already-added flows
            salary_amount: Optional[float] = None
            for entry in state.daily_flows.get(base_date, []):
                if entry.amount > 0:
                    salary_amount = entry.amount
                    break
            if salary_amount is None and salary_override_amount:
                salary_amount = salary_override_amount
            if salary_amount:
                d = base_date + timedelta(days=30)
                while d <= end_date:
                    if not any(abs((d - s).days) <= 5 for s in confirmed_salary_dates):
                        state.daily_flows[d].append(DailyEntry(
                            date=d,
                            amount=salary_amount,
                            event_id=f"salary_proj_{d}",
                            description="Projected salary",
                            is_recurring=True,
                            category="income",
                            flexibility="fixed",
                        ))
                    d += timedelta(days=30)

    state.compute_balance_series()
    return state


# ---------------------------------------------------------------------------
# Monthly category helpers
# ---------------------------------------------------------------------------

def _compute_monthly_category_averages(
    events_df: pd.DataFrame,
    request_date: date,
    home_currency: str,
    lookback_months: int = 3,
) -> List[MonthlyCategory]:
    """
    For every debit category in the user's settled history (looking back
    `lookback_months` months from request_date), compute the average monthly
    spending amount and the typical spending interval.

    Only categories with at least 2 distinct months of activity are included
    (to avoid projecting one-time large purchases as recurring).

    Returns a list of MonthlyCategory objects.
    """
    cutoff = _months_before(request_date, lookback_months)

    hist = events_df[
        (events_df["status"] == "settled")
        & events_df["settlement_date"].notna()
        & (events_df["settlement_date"] > cutoff)
        & (events_df["settlement_date"] <= request_date)
        & events_df["amount"].notna()
        & (events_df["direction"] == "debit")
        & events_df["category"].notna()
    ].copy()

    if hist.empty:
        return []

    hist["_month"] = hist["settlement_date"].apply(lambda d: (d.year, d.month))
    hist["_converted"] = hist.apply(
        lambda r: convert_to_home_currency(
            float(r["amount"]),
            str(r.get("currency", home_currency)),
            home_currency,
            r["settlement_date"],
        ),
        axis=1,
    )

    result: List[MonthlyCategory] = []

    for cat, grp in hist.groupby(hist["category"].fillna("").str.strip()):
        if not cat:
            continue
        # Monthly totals
        monthly = grp.groupby("_month")["_converted"].sum()
        if len(monthly) < 2:
            continue  # need at least 2 months to project
        # Use a robust baseline for flexible spending. Essential categories
        # receive a conservative upper-quantile reserve so one unusually cheap
        # month does not make a future payment look safe.
        if str(cat).strip().lower() in ESSENTIAL_CATEGORIES:
            avg = float(monthly.quantile(0.75))
        else:
            avg = float(monthly.median())
        if avg <= 0:
            continue

        # Compute average inter-event interval for timing
        grp_sorted = grp.sort_values("settlement_date")
        dates_list = list(grp_sorted["settlement_date"])
        if len(dates_list) >= 2:
            intervals = [(dates_list[i+1] - dates_list[i]).days for i in range(len(dates_list)-1)]
            avg_interval = sum(intervals) / len(intervals)
        else:
            avg_interval = 30.0
        # Snap to standard intervals for clean projection
        # Use same standards as _snap_interval: 7, 14, 30, 60
        snap_standards = [7, 14, 30, 60, 90]
        snapped_interval = min(snap_standards, key=lambda b: abs(b - avg_interval))

        last_row = grp_sorted.iloc[-1]
        event_ids = [str(e) for e in grp_sorted["event_id"].tolist()]

        flex_counts = grp["flexibility"].fillna("fixed").value_counts()
        flex = str(flex_counts.index[0]) if not flex_counts.empty else "fixed"

        min_allowed: Optional[float] = None
        if pd.notna(last_row.get("minimum_allowed_amount")):
            min_allowed = float(last_row["minimum_allowed_amount"])

        result.append(MonthlyCategory(
            category=cat,
            monthly_amount=avg,
            avg_interval_days=snapped_interval,
            last_date=last_row["settlement_date"],
            event_ids=event_ids,
            flexibility=flex,
            min_allowed_amount=min_allowed,
        ))

    return result


def _strict_monthly_projection_for_category(
    state: FinancialState,
    category: str,
    request_date: date,
    end_date: date,
) -> float:
    """
    Compute the average projected monthly debit in a category from the
    already-added strict recurring patterns.
    Returns 0 if no such flows exist.
    """
    total = 0.0
    for d, entries in state.daily_flows.items():
        if d < request_date or d > end_date:
            continue
        for e in entries:
            if e.category == category and e.amount < 0 and e.is_recurring:
                total += abs(e.amount)
    months = FORECAST_DAYS / 30.0
    return total / months if months > 0 else 0.0


def _inject_category_spend(
    state: FinancialState,
    category: str,
    interval_days: int,
    per_interval_amount: float,
    base_event_id: str,
    request_date: date,
    end_date: date,
    flexibility: str,
    description: str,
    prefix: str,
    last_category_date: Optional[date] = None,
) -> None:
    """
    Inject periodic lump-sum debits for a category into state.daily_flows.

    Uses the actual average inter-event interval (7, 14, 30, or 60 days)
    to correctly time the projections. This ensures that weekly transport
    costs are projected weekly, not as a single monthly lump.

    Timing: starts `interval_days` after the last known occurrence.
    """
    if last_category_date and last_category_date <= request_date:
        first = last_category_date + timedelta(days=interval_days)
        # Advance to be strictly after request_date
        while first <= request_date:
            first = _advance_recurrence_date(first, interval_days)
    else:
        first = request_date + timedelta(days=interval_days)

    d = first
    while d <= end_date:
        synth_id = f"{prefix}_{base_event_id}_{d}"
        state.daily_flows[d].append(DailyEntry(
            date=d,
            amount=-per_interval_amount,
            event_id=synth_id,
            description=description,
            is_recurring=True,
            category=category,
            flexibility=flexibility,
        ))
        d = _advance_recurrence_date(d, interval_days)


# Keep old name as alias for backward compatibility
def _inject_monthly_lump(
    state: FinancialState,
    category: str,
    monthly_amount: float,
    base_event_id: str,
    home_currency: str,
    currency: str,
    request_date: date,
    end_date: date,
    flexibility: str,
    description: str,
    prefix: str,
    last_category_date: Optional[date] = None,
) -> None:
    """Wrapper that calls _inject_category_spend with monthly (30-day) intervals."""
    _inject_category_spend(
        state=state,
        category=category,
        interval_days=30,
        per_interval_amount=monthly_amount,
        base_event_id=base_event_id,
        request_date=request_date,
        end_date=end_date,
        flexibility=flexibility,
        description=description,
        prefix=prefix,
        last_category_date=last_category_date,
    )


def _months_before(d: date, months: int) -> date:
    """Return the date `months` calendar months before `d`."""
    y = d.year
    m = d.month - months
    while m <= 0:
        m += 12
        y -= 1
    # Clamp day to valid range for the resulting month
    import calendar
    last_day = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, last_day))


# ---------------------------------------------------------------------------
# Recurrence detection helpers
# ---------------------------------------------------------------------------

def _detect_recurring_patterns(
    events_df: pd.DataFrame,
    as_of_date: date,
    home_currency: str,
) -> List[RecurringPattern]:
    """
    Identify strictly periodic DEBIT expense streams from settled history.

    Grouping key: category + direction + normalised description (first 4 words).
    Qualifying criteria:
      - >= MIN_RECURRENCE_COUNT settled occurrences on or before as_of_date
      - All inter-occurrence intervals within INTERVAL_TOLERANCE + 20% of avg

    These are pure single-description periodic patterns (rent, utilities,
    subscriptions, loan payments).  Variable-category spending (groceries,
    transport) is handled separately by _compute_monthly_category_averages.

    Returns a list of RecurringPattern objects with snapped interval_days.
    """
    hist = events_df[
        (events_df["status"] == "settled")
        & events_df["settlement_date"].notna()
        & (events_df["settlement_date"] <= as_of_date)
        & events_df["amount"].notna()
        & (events_df["direction"] == "debit")
    ].copy()

    if hist.empty:
        return []

    def _norm(desc: str) -> str:
        return " ".join(str(desc).lower().split()[:4])

    hist["_key"] = (
        hist["category"].fillna("").str.strip()
        + "|debit|"
        + hist["description"].fillna("").apply(_norm)
    )

    patterns: List[RecurringPattern] = []
    for key, grp in hist.groupby("_key"):
        grp_sorted = grp.sort_values("settlement_date")
        dates = list(grp_sorted["settlement_date"])
        if len(dates) < MIN_RECURRENCE_COUNT:
            continue

        intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        avg = sum(intervals) / len(intervals)
        tolerance = INTERVAL_TOLERANCE + avg * 0.20
        if not all(abs(iv - avg) <= tolerance for iv in intervals):
            continue

        interval_days = _snap_interval(avg)
        if interval_days <= 0:
            continue

        last_row = grp_sorted.iloc[-1]

        # Recency check: only project patterns whose last occurrence is
        # recent enough to still be active.  Allow 1.5x the interval as
        # grace period (e.g. a monthly expense that was 6 weeks ago is
        # still considered active).
        recency_cutoff = as_of_date - timedelta(days=max(PATTERN_RECENCY_DAYS, interval_days * 2))
        if dates[-1] < recency_cutoff:
            continue

        patterns.append(RecurringPattern(
            description=str(last_row.get("description", "")),
            category=str(key.split("|")[0]),
            direction="debit",
            avg_amount=abs(float(grp_sorted["amount"].mean())),
            interval_days=interval_days,
            last_date=dates[-1],
            event_ids=[str(e) for e in grp_sorted["event_id"].tolist()],
            flexibility=str(last_row.get("flexibility", "fixed")),
            minimum_allowed_amount=(
                float(last_row["minimum_allowed_amount"])
                if pd.notna(last_row.get("minimum_allowed_amount"))
                else None
            ),
            currency=str(last_row.get("currency", home_currency)),
            home_currency=home_currency,
        ))

    return patterns


def _detect_salary_patterns(
    events_df: pd.DataFrame,
    as_of_date: date,
    home_currency: str,
) -> List[RecurringPattern]:
    """
    Identify regular salary / income CREDIT streams from settled history.

    Same method as _detect_recurring_patterns but restricted to
    event_type in ("salary", "income") and direction = "credit".
    Requires >= MIN_RECURRENCE_COUNT settled occurrences.

    Salary projection is SKIPPED for a pattern when its most recent event
    description contains termination keywords (e.g. "Final employer payroll",
    "employment ended").  Such keywords indicate the income stream has stopped
    and should not be extrapolated forward.
    """
    from message_parser import SALARY_TERMINATION_SIGNALS

    hist = events_df[
        (events_df["status"] == "settled")
        & events_df["settlement_date"].notna()
        & (events_df["settlement_date"] <= as_of_date)
        & events_df["amount"].notna()
        & (events_df["direction"] == "credit")
        & events_df["event_type"].isin(["salary", "income"])
    ].copy()
    # Hypothesis: only explicit payroll-like streams are safe to project.
    payroll_text = hist["description"].fillna("").str.lower()
    hist = hist[payroll_text.str.contains("salary|payroll|wage|base pay|monthly pay", regex=True)]


    if hist.empty:
        return []

    def _norm(desc: str) -> str:
        return " ".join(str(desc).lower().split()[:3])

    hist["_key"] = (
        hist["event_type"].fillna("").str.strip()
        + "|credit|"
        + hist["description"].fillna("").apply(_norm)
    )

    patterns: List[RecurringPattern] = []
    for key, grp in hist.groupby("_key"):
        grp_sorted = grp.sort_values("settlement_date")
        dates = list(grp_sorted["settlement_date"])
        if len(dates) < MIN_RECURRENCE_COUNT:
            continue

        intervals = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        avg = sum(intervals) / len(intervals)
        tolerance = INTERVAL_TOLERANCE + avg * 0.20
        if not all(abs(iv - avg) <= tolerance for iv in intervals):
            continue

        interval_days = _snap_interval(avg)
        if interval_days <= 0:
            continue

        last_row = grp_sorted.iloc[-1]

        # Termination check
        last_occurrence_date = dates[-1]
        later_income = events_df[
            events_df["event_type"].isin(["salary", "income"])
            & (events_df["status"] == "settled")
            & events_df["settlement_date"].notna()
            & (events_df["settlement_date"] >= last_occurrence_date)
        ]
        terminated = False
        for _, later_row in later_income.iterrows():
            desc_lower = str(later_row.get("description", "")).lower()
            if any(re.search(p, desc_lower) for p in SALARY_TERMINATION_SIGNALS):
                terminated = True
                break
        if terminated:
            continue

        patterns.append(RecurringPattern(
            description=str(last_row.get("description", "")),
            category=str(last_row.get("category", "income")),
            direction="credit",
            avg_amount=abs(float(last_row["amount"])),
            interval_days=interval_days,
            last_date=dates[-1],
            event_ids=[str(e) for e in grp_sorted["event_id"].tolist()],
            flexibility="fixed",
            minimum_allowed_amount=None,
            currency=str(last_row.get("currency", home_currency)),
            home_currency=home_currency,
        ))
    return patterns


def _snap_interval(days: float) -> int:
    """
    Round an average interval to the nearest standard recurrence period.

    Standard periods: 7 (weekly), 14 (bi-weekly), 28/30/31 (monthly),
                      60 (bi-monthly), 90 (quarterly), 180 (semi-annual),
                      365 (annual).

    Falls back to round(days) if no standard period is within tolerance.
    """
    standards = [7, 14, 28, 30, 31, 60, 90, 180, 365]
    closest = min(standards, key=lambda b: abs(b - days))
    if abs(closest - days) <= INTERVAL_TOLERANCE + days * 0.15:
        return closest
    return max(1, round(days))


def _add_months(d: date, months: int) -> date:
    """Advance a calendar recurrence without drifting its day of month."""
    import calendar

    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def _advance_recurrence_date(d: date, interval_days: int) -> date:
    """Advance standard calendar cadences; keep weekly/irregular ones daily."""
    calendar_months = {
        28: 1, 30: 1, 31: 1,
        60: 2, 90: 3, 180: 6, 365: 12,
    }
    months = calendar_months.get(interval_days)
    return _add_months(d, months) if months is not None else d + timedelta(days=interval_days)


def _project_occurrences(
    last_date: date,
    interval_days: int,
    from_date: date,
    to_date: date,
) -> List[date]:
    """Generate future occurrences using calendar-aware standard cadences."""
    occurrences: List[date] = []
    d = _advance_recurrence_date(last_date, interval_days)
    while d <= to_date:
        if d >= from_date:
            occurrences.append(d)
        d = _advance_recurrence_date(d, interval_days)
    return occurrences
