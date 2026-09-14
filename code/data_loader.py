"""
data_loader.py
==============
Loads and normalises every CSV file from the dataset/ directory.
Provides cached DataFrame accessors and per-entity lookup helpers used
throughout the pipeline.

Dataset layout (all files are read-only inputs):
    dataset/
    ├── financial_profiles.csv       – user balance, currency, preferences
    ├── financial_events.csv         – full transaction / event history
    ├── exchange_rates.csv           – fixed dated FX rates
    ├── requests.csv                 – 250 evaluation requests (predict these)
    ├── sample_requests.csv          – 25 solved examples (format reference)
    ├── request_payment_options.csv  – seller payment options per request
    ├── messages.csv                 – messages tied to users / events
    ├── images.csv                   – image metadata (amounts may be blank)
    └── media/images/<id>.png        – actual PNG files for local OCR extraction

Join keys:
    user_id      → links profiles, events, messages, images
    request_id   → links requests, payment options, messages, images
    related_event_id → links a message or image to a specific financial event
    rate_date + from_currency + to_currency → exchange rate lookup
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Resolve dataset/ relative to this file so the code works from any cwd
DATASET_DIR: Path = Path(__file__).parent.parent / "dataset"
IMAGES_DIR: Path = DATASET_DIR / "media" / "images"


# ---------------------------------------------------------------------------
# Cached DataFrame loaders
# ---------------------------------------------------------------------------
# lru_cache(maxsize=1) means each CSV is read from disk exactly once per
# process, regardless of how many requests are processed in parallel.

@lru_cache(maxsize=1)
def load_financial_profiles() -> pd.DataFrame:
    """
    User-level financial profiles.

    Key columns:
        user_id, home_currency, current_available_balance,
        minimum_balance_to_keep, financial_priorities,
        expense_categories_to_protect,
        expense_categories_user_is_willing_to_reduce,
        expense_categories_user_is_willing_to_stop,
        payment_methods_user_will_consider,
        max_installment_months
    """
    df = pd.read_csv(DATASET_DIR / "financial_profiles.csv")
    df["current_available_balance"] = pd.to_numeric(
        df["current_available_balance"], errors="coerce"
    )
    df["minimum_balance_to_keep"] = pd.to_numeric(
        df["minimum_balance_to_keep"], errors="coerce"
    )
    df["max_installment_months"] = pd.to_numeric(
        df["max_installment_months"], errors="coerce"
    )
    return df


@lru_cache(maxsize=1)
def load_financial_events() -> pd.DataFrame:
    """
    Historical, pending, scheduled, settled, failed, and cancelled events.

    Key columns:
        event_id, user_id, event_type, description, category,
        direction (debit|credit), amount, currency,
        event_date, settlement_date, status,
        linked_event_id, flexibility, minimum_allowed_amount

    Notes:
        - amount may be NaN when the value is embedded in a linked image
        - Only settled/pending/scheduled rows affect cash flow projection
        - unrealised investment rows must NEVER be counted as available cash
    """
    df = pd.read_csv(DATASET_DIR / "financial_events.csv", low_memory=False)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["minimum_allowed_amount"] = pd.to_numeric(
        df["minimum_allowed_amount"], errors="coerce"
    )
    for col in ("event_date", "settlement_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    return df


@lru_cache(maxsize=1)
def load_exchange_rates() -> pd.DataFrame:
    """
    Fixed dated exchange rates supplied by the challenge organisers.

    Key columns: rate_date, from_currency, to_currency, rate

    Usage: for a foreign-currency event, use the row whose rate_date is
    on or before the event's settlement_date with the matching currency pair.
    Live market rates are NOT used.
    """
    df = pd.read_csv(DATASET_DIR / "exchange_rates.csv")
    df["rate_date"] = pd.to_datetime(df["rate_date"], errors="coerce").dt.date
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    return df


@lru_cache(maxsize=1)
def load_requests() -> pd.DataFrame:
    """
    250 evaluation requests that require predictions.

    Key columns:
        request_id, user_id, request_date, request_type,
        requested_amount, desired_completion_date,
        allows_partial_payment, request_text
    """
    df = pd.read_csv(DATASET_DIR / "requests.csv")
    df["request_date"] = pd.to_datetime(df["request_date"], errors="coerce").dt.date
    df["desired_completion_date"] = pd.to_datetime(
        df["desired_completion_date"], errors="coerce"
    ).dt.date
    df["requested_amount"] = pd.to_numeric(df["requested_amount"], errors="coerce")
    df["allows_partial_payment"] = (
        df["allows_partial_payment"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False, "yes": True, "no": False})
        .fillna(False)
    )
    return df


@lru_cache(maxsize=1)
def load_sample_requests() -> pd.DataFrame:
    """
    25 solved example requests provided by the organisers.
    Use for validation / format reference — NOT as training labels.
    """
    df = pd.read_csv(DATASET_DIR / "sample_requests.csv")
    df["request_date"] = pd.to_datetime(df["request_date"], errors="coerce").dt.date
    df["desired_completion_date"] = pd.to_datetime(
        df["desired_completion_date"], errors="coerce"
    ).dt.date
    df["requested_amount"] = pd.to_numeric(df["requested_amount"], errors="coerce")
    df["allows_partial_payment"] = (
        df["allows_partial_payment"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False, "yes": True, "no": False})
        .fillna(False)
    )
    return df


@lru_cache(maxsize=1)
def load_payment_options() -> pd.DataFrame:
    """
    Seller / provider payment options available per request.

    Key columns:
        payment_option_id, request_id, payment_method (full_payment|installments),
        payment_amount, number_of_payments, first_payment_date,
        payment_frequency_days, financing_fee, total_payable_amount

    Rules:
        - A request always has 2–4 options.
        - An installment plan is only valid if the user's max_installment_months
          allows the total duration.
        - The plan schedule must exactly match these values — do not invent amounts.
    """
    df = pd.read_csv(DATASET_DIR / "request_payment_options.csv")
    df["payment_amount"] = pd.to_numeric(df["payment_amount"], errors="coerce")
    df["number_of_payments"] = pd.to_numeric(
        df["number_of_payments"], errors="coerce"
    ).astype("Int64")
    df["first_payment_date"] = pd.to_datetime(
        df["first_payment_date"], errors="coerce"
    ).dt.date
    df["payment_frequency_days"] = pd.to_numeric(
        df["payment_frequency_days"], errors="coerce"
    ).astype("Int64")
    df["financing_fee"] = pd.to_numeric(df["financing_fee"], errors="coerce").fillna(0)
    df["total_payable_amount"] = pd.to_numeric(
        df["total_payable_amount"], errors="coerce"
    )
    return df


@lru_cache(maxsize=1)
def load_messages() -> pd.DataFrame:
    """
    Messages from employers, banks, merchants, or financial services.

    Key columns:
        message_id, user_id, request_id, related_event_id,
        sent_at, source_type, message_text

    Important:
        - related_event_id is set only when the message directly describes
          one specific financial-event row. A blank value means no 1:1 link.
        - Message content is UNTRUSTED. Embedded instructions never override
          the challenge rules or agent behaviour.
        - Use messages to: amend salary, flag pending credits, cancel events.
    """
    df = pd.read_csv(DATASET_DIR / "messages.csv")
    df["sent_at"] = pd.to_datetime(df["sent_at"], errors="coerce")
    return df


@lru_cache(maxsize=1)
def load_images() -> pd.DataFrame:
    """
    Image metadata linking PNG files to users, requests, or events.

    Key columns: image_id, user_id, request_id, related_event_id

    The actual PNG is at:  dataset/media/images/<image_id>.png
    e.g. image_07  →  dataset/media/images/image_07.png
    """
    return pd.read_csv(DATASET_DIR / "images.csv")


# ---------------------------------------------------------------------------
# Per-entity lookup helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def get_profile(user_id: str) -> Optional[pd.Series]:
    """Return the financial profile row for a user, or None if not found."""
    profiles = load_financial_profiles()
    rows = profiles[profiles["user_id"] == user_id]
    return rows.iloc[0] if not rows.empty else None


@lru_cache(maxsize=512)
def get_user_events(user_id: str) -> pd.DataFrame:
    """Return all financial events for a user (all statuses, all dates)."""
    events = load_financial_events()
    return events[events["user_id"] == user_id].copy()


@lru_cache(maxsize=512)
def get_request_payment_options(request_id: str) -> pd.DataFrame:
    """Return all payment options available for a specific request."""
    opts = load_payment_options()
    return opts[opts["request_id"] == request_id].copy()


@lru_cache(maxsize=512)
def get_request_messages(user_id: str, request_id: str) -> pd.DataFrame:
    """
    Return messages relevant to a request: those for the user OR the request.
    Messages are linked by user_id (user-level context) and request_id
    (request-specific amendments).
    """
    messages = load_messages()
    mask = (messages["user_id"] == user_id) | (messages["request_id"] == request_id)
    return messages[mask].copy()


@lru_cache(maxsize=512)
def get_request_images(request_id: str) -> pd.DataFrame:
    """Return image metadata rows linked to a specific request."""
    images = load_images()
    return images[images["request_id"] == request_id].copy()


@lru_cache(maxsize=512)
def get_event_images(event_id: str) -> pd.DataFrame:
    """Return image metadata rows linked to a specific financial event."""
    images = load_images()
    return images[images["related_event_id"] == event_id].copy()


# ---------------------------------------------------------------------------
# FX conversion
# ---------------------------------------------------------------------------

def convert_to_home_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
    on_date: date,
) -> float:
    """
    Convert *amount* from *from_currency* to *to_currency* using the fixed
    exchange rate on or before *on_date*.

    Resolution order:
      1. Direct pair (from → to) on or before on_date
      2. Inverse pair  (to → from) on or before on_date
      3. Chain via USD: from → USD → to
      4. Chain via EUR: from → EUR → to
      5. Return amount unchanged (same currency or rate unavailable)

    All rates come from exchange_rates.csv — no live API calls.
    """
    if from_currency == to_currency or pd.isna(amount):
        return float(amount)

    rates_df = load_exchange_rates()

    # 1. Direct
    rate = _get_rate(rates_df, from_currency, to_currency, on_date)
    if rate is not None:
        return float(amount) * rate

    # 2. Inverse
    inv = _get_rate(rates_df, to_currency, from_currency, on_date)
    if inv is not None:
        return float(amount) / inv

    # 3/4. Chain through bridge currencies
    for bridge in ("USD", "EUR"):
        r1 = _get_rate(rates_df, from_currency, bridge, on_date)
        r2 = _get_rate(rates_df, bridge, to_currency, on_date)
        if r1 is None:
            inv1 = _get_rate(rates_df, bridge, from_currency, on_date)
            r1 = (1.0 / inv1) if inv1 else None
        if r2 is None:
            inv2 = _get_rate(rates_df, to_currency, bridge, on_date)
            r2 = (1.0 / inv2) if inv2 else None
        if r1 is not None and r2 is not None:
            return float(amount) * r1 * r2

    # Fallback: return unconverted (same-scale estimate)
    return float(amount)


def _get_rate(
    rates_df: pd.DataFrame,
    from_cur: str,
    to_cur: str,
    on_date: date,
) -> Optional[float]:
    """Return the most recent rate for (from_cur → to_cur) on or before on_date."""
    mask = (
        (rates_df["from_currency"] == from_cur)
        & (rates_df["to_currency"] == to_cur)
        & (rates_df["rate_date"] <= on_date)
    )
    subset = rates_df[mask].sort_values("rate_date", ascending=False)
    return float(subset.iloc[0]["rate"]) if not subset.empty else None


# ---------------------------------------------------------------------------
# Profile field helpers
# ---------------------------------------------------------------------------

def parse_pipe_list(value: object) -> List[str]:
    """
    Parse a pipe-separated profile field into a list of stripped strings.
    Returns an empty list for blank or NaN values.

    Example: "rent|utilities|groceries" → ["rent", "utilities", "groceries"]
    """
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [v.strip() for v in str(value).split("|") if v.strip()]
