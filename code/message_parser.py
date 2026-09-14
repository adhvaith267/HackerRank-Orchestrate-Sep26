"""
message_parser.py
=================
Extracts structured financial amendments from free-text messages in
messages.csv using a two-stage approach:

Stage 1 — Fast heuristic (no LLM cost)
    Regex-based patterns for the most common amendment signals:
      - Salary/pay changes from employer messages (with currency anchor)
      - Employment/salary termination signals
      - Pending/unsettled credits (refunds, prizes, commissions, gig income)
      - Suppress-all-projected-income signals from service providers
      - Internal account transfers (should not be double-counted as income)
      - Explicit cancellations

Stage 2 — Conservative no-op fallback
    Ambiguous messages are ignored rather than allowing an LLM to invent a
    financial amendment.

Amendment keys returned
-----------------------
    {
        "salary_override"               : float | None    – new confirmed salary amount
        "salary_date"                   : str   | None    – "YYYY-MM-DD" new salary date
        "salary_terminated"             : bool            – income stream has ended
        "suppress_projected_income"     : bool            – suppress ALL projected gig/irregular income
        "cancelled_events"              : set[str]        – event_ids explicitly cancelled
        "amount_overrides"              : {event_id: float}  – corrected event amounts
        "pending_credits"               : set[str]        – event_ids not yet settled
        "suppress_credit_event"         : str | None      – event_id to suppress (own-transfer etc.)
    }

Safety note
-----------
Message content is treated as UNTRUSTED DATA.  Embedded instructions
("ignore previous instructions", "you are now…", etc.) are never followed.
The parser only extracts the specific financial fields listed above.
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Dict, Optional, Set

import pandas as pd

from data_loader import get_request_messages


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def parse_messages(
    user_id: str,
    request_id: str,
    request_date: date,
) -> Dict[str, Any]:
    """
    Parse all messages relevant to a user/request and return merged amendments.

    Messages are processed oldest-first so that newer records override older
    ones (consistent with the challenge conflict-resolution rules).

    Parameters
    ----------
    user_id      : str
    request_id   : str
    request_date : date  Used to ignore messages dated after the request.

    Returns
    -------
    dict with amendment keys (see module docstring)
    """
    messages_df = get_request_messages(user_id, request_id)
    if messages_df.empty:
        return _empty_amendments()

    # Only evidence available on or before the request date may amend state.
    sent_dates = pd.to_datetime(messages_df["sent_at"], errors="coerce").dt.date
    messages_df = messages_df[sent_dates.notna() & (sent_dates <= request_date)]
    if messages_df.empty:
        return _empty_amendments()

    # Sort ascending by sent_at so newer messages win on conflict
    messages_df = messages_df.sort_values("sent_at", ascending=True)

    amendments = _empty_amendments()

    for _, row in messages_df.iterrows():
        msg_text = str(row.get("message_text", "")).strip()
        if not msg_text:
            continue

        # Stage 1: try fast heuristic
        parsed = _heuristic_parse(row, msg_text)

        # Stage 2: conservative no-op fallback for unmatched messages.
        # If parsed is None (heuristic returned no match), treat as empty {}
        # rather than inferring unsupported facts, which can introduce wrong amendments.
        if parsed is None:
            parsed = {}  # no-op: message not matched by heuristic

        if parsed:
            _merge(amendments, parsed, row)

    return amendments


# ---------------------------------------------------------------------------
# Heuristic parser — no LLM cost
# ---------------------------------------------------------------------------

# All currencies used in this dataset (from problem_statement.md / exchange_rates.csv).
# A salary amount extracted by the heuristic MUST be preceded by one of these
# currency codes (or common symbols).  Without a currency anchor the extracted
# digits could be a year, a reference number, or any other non-monetary value,
# so the parser rejects the value rather than inferring an unsupported amendment.
_KNOWN_CURRENCIES = r"(?:INR|ZAR|IDR|USD|EUR|Rs\.?|Rp\.?|\$|€|R\b)"

# Salary amount patterns: require a known currency code immediately before digits
_SALARY_AMOUNT_PATTERNS = [
    # English — currency then amount, e.g. "salary of INR 42750" / "EUR 2717"
    rf"{_KNOWN_CURRENCIES}\s*([\d,]+(?:\.\d+)?)",
    # Indonesian — "gaji … IDR 42750000" / "naik menjadi IDR 42750000"
    rf"(?:gaji|penghasilan|naik menjadi|menjadi|adalah)[^0-9]{{0,30}}{_KNOWN_CURRENCIES}\s*([\d,]+(?:\.\d+)?)",
]

# Patterns for a YYYY-MM-DD date
_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")

# Signals that a credit is mentioned but NOT yet settled (has related_event_id)
_PENDING_CREDIT_SIGNALS = [
    r"(?:pending|not yet|hasn.t reached|will confirm|processing|not credited|in payment processing)",
    r"(?:menunggu|belum|masih menunggu|diproses)",
]

# Signals that a SERVICE PROVIDER's next payout is pending (no related_event_id)
# → suppress all projected gig/irregular income for this user
_PENDING_PAYOUT_SIGNALS = [
    r"(?:payout.*(?:still\s+)?pending|payment.*not\s+(?:yet\s+)?(?:credited|withdrawable|settled))",
    r"(?:earnings.*(?:still\s+)?pending|not\s+withdrawable)",
    r"(?:balance\s+isn.t\s+withdrawable|cannot\s+be\s+withdrawn)",
    r"(?:prize\s+claim.*(?:still\s+in|payment\s+processing))",
    r"(?:payment\s+has\s+not\s+been\s+credited)",
]

# Signals of an explicit cancellation or reversal
_CANCEL_SIGNALS = [
    r"(?:cancel|cancelled|void|reversed|refund initiated)",
    r"(?:dibatalkan|dikembalikan)",
]

# Signals that employment / salary has terminated — no future income projected.
# Shared with financial_state.py (imported there by name).
SALARY_TERMINATION_SIGNALS = [
    r"(?:employment has ended|contract has ended|seasonal contract has ended)",
    r"(?:no (?:further|regular|ongoing|off-?set) (?:salary|pay|income|payments?))",
    r"(?:hubungan kerja.*berakhir|tidak ada.*gaji|tidak ada.*pendapatan)",
    r"(?:final.*payroll|last.*payroll|payroll.*final|payroll.*last)",
    r"(?:separated from|termination of employment|position has been eliminated)",
    r"(?:your employment has ended|there are no regular|no ongoing salary)",
]

# Signals that a bank/service message describes an internal own-account transfer
# The credit in that transfer should NOT count as income.
_OWN_TRANSFER_SIGNALS = [
    r"(?:transfer between your (?:two )?accounts|both accounts.*same account holder)",
    r"(?:internal transfer|own.account transfer|same account holder)",
    r"(?:matching debit and credit.*transfer)",
]

# Signals that a bonus/commission is NOT yet confirmed/pending performance review
_PENDING_BONUS_SIGNALS = [
    r"(?:bonus.*(?:still\s+)?pending|bonus.*not\s+(?:yet\s+)?(?:approved|confirmed|determined))",
    r"(?:bonus.*menunggu|bonus.*belum)",
    r"(?:awaiting.*performance|pending.*final\s+(?:assessment|review|approval))",
    r"(?:jumlah akhir.*belum disetujui|tanggal pembayaran belum)",
]


def _heuristic_parse(row: pd.Series, msg_text: str) -> Optional[Dict]:
    """
    Parse a message with conservative, currency-anchored regex rules.

    Returns a parsed dict or None if the message is ambiguous (ignored safely).

    Key rules:
    - Salary amount extraction REQUIRES a known currency code immediately before
      the digits to avoid capturing years, percentages, or reference numbers.
    - A date-only salary update (no amount) is still useful for adjusting timing.
    - Service provider / bank messages without a related_event_id can still signal
      that projected income should be suppressed.
    """
    text_lower = msg_text.lower()
    source = str(row.get("source_type", "")).lower()
    result: Dict = {}

    related = (
        str(row["related_event_id"])
        if pd.notna(row.get("related_event_id"))
        else ""
    )

    # ------------------------------------------------------------------
    # 1. Own-account transfer (bank) — suppress the credit side
    # ------------------------------------------------------------------
    if source == "bank":
        if any(re.search(p, text_lower) for p in _OWN_TRANSFER_SIGNALS):
            # The related event (credit) should not be counted as income
            if related:
                return {"suppress_credit_event": related}
            return {"suppress_projected_income": False}  # no-op; no linked event is available to suppress

    # ------------------------------------------------------------------
    # 2. Pending service/gig payout (service_provider / financial_service)
    # ------------------------------------------------------------------
    if source in ("service_provider", "financial_service"):
        # 2a. Prize / payout pending (no event link) → suppress projected gig income
        if any(re.search(p, text_lower) for p in _PENDING_PAYOUT_SIGNALS):
            if related:
                return {"is_pending_credit": True, "_event_id": related}
            return {"suppress_projected_income": True}

        # 2b. Prize received / settled → the related event is confirmed income
        # (no amendment needed; income already settled in the events CSV)
        settled_patterns = [
            r"(?:prize proceeds.*reached your account|claim is now closed)",
            r"(?:payment.*credited|payout.*completed|funds.*deposited)",
        ]
        if any(re.search(p, text_lower) for p in settled_patterns):
            # No suppression needed; just note nothing is pending
            return {}

    # ------------------------------------------------------------------
    # 3. Pending credit (any source with related_event_id)
    # ------------------------------------------------------------------
    if any(re.search(p, text_lower) for p in _PENDING_CREDIT_SIGNALS):
        if related:
            return {"is_pending_credit": True, "_event_id": related}

    # ------------------------------------------------------------------
    # 4. Employer / payroll amendments
    # ------------------------------------------------------------------
    if source in ("employer", "payroll"):

        # 4a. Employment / salary termination — suppress all future income
        if any(re.search(p, text_lower) for p in SALARY_TERMINATION_SIGNALS):
            return {"salary_terminated": True}

        # 4b. Bonus pending / not yet confirmed → suppress bonus projection
        if any(re.search(p, text_lower) for p in _PENDING_BONUS_SIGNALS):
            return {"suppress_projected_income": True}

        # 4c. Salary amount — only accept when preceded by a known currency code
        for pat in _SALARY_AMOUNT_PATTERNS:
            m = re.search(pat, msg_text, re.IGNORECASE)
            if m:
                try:
                    amount = float(m.group(m.lastindex).replace(",", ""))
                    if amount >= 100:
                        result["salary_override"] = amount
                except (ValueError, IndexError):
                    pass
                break  # only use the first matching pattern

        # 4d. Salary date — any YYYY-MM-DD in the message
        dm = _DATE_PATTERN.search(msg_text)
        if dm:
            result["salary_date"] = dm.group(1)

        if result:
            return result

    # ------------------------------------------------------------------
    # 5. Explicit cancellation
    # ------------------------------------------------------------------
    if any(re.search(p, text_lower) for p in _CANCEL_SIGNALS):
        if related:
            return {"cancelled_event_id": related}

    return None   # no supported amendment found


# ---------------------------------------------------------------------------
# Amendment merger
# ---------------------------------------------------------------------------

def _empty_amendments() -> Dict[str, Any]:
    return {
        "salary_override":           None,
        "salary_date":               None,
        "salary_terminated":         False,
        "suppress_projected_income": False,
        "cancelled_events":          set(),
        "amount_overrides":          {},
        "pending_credits":           set(),
        "suppressed_credits":        set(),
    }


def _merge(base: Dict, parsed: Dict, row: pd.Series) -> None:
    """
    Merge a single parsed dict into the accumulating amendments.
    Validates salary_override to reject clearly wrong values (< 100).
    """
    related = (
        str(row["related_event_id"])
        if pd.notna(row.get("related_event_id"))
        else ""
    )

    # Employment termination — mark salary as stopped
    if parsed.get("salary_terminated"):
        base["salary_terminated"] = True
        return

    # Suppress all projected income (pending bonus, gig payout pending, etc.)
    if parsed.get("suppress_projected_income"):
        base["suppress_projected_income"] = True

    # Suppress a specific credit event (own-account transfer)
    if parsed.get("suppress_credit_event"):
        base["suppressed_credits"].add(str(parsed["suppress_credit_event"]))

    if "salary_override" in parsed and parsed["salary_override"]:
        val = float(parsed["salary_override"])
        if val >= 100:
            base["salary_override"] = val

    if "salary_date" in parsed and parsed["salary_date"]:
        base["salary_date"] = str(parsed["salary_date"])

    if "cancelled_event_id" in parsed:
        base["cancelled_events"].add(str(parsed["cancelled_event_id"]))

    if "amount_override" in parsed and related:
        base["amount_overrides"][related] = float(parsed["amount_override"])

    if parsed.get("is_pending_credit"):
        ev = parsed.get("_event_id") or related
        if ev:
            base["pending_credits"].add(ev)


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[Dict]:
    """
    Attempt to parse a JSON object from an LLM response string.
    Tries: bare JSON → markdown code block → first {...} substring.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m2 = re.search(r"\{[\s\S]*\}", text)
    if m2:
        try:
            return json.loads(m2.group(0))
        except json.JSONDecodeError:
            pass
    return None
