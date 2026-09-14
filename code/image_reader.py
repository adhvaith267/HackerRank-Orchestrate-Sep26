"""Local Tesseract OCR for blank financial-event amounts."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from data_loader import DATASET_DIR, get_event_images

IMAGES_DIR = DATASET_DIR / "media" / "images"
_NUMBER_RE = re.compile(r"(?<!\w)(?:[A-Z]{2,4}\s*)?[$€₹]?\s*(\d[\d,\s]*(?:\.\d+)?)")
_GENERIC_TERMS = ("total", "amount due", "net pay", "salary", "gross", "payable", "payment", "invoice", "receipt", "paid", "balance due")


def image_path_for(image_id: str) -> Path:
    return IMAGES_DIR / f"{image_id}.png"


def _preprocess(path: Path) -> Image.Image:
    image = Image.open(path).convert("L")
    image = image.resize((image.width * 2, image.height * 2))
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Sharpness(image).enhance(1.5)
    return image.filter(ImageFilter.SHARPEN)


@lru_cache(maxsize=None)
def extract_image_text(image_id: str) -> str:
    path = image_path_for(image_id)
    if not path.exists():
        return ""
    try:
        image = _preprocess(path)
        texts = [pytesseract.image_to_string(image, config="--oem 3 --psm 6")]
        # Receipts often place the total in a small, low-contrast footer.
        footer = image.crop((image.width // 3, image.height * 2 // 3, image.width, image.height))
        texts.append(pytesseract.image_to_string(footer, config="--oem 3 --psm 11"))
        return "\n".join(texts)
    except pytesseract.TesseractNotFoundError:
        return ""


_WORD_VALUES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000, "lakh": 100000, "million": 1000000}


def _words_value(text: str) -> Optional[float]:
    total = current = 0
    found = False
    for token in re.findall(r"[a-z]+", text.lower()):
        if token in {"and", "rupees", "paisa", "only"}:
            continue
        if token not in _WORD_VALUES:
            continue
        found = True
        value = _WORD_VALUES[token]
        if value in (100, 1000, 100000, 1000000):
            current = max(current, 1) * value
            if value >= 1000:
                total += current
                current = 0
        else:
            current += value
    return float(total + current) if found else None


def _value(raw: str) -> Optional[float]:
    cleaned = raw.replace(" ", "")
    if "." not in cleaned and "," in cleaned and len(cleaned.rsplit(",", 1)[1]) == 2:
        cleaned = cleaned.replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if value <= 0 or (value.is_integer() and 1900 <= value <= 2100):
        return None
    return value


def parse_primary_amount(text: str, preferred_terms: Iterable[str] = ()) -> Tuple[Optional[float], float]:
    terms = tuple(term.lower() for term in preferred_terms)
    candidates = []
    lowered_text = text.lower()
    word_match = re.search(r"amount in (.+?) rupees", lowered_text, re.S)
    if word_match:
        word_amount = _words_value(word_match.group(1))
        if word_amount:
            candidates.append((150, word_amount))
    written_total = re.search(
        r"(?:indian\s+)?rupees?\s+(.+)\s+and\s+([a-z -]+)\s+paise",
        lowered_text,
    )
    if written_total:
        whole = _words_value(written_total.group(1))
        paisa = _words_value(written_total.group(2))
        if whole is not None and paisa is not None:
            candidates.append((400, whole + paisa / 100.0))
    for line in text.splitlines():
        normalized = re.sub(r"(?<=\d)-(?=\d{3}\b)", "", line)
        lower = normalized.lower()
        active_terms = terms or _GENERIC_TERMS
        score_base = sum(lower.count(term) * ((len(active_terms) - index) * 100) for index, term in enumerate(active_terms))
        if "total" in lower:
            for split in re.finditer(r"(?<!\d)(\d)[^0-9]{0,4}(\d{3})(?!\d)", normalized):
                if not split.group(2).startswith("0"):
                    candidates.append((score_base + 30, float(split.group(1) + split.group(2))))
        for match in _NUMBER_RE.finditer(normalized):
            amount = _value(match.group(1))
            if amount is None:
                continue
            score = score_base + (1 if amount >= 1000 else 0) + (50 if "." in match.group(1) else 0)
            if any(symbol in match.group(0) for symbol in ("$", "€", "₹")):
                score += 3
            candidates.append((score, amount))
    if not candidates:
        return None, 0.0
    score, amount = max(candidates, key=lambda item: (item[0], item[1]))
    return amount, min(0.99, 0.45 + score / 100.0)


def extract_amount_from_image(image_id: str, preferred_terms: Iterable[str] = ()) -> Optional[float]:
    return parse_primary_amount(extract_image_text(image_id), preferred_terms)[0]


def audit_image(image_id: str, preferred_terms: Iterable[str] = ()) -> Dict[str, object]:
    text = extract_image_text(image_id)
    amount, confidence = parse_primary_amount(text, preferred_terms)
    return {"image_id": image_id, "amount": amount, "confidence": confidence, "text": text}


def _preferred_terms(row: pd.Series) -> Tuple[str, ...]:
    event_type = str(row.get("event_type", "")).lower()
    description = str(row.get("description", "")).lower()
    if event_type in {"salary", "income"}:
        return ("net pay", "salary")
    if "rent" in description:
        return ("amount received", "amountreceived", "total amount to be received")
    if "grocery" in description or "bulk" in description or "pantry" in description:
        return ("cash paid", "anount", "item bill", "grand total", "total amount", "total amount received", "amount payable", "balance due", "net amount", "total")
    return ("anount", "item bill", "grand total", "total amount", "total amount received", "amount payable", "balance due", "net amount", "total")


async def fill_blank_event_amounts(user_id: str, request_id: str, events_df: pd.DataFrame) -> Dict[str, float]:
    result: Dict[str, float] = {}
    blank_events = events_df[events_df["amount"].isna() & (events_df["user_id"] == user_id)]
    for _, row in blank_events.iterrows():
        event_id = str(row["event_id"])
        image_rows = get_event_images(event_id)
        if image_rows.empty:
            continue
        image_id = str(image_rows.iloc[0]["image_id"])
        amount = extract_amount_from_image(image_id, _preferred_terms(row))
        if amount is not None:
            result[event_id] = amount
    return result


def _parse_amount(text: str) -> Optional[float]:
    return parse_primary_amount(text)[0]
