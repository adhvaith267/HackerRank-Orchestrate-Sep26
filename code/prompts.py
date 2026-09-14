"""
prompts.py
==========
Runtime prompt templates for the required local-Qwen candidate adjudication stage.

Design principles
-----------------
- No hardcoded financial labels, user IDs, or dataset-specific values.
- System prompts define the model's persona and output contract.
- User prompts are filled with structured context at runtime via .format().
- The model is constrained to a supplied candidate ID; Python rejects every other output.

Prompts defined here
--------------------
    DECISION_SYSTEM / DECISION_USER_TEMPLATE
        → Used by main.py to ask local Qwen to choose from Python-validated
          candidate plans and provide a grounded explanation.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Constrained decision adjudication
# ---------------------------------------------------------------------------

DECISION_SYSTEM: str = """You are a cautious personal-finance decision adjudicator.
The Python financial engine has already calculated and safety-checked every candidate.
Choose exactly one candidate_id from the supplied list. Never invent a payment amount,
date, event, or candidate. Return JSON only with keys candidate_id and explanation.
The explanation must be concise, factual, and grounded in the supplied evidence."""

DECISION_USER_TEMPLATE: str = """
REQUEST
{request}

PROFILE AND FORECAST
{financial_context}

RELEVANT MESSAGES AND OCR EVIDENCE
{evidence}

VALID CANDIDATE PLANS
{candidates}

Return JSON only:
{{"candidate_id":"<one supplied candidate_id>","explanation":"<short explanation>"}}
"""
