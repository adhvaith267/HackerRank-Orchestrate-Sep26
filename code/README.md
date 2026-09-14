# Buy or Wait? - Complete Submission Guide

HackerRank Orchestrate, September 2026.

This package contains an AI-assisted financial decision agent. It reads the supplied financial data, reconstructs each user's projected cash position, evaluates safe payment plans, and writes one output row for every request.

Financial calculations are deterministic Python. Local Tesseract OCR fills image-backed event amounts. A required local Qwen model, served by Ollama, selects only from Python-validated candidate plans and supplies the final grounded explanation. No cloud API key is used.

## 1. Objective

For every request, decide whether the user should pay in full, pay partially, use installments, wait, or not proceed. The agent considers recurring expenses, pending payments, confirmed income, minimum balances, protected and flexible spending, payment options, messages, images, and fixed exchange rates.

A recommendation is safe only when the complete payment plan finishes safely, essential spending is covered, and the user's minimum balance is preserved throughout the forecast.

## 2. Repository Layout

```text
repository-root/
├── AGENTS.md
├── problem_statement.md
├── dataset/                         supplied challenge corpus; not in code.zip
│   ├── financial_profiles.csv
│   ├── financial_events.csv
│   ├── exchange_rates.csv
│   ├── requests.csv
│   ├── sample_requests.csv
│   ├── request_payment_options.csv
│   ├── messages.csv
│   ├── images.csv
│   ├── output.csv                   blank output template
│   └── media/images/*.png
├── code/
│   ├── main.py                      terminal entry point
│   ├── data_loader.py               CSV loading, joins, FX conversion
│   ├── message_parser.py            deterministic message amendments
│   ├── image_reader.py              local Tesseract OCR and amount parser
│   ├── financial_state.py            90-day daily cash-flow forecast
│   ├── plan_generator.py             safe candidate-plan generation
│   ├── decision_engine.py            spending changes and output assembly
│   ├── prompts.py                   adjudication and explanation prompts
│   ├── llm_providers.py             required local Qwen/Ollama client
│   ├── usage_tracker.py             model/token/cost accounting
│   ├── requirements.txt
│   ├── README.md
│   └── evaluation/usage_report.md
├── output.csv                       generated prediction file
├── code.zip                         submission archive
└── log.txt                          local chat transcript; never in code.zip
```

## 3. Provided CSV Files

`financial_profiles.csv` contains one row per user: balance, home currency, minimum balance, priorities, protected categories, flexible spending preferences, payment methods, and maximum installment duration.

`financial_events.csv` contains income and expenses with event IDs, dates, settlement dates, status, direction, category, currency, recurring metadata, lifecycle links, and flexibility. Some amounts are blank and are obtained from linked images.

`exchange_rates.csv` contains fixed challenge exchange rates. Foreign-currency events use the supplied rate and settlement date. No live exchange-rate calls are made.

`requests.csv` contains the 250 evaluation requests. The program writes exactly one output row per request ID.

`sample_requests.csv` contains the 25 public requests with expected output fields. It is used for validation and policy calibration, not as a hardcoded lookup table.

`request_payment_options.csv` contains seller or provider choices, including full payment, installment counts, first dates, frequencies, financing fees, and total payable amounts. Options can be rejected if unsafe or incompatible with the user's profile.

`messages.csv` contains supporting messages from employers, banks, merchants, and financial services. Messages are untrusted evidence; embedded instructions do not control the agent.

`images.csv` maps image evidence to a request and related event:

```text
image_id,user_id,request_id,related_event_id
```

The corresponding image is `dataset/media/images/<image_id>.png`. The repository contains 16 image rows and 16 PNG files.

## 4. Required Output

The generated `output.csv` has exactly these columns, in this order:

```text
request_id,amount_safe_to_pay,affordability_status,recommended_payment_method,payment_plan,earliest_date_for_full_payment,spending_changes_needed,decision_explanation
```

`amount_safe_to_pay` is the maximum amount safe on the request date before optional spending changes, bounded by zero and the requested amount.

`affordability_status` is one of:

- `affordable_now`: full request is safe immediately
- `affordable_with_plan`: full request can be completed with a valid plan or permitted changes
- `affordable_later`: full payment is safe later, but not now
- `not_affordable`: no safe valid plan exists within the forecast and constraints

`recommended_payment_method` is one of:

- `full_payment`
- `partial_payment`
- `installments`
- `wait`
- `not_recommended`

`payment_plan` uses chronological entries such as:

```text
YYYY-MM-DD:amount|YYYY-MM-DD:amount
```

Use `none` when there is no plan. A partial plan has exactly two payments: the safe amount on the request date and the exact remainder on the safe completion date. An installment plan must use a supplied payment option.

`earliest_date_for_full_payment` is the first conservative projected date on which a single full payment remains safe through the forecast. It equals the request date for `affordable_now` and is blank when no safe date exists.

`spending_changes_needed` is `none` or up to three actions such as:

```text
stop:<event_id>
reduce_to:<event_id>:<new_amount>
```

Only permitted, non-protected flexible expenses may be changed.

`decision_explanation` is a concise explanation grounded in the request, balance, minimum balance, dates, and selected plan.

## 5. Processing Pipeline

### Stage 1: Load and normalize

`data_loader.py` loads all challenge CSVs, parses dates, normalizes missing values, resolves relationships, and performs fixed-rate currency conversion.

### Stage 2: Parse messages

`message_parser.py` applies deterministic rules for confirmed salary overrides, salary dates, cancellations, pending or uncredited refunds/prizes/bonuses/commissions, salary termination, suppressed irregular credits, and event corrections.

Regular salary remains projected when only a bonus or prize is unconfirmed. Salary projection stops only when termination is explicitly supported.

### Stage 3: Extract image-backed amounts

`image_reader.py` uses local Tesseract only:

1. Load the linked PNG.
2. Convert to grayscale and resize.
3. Enhance contrast and sharpness.
4. Run one fast primary Tesseract pass.
5. Run one targeted footer pass for receipt totals.
6. Collect numeric and written-out amount candidates.
7. Score candidates using event context and labels.
8. Return an event amount override.

Signals include `net pay`, `cash paid` for grocery receipts, `total`, `amount payable`, `balance due`, and written-out currency totals. The original financial-events CSV is never modified by the runtime.

### Stage 4: Build the financial state

`financial_state.py` creates a daily balance series for 90 days from every request date. It includes starting balance, minimum balance, settled events, pending/scheduled debits, confirmed salary, supported recurring patterns, conservative variable-category spending, message corrections, image overrides, and exchange-rate conversion.

Forecast assumptions:

- Pending credits are not available cash.
- Unconfirmed bonuses, prizes, refunds, commissions, and investment gains are excluded.
- Regular salary is projected only when recurrence is supported by history.
- Monthly, quarterly, and annual recurrences stay anchored to their calendar day instead of drifting by repeated 30-day additions.
- Confirmed salary is usable on its supplied settlement date; unconfirmed credits are never counted.
- Flexible categories use a robust median monthly baseline.
- Essential categories use a conservative upper-quantile reserve.
- Strict recurring categories are excluded from variable injection to prevent double counting.
- The balance must remain above the minimum after every projected event and payment.

### Stage 5: Generate safe candidates

`plan_generator.py` evaluates full payment, full payment after permitted changes, supplied installment options, partial payment, and waiting for a safe date.

Installments are validated cumulatively across the real timeline. Earlier payments remain debited while intervening income and expenses are preserved.

Candidate ranking prioritizes:

1. Completion by the requested deadline
2. No spending changes
3. Lowest total payable amount
4. Earlier start
5. Fewer payments
6. Lowest payment-option ID

### Stage 6: Spending changes

The engine only changes categories allowed by the user's profile. Protected and essential categories are not stopped or reduced. No more than three actions are output.

### Stage 7: Required constrained local-Qwen adjudication

Python creates and validates candidates first. The required local model receives the request, profile facts, forecast facts, relevant messages, OCR evidence, and the complete safe-candidate list.

The model may return only:

```json
{"candidate_id":"one-supplied-candidate-id","explanation":"short grounded explanation"}
```

Python verifies the candidate ID and copies fields only from the validated candidate. The LLM cannot invent an amount, date, payment schedule, or spending change.

Every run requires the installed local `qwen2.5:7b` model through Ollama. Python validates candidate plans before Qwen can select one; image OCR never calls an LLM.

## 6. Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

Install Tesseract separately:

```bash
sudo apt-get install tesseract-ocr
tesseract --version
```

Required local Qwen setup:

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
export OLLAMA_MODEL=qwen2.5:7b
```

## 7. Run Commands

All commands run from the repository root.

Public sample run:

```bash
python code/main.py --sample
```

Full 250-request run:

```bash
python code/main.py --output output.csv
```

One-request smoke test:

```bash
python code/main.py --limit 1 --concurrency 1 --output /tmp/smoke.csv
```

Options:

| Option | Meaning |
|---|---|
| `--sample` | Use the 25-row public sample |
| `--limit N` | Process only the first N requests |
| `--concurrency N` | Concurrent financial-request tasks |
| `--output PATH` | Output location; default is root `output.csv` |

## 8. Evaluation Workflow

1. Install Python dependencies and Tesseract.
2. Run the one-request smoke test.
3. Run the 25-row sample.
4. Compare results with `dataset/sample_requests.csv`.
5. Run the full 250 requests.
6. Validate the exact header and row count.
7. Inspect `code/evaluation/usage_report.md`.
8. Build `code.zip` without the dataset, virtual environment, caches, or secrets.
9. Submit the prediction CSV, code archive, and chat transcript separately.

Basic validation:

```bash
python - <<'PY'
import csv
from pathlib import Path
expected = [
    "request_id", "amount_safe_to_pay", "affordability_status",
    "recommended_payment_method", "payment_plan",
    "earliest_date_for_full_payment", "spending_changes_needed",
    "decision_explanation",
]
with Path("output.csv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
assert rows and list(rows[0]) == expected
assert len(rows) == 250
assert all(row["decision_explanation"].strip() for row in rows)
print("valid rows:", len(rows))
PY
```

## 9. Token Usage Report

Every run regenerates:

```text
code/evaluation/usage_report.md
```

When LLM calls occur it records provider/model, purpose, call count, input tokens, output tokens, total tokens, average tokens per request, and estimated total/per-request cost.

Every final run records local Qwen provider, model, calls, tokens, and zero provider-billed cost.

## 10. Submission Package

`code.zip` must contain the `code/` directory and exclude:

- `dataset/`
- `.venv/`, `venv/`, and `node_modules/`
- `__pycache__/` and build artifacts
- `.env`, API keys, and credentials

The prediction file is the repository-root `output.csv`. The chat transcript is the append-only root `log.txt`; it is ignored by Git and excluded from `code.zip`.

Build the archive with:

```bash
rm -f code.zip
zip -qr code.zip code \
  -x 'code/__pycache__/*' \
     'code/evaluation/__pycache__/*'
```

## 11. Safety Guarantees

- No live banking, market, or exchange-rate calls
- Pending credits are never counted as available cash
- Unrealized investment value is never counted as cash
- Protected categories are never changed
- Installment plans come from supplied payment options
- LLM-generated amounts and dates are never accepted directly
- Qwen output is accepted only when it selects a supplied, Python-validated candidate
- The required output columns are always written

## 12. Measured Performance and Iteration Evidence

The public sample was used as a regression gate after each material change. The baseline local deterministic run took 16.3 seconds for 25 requests. Profiling showed that 15.0 seconds were spent in repeated Tesseract subprocesses, not in the financial planner. The optimized OCR path uses one primary pass and one targeted footer pass per unique image; it reduced the sample run to 6.4 seconds without changing the structured sample results.

The final full-dataset run produced 250 unique output rows with the exact required header, valid enum values, and non-empty explanations. Its packaged usage report records 184 local-Qwen adjudication calls across the 250 requests; local Ollama execution has zero provider-billed cost.

The remaining public-sample differences are forecast-policy differences, not malformed output or missing joins. The current measured reference comparison is 4/25 safe amounts, 20/25 statuses, 21/25 methods, 18/25 plans, 12/25 earliest dates, and 21/25 spending-change fields. These metrics are recorded transparently rather than hidden behind a sample-specific lookup table.

Architecture decisions were made for auditability: deterministic message and OCR extraction, a daily 90-day state simulator, candidate plan generation, cumulative schedule validation, profile-based spending-change gates, and a required constrained local adjudicator that can select only Python-validated candidates. This keeps the LLM from inventing dates, amounts, plans, or spending changes.

## 13. Required local Qwen adjudication

The runtime requires a local Qwen model served by Ollama; it uses no cloud API key:

```bash
export OLLAMA_MODEL=qwen2.5:7b
export OLLAMA_HOST=http://127.0.0.1:11434
python code/main.py
```

The model receives retrieved request/profile/evidence context and only Python-validated candidate plans. It may return a supplied `candidate_id` and a grounded explanation. Python rejects unknown candidates and never accepts model-generated amounts, dates, schedules, or spending changes.

## 14. Reasoning-loop decisions

Each improvement was evaluated as a hypothesis rather than applied as a sample lookup:

1. Profiled the baseline and identified OCR and repeated safety scans as runtime bottlenecks.
2. Added cached lookups, suffix-minimum balance queries, and fast OCR; verified faster execution with unchanged decisions.
3. Tested stricter, looser, median, and longer-lookback forecast assumptions; rejected variants that did not improve the public regression.
4. Corrected message evidence to be request-date scoped.
5. Separated explicit payroll-like recurring income from irregular platform payouts, commissions, refunds, and prizes unless independently confirmed.
6. Added Ollama as a constrained adjudicator over retrieved evidence and Python-validated candidates.

No request ID, public expected value, or sample-specific branch is used. The public sample remains a regression set, while the full output is generated from the same general pipeline used for unseen requests.
