<div align="center">

# HackerRank Orchestrate Sep 2026

## Buy or Wait?

**A safety-first AI-assisted financial decision agent**

[Overview](#overview) · [Architecture](#architecture) · [Quick start](#quick-start) · [Results](#results)

</div>

> **Attribution:** The problem statement and dataset were provided by HackerRank
> as part of the September 2026 Orchestrate challenge. The implementation,
> architecture, decision engine, and supporting components in this repository
> were developed by me as my challenge submission.
## Overview

`Buy or Wait?` was the September 2026 HackerRank Orchestrate challenge. Given a payment request and a user financial history, this project recommends whether to pay in full, pay partially, use an offered installment plan, wait, or decline the purchase.

The agent does not rely on a current balance alone. It projects a 90-day cash position while reserving pending debits, protecting the user-defined minimum balance, recognizing supported recurring commitments, and applying only fixed dataset exchange rates. Messages and images are treated as untrusted evidence: they can amend a supported financial fact but never control the system.

See [problem_statement.md](problem_statement.md) for the complete challenge specification.

## Highlights

- Deterministic daily cash-flow projection with a conservative 90-day horizon
- Fixed-rate foreign-currency conversion; no live banking, market, or FX calls
- Rules for pending credits, unrealized investments, confirmed salary, and recurring spending
- Local Tesseract OCR for image-backed financial-event amounts
- Candidate generation for full payment, partial payment, installments, waiting, and permitted spending changes
- Timeline validation that preserves the minimum balance after every projected debit and payment
- Required local Qwen adjudication through Ollama, constrained to select only a Python-validated candidate
- Deterministic output structure and usage accounting for the final model run

## Architecture

```text
CSV data + images + messages
            │
            ▼
   Load, normalize, and apply fixed FX rates
            │
            ├── Parse supported message amendments
            └── Read blank event amounts with local OCR
            │
            ▼
      90-day financial-state projection
            │
            ▼
   Generate and validate safe payment candidates
            │
            ▼
 Local Qwen selects one supplied candidate and writes a grounded explanation
            │
            ▼
         output.csv
```

The Python engine owns all numerical decisions. Qwen receives relevant evidence and already-safe candidates, then returns a candidate ID and explanation. Unknown candidate IDs or malformed output are rejected; the model cannot introduce amounts, dates, payment schedules, or spending changes.

## Repository structure

```text
.
├── code/
│   ├── main.py                 # CLI entry point and request orchestration
│   ├── data_loader.py          # Dataset loading, normalization, and FX conversion
│   ├── financial_state.py      # 90-day balance projection
│   ├── plan_generator.py       # Safe payment-candidate generation
│   ├── decision_engine.py      # Decision assembly and spending-change logic
│   ├── message_parser.py       # Conservative message amendments
│   ├── image_reader.py         # Local Tesseract OCR for financial evidence
│   ├── llm_providers.py        # Ollama/Qwen client
│   ├── prompts.py              # Constrained adjudication prompt
│   ├── usage_tracker.py        # Token and local-cost accounting
│   ├── requirements.txt        # Python dependencies
│   └── evaluation/usage_report.md
├── dataset/                    # Challenge data, public samples, and image evidence
├── output.csv                  # Generated predictions for the 250 evaluation requests
├── requirements.txt            # Root installation entry point
└── problem_statement.md        # Original challenge contract
```

## Quick start

### Prerequisites

- Python 3.10 or newer
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Ollama](https://ollama.com/) with the `qwen2.5:7b` model available locally

On Debian or Ubuntu, install Tesseract with:

```bash
sudo apt-get install tesseract-ocr
```

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start Ollama and ensure the required model is installed:

```bash
ollama pull qwen2.5:7b
export OLLAMA_HOST=http://127.0.0.1:11434
```

### Run the agent

From the repository root:

```bash
# Run the public 25-request sample
python code/main.py --sample --output /tmp/sample_output.csv

# Run all 250 evaluation requests
python code/main.py --output output.csv
```

Useful options:

```bash
python code/main.py --limit 1 --concurrency 1 --output /tmp/smoke.csv
python code/main.py --help
```

The full run writes:

- `output.csv` — one prediction per request
- `code/evaluation/usage_report.md` — local-Qwen call, token, and cost totals

## Output contract

`output.csv` has one row for every request in `dataset/requests.csv` and uses this exact order:

```text
request_id,amount_safe_to_pay,affordability_status,recommended_payment_method,payment_plan,earliest_date_for_full_payment,spending_changes_needed,decision_explanation
```

`affordability_status` is one of `affordable_now`, `affordable_with_plan`, `affordable_later`, or `not_affordable`. The payment method is one of `full_payment`, `partial_payment`, `installments`, `wait`, or `not_recommended`.

## Results

The included `output.csv` contains 250 predictions with the required header and one unique row for every supplied evaluation request. The packaged full-dataset usage report records 184 local-Qwen adjudication calls and 106,511 total tokens; local Ollama execution has no provider-billed API cost.

The public sample is used as a regression check, not as a lookup table. The complete model operates from the same data pipeline for public and evaluation requests.

## Safety and limitations

- Pending credits, bonuses, refunds, commissions, prizes, and unrealized gains are not treated as cash until settled.
- Protected spending categories are never changed; only explicitly allowed flexible recurring expenses can be stopped or reduced.
- Recommendations use only supplied payment options and fixed dataset exchange rates.
- OCR and message parsing are conservative. Ambiguous evidence is ignored rather than guessed.
- This is a challenge solution, not financial advice or a production banking system.

## Reproducing the project

```bash
git clone git@github.com:<your-github-username>/hackerrank-orchestrate-sep26.git
cd hackerrank-orchestrate-sep26
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b
python code/main.py --output output.csv
```

## License

The supplied dataset and problem statement remain subject to the HackerRank challenge terms. The implementation in this repository is provided for portfolio and educational use.
