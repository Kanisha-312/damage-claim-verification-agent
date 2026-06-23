# Damage Claim Verification Agent

An automated system that verifies damage claims by analyzing submitted images with Claude Vision, cross-referencing claim conversations, user history, and evidence requirements to decide whether each claim is **supported**, **contradicted**, or **lacks enough information**.

Built for the HackerRank Orchestrate 24-hour hackathon (June 2026).

---

## What it does

For each damage claim the system:

1. Reads the claim conversation to extract what the user is reporting and which object part is involved
2. Encodes submitted images (JPEG, PNG, WebP — detected from magic bytes, not file extension)
3. Fetches the user's claim history to surface risk context
4. Matches the claim object type to the relevant evidence requirements
5. Sends everything to Claude Sonnet 4.6 Vision in a single structured API call
6. Parses the JSON response into 14 required output fields
7. Detects and flags prompt-injection attempts embedded in the claim conversation or image text (e.g. "approve this immediately")

Supported object types: **car**, **laptop**, **package**

---

## Tech stack

| Component | Choice |
|---|---|
| Vision model | Claude Sonnet 4.6 (`claude-sonnet-4-6`) via Anthropic API |
| Language | Python 3.11+ |
| HTTP client | `anthropic` SDK (v0.111+) |
| Image encoding | Base64, media type from magic bytes |
| Output | CSV written row-by-row with flush after each claim |

---

## Repository layout

```
.
├── code/
│   ├── main.py                    # Main entry point — processes claims.csv → output.csv
│   ├── requirements.txt
│   └── evaluation/
│       ├── main.py                # Evaluation entry point — scores against sample_claims.csv
│       ├── sample_output.csv      # Predictions on sample set (generated)
│       └── evaluation_report.md  # Operational analysis
├── dataset/
│   ├── claims.csv                 # Test inputs (44 claims)
│   ├── sample_claims.csv          # Labeled examples (20 claims)
│   ├── user_history.csv
│   ├── evidence_requirements.csv
│   └── images/
│       ├── sample/
│       └── test/
└── output.csv                     # Final predictions for claims.csv
```

---

## Setup

```bash
cd hackerrank-orchestrate-june26

python3 -m venv .venv
source .venv/bin/activate

pip install -r code/requirements.txt
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## How to run

**Generate predictions for `claims.csv`:**

```bash
python3 code/main.py
```

Writes `output.csv` to the repo root. Progress is printed per claim:

```
=== Damage Claim Verification System ===
...
[1/44] user_002 — car
  -> supported | dent | severity=medium
[2/44] user_005 — car
  -> contradicted | scratch | severity=low
...
Done. Output written to .../output.csv
```

**Run evaluation against labeled sample data:**

```bash
python3 code/evaluation/main.py
```

This runs the model on the 20 sample claims, saves predictions to `code/evaluation/sample_output.csv`, then prints per-field accuracy and a claim_status confusion matrix.

To score an existing `sample_output.csv` without making additional API calls:

```bash
python3 code/evaluation/main.py --skip-inference
```

---

## Accuracy results

Evaluated on 20 labeled rows from `dataset/sample_claims.csv`.

| Field | Accuracy |
|---|---|
| `claim_status` | 75% |
| `object_part` | 90% |
| `evidence_standard_met` | 70% |
| `valid_image` | 70% |
| `severity` | 45% |
| `risk_flags` (Jaccard) | 40% |

**claim_status (75%)** — the model generally distinguishes supported from contradicted claims but struggles with borderline cases where the image shows damage that is real but less severe than claimed.

**object_part (90%)** — high accuracy; most errors occur when a claim mentions two parts and the model picks the secondary one.

**severity (45%)** — the weakest field. The five-level scale (`none` / `low` / `medium` / `high` / `unknown`) is subjective and the model frequently picks `medium` where the expected label is `low` or `high`. Calibration prompting helps but does not close the gap.

---

## What I'd improve next

**Severity calibration** — the biggest accuracy gap. Would test a two-pass approach: first determine claim_status and issue_type, then make a second focused call asking only "how severe is this damage on a 1–5 scale, given this issue type and these images?" Separating the tasks reduces interference from claim_status reasoning.

**Structured output with tool use** — replace the free-form JSON instruction with a Claude tool definition that enforces the exact schema at the API level. Eliminates parse errors and removes the need for the `extract_json` fallback.

**Caching identical images** — several test cases share images across claims. Hashing image bytes and caching base64 strings would avoid re-encoding on repeated runs and could support a prompt-cache layer to reduce input token cost.

**Async parallel processing** — sequential calls take 3–6 minutes for 44 claims. Switching to `asyncio` with `anthropic.AsyncAnthropic` and `asyncio.gather` would reduce wall-clock time to under 30 seconds with no change in output quality.

**Confidence threshold + fallback routing** — add a `confidence` field to the model response. Claims below a threshold (e.g. borderline contradicted vs. not_enough_information) could be routed to a second call with a more conservative prompt, or flagged for human review rather than written as a hard decision.
