"""
Evaluation script for the damage claim verification system.

Runs the model on dataset/sample_claims.csv (which has expected outputs),
saves predictions to evaluation/sample_output.csv, then prints per-field
accuracy against the expected values.

Usage:
    ANTHROPIC_API_KEY=<key> python code/evaluation/main.py
    python code/evaluation/main.py --skip-inference   # reuse existing sample_output.csv
"""

import csv
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CODE_DIR))

import anthropic as anthropic_lib
from main import (
    DATASET_DIR,
    FALLBACK_RESULT,
    OUTPUT_COLUMNS,
    load_csv,
    process_claim,
)

EVAL_DIR = Path(__file__).parent
SAMPLE_OUTPUT = EVAL_DIR / "sample_output.csv"

EVAL_FIELDS = [
    "claim_status",
    "evidence_standard_met",
    "issue_type",
    "object_part",
    "severity",
    "valid_image",
]


# ---------------------------------------------------------------------------
# Inference on sample data
# ---------------------------------------------------------------------------

def run_inference(sample_claims: list[dict]) -> list[dict]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.", flush=True)
        sys.exit(1)

    client = anthropic_lib.Anthropic(api_key=api_key)
    user_history = {r["user_id"]: r for r in load_csv(DATASET_DIR / "user_history.csv")}
    evidence_reqs = load_csv(DATASET_DIR / "evidence_requirements.csv")

    rows = []
    for i, claim in enumerate(sample_claims, 1):
        print(f"[{i}/{len(sample_claims)}] {claim['user_id']} — {claim['claim_object']}", flush=True)
        input_row = {
            "user_id": claim["user_id"],
            "image_paths": claim["image_paths"],
            "user_claim": claim["user_claim"],
            "claim_object": claim["claim_object"],
        }
        try:
            result = process_claim(client, input_row, user_history, evidence_reqs)
            print(f"  -> {result['claim_status']} | {result['issue_type']} | severity={result['severity']}", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            result = dict(FALLBACK_RESULT)
        rows.append({**input_row, **result})

    with open(SAMPLE_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPredictions saved to {SAMPLE_OUTPUT}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def normalise(val: str) -> str:
    return val.strip().lower()


def flag_jaccard(pred: str, expected: str) -> float:
    """Jaccard similarity for semicolon-separated flag sets."""
    p = set(normalise(pred).split(";")) - {"none", ""}
    e = set(normalise(expected).split(";")) - {"none", ""}
    if not p and not e:
        return 1.0
    if not p or not e:
        return 0.0
    return len(p & e) / len(p | e)


def compute_metrics(predictions: list[dict], expected: list[dict]) -> dict:
    # Index expected by user_id + image_paths as key (handles duplicates)
    exp_map: dict[tuple, dict] = {}
    for row in expected:
        key = (row["user_id"], row["image_paths"])
        exp_map[key] = row

    field_matches: dict[str, list[bool]] = {f: [] for f in EVAL_FIELDS}
    flag_scores: list[float] = []
    status_confusion: dict[str, dict[str, int]] = {}
    matched = 0

    for pred in predictions:
        key = (pred["user_id"], pred["image_paths"])
        exp = exp_map.get(key)
        if exp is None:
            continue
        matched += 1

        for field in EVAL_FIELDS:
            field_matches[field].append(
                normalise(pred.get(field, "")) == normalise(exp.get(field, ""))
            )

        flag_scores.append(flag_jaccard(pred.get("risk_flags", "none"), exp.get("risk_flags", "none")))

        pred_status = normalise(pred.get("claim_status", ""))
        exp_status = normalise(exp.get("claim_status", ""))
        status_confusion.setdefault(exp_status, {}).setdefault(pred_status, 0)
        status_confusion[exp_status][pred_status] += 1

    return {
        "matched": matched,
        "total": len(predictions),
        "field_accuracy": {f: (sum(v) / len(v) * 100) if v else 0.0 for f, v in field_matches.items()},
        "risk_flags_jaccard": (sum(flag_scores) / len(flag_scores) * 100) if flag_scores else 0.0,
        "status_confusion": status_confusion,
    }


def print_report(metrics: dict) -> None:
    print("\n" + "=" * 52)
    print("  EVALUATION RESULTS")
    print("=" * 52)
    print(f"  Matched claims : {metrics['matched']} / {metrics['total']}")
    print()
    print(f"  {'Field':<28} {'Accuracy':>8}")
    print(f"  {'-'*28} {'-'*8}")
    for field, acc in metrics["field_accuracy"].items():
        print(f"  {field:<28} {acc:>7.1f}%")
    print(f"  {'risk_flags (Jaccard)':<28} {metrics['risk_flags_jaccard']:>7.1f}%")
    print()

    confusion = metrics["status_confusion"]
    if confusion:
        labels = sorted({s for row in confusion.values() for s in row} | set(confusion.keys()))
        col_w = max(len(l) for l in labels) + 2
        header = f"  {'exp \\ pred':<20}" + "".join(f"{l:>{col_w}}" for l in labels)
        print("  Claim status confusion matrix (rows=expected, cols=predicted):")
        print(header)
        for exp_label in labels:
            row_str = f"  {exp_label:<20}"
            for pred_label in labels:
                count = confusion.get(exp_label, {}).get(pred_label, 0)
                row_str += f"{count:>{col_w}}"
            print(row_str)

    print("=" * 52)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    skip_inference = "--skip-inference" in sys.argv

    sample_claims = load_csv(DATASET_DIR / "sample_claims.csv")

    if skip_inference and SAMPLE_OUTPUT.exists():
        print(f"Loading existing predictions from {SAMPLE_OUTPUT}", flush=True)
        predictions = load_csv(SAMPLE_OUTPUT)
    else:
        if skip_inference:
            print(f"sample_output.csv not found — running inference anyway.", flush=True)
        print(f"Running model on {len(sample_claims)} sample claims...\n", flush=True)
        predictions = run_inference(sample_claims)

    metrics = compute_metrics(predictions, sample_claims)
    print_report(metrics)


if __name__ == "__main__":
    main()
