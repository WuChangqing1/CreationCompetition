"""Aggregate actual fold rows into mean ± sample-standard-deviation summaries."""

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "Accuracy", "Macro_F1", "Weighted_F1", "Precision", "Recall",
    "Positive_Recall", "Specificity", "ROC_AUC", "Parameters",
    "Model_Size_MB", "Inference_Latency_ms", "Peak_VRAM_MB",
]
GROUP_COLUMNS = [
    "Run_ID", "DatasetYear", "Cohort", "Model", "Track", "Task", "AudioFeature", "VideoFeature",
    "UsePersonality", "SplitWindow", "Seed",
]


def _numeric(values):
    result = []
    for value in values:
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            continue
    return result


def aggregate_results(rows):
    grouped = OrderedDict()
    for row in rows:
        key = tuple(str(row.get(column, "N/A")) for column in GROUP_COLUMNS)
        grouped.setdefault(key, []).append(row)
    summary = []
    for key, model_rows in grouped.items():
        passed = [row for row in model_rows if str(row.get("Status", "PASS")).upper() == "PASS"]
        status = str(model_rows[-1].get("Status", "N/A")).upper()
        output = {column: value for column, value in zip(GROUP_COLUMNS, key)}
        output.update({"Folds": len(passed), "Status": "PASS" if passed else status})
        for metric in METRICS:
            values = _numeric(row.get(metric) for row in passed)
            if values:
                standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                output[metric] = f"{float(np.mean(values)):.4f} ± {standard_deviation:.4f}"
            else:
                output[metric] = status if not passed else "N/A"
        summary.append(output)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate actual MPDD fold results")
    parser.add_argument("--input", type=Path, default=Path("experiments/results/raw_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/summary.csv"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"Raw result file does not exist: {args.input}")
    rows = pd.read_csv(args.input).to_dict("records")
    summary = aggregate_results(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(args.output)


if __name__ == "__main__":
    main()
