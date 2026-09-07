"""Run models independently so one failure never stops the remaining list."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.errors import OptionalDependencyError


def run_models(model_names, runner, report_path):
    report = []
    for model_name in model_names:
        started = time.perf_counter()
        status, error = "PASS", ""
        try:
            runner(model_name)
        except OptionalDependencyError as exc:
            status, error = "SKIPPED", str(exc)
        except NotImplementedError as exc:
            status, error = "N/A", str(exc)
        except Exception as exc:
            status, error = "FAILED", str(exc)
        row = {
            "model": model_name,
            "status": status,
            "duration": time.perf_counter() - started,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        report.append(row)
        print(f"[{status}] {model_name}" + (f": {error}" if error else ""))
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run multiple MPDD models with failure isolation")
    parser.add_argument("--models", required=True, help="Comma-separated model names")
    parser.add_argument("--dataset-year", default="2025", choices=["2025", "2026"])
    parser.add_argument("--cohort", choices=["Elder", "Young"])
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--track", default="Track1")
    parser.add_argument("--task", default="binary", choices=["binary"])
    parser.add_argument("--audio-feature", default="mfccs")
    parser.add_argument("--video-feature", default="densenet")
    parser.add_argument("--use-personality", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-window", default="1s")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "experiments" / "results" / "run_report.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model_names = [name.strip().lower() for name in args.models.split(",") if name.strip()]

    def runner(model_name):
        command = [
            sys.executable, str(ROOT / "experiments" / "run_model_cv.py"),
            "--dataset-year", args.dataset_year,
            "--model", model_name, "--track", args.track, "--task", args.task,
            "--audio-feature", args.audio_feature, "--video-feature", args.video_feature,
            "--split-window", args.split_window, "--folds", str(args.folds), "--seed", str(args.seed),
            "--use-personality" if args.use_personality else "--no-use-personality",
        ]
        if args.cohort:
            command.extend(["--cohort", args.cohort])
        if args.data_root:
            command.extend(["--data-root", str(args.data_root)])
        if args.device:
            command.extend(["--device", args.device])
        if args.tiny:
            command.append("--tiny")
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            if "OptionalDependencyError" in message:
                raise OptionalDependencyError(message)
            if "NotImplementedError" in message:
                raise NotImplementedError(message)
            raise RuntimeError(message or f"exit code {result.returncode}")

    run_models(model_names, runner, args.report)


if __name__ == "__main__":
    main()
