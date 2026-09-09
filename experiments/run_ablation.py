"""Run isolated OurModel ablation variants through the existing CV pipeline."""

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocols import PROTOCOLS, resolve_protocol
from experiments.run_model_cv import load_model_config, main as run_model_cv
from experiments.dataset_layouts import resolve_dataset, resolve_independent_test
from experiments.independent_test import upsert_result_rows, write_prediction_csv
from experiments.independent_test_models import load_saved_fold_records
from experiments.run_independent_test import (
    _artifact_dirs,
    _read_csv,
    _write_confusion_csv,
    evaluate_one_model,
)
from models.ourablation_model import ABLATION_VARIANTS, normalize_ablation_variant


def build_variant_config(variant, protocol_name):
    variant = normalize_ablation_variant(variant)
    base = load_model_config("our")
    _, config = resolve_protocol(protocol_name, "our", base)
    config["ablation_variant"] = variant
    if variant == "no_aux_focal":
        config["focal_weight"] = 0.0
        config["loss_function"] = "CrossEntropyLoss"
    return config


def parse_variants(value):
    variants = [normalize_ablation_variant(item) for item in value.split(",") if item.strip()]
    if not variants:
        raise argparse.ArgumentTypeError("--variants must contain at least one variant")
    return list(dict.fromkeys(variants))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run OurModel ablations without modifying the original OurModel"
    )
    parser.add_argument("--variants", type=parse_variants, default=list(ABLATION_VARIANTS))
    parser.add_argument("--stage", choices=("cv", "independent-test"), default="cv")
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), default="legacy_bicfnet")
    parser.add_argument("--dataset-year", choices=("2025", "2026"), required=True)
    parser.add_argument("--cohort", choices=("Elder", "Young"), default="Elder")
    parser.add_argument("--track", choices=("Track1", "Track2"), default="Track1")
    parser.add_argument("--task", choices=("binary",), default="binary")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--audio-feature", default="mfccs")
    parser.add_argument("--video-feature", default="densenet")
    parser.add_argument("--personality-id-source", choices=("filename", "subject_id"), default="subject_id")
    parser.add_argument("--split-window", default="1s")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--splits-dir", type=Path)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measure", type=int, default=50)
    parser.add_argument("--independent-output-name", default="independent_test_results.csv")
    args = parser.parse_args(argv)
    args.results_dir = args.results_dir or ROOT / "experiments" / "results_ablation" / args.dataset_year
    return args


def combine_variant_csvs(results_dir, variants, source_name, output_name):
    rows = []
    fieldnames = None
    for variant in variants:
        source = Path(results_dir) / variant / source_name
        if not source.is_file():
            continue
        with source.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            current_fields = list(reader.fieldnames or ())
            if fieldnames is None:
                fieldnames = ["Ablation", *current_fields]
            elif current_fields != fieldnames[1:]:
                raise ValueError(f"CSV headers disagree across ablation variants: {source}")
            rows.extend({"Ablation": variant, **row} for row in reader)
    if fieldnames is None:
        raise FileNotFoundError(f"No {source_name} files found below {results_dir}")
    output = Path(results_dir) / output_name
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    return output


def _cv_arguments(args, variant, config_path, variant_results_dir):
    command = [
        "--model", "ourablation",
        "--config", str(config_path),
        "--protocol", args.protocol,
        "--dataset-year", args.dataset_year,
        "--cohort", args.cohort,
        "--track", args.track,
        "--task", args.task,
        "--data-root", str(args.data_root),
        "--audio-feature", args.audio_feature,
        "--video-feature", args.video_feature,
        "--use-personality",
        "--personality-id-source", args.personality_id_source,
        "--split-window", args.split_window,
        "--folds", str(args.folds),
        "--device", args.device,
        "--results-dir", str(variant_results_dir),
        "--warmup", str(args.warmup),
        "--measure", str(args.measure),
    ]
    if args.seed is not None:
        command.extend(("--seed", str(args.seed)))
    if args.splits_dir is not None:
        command.extend(("--splits-dir", str(args.splits_dir)))
    if args.tiny:
        command.append("--tiny")
    return command


def _independent_args(args, variant_results_dir):
    protocol = PROTOCOLS[args.protocol]
    seed = protocol["seed"] if args.seed is None else args.seed
    split_root = ROOT / "experiments" / "splits"
    if args.protocol == "legacy_bicfnet":
        split_root = split_root / args.protocol
    splits_dir = args.splits_dir or split_root / args.dataset_year / args.cohort
    return SimpleNamespace(
        dataset_year=args.dataset_year,
        cohort=args.cohort,
        track=args.track,
        task=args.task,
        data_root=args.data_root.resolve(),
        audio_feature=args.audio_feature,
        video_feature=args.video_feature,
        use_personality=True,
        personality_id_source=args.personality_id_source,
        split_window=args.split_window,
        folds=args.folds,
        seed=seed,
        device=args.device,
        batch_size=int(protocol.get("batch_size", 8)),
        feature_max_len=int(protocol["feature_max_len"]),
        feature_max_len_fallback=int(protocol["feature_max_len"]),
        results_dir=variant_results_dir.resolve(),
        cv_results=(variant_results_dir / "raw_results.csv").resolve(),
        output=(variant_results_dir / args.independent_output_name).resolve(),
        splits_dir=Path(splits_dir).resolve(),
        classes=2,
        subject_aggregation=protocol["subject_aggregation"],
        include_legacy_voted_event=args.protocol == "legacy_bicfnet",
    )


def _run_independent_variant(args, variant, variant_results_dir):
    eval_args = _independent_args(args, variant_results_dir)
    cv_rows = _read_csv(eval_args.cv_results)
    training_paths = resolve_dataset(
        eval_args.data_root, eval_args.dataset_year, eval_args.cohort,
        eval_args.split_window, eval_args.audio_feature, eval_args.video_feature,
    )
    test_paths = resolve_independent_test(
        eval_args.data_root, eval_args.dataset_year, eval_args.cohort,
        eval_args.split_window, eval_args.audio_feature, eval_args.video_feature,
    )
    result = evaluate_one_model(
        eval_args, "ourablation", cv_rows, training_paths, test_paths,
        load_saved_fold_records(eval_args),
    )
    prediction_dir, confusion_dir = _artifact_dirs(eval_args.output)
    for row in result["rows"]:
        level = row["EvaluationLevel"]
        ids, labels, probabilities = result[level]
        stem = f"{eval_args.dataset_year}_{eval_args.cohort}_{variant}_{level}"
        write_prediction_csv(
            prediction_dir / f"{stem}.csv", ids, labels, probabilities,
            "ourablation", level,
        )
        _write_confusion_csv(confusion_dir / f"{stem}.csv", row)
    upsert_result_rows(eval_args.output, result["rows"])
    return eval_args.output


def main(argv=None):
    args = parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        variant_results_dir = args.results_dir / variant
        variant_results_dir.mkdir(parents=True, exist_ok=True)
        print(f"[RUN] {variant}")
        if args.stage == "cv":
            config_path = variant_results_dir / "ablation_config.json"
            config_path.write_text(
                json.dumps(build_variant_config(variant, args.protocol), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            run_model_cv(_cv_arguments(args, variant, config_path, variant_results_dir))
        else:
            _run_independent_variant(args, variant, variant_results_dir)
        print(f"[PASS] {variant}")
    if args.stage == "cv":
        combined = combine_variant_csvs(
            args.results_dir, ABLATION_VARIANTS, "raw_results.csv", "ablation_raw_results.csv"
        )
    else:
        combined = combine_variant_csvs(
            args.results_dir, ABLATION_VARIANTS, args.independent_output_name,
            "ablation_independent_test_results.csv",
        )
    print(str(combined))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
