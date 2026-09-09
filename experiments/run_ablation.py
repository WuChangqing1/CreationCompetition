"""Train and test OurModel ablations with one legacy-style holdout split."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.data_utils import build_subject_labels, extract_subject_id, filter_entries_by_subjects
from experiments.dataset_layouts import resolve_dataset, resolve_independent_test
from experiments.legacy_comparison import HISTORICAL_SEED, load_historical_split
from experiments.protocols import resolve_protocol
from experiments.run_model_cv import load_model_config
from models.ourablation_model import ABLATION_VARIANTS, normalize_ablation_variant


def build_variant_config(variant, protocol_name="legacy_bicfnet"):
    variant = normalize_ablation_variant(variant)
    base = load_model_config("our")
    _, config = resolve_protocol(protocol_name, "our", base)
    config["ablation_variant"] = variant
    if variant == "no_aux_focal":
        config["focal_weight"] = 0.0
        config["loss_function"] = "CrossEntropyLoss"
    return config


def parse_variants(value):
    try:
        variants = [normalize_ablation_variant(item) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not variants:
        raise argparse.ArgumentTypeError("--variants must contain at least one variant")
    return list(dict.fromkeys(variants))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run OurModel ablations once on the legacy train/validation protocol; no 5-Fold"
    )
    parser.add_argument("--variants", type=parse_variants, default=list(ABLATION_VARIANTS))
    parser.add_argument("--dataset-year", choices=("2025", "2026"), required=True)
    parser.add_argument("--cohort", choices=("Elder", "Young"), default="Elder")
    parser.add_argument("--track", choices=("Track1",), default="Track1")
    parser.add_argument("--task", choices=("binary",), default="binary")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--audio-feature", default="mfccs")
    parser.add_argument("--video-feature", default="densenet")
    parser.add_argument("--personality-id-source", choices=("filename", "subject_id"))
    parser.add_argument("--split-window", default="1s")
    parser.add_argument("--seed", type=int, default=HISTORICAL_SEED)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output-dir", "--results-dir", dest="output_dir", type=Path)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args(argv)
    args.data_root = args.data_root.expanduser().resolve()
    if args.personality_id_source is None:
        args.personality_id_source = "filename" if args.dataset_year == "2025" else "subject_id"
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
    return args


def split_2026_subject_holdout(entries, seed=HISTORICAL_SEED, validation_ratio=0.1):
    """Create one reproducible stratified holdout without splitting a subject."""
    from sklearn.model_selection import train_test_split

    subject_labels = build_subject_labels(entries, "bin_category")
    subject_ids = sorted(subject_labels)
    labels = [subject_labels[subject_id] for subject_id in subject_ids]
    train_ids, validation_ids = train_test_split(
        subject_ids, test_size=validation_ratio, random_state=int(seed),
        shuffle=True, stratify=labels,
    )
    train_entries = filter_entries_by_subjects(entries, train_ids)
    validation_entries = filter_entries_by_subjects(entries, validation_ids)
    if {extract_subject_id(row) for row in train_entries} & {
        extract_subject_id(row) for row in validation_entries
    }:
        raise RuntimeError("2026 single holdout contains subject leakage")
    return train_entries, validation_entries


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


def _default_output_dir(year):
    stamp = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())
    return ROOT / "experiments" / "results_ablation_legacy_single_split" / year / stamp


def _runtime_args(cli_args, output_dir):
    return SimpleNamespace(
        data_root=cli_args.data_root, output_dir=output_dir,
        dataset_year=cli_args.dataset_year, cohort=cli_args.cohort,
        track=cli_args.track, task=cli_args.task,
        audio_feature=cli_args.audio_feature, video_feature=cli_args.video_feature,
        use_personality=True, personality_id_source=cli_args.personality_id_source,
        split_window=cli_args.split_window, seed=cli_args.seed,
        feature_max_len=26, batch_size=8,
        epochs=1 if cli_args.tiny else cli_args.epochs,
        classes=2, device=cli_args.device,
    )


def _resolve_single_split(args):
    training = resolve_dataset(
        args.data_root, args.dataset_year, args.cohort, args.split_window,
        args.audio_feature, args.video_feature,
    )
    testing = resolve_independent_test(
        args.data_root, args.dataset_year, args.cohort, args.split_window,
        args.audio_feature, args.video_feature,
    )
    if args.dataset_year == "2025":
        training_json = training["track_root"] / "Training" / "labels" / "Training_Validation_files.json"
        train_entries, validation_entries = load_historical_split(training_json)
        split_name = "historical_frozen_292_45"
    else:
        train_entries, validation_entries = split_2026_subject_holdout(
            training["entries"], seed=args.seed, validation_ratio=0.1
        )
        split_name = "subject_stratified_single_90_10"
    return training, testing, train_entries, validation_entries, split_name


def _make_dataset(entries, args, paths):
    from experiments.run_model_cv import make_dataset

    return make_dataset(entries, args, paths)


def _write_result(path, row):
    from experiments.independent_test import RESULT_COLUMNS

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerow({column: row[column] for column in RESULT_COLUMNS})
    temporary.replace(path)


def _test_result(variant, probabilities, test_entries, args, run_id):
    from experiments.independent_test import (
        aggregate_subject_predictions, broadcast_subject_predictions,
        build_result_row, write_prediction_csv,
    )

    labels = np.asarray([int(entry["bin_category"]) for entry in test_entries], dtype=int)
    subject_ids = [extract_subject_id(entry) for entry in test_entries]
    unique_ids, _, subject_probabilities = aggregate_subject_predictions(
        subject_ids, labels, probabilities, method="majority_vote"
    )
    voted_probabilities = broadcast_subject_predictions(subject_ids, unique_ids, subject_probabilities)
    row = build_result_row(
        year=args.dataset_year, cohort=args.cohort, model="ourablation",
        level="legacy_voted_event", labels=labels, probabilities=voted_probabilities,
        folds=0, run_id=run_id, device=args.device, seed=args.seed,
    )
    write_prediction_csv(
        args.output_dir / "predictions" / f"{variant}.csv",
        subject_ids, labels, voted_probabilities, "ourablation", "legacy_voted_event",
    )
    return row


def _train_and_test_variant(variant, args, paths):
    import torch
    from torch.utils.data import DataLoader

    from experiments.evaluator import evaluate_predictions
    from experiments.model_registry import create_experiment_model
    from experiments.run_model_cv import (
        build_opt, evaluate_torch, infer_feature_dims,
        is_better_validation, set_random_seed,
    )

    training_paths, test_paths, train_entries, validation_entries, _ = paths
    set_random_seed(args.seed)
    config = build_variant_config(variant)
    config.update({
        "seed": args.seed, "feature_max_len": 26, "batch_size": 8,
        "epochs": args.epochs, "checkpoint_selection": "val_macro_f1",
        "personality_id_source": args.personality_id_source,
        "protocol": "legacy_single_split",
    })
    train_dataset = _make_dataset(train_entries, args, training_paths)
    validation_dataset = _make_dataset(validation_entries, args, training_paths)
    test_dataset = _make_dataset(test_paths["entries"], args, test_paths)
    run_id = f"legacy-single-split-{args.dataset_year}-{variant}-seed{args.seed}"
    artifact_dir = args.output_dir / variant
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to start a formal ablation run on CPU")
    input_dim_a, input_dim_v = infer_feature_dims(training_paths["entries"], training_paths)
    opt = build_opt("ourablation", config, input_dim_a, input_dim_v, 2, 26, run_id)
    opt.use_personality = True
    opt.device = str(device)
    model = create_experiment_model("ourablation", opt=opt)
    model.to(device)

    generator = torch.Generator().manual_seed(args.seed)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, generator=generator, pin_memory=pin_memory
    )
    validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=pin_memory)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model.optimizer, T_max=args.epochs, eta_min=float(config.get("scheduler_eta_min", 1e-6))
    )
    best_macro_f1 = None
    best_epoch = None
    best_state = None
    with (artifact_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as history_handle:
        writer = csv.DictWriter(
            history_handle,
            fieldnames=("epoch", "validation_macro_f1", "best_epoch", "best_macro_f1"),
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train(True)
            for batch in train_loader:
                model.set_input(batch)
                model.optimize_parameters(epoch - 1)
            labels, predictions, probabilities = evaluate_torch(model, validation_loader, device)
            macro_f1 = evaluate_predictions(labels, predictions, probabilities)["Macro_F1"]
            if is_better_validation(macro_f1, best_macro_f1):
                best_macro_f1 = macro_f1
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            writer.writerow({
                "epoch": epoch, "validation_macro_f1": macro_f1,
                "best_epoch": best_epoch, "best_macro_f1": best_macro_f1,
            })
            history_handle.flush()
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(
                    f"[{variant}] epoch {epoch:03d}/{args.epochs} "
                    f"val_macro_f1={macro_f1:.4f} best={best_macro_f1:.4f}@{best_epoch}"
                )
            scheduler.step()

    model.load_state_dict(best_state, strict=True)
    checkpoint = {
        "model_name": "ourablation", "model_state_dict": best_state,
        "config": {
            **vars(opt), "ablation_variant": variant,
            "best_epoch": best_epoch, "best_val_macro_f1": best_macro_f1,
        },
        "feature_config": {
            "dataset_year": args.dataset_year, "cohort": args.cohort,
            "track": args.track, "task": args.task,
            "audio_feature": args.audio_feature, "video_feature": args.video_feature,
            "use_personality": True, "split_window": args.split_window,
            "device": str(device), "personality_id_source": args.personality_id_source,
            "protocol": "legacy_single_split", "ablation_variant": variant,
        },
        "fold": None, "seed": args.seed,
    }
    torch.save(checkpoint, artifact_dir / "checkpoint.pth")
    _, _, test_probabilities = evaluate_torch(model, test_loader, device)
    row = _test_result(variant, test_probabilities, test_paths["entries"], args, run_id)
    _write_result(artifact_dir / "result.csv", row)
    print(
        f"[PASS] {variant}: Accuracy={float(row['Accuracy']):.4f}, "
        f"Macro-F1={float(row['Macro_F1']):.4f}, "
        f"CM=[[{row['TN']},{row['FP']}],[{row['FN']},{row['TP']}]]"
    )
    return row


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing to start formal legacy ablations")
    args.output_dir = args.output_dir or _default_output_dir(args.dataset_year)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runtime = _runtime_args(args, args.output_dir)
    paths = _resolve_single_split(runtime)
    _, _, train_entries, validation_entries, split_name = paths
    (args.output_dir / "run_config.json").write_text(
        json.dumps({
            "protocol": "legacy_single_split_ablation",
            "dataset_year": args.dataset_year,
            "cross_validation": False, "folds": 0, "split": split_name,
            "train_samples": len(train_entries),
            "validation_samples": len(validation_entries),
            "seed": args.seed, "epochs": runtime.epochs,
            "feature_max_len": 26, "batch_size": 8, "device": args.device,
            "personality_id_source": runtime.personality_id_source,
            "variants": args.variants,
            "subject_aggregation": "majority_vote_then_event_backfill",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for variant in args.variants:
        print(f"[RUN] {variant}: legacy single split, no 5-Fold")
        _train_and_test_variant(variant, runtime, paths)
    combined = combine_variant_csvs(
        args.output_dir, args.variants, "result.csv", "ablation_results.csv"
    )
    print(f"[PASS] legacy single-split ablations complete: {combined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
