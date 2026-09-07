"""Create reusable stratified folds whose indivisible unit is a subject."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.model_selection import StratifiedKFold

from experiments.data_utils import build_subject_labels


LABEL_KEYS = {"binary": "bin_category", "ternary": "tri_category", "quinary": "pen_category"}


def validate_no_subject_leakage(train_ids, val_ids):
    overlap = sorted(set(map(str, train_ids)) & set(map(str, val_ids)))
    if overlap:
        raise ValueError(f"Subject leakage detected. Overlap: {overlap}")


def _distribution(ids, subject_labels):
    return dict(sorted(Counter(str(subject_labels[subject_id]) for subject_id in ids).items()))


def create_subject_folds(entries, label_key="bin_category", folds=5, seed=3407):
    subject_labels = build_subject_labels(entries, label_key)
    subject_ids = sorted(subject_labels)
    labels = [subject_labels[subject_id] for subject_id in subject_ids]
    counts = Counter(labels)
    fingerprint = hashlib.sha256(
        json.dumps(subject_labels, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    minimum = min(counts.values(), default=0)
    if minimum < folds:
        raise ValueError(
            f"Cannot create {folds} stratified folds: smallest class has {minimum} subjects"
        )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results = []
    for fold_index, (train_indices, val_indices) in enumerate(splitter.split(subject_ids, labels), 1):
        train_ids = [subject_ids[index] for index in train_indices]
        val_ids = [subject_ids[index] for index in val_indices]
        validate_no_subject_leakage(train_ids, val_ids)
        results.append({
            "fold": fold_index,
            "seed": seed,
            "folds": folds,
            "label_key": label_key,
            "dataset_fingerprint": fingerprint,
            "train_ids": train_ids,
            "val_ids": val_ids,
            "train_class_distribution": _distribution(train_ids, subject_labels),
            "val_class_distribution": _distribution(val_ids, subject_labels),
        })
    return results


def write_subject_folds(folds, output_dir, force=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"fold_{fold['fold']}.json" for fold in folds]
    existing = [path.exists() for path in paths]
    unrelated = list(output_dir.glob("fold_*.json"))
    if not force and (any(existing) or unrelated):
        if not all(existing) or len(unrelated) != len(paths):
            raise ValueError(
                "Existing split cache is partial or has a different fold count; use --force to replace it"
            )
        required = (
            "fold", "seed", "folds", "label_key", "dataset_fingerprint",
            "train_ids", "val_ids", "train_class_distribution", "val_class_distribution",
        )
        for expected, path in zip(folds, paths):
            actual = json.loads(path.read_text(encoding="utf-8"))
            if any(actual.get(key) != expected.get(key) for key in required):
                raise ValueError(
                    "Existing split cache is incompatible with the requested task, seed, folds, or dataset; "
                    "use --force to replace it"
                )
        return paths
    for fold, path in zip(folds, paths):
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(fold, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Create reusable subject-level stratified folds")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--train-json", type=Path)
    parser.add_argument("--task", choices=sorted(LABEL_KEYS), default="binary")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/splits"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train_json = args.train_json
    if train_json is None:
        if args.data_root is None:
            raise SystemExit("Provide --data-root or --train-json")
        train_json = args.data_root / "Training" / "labels" / "Training_Validation_files.json"
    entries = json.loads(train_json.read_text(encoding="utf-8"))
    folds = create_subject_folds(entries, LABEL_KEYS[args.task], args.folds, args.seed)
    paths = write_subject_folds(folds, args.output_dir, args.force)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
