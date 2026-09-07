"""Resolve MPDD 2025 and MPDD-AVG 2026 datasets into one experiment manifest."""

import csv
import json
import re
from pathlib import Path


COHORT_2025 = {"Elder": "MPDD-Elderly", "Young": "MPDD-Young"}
AUDIO_ALIASES_2026 = {
    "mfccs": ("mfcc", "mfcc64"),
    "mfcc": ("mfcc", "mfcc64"),
    "wav2vec": ("wav2vec", "wav2vec2"),
    "opensmile": ("opensmile",),
}


def _find_child(parent, names):
    children = {child.name.lower(): child for child in parent.iterdir()} if parent.exists() else {}
    for name in names:
        if name.lower() in children:
            return children[name.lower()]
    raise FileNotFoundError(f"None of {names} exists under {parent}")


def _label(row, *names):
    lowered = {key.lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value not in (None, ""):
            return int(float(value))
    raise KeyError(f"Missing label columns {names}")


def _event_number(path):
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        raise ValueError(f"Cannot infer event number from {path.name}")
    return int(numbers[-1])


def build_2026_entries(track_root, audio_root, video_root, label_csv):
    with Path(label_csv).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = {}
    for row in rows:
        subject = str(row.get("ID") or row.get("id"))
        labels[subject] = {
            "bin_category": _label(row, "label2", "binary_gt"),
            "tri_category": _label(row, "label3", "ternary_gt"),
        }

    entries = []
    for subject, subject_labels in sorted(labels.items(), key=lambda item: int(item[0])):
        audio_dir = Path(audio_root) / subject
        video_dir = Path(video_root) / subject
        audio_files = {_event_number(path): path for path in audio_dir.rglob("*.npy")}
        video_files = {_event_number(path): path for path in video_dir.rglob("*.npy")}
        shared = sorted(set(audio_files) & set(video_files))
        if not shared:
            raise FileNotFoundError(f"No paired audio/video events for subject {subject}")
        for event in shared:
            entries.append({
                "subject_id": subject,
                "audio_feature_path": audio_files[event].relative_to(audio_root).as_posix(),
                "video_feature_path": video_files[event].relative_to(video_root).as_posix(),
                **subject_labels,
            })
    return entries


def resolve_dataset(data_root, dataset_year, cohort, split_window, audio_feature, video_feature):
    root = Path(data_root)
    if dataset_year == "2025":
        track_root = root if (root / "Training").exists() else root / COHORT_2025[cohort]
        training = track_root / "Training"
        entries = json.loads((training / "labels" / "Training_Validation_files.json").read_text(encoding="utf-8"))
        return {
            "entries": entries,
            "personality": training / "individualEmbedding" / "descriptions_embeddings_with_ids.npy",
            "audio": _find_child(training / split_window / "Audio", (audio_feature,)),
            "video": _find_child(training / split_window / "Visual", (video_feature,)),
            "track_root": track_root,
        }

    track_root = root / "MPDD-AVG2026-trainval" / cohort
    audio_base = _find_child(track_root / "Audio" / "train", AUDIO_ALIASES_2026.get(audio_feature, (audio_feature,)))
    video_base = _find_child(track_root / "Video" / "train", (video_feature,))
    label_csv = track_root / "split_labels_train.csv"
    return {
        "entries": build_2026_entries(track_root, audio_base, video_base, label_csv),
        "personality": track_root / "descriptions_embeddings_with_ids.npy",
        "audio": audio_base,
        "video": video_base,
        "track_root": track_root,
    }
