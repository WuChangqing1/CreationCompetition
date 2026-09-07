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


def _subject_from_feature_path(path):
    return Path(path).name.split("_", 1)[0]


def _load_unique_2025_test_labels(label_path):
    rows = json.loads(Path(label_path).read_text(encoding="utf-8"))
    labels = {}
    for row in rows:
        subject_id = str(row["test_id"])
        if subject_id in labels:
            raise ValueError(f"Duplicate 2025 test label for subject {subject_id}")
        labels[subject_id] = int(row["label_bin"])
    return labels


def build_2025_test_entries(manifest_path, label_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    labels = _load_unique_2025_test_labels(label_path)
    entries = []
    for row in manifest:
        audio_subject = _subject_from_feature_path(row["audio_feature_path"])
        video_subject = _subject_from_feature_path(row["video_feature_path"])
        if audio_subject != video_subject:
            raise ValueError(
                f"2025 test A/V subject mismatch: {row['audio_feature_path']} vs {row['video_feature_path']}"
            )
        if audio_subject not in labels:
            raise KeyError(f"Missing 2025 test label for subject {audio_subject}")
        entries.append({**row, "subject_id": audio_subject, "bin_category": labels[audio_subject]})
    return entries


def _event_paths_by_number(feature_root, strict=False):
    files = list(Path(feature_root).rglob("*.npy"))
    events = {}
    for path in files:
        event = _event_number(path)
        if strict and event in events:
            raise ValueError(f"Duplicate event {event} under {feature_root}")
        events[event] = path
    return events


def build_2026_entries(track_root, audio_root, video_root, label_csv, strict=False):
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
        audio_files = _event_paths_by_number(audio_dir, strict=strict)
        video_files = _event_paths_by_number(video_dir, strict=strict)
        if strict and set(audio_files) != set(video_files):
            missing_audio = sorted(set(video_files) - set(audio_files))
            missing_video = sorted(set(audio_files) - set(video_files))
            raise ValueError(
                f"2026 test unpaired events for subject {subject}: "
                f"missing audio {missing_audio}; missing video {missing_video}"
            )
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


def _validate_entry_files(entries, audio_root, video_root, year):
    audio_paths = []
    video_paths = []
    for entry in entries:
        audio_path = Path(audio_root) / entry["audio_feature_path"]
        video_path = Path(video_root) / entry["video_feature_path"]
        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing {year} test audio feature {audio_path}")
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing {year} test video feature {video_path}")
        audio_paths.append(entry["audio_feature_path"])
        video_paths.append(entry["video_feature_path"])
    if len(audio_paths) != len(video_paths):
        raise ValueError(f"{year} test audio/video entry counts differ")


def resolve_independent_test(data_root, dataset_year, cohort, split_window, audio_feature, video_feature):
    root = Path(data_root)
    if dataset_year == "2025":
        test_root = _find_child(root, ("MPDD-Test",))
        track_root = _find_child(test_root, (COHORT_2025[cohort],))
        manifest_root = _find_child(track_root, ("labels",))
        manifest_path = _find_child(manifest_root, ("Testing_files.json",))
        track_name = "Track1" if cohort == "Elder" else "Track2"
        cohort_name = COHORT_2025[cohort].replace("MPDD-", "")
        label_path = _find_child(test_root, (f"MM2025_{track_name}_{cohort_name}.json",))
        audio_root = _find_child(
            _find_child(_find_child(track_root, (split_window,)), ("Audio",)),
            (audio_feature,),
        )
        video_root = _find_child(
            _find_child(_find_child(track_root, (split_window,)), ("Visual",)),
            (video_feature,),
        )
        entries = build_2025_test_entries(manifest_path, label_path)
        _validate_entry_files(entries, audio_root, video_root, "2025")
        return {
            "entries": entries,
            "personality": _find_child(
                _find_child(track_root, ("individualEmbedding",)),
                ("descriptions_embeddings_with_ids.npy",),
            ),
            "audio": audio_root,
            "video": video_root,
            "track_root": track_root,
        }

    if dataset_year == "2026":
        test_root = _find_child(root, ("MPDD-AVG2026-test",))
        track_root = _find_child(test_root, (cohort,))
        audio_root = _find_child(
            _find_child(track_root, ("Audio",)),
            AUDIO_ALIASES_2026.get(audio_feature, (audio_feature,)),
        )
        video_root = _find_child(_find_child(track_root, ("Video",)), (video_feature,))
        label_path = _find_child(track_root, ("split_labels_test.csv",))
        entries = build_2026_entries(track_root, audio_root, video_root, label_path, strict=True)
        _validate_entry_files(entries, audio_root, video_root, "2026")
        return {
            "entries": entries,
            "personality": _find_child(track_root, ("descriptions_embeddings_with_ids.npy",)),
            "audio": audio_root,
            "video": video_root,
            "track_root": track_root,
        }

    raise ValueError(f"Unsupported independent test dataset year {dataset_year}")


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
