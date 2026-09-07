"""Stable subject and label helpers shared by experiment scripts."""

from pathlib import Path


def extract_subject_id(entry):
    if entry.get("subject_id") not in (None, ""):
        return str(entry["subject_id"])
    feature_path = entry.get("audio_feature_path")
    if feature_path:
        filename = Path(feature_path).name
        return filename.split("_", 1)[0].removesuffix(".npy")
    if entry.get("id") not in (None, ""):
        return str(entry["id"])
    raise ValueError("Entry has no subject_id, audio_feature_path, or id")


def build_subject_labels(entries, label_key):
    labels = {}
    for entry in entries:
        if label_key not in entry:
            raise KeyError(f"Missing label key: {label_key}")
        subject_id = extract_subject_id(entry)
        label = int(entry[label_key])
        if subject_id in labels and labels[subject_id] != label:
            raise ValueError(
                f"Subject {subject_id} has conflicting labels: {labels[subject_id]} and {label}"
            )
        labels[subject_id] = label
    return labels


def filter_entries_by_subjects(entries, subject_ids):
    wanted = {str(subject_id) for subject_id in subject_ids}
    return [entry for entry in entries if extract_subject_id(entry) in wanted]
