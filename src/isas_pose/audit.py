"""Dataset preflight and immutable file fingerprints; no model dependency."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FILE_PATTERN, make_bags, make_windows


def subject_files(directory: Path) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {directory}")
    files: dict[int, Path] = {}
    for path in sorted(directory.glob("keypoints_with_labels_*.csv")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match:
            subject_id = int(match.group(1))
            if subject_id in files:
                raise ValueError(f"Duplicate subject ID: {subject_id}")
            files[subject_id] = path
    if len(files) < 2:
        raise ValueError("At least two labeled subject CSV files are required")
    return files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_fingerprints(directory: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in subject_files(directory).values()}


def inspect_data(directory: Path, window_size: int, stride: int, bag_size: int) -> dict:
    """Validate each CSV and report counts before any expensive training."""
    subjects = []
    for uid, path in subject_files(directory).items():
        frame = pd.read_csv(path)
        windows = make_windows(frame, window_size, stride, labeled=True)
        # Bag construction depends only on adjacency; dummy embeddings suffice.
        bags, _ = make_bags(np.zeros((len(windows.poses), 1), dtype=np.float32),
                            windows.labels, bag_size, frame_ids=windows.frame_ids,
                            expected_step=stride)
        labels = frame["Action Label"].astype("string").str.strip()
        missing = labels.isna() | labels.eq("").fillna(False) | labels.str.casefold().eq("none").fillna(False)
        subjects.append({
            "subject_id": uid,
            "file": path.name,
            "sha256": sha256_file(path),
            "rows": len(frame),
            "excluded_label_rows": int(missing.sum()),
            "windows": len(windows.poses),
            "bags": len(bags),
            "window_label_counts": dict(sorted(Counter(windows.labels).items())),
        })
    return {"window_size": window_size, "stride": stride, "bag_size": bag_size,
            "subjects": subjects}


def write_audit(report: dict, output: Path | None) -> None:
    result = json.dumps(report, indent=2, ensure_ascii=False)
    if output is None:
        print(result)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result + "\n", encoding="utf-8")
