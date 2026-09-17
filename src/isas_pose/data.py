"""CSV schema, pose normalization, temporal windows and subject-safe bags."""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

JOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COORD_COLUMNS = tuple(f"{joint}_{axis}" for joint in JOINTS for axis in ("x", "y"))
FILE_PATTERN = re.compile(r"keypoints_with_labels_(\d+)\.csv$")
NORMAL_LABELS = frozenset({"Sitting quietly", "Using phone", "Walking", "Eating snacks"})
UNUSUAL_LABELS = frozenset({"Head banging", "Throwing things", "Attacking", "Biting nails"})
ACTIVITY_LABELS = NORMAL_LABELS | UNUSUAL_LABELS


@dataclass
class Windows:
    poses: np.ndarray  # [N, window, 17, 2]
    labels: list[str]
    frame_ids: np.ndarray  # first frame of each window


def validate_frame(df: pd.DataFrame, *, labeled: bool) -> None:
    required = {"frame_id", *COORD_COLUMNS}
    if labeled:
        required.add("Action Label")
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("CSV has no rows")
    if df["frame_id"].isna().any() or df["frame_id"].duplicated().any():
        raise ValueError("frame_id must be present and unique within each CSV")
    if not pd.api.types.is_numeric_dtype(df["frame_id"]):
        raise ValueError("frame_id must be numeric")
    if not df["frame_id"].is_monotonic_increasing:
        raise ValueError("frame_id must be increasing")
    try:
        coordinates = df.loc[:, COORD_COLUMNS].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("Joint coordinates must be numeric") from exc
    if not np.isfinite(coordinates).all():
        raise ValueError("Joint coordinates must be finite")


def normalize_pose(df: pd.DataFrame) -> np.ndarray:
    validate_frame(df, labeled=False)
    poses = df.loc[:, COORD_COLUMNS].to_numpy(dtype=np.float32).reshape(-1, len(JOINTS), 2)
    hips = (poses[:, JOINTS.index("left_hip")] + poses[:, JOINTS.index("right_hip")]) / 2
    shoulders = poses[:, JOINTS.index("left_shoulder")] - poses[:, JOINTS.index("right_shoulder")]
    scale = np.linalg.norm(shoulders, axis=1) + 1e-6
    return ((poses - hips[:, None, :]) / scale[:, None, None]).astype(np.float32)


def make_windows(df: pd.DataFrame, window_size: int, stride: int, *, labeled: bool) -> Windows:
    validate_frame(df, labeled=labeled)
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be positive")
    if labeled:
        df = df.copy()
        df["Action Label"] = df["Action Label"].replace({"Throwing": "Throwing things"})
    if len(df) < window_size:
        raise ValueError(f"Need at least {window_size} frames; got {len(df)}")
    poses = normalize_pose(df)
    labels = df["Action Label"].astype("string").str.strip().tolist() if labeled else []
    if labeled:
        unknown = sorted({label for label in labels
                          if not pd.isna(label) and label != "" and label.casefold() != "none"
                          and label not in ACTIVITY_LABELS})
        if unknown:
            raise ValueError(f"Unknown activity labels: {unknown}")
    segments, targets, frames = [], [], []
    frame_ids = df["frame_id"].to_numpy()
    for start in range(0, len(df) - window_size + 1, stride):
        if np.any(np.diff(frame_ids[start:start + window_size]) != 1):
            continue
        if labeled and any(pd.isna(label) or label == "" or label.casefold() == "none"
                           for label in labels[start:start + window_size]):
            continue
        segments.append(poses[start:start + window_size])
        frames.append(df["frame_id"].iloc[start])
        if labeled:
            targets.append(Counter(labels[start:start + window_size]).most_common(1)[0][0])
    if not segments:
        raise ValueError("No usable windows: check None labels and window size")
    return Windows(np.stack(segments), targets, np.asarray(frames))


def activity_type(label: str) -> str:
    """Map one of the eight tutorial activities to normal or unusual."""
    if label in NORMAL_LABELS:
        return "normal"
    if label in UNUSUAL_LABELS:
        return "unusual"
    raise ValueError(f"Unknown activity label for binary evaluation: {label!r}")


def load_subjects(directory: Path, window_size: int, stride: int) -> dict[int, Windows]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {directory}")
    subjects = {}
    for path in sorted(directory.glob("keypoints_with_labels_*.csv")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match:
            uid = int(match.group(1))
            if uid in subjects:
                raise ValueError(f"Duplicate subject ID: {uid}")
            subjects[uid] = make_windows(pd.read_csv(path), window_size, stride, labeled=True)
    if len(subjects) < 2:
        raise ValueError("At least two labeled subject CSV files are required")
    return subjects


def make_bags(embeddings: np.ndarray, labels: list[str] | None, bag_size: int,
              *, frame_ids: np.ndarray | None = None, expected_step: int | None = None):
    """Return adjacent embedding bags and middle-window labels if available."""
    if bag_size < 1 or bag_size % 2 != 1:
        raise ValueError("bag_size must be a positive odd number")
    if len(embeddings) < bag_size:
        raise ValueError(f"Need at least {bag_size} windows to form a bag")
    if labels is not None and len(labels) != len(embeddings):
        raise ValueError("labels and embeddings have different lengths")
    if frame_ids is not None and (len(frame_ids) != len(embeddings) or expected_step is None):
        raise ValueError("frame_ids and expected_step must match the embeddings")
    starts = range(len(embeddings) - bag_size + 1)
    if frame_ids is not None:
        starts = [i for i in starts
                  if np.all(np.diff(frame_ids[i:i + bag_size]) == expected_step)]
    if not starts:
        raise ValueError("No adjacent windows remain to form a bag")
    bags = np.stack([embeddings[i:i + bag_size] for i in starts])
    targets = None if labels is None else [labels[i + bag_size // 2] for i in starts]
    return bags.astype(np.float32), targets
