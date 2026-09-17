"""Explicit output contracts from the tutorial and challenge website."""

import pandas as pd


def format_predictions(frame: pd.DataFrame, labels: pd.Series, *, output_format: str,
                       participant_id: str | None = None,
                       timestamp_column: str = "timestamp") -> pd.DataFrame:
    if len(frame) != len(labels) or labels.isna().any():
        raise ValueError("Every input row needs exactly one predicted activity label")
    if output_format == "test-file":
        result = frame.copy()
        result["Action Label"] = labels.to_numpy()
        return result
    if output_format != "submission":
        raise ValueError(f"Unknown output format: {output_format}")
    if timestamp_column not in frame:
        raise ValueError(f"Submission requires timestamp column {timestamp_column!r}")
    if participant_id is None:
        if "participant_id" not in frame:
            raise ValueError("Submission requires --participant-id or an input participant_id column")
        ids = frame["participant_id"]
    else:
        ids = pd.Series([participant_id] * len(frame), index=frame.index)
    timestamps = frame[timestamp_column]
    if ids.isna().any() or timestamps.isna().any():
        raise ValueError("participant_id and timestamp cannot contain missing values")
    return pd.DataFrame({"participant_id": ids.to_numpy(),
                         "timestamp": timestamps.to_numpy(),
                         "predicted_label": labels.to_numpy()})
