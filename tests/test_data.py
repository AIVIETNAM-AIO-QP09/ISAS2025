import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from isas_pose.data import COORD_COLUMNS, activity_type, make_bags, make_windows, normalize_pose
from isas_pose.audit import inspect_data
from isas_pose.export import format_predictions


def frame(rows=7):
    data = {column: np.ones(rows, dtype=float) for column in COORD_COLUMNS}
    data["left_shoulder_x"] = np.full(rows, 2.0)
    data["right_shoulder_x"] = np.zeros(rows)
    data["frame_id"] = np.arange(rows)
    data["Action Label"] = ["Walking"] * (rows - 2) + ["Throwing"] * 2
    return pd.DataFrame(data)


class DataTests(unittest.TestCase):
    def test_normalization_centers_hips(self):
        poses = normalize_pose(frame())
        np.testing.assert_allclose((poses[:, 11] + poses[:, 12]) / 2, 0, atol=1e-6)

    def test_windows_and_label_alias(self):
        windows = make_windows(frame(), 3, 2, labeled=True)
        self.assertEqual(windows.poses.shape, (3, 3, 17, 2))
        self.assertEqual(windows.frame_ids.tolist(), [0, 2, 4])
        self.assertEqual(windows.labels[-1], "Throwing things")

    def test_rejects_duplicate_frames(self):
        df = frame()
        df.loc[1, "frame_id"] = 0
        with self.assertRaisesRegex(ValueError, "unique"):
            make_windows(df, 3, 1, labeled=True)

    def test_bags_do_not_cross_boundaries(self):
        first, first_labels = make_bags(np.zeros((4, 2)), ["a"] * 4, 3)
        second, second_labels = make_bags(np.ones((4, 2)), ["b"] * 4, 3)
        self.assertEqual((len(first), len(second)), (2, 2))
        self.assertEqual(first_labels + second_labels, ["a", "a", "b", "b"])
        self.assertTrue(np.all(first == 0) and np.all(second == 1))

    def test_rejects_short_input(self):
        with self.assertRaisesRegex(ValueError, "Need at least"):
            make_windows(frame(2), 3, 1, labeled=True)

    def test_rejects_missing_joint(self):
        with self.assertRaisesRegex(ValueError, "left_wrist_x"):
            make_windows(frame().drop(columns="left_wrist_x"), 3, 1, labeled=True)

    def test_rejects_even_bag_size(self):
        with self.assertRaisesRegex(ValueError, "odd"):
            make_bags(np.ones((4, 2)), None, 2)

    def test_none_rows_do_not_enter_training_windows(self):
        df = frame()
        df.loc[3, "Action Label"] = "None"
        windows = make_windows(df, 3, 1, labeled=True)
        self.assertEqual(windows.frame_ids.tolist(), [0, 4])

    def test_activity_type(self):
        self.assertEqual(activity_type("Walking"), "normal")
        self.assertEqual(activity_type("Throwing things"), "unusual")

    def test_output_contracts(self):
        df = frame(3)
        df["timestamp"] = ["t0", "t1", "t2"]
        labels = pd.Series(["Walking", "Walking", "Attacking"])
        filled = format_predictions(df, labels, output_format="test-file")
        self.assertEqual(filled["Action Label"].tolist(), labels.tolist())
        submitted = format_predictions(df, labels, output_format="submission", participant_id="P5")
        self.assertEqual(submitted.columns.tolist(),
                         ["participant_id", "timestamp", "predicted_label"])
        self.assertEqual(submitted["participant_id"].tolist(), ["P5"] * 3)

    def test_submission_requires_timestamp(self):
        with self.assertRaisesRegex(ValueError, "timestamp"):
            format_predictions(frame(), pd.Series(["Walking"] * 7),
                               output_format="submission", participant_id="P5")

    def test_bag_does_not_cross_excluded_window_gap(self):
        with self.assertRaisesRegex(ValueError, "No adjacent windows"):
            make_bags(np.ones((3, 2)), ["Walking"] * 3, 3,
                      frame_ids=np.array([0, 1, 4]), expected_step=1)

    def test_window_does_not_cross_frame_gap(self):
        df = frame()
        df.loc[3:, "frame_id"] += 10
        windows = make_windows(df, 3, 1, labeled=True)
        self.assertEqual(windows.frame_ids.tolist(), [0, 13, 14])

    def test_unknown_activity_is_rejected(self):
        df = frame()
        df.loc[0, "Action Label"] = "Unknown action"
        with self.assertRaisesRegex(ValueError, "Unknown activity"):
            make_windows(df, 3, 1, labeled=True)

    def test_inspect_reports_counts_and_hashes(self):
        root = Path("example")
        paths = {uid: root / f"keypoints_with_labels_{uid}.csv" for uid in (1, 2)}
        with patch("isas_pose.audit.subject_files", return_value=paths), \
             patch("isas_pose.audit.pd.read_csv", return_value=frame()), \
             patch("isas_pose.audit.sha256_file", return_value="a" * 64):
            report = inspect_data(root, 3, 1, 3)
        self.assertEqual(len(report["subjects"]), 2)
        self.assertEqual(report["subjects"][0]["rows"], 7)
        self.assertEqual(report["subjects"][0]["bags"], 3)
        self.assertEqual(len(report["subjects"][0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
