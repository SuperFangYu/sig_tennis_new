import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from qualisys.compare_angles import _read_events, compare_trial
from qualisys.config import ANGLE_COLUMNS
from qualisys.run_qualisys import build_qualisys_angles, read_qtm_3d_tsv
from qualisys.trials import Trial, load_trials


class QualisysFrameworkTests(unittest.TestCase):
    def test_manifest_pairs_video_and_3d_without_force_plate_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trial_dir = root / "trial_001"
            trial_dir.mkdir()
            (trial_dir / "video.mp4").touch()
            (trial_dir / "markers_3d.tsv").touch()
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "trial_id",
                        "participant_id",
                        "action",
                        "handedness",
                        "view_plane",
                        "video_file",
                        "qualisys_3d_file",
                        "marker_map_file",
                        "notes",
                    ]
                )
                writer.writerow(
                    [
                        "trial_001",
                        "P001",
                        "forehand",
                        "right",
                        "ZY",
                        "trial_001/video.mp4",
                        "trial_001/markers_3d.tsv",
                        "",
                        "paired",
                    ]
                )

            trials = load_trials(manifest)
            self.assertEqual(len(trials), 1)
            self.assertEqual(trials[0].trial_id, "trial_001")
            self.assertEqual(trials[0].video_path, (trial_dir / "video.mp4").resolve())

    def test_qtm_reader_rejects_analog_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "analog.tsv"
            path.write_text(
                "FILE_VERSION\t2.0.0\nDATA_INCLUDED\tAnalog\nFrame\tTime\tForce X\n1\t0\t1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "不是 Qualisys 3D"):
                read_qtm_3d_tsv(path)

    def test_build_qualisys_angles_uses_zy_internal_angles(self):
        points = {
            "left_shoulder": (0.0, 1.0),
            "left_elbow": (-1.0, 1.0),
            "left_wrist": (-1.0, 0.0),
            "left_hip": (0.0, 0.0),
            "left_knee": (1.0, 0.0),
            "left_ankle": (1.0, -1.0),
            "right_shoulder": (10.0, 1.0),
            "right_elbow": (9.0, 1.0),
            "right_wrist": (9.0, 0.0),
            "right_hip": (10.0, 0.0),
            "right_knee": (11.0, 0.0),
            "right_ankle": (11.0, -1.0),
        }
        data = {"Frame": [10, 11, 12], "Time": [5.0, 5.01, 5.02]}
        marker_map = {}
        for joint, (z_value, y_value) in points.items():
            marker = f"marker_{joint}"
            marker_map[joint] = [marker]
            data[f"{marker} Z"] = [z_value] * 3
            data[f"{marker} Y"] = [y_value] * 3
        angles = build_qualisys_angles(pd.DataFrame(data), marker_map)

        self.assertEqual(list(angles.columns), ["frame", "time", *ANGLE_COLUMNS])
        np.testing.assert_allclose(angles["time"], [0.0, 0.01, 0.02], atol=1e-12)
        np.testing.assert_allclose(angles[list(ANGLE_COLUMNS)], 90.0, atol=1e-12)

    def test_compare_trial_uses_one_time_map_for_all_angles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            intermediate_root = root / "intermediate"
            output_root = root / "output"
            trial = Trial(
                trial_id="trial_001",
                participant_id="P001",
                action="forehand",
                handedness="right",
                view_plane="ZY",
                video_path=root / "unused.mp4",
                qualisys_3d_path=root / "unused.tsv",
            )
            with (
                patch("qualisys.trials.INTERMEDIATE_ROOT", intermediate_root),
                patch("qualisys.trials.OUTPUT_ROOT", output_root),
            ):
                trial.ensure_output_dirs()
                time = np.arange(0.0, 3.01, 0.1)
                base = pd.DataFrame({"frame": np.arange(len(time)) + 1, "time": time})
                for index, angle in enumerate(ANGLE_COLUMNS):
                    base[angle] = 80.0 + index + 5.0 * time
                base.to_csv(trial.intermediate_dir / "rtmpose_8_angles.csv", index=False)
                base.to_csv(trial.intermediate_dir / "qualisys_8_angles.csv", index=False)
                events = pd.DataFrame(
                    [(1, "start", 0.5), (1, "contact", 1.5), (1, "end", 2.5)],
                    columns=["repetition", "event", "time"],
                )
                events.to_csv(trial.intermediate_dir / "video_events.csv", index=False)
                events.to_csv(trial.intermediate_dir / "qualisys_events.csv", index=False)

                outputs = compare_trial(trial)
                aligned = pd.read_csv(outputs["aligned_angles"])
                metrics = pd.read_csv(outputs["angle_metrics"])
                qc = pd.read_csv(outputs["alignment_qc"])

            self.assertTrue(np.allclose(qc["slope"], 1.0))
            self.assertTrue(np.allclose(qc["intercept"], 0.0, atol=1e-12))
            for angle in ANGLE_COLUMNS:
                self.assertTrue(np.allclose(aligned[f"{angle}_error"], 0.0, atol=1e-10))
            self.assertTrue(np.allclose(metrics["mae"], 0.0, atol=1e-10))

    def test_event_order_must_be_start_contact_end(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.csv"
            pd.DataFrame(
                [(1, "start", 1.0), (1, "contact", 0.5), (1, "end", 2.0)],
                columns=["repetition", "event", "time"],
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "start < contact < end"):
                _read_events(path, "视频")


if __name__ == "__main__":
    unittest.main()
