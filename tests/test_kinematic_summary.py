"""运动学摘要、阶段明细与图表生成单元测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from algorithm.common.kinematic_plots import (
    generate_segment_kinematic_charts,
    plot_lower_limb_angles,
    plot_racket_kinematics,
    plot_trunk_rotation,
    plot_upper_limb_angles,
)
from algorithm.common.segmentation_helpers import (
    build_kinematic_phase_summary,
    build_kinematic_summary,
    pick_side_columns,
    write_kinematic_phase_summary_csv,
    write_kinematic_summary_csv,
)


def _synthetic_df(n: int = 60) -> pd.DataFrame:
    time_axis = np.linspace(0, 2.0, n)
    df = pd.DataFrame(
        {
            "frame": np.arange(n),
            "time": time_axis,
            "x_clean": np.linspace(100, 400, n),
            "y_clean": np.linspace(200, 100, n),
            "right_shoulder_angle": 90 + 20 * np.sin(np.linspace(0, 3, n)),
            "right_elbow_angle": 120 + 15 * np.sin(np.linspace(0, 2, n)),
            "left_elbow_angle": 110 + 10 * np.sin(np.linspace(0, 2, n)),
            "right_hip_angle": 160 + 10 * np.sin(np.linspace(0, 2, n)),
            "right_knee_angle": 150 + 12 * np.sin(np.linspace(0, 2, n)),
            "left_knee_angle": 155 + 8 * np.sin(np.linspace(0, 2, n)),
            "shoulder_hip_angle": 30 + 10 * np.sin(np.linspace(0, 2, n)),
            "trunk_angle": 5 + 8 * np.sin(np.linspace(0, 2, n)),
            "shoulder_line_angle": 2 + 5 * np.sin(np.linspace(0, 2, n)),
            "racket_head_speed": 100 + 200 * np.sin(np.linspace(0, 4, n)),
            "racket_head_acc": 50 + 80 * np.abs(np.sin(np.linspace(0, 4, n))),
            "racket_head_to_wrist_dist": 40 + 10 * np.sin(np.linspace(0, 2, n)),
            "racket_head_wrist_y_diff": -5 + 3 * np.sin(np.linspace(0, 2, n)),
            "racket_long_axis_angle": 45 + 20 * np.sin(np.linspace(0, 2, n)),
            "racket_width_axis_angle": 10 + 5 * np.sin(np.linspace(0, 2, n)),
            "racket_head_rel_body_x": 20 + 5 * np.sin(np.linspace(0, 2, n)),
            "racket_head_rel_body_y": -10 + 3 * np.sin(np.linspace(0, 2, n)),
            "racket_valid_kpt_count": np.full(n, 5.0),
            "racket_mean_conf": np.full(n, 0.9),
        }
    )
    df.loc[0, "right_elbow_angle"] = np.nan
    return df


class TestKinematicSummary(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _synthetic_df()
        self.time_axis = self.df["time"].to_numpy(dtype=np.float64)
        self.speed = self.df["racket_head_speed"].to_numpy(dtype=np.float64)
        self.side_cols = pick_side_columns("right")

    def test_build_summary_missing_columns_no_crash(self) -> None:
        sparse = self.df[["frame", "time", "x_clean", "y_clean"]].copy()
        sparse["racket_head_speed"] = self.speed
        summary = build_kinematic_summary(
            "forehand",
            sparse,
            self.side_cols,
            5,
            30,
            50,
            sparse["time"].to_numpy(dtype=np.float64),
            self.speed,
            score=0.5,
            segment_id=1,
        )
        self.assertEqual(summary["segment_id"], 1)
        self.assertEqual(summary["action_type"], "forehand")
        self.assertTrue(np.isnan(summary["racket_elbow_angle_range"]))
        self.assertIn("quality_score", summary)

    def test_build_summary_extended_fields(self) -> None:
        summary = build_kinematic_summary(
            "forehand",
            self.df,
            self.side_cols,
            5,
            30,
            50,
            self.time_axis,
            self.speed,
            score=0.8,
            segment_id=2,
            extra_fields={"notes": "test"},
        )
        self.assertEqual(summary["start_frame"], 5)
        self.assertEqual(summary["contact_frame"], 30)
        self.assertEqual(summary["end_frame"], 50)
        self.assertIn("racket_head_acc_peak", summary)
        self.assertIn("racket_head_to_wrist_dist_mean", summary)
        self.assertEqual(summary["notes"], "test")

    def test_phase_summary_forehand_row_count(self) -> None:
        rows = build_kinematic_phase_summary(
            "forehand",
            1,
            self.df,
            self.side_cols,
            5,
            30,
            50,
            self.time_axis,
            self.speed,
        )
        self.assertGreaterEqual(len(rows), 3)
        phases = {r["phase_name"] for r in rows}
        self.assertIn("preparation", phases)
        self.assertIn("contact", phases)

    def test_phase_summary_serve_row_count(self) -> None:
        rows = build_kinematic_phase_summary(
            "serve",
            1,
            self.df,
            self.side_cols,
            5,
            30,
            50,
            self.time_axis,
            self.speed,
            event_times={
                "toss_time": float(self.time_axis[10]),
                "drop_time": float(self.time_axis[20]),
                "hit_time": float(self.time_axis[30]),
            },
        )
        phases = {r["phase_name"] for r in rows}
        self.assertIn("toss", phases)
        self.assertIn("racket_drop", phases)

    def test_write_summary_csv(self) -> None:
        summary = build_kinematic_summary(
            "volley",
            self.df,
            self.side_cols,
            5,
            30,
            50,
            self.time_axis,
            self.speed,
            segment_id=1,
            extra_fields={"elbow_stability_score": 0.9},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_kinematic_summary_csv([summary], Path(tmp) / "test_summary.csv")
            self.assertTrue(Path(path).is_file())
            loaded = pd.read_csv(path)
            self.assertEqual(len(loaded), 1)
            self.assertIn("segment_id", loaded.columns)

    def test_write_phase_csv(self) -> None:
        rows = build_kinematic_phase_summary(
            "volley",
            1,
            self.df,
            self.side_cols,
            5,
            30,
            50,
            self.time_axis,
            self.speed,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_kinematic_phase_summary_csv(rows, Path(tmp) / "test_phase.csv")
            self.assertTrue(Path(path).is_file())
            loaded = pd.read_csv(path)
            self.assertGreater(len(loaded), 0)


class TestKinematicPlots(unittest.TestCase):
    def setUp(self) -> None:
        self.df = _synthetic_df()
        self.time_axis = self.df["time"].to_numpy(dtype=np.float64)
        self.speed = self.df["racket_head_speed"].to_numpy(dtype=np.float64)
        self.side_cols = pick_side_columns("right")

    def test_plot_functions_write_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            plot_upper_limb_angles(
                base / "upper.png",
                self.df,
                self.side_cols,
                self.time_axis,
                5,
                50,
                30,
                "测试上肢",
            )
            plot_lower_limb_angles(
                base / "lower.png",
                self.df,
                self.side_cols,
                self.time_axis,
                5,
                50,
                30,
                "测试下肢",
            )
            plot_trunk_rotation(
                base / "trunk.png",
                self.df,
                self.time_axis,
                5,
                50,
                30,
                "测试躯干",
            )
            plot_racket_kinematics(
                base / "racket.png",
                self.df,
                self.time_axis,
                self.speed,
                5,
                50,
                30,
                "测试球拍",
            )
            for name in ("upper.png", "lower.png", "trunk.png", "racket.png"):
                self.assertTrue((base / name).is_file())
                self.assertGreater((base / name).stat().st_size, 100)

    def test_generate_segment_charts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            paths = generate_segment_kinematic_charts(
                out,
                "testvid",
                1,
                "正手",
                self.df,
                self.side_cols,
                self.time_axis,
                self.speed,
                5,
                50,
                30,
            )
            self.assertEqual(len(paths), 4)
            for p in paths.values():
                self.assertTrue(Path(p).is_file())


class TestBuildArtifactList(unittest.TestCase):
    def test_includes_kinematic_kinds(self) -> None:
        from backend.services.pipeline import build_artifact_list

        result = {
            "run_id": "test",
            "video_name": "vid",
            "artifacts": {
                "racket_csv": "",
                "kinematic_summary_csv": str(Path("data/outputs/r/vid/vid_forehand_kinematic_summary.csv")),
                "kinematic_phase_summary_csv": str(Path("data/outputs/r/vid/vid_forehand_kinematic_phase_summary.csv")),
                "upper_limb_charts": [str(Path("data/outputs/r/vid/vid_upper_limb_angles_1.png"))],
                "lower_limb_charts": [],
                "trunk_rotation_charts": [],
                "racket_kinematic_charts": [],
            },
        }
        items = build_artifact_list(result)
        kinds = {x["kind"] for x in items}
        self.assertIn("kinematic_summary_csv", kinds)
        self.assertIn("kinematic_phase_summary_csv", kinds)
        self.assertIn("upper_limb_angle_chart", kinds)


if __name__ == "__main__":
    unittest.main()
