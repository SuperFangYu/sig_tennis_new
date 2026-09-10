import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from qualisys.config import ANGLE_COLUMNS
from qualisys.core.action_profiles import get_action_profile
from qualisys.core.alignment import (
    derive_manual_repetition_windows,
    derive_repetition_windows,
    map_video_times_to_qualisys,
    match_repetition_peaks,
    prepare_speed_signal,
    select_repetition_peaks_in_ranges,
)
from qualisys.core.io import create_run_directory
from qualisys.core.metrics import align_angle_curves, calculate_angle_metrics
from qualisys.core.qtm import build_qualisys_angles, read_qtm_3d_tsv
from qualisys.core.runner import _select_video_time_range, _validate_video_repetition_ranges


def _gaussian_speed(time: np.ndarray, peaks: list[float]) -> np.ndarray:
    speed = np.full(len(time), 0.03, dtype=float)
    for peak in peaks:
        speed += np.exp(-0.5 * np.square((time - peak) / 0.10))
    return speed


class QualisysFrameworkTests(unittest.TestCase):
    def test_all_five_action_profiles_exist(self):
        for action in (
            "forehand",
            "backhand",
            "forehand_volley",
            "backhand_volley",
            "serve",
        ):
            self.assertGreater(get_action_profile(action).max_pre_s, 0)

    def test_every_run_creates_a_new_timestamped_output_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixed = datetime(2026, 9, 8, 15, 30, 12, 123000)
            first = create_run_directory(
                "forehand", "fy_zs_1", output_root=Path(temp_dir), now=fixed
            )
            second = create_run_directory(
                "forehand", "fy_zs_1", output_root=Path(temp_dir), now=fixed
            )
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())
            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, Path(temp_dir) / "forehand" / "fy_zs_1")

    def test_video_time_range_keeps_absolute_video_seconds(self):
        source = pd.DataFrame(
            {
                "time": np.arange(0.0, 10.1, 0.1),
                "racket_head_speed": np.ones(101),
            }
        )
        selected = _select_video_time_range(source, (2.0, 7.0))
        self.assertAlmostEqual(float(selected["time"].iloc[0]), 2.0)
        self.assertAlmostEqual(float(selected["time"].iloc[-1]), 7.0)
        with self.assertRaisesRegex(ValueError, "开始秒 < 结束秒"):
            _select_video_time_range(source, (7.0, 2.0))

    def test_manual_video_repetition_ranges_select_one_peak_per_action(self):
        profile = get_action_profile("serve")
        time = np.arange(0.0, 30.0, 0.02)
        expected_peaks = [2.0, 8.0, 14.0, 20.0, 26.0]
        ranges = ((1.0, 3.0), (7.0, 9.0), (13.0, 15.0), (19.0, 21.0), (25.0, 27.0))
        source = pd.DataFrame(
            {"time": time, "racket_head_speed": _gaussian_speed(time, expected_peaks)}
        )
        normalized = _validate_video_repetition_ranges(
            source,
            ranges,
            expected_repetitions=5,
        )
        prepared = prepare_speed_signal(
            source,
            profile=profile,
            expected_repetitions=5,
        )
        indices = select_repetition_peaks_in_ranges(prepared, normalized)
        np.testing.assert_allclose(
            prepared.frame.loc[indices, "time"].to_numpy(float),
            expected_peaks,
            atol=0.04,
        )

    def test_manual_repetition_windows_keep_user_boundaries(self):
        profile = get_action_profile("serve")
        video_time = np.arange(0.0, 30.0, 0.02)
        video_peaks = [2.0, 8.0, 14.0, 20.0, 26.0]
        ranges = ((1.0, 3.0), (7.0, 9.0), (13.0, 15.0), (19.0, 21.0), (25.0, 27.0))
        qtm_time = np.arange(0.0, 32.0, 0.01)
        slope, intercept = 1.05, 0.4
        qtm_peaks = [slope * value + intercept for value in video_peaks]
        video = prepare_speed_signal(
            pd.DataFrame(
                {"time": video_time, "racket_head_speed": _gaussian_speed(video_time, video_peaks)}
            ),
            profile=profile,
            expected_repetitions=5,
        )
        qtm = prepare_speed_signal(
            pd.DataFrame(
                {"time": qtm_time, "racket_head_speed": _gaussian_speed(qtm_time, qtm_peaks)}
            ),
            profile=profile,
            expected_repetitions=5,
        )
        fixed = select_repetition_peaks_in_ranges(video, ranges)
        result = match_repetition_peaks(
            video,
            qtm,
            expected_repetitions=5,
            fixed_video_peak_indices=fixed,
        )
        windows = derive_manual_repetition_windows(ranges, result)
        np.testing.assert_allclose(windows["video_start_time"], [value[0] for value in ranges])
        np.testing.assert_allclose(windows["video_end_time"], [value[1] for value in ranges])

    def test_manual_video_repetition_ranges_reject_overlap(self):
        source = pd.DataFrame({"time": np.arange(0.0, 10.1, 0.1)})
        with self.assertRaisesRegex(ValueError, "不能重叠"):
            _validate_video_repetition_ranges(
                source,
                ((0.5, 2.0), (1.5, 3.0), (3.5, 4.5), (5.0, 6.0), (7.0, 8.0)),
                expected_repetitions=5,
            )

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

    def test_piecewise_map_preserves_a_linear_five_peak_case(self):
        profile = get_action_profile("forehand")
        video_time = np.arange(0.0, 14.0, 0.02)
        video_peaks = [1.5, 4.0, 6.5, 9.0, 11.5]
        slope = 1.228
        intercept = 0.70
        qtm_time = np.arange(0.0, 16.0, 0.01)
        qtm_peaks = [slope * value + intercept for value in video_peaks]
        video = prepare_speed_signal(
            pd.DataFrame(
                {"time": video_time, "racket_head_speed": _gaussian_speed(video_time, video_peaks)}
            ),
            profile=profile,
            expected_repetitions=5,
        )
        qtm = prepare_speed_signal(
            pd.DataFrame(
                {"time": qtm_time, "racket_head_speed": _gaussian_speed(qtm_time, qtm_peaks)}
            ),
            profile=profile,
            expected_repetitions=5,
        )

        result = match_repetition_peaks(video, qtm, expected_repetitions=5)
        windows = derive_repetition_windows(video, result, profile=profile)

        self.assertAlmostEqual(result.slope, slope, places=2)
        self.assertAlmostEqual(result.intercept, intercept, delta=0.015)
        self.assertLess(result.rmse_seconds, 0.02)
        self.assertEqual(len(result.anchors), 5)
        self.assertEqual(len(result.segments), 4)
        self.assertEqual(len(windows), 5)
        np.testing.assert_allclose(
            map_video_times_to_qualisys(video_peaks, result),
            qtm_peaks,
            atol=0.015,
        )

    def test_piecewise_map_accepts_nonlinear_clock_intervals(self):
        profile = get_action_profile("serve")
        video_time = np.arange(0.0, 30.0, 0.02)
        video_peaks = [2.0, 8.0, 14.0, 20.0, 26.0]
        qtm_time = np.arange(0.0, 32.0, 0.01)
        qtm_peaks = [1.0, 8.6, 14.8, 22.4, 29.2]
        video = prepare_speed_signal(
            pd.DataFrame(
                {"time": video_time, "racket_head_speed": _gaussian_speed(video_time, video_peaks)}
            ),
            profile=profile,
            expected_repetitions=5,
        )
        qtm = prepare_speed_signal(
            pd.DataFrame(
                {"time": qtm_time, "racket_head_speed": _gaussian_speed(qtm_time, qtm_peaks)}
            ),
            profile=profile,
            expected_repetitions=5,
        )

        result = match_repetition_peaks(video, qtm, expected_repetitions=5)

        self.assertGreater(result.rmse_seconds, 0.20)
        np.testing.assert_allclose(
            map_video_times_to_qualisys(video_peaks, result),
            qtm_peaks,
            atol=0.02,
        )
        np.testing.assert_allclose(
            result.segments["local_slope"],
            np.diff(qtm_peaks) / np.diff(video_peaks),
            atol=0.01,
        )

    def test_all_angles_share_one_time_map_and_zero_error(self):
        slope = 1.02
        intercept = 0.6
        video_time = np.arange(0.0, 10.01, 0.1)
        qtm_time = np.arange(0.0, 11.01, 0.05)
        video_peaks = [1.0, 3.0, 5.0, 7.0, 9.0]
        qtm_peaks = [slope * value + intercept for value in video_peaks]
        profile = get_action_profile("forehand")
        video_signal = prepare_speed_signal(
            pd.DataFrame(
                {"time": video_time, "racket_head_speed": _gaussian_speed(video_time, video_peaks)}
            ),
            profile=profile,
            expected_repetitions=5,
        )
        qtm_signal = prepare_speed_signal(
            pd.DataFrame(
                {"time": qtm_time, "racket_head_speed": _gaussian_speed(qtm_time, qtm_peaks)}
            ),
            profile=profile,
            expected_repetitions=5,
        )
        alignment = match_repetition_peaks(
            video_signal,
            qtm_signal,
            expected_repetitions=5,
        )
        rtmpose = pd.DataFrame({"frame": np.arange(len(video_time)), "time": video_time})
        qualisys = pd.DataFrame({"frame": np.arange(len(qtm_time)), "time": qtm_time})
        for index, angle in enumerate(ANGLE_COLUMNS):
            rtmpose[angle] = 70.0 + index + 2.0 * video_time
            qualisys[angle] = 70.0 + index + 2.0 * ((qtm_time - intercept) / slope)
        ranges = ((0.5, 1.5), (2.5, 3.5), (4.5, 5.5), (6.5, 7.5), (8.5, 9.5))
        windows = derive_manual_repetition_windows(
            ranges,
            alignment,
        )

        aligned = align_angle_curves(
            rtmpose,
            qualisys,
            windows,
            alignment=alignment,
        )
        metrics = calculate_angle_metrics(aligned)

        for angle in ANGLE_COLUMNS:
            np.testing.assert_allclose(aligned[f"{angle}_error"], 0.0, atol=0.05)
        self.assertLess(float(metrics["mae_deg"].max()), 0.05)


if __name__ == "__main__":
    unittest.main()
