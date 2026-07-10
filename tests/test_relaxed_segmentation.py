"""发球与截击宽松筛选的回归测试。"""

from __future__ import annotations

import unittest
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

# 这些测试只覆盖数值判定逻辑；最小测试环境可不安装绘图、视频与 SciPy。
try:
    import matplotlib.pyplot  # noqa: F401
except ModuleNotFoundError:
    matplotlib_stub = types.ModuleType("matplotlib")
    pyplot_stub = MagicMock()
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = pyplot_stub

try:
    import moviepy  # noqa: F401
except ModuleNotFoundError:
    moviepy_stub = types.ModuleType("moviepy")
    moviepy_stub.VideoFileClip = MagicMock()
    sys.modules["moviepy"] = moviepy_stub

try:
    import scipy.signal  # noqa: F401
except ModuleNotFoundError:
    scipy_stub = types.ModuleType("scipy")
    signal_stub = types.ModuleType("scipy.signal")
    signal_stub.find_peaks = MagicMock()
    signal_stub.savgol_filter = MagicMock()
    scipy_stub.signal = signal_stub
    sys.modules["scipy"] = scipy_stub
    sys.modules["scipy.signal"] = signal_stub

from algorithm.common.segmentation_helpers import pick_side_columns
from algorithm.serve.segmentation import _detect_event_chain, _evaluate_serve_candidate
from algorithm.volley.segmentation import _evaluate_volley_candidate


def _angle_columns(n: int) -> dict[str, np.ndarray]:
    phase = np.linspace(0.0, 3.0, n)
    return {
        "right_shoulder_angle": 90.0 + 12.0 * np.sin(phase),
        "right_elbow_angle": 120.0 + 15.0 * np.sin(phase),
        "right_hip_angle": 150.0 + 10.0 * np.sin(phase),
        "right_knee_angle": 145.0 + 12.0 * np.sin(phase),
        "left_knee_angle": 150.0 + 8.0 * np.sin(phase),
        "shoulder_hip_angle": 25.0 + 8.0 * np.sin(phase),
        "trunk_angle": 5.0 + 6.0 * np.sin(phase),
    }


class TestRelaxedServeSegmentation(unittest.TestCase):
    def test_incomplete_event_chain_falls_back_to_peak_anchor(self) -> None:
        n, fps, peak = 120, 30, 90
        time_axis = np.arange(n, dtype=np.float64) / fps
        toss_wrist_y = np.full(n, 260.0)
        toss_wrist_y[70] = 210.0
        racket_y = np.full(n, 180.0)
        racket_y[80] = 330.0
        racket_y[peak] = 80.0
        speed = np.full(n, 50.0)
        speed[75] = 900.0  # 速度峰早于挠背点，使严格事件链顺序失败。

        self.assertIsNone(
            _detect_event_chain(peak, toss_wrist_y, racket_y, speed, time_axis, fps)
        )

        data = {
            "time": time_axis,
            "racket_long_axis_angle": np.linspace(10.0, 80.0, n),
            **_angle_columns(n),
        }
        df = pd.DataFrame(data)
        body_y = np.full(n, 350.0)
        ok, event, reason = _evaluate_serve_candidate(
            df,
            peak,
            toss_wrist_y,
            racket_y,
            body_y,
            speed,
            time_axis,
            fps,
            pick_side_columns("right"),
            True,
        )

        self.assertTrue(ok, reason)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["chain_mode"], "relaxed")
        self.assertIn("事件链顺序不完整", event["relaxed_reasons"])
        self.assertLess(event["toss_idx"], event["drop_idx"])
        self.assertLess(event["drop_idx"], event["hit_idx"])


class TestRelaxedVolleySegmentation(unittest.TestCase):
    def test_x_displacement_is_not_a_hard_filter(self) -> None:
        n, fps, contact = 90, 30, 45
        time_axis = np.arange(n, dtype=np.float64) / fps
        speed = np.full(n, 60.0)
        speed[contact] = 900.0
        acc = np.full(n, 50.0)
        acc[contact] = 800.0

        for x_span in (5.0, 400.0):
            with self.subTest(x_span=x_span):
                df = pd.DataFrame(
                    {
                        "time": time_axis,
                        "x_clean": np.linspace(100.0, 100.0 + x_span, n),
                        "y_clean": np.full(n, 180.0),
                        "racket_head_wrist_y_diff": np.full(n, -10.0),
                        **_angle_columns(n),
                    }
                )
                ok, _, _, _, _, _, reason = _evaluate_volley_candidate(
                    df,
                    contact,
                    max(0, contact - 12),
                    min(n - 1, contact + 12),
                    df["x_clean"].to_numpy(dtype=np.float64),
                    df["y_clean"].to_numpy(dtype=np.float64),
                    speed,
                    acc,
                    fps,
                    pick_side_columns("right"),
                    True,
                )
                self.assertTrue(ok, reason)


if __name__ == "__main__":
    unittest.main()
