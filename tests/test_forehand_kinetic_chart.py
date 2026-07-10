"""正手动力链图：接触帧索引与落盘绘图（无视频依赖）。"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from algorithm.forehand.plot_kinetic_chain import compute_contact_index, save_forehand_kinetic_chart


class TestForehandKineticChart(unittest.TestCase):
    def test_compute_contact_index_peak_in_window(self) -> None:
        v = np.array([0.0, 1.0, 9.0, 3.0, 0.0], dtype=np.float64)
        self.assertEqual(compute_contact_index(v, 1, 3), 2)

    def test_compute_contact_index_degenerate(self) -> None:
        v = np.ones(5, dtype=np.float64)
        self.assertEqual(compute_contact_index(v, 2, 1), 2)

    def test_save_forehand_kinetic_chart_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "demo_kinetic_chain_1.png"
            t = np.linspace(0.0, 0.5, 30)
            knee = 160 + 10 * np.sin(t * 20)
            sh = 25 + 3 * np.cos(t * 15)
            elbow = 100 + 20 * np.sin(t * 10)
            save_forehand_kinetic_chart(
                out,
                t,
                knee,
                sh,
                elbow,
                t_contact=0.28,
                title="单元测试动力链图",
            )
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 5000)


if __name__ == "__main__":
    unittest.main()
