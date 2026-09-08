"""生成只用于检查时间对齐的双速度曲线图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_alignment_qc_plot(
    video_signal: pd.DataFrame,
    qtm_signal: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    for axis, data, title in (
        (axes[0], video_signal, "Video racket-head speed"),
        (axes[1], qtm_signal, "Qualisys racket-head speed (ZY)"),
    ):
        axis.plot(data["time"], data["speed_normalized"], color="#4063d8", linewidth=1.4)
        candidates = data[data["candidate_peak"] == 1]
        selected = data[data["selected_anchor"] == 1]
        axis.scatter(
            candidates["time"], candidates["speed_normalized"],
            s=24, color="#f2a900", label="candidate"
        )
        axis.scatter(
            selected["time"], selected["speed_normalized"],
            s=52, color="#d62728", marker="x", label="selected anchor"
        )
        axis.set_title(title)
        axis.set_ylabel("normalized speed")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[1].set_xlabel("time (s, each recording starts at 0)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


__all__ = ["save_alignment_qc_plot"]
