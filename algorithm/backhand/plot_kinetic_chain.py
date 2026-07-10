"""
反手动力链与拍头速度/重心图：Matplotlib 绘图与落盘。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from algorithm.forehand.plot_kinetic_chain import phase_background_times

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def save_backhand_kinetic_chart(
    output_path: Union[str, Path],
    time_seg: np.ndarray,
    knee_deg: np.ndarray,
    shoulder_hip_deg: np.ndarray,
    elbow_deg: np.ndarray,
    t_contact: float,
    title: str,
    y_label: str = "角度 (°)",
) -> str:
    output_path = Path(output_path)
    time_seg = np.asarray(time_seg, dtype=np.float64)
    time_start = float(time_seg[0])
    time_end = float(time_seg[-1])
    if time_end < time_start:
        time_start, time_end = time_end, time_start

    t_green_start, t_green_end, tc = phase_background_times(time_start, time_end, t_contact)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.axvspan(time_start, t_green_start, color="gold", alpha=0.15, label="准备/引拍", zorder=0)
    ax.axvspan(t_green_start, t_green_end, color="limegreen", alpha=0.15, label="挥拍/击球", zorder=0)
    ax.axvspan(t_green_end, time_end, color="skyblue", alpha=0.15, label="随挥", zorder=0)

    ax.plot(time_seg, knee_deg, color="#e6550d", linewidth=1.8, label="膝角", zorder=2)
    ax.plot(time_seg, shoulder_hip_deg, color="#756bb1", linewidth=1.8, label="肩髋夹角", zorder=2)
    ax.plot(time_seg, elbow_deg, color="#3182bd", linewidth=1.8, label="肘角", zorder=2)

    if time_start <= tc <= time_end:
        ax.axvline(tc, color="black", linestyle="--", linewidth=1.2, label="击球瞬间", zorder=3)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(time_start, time_end)
    ax.margins(y=0.1)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)
    return str(output_path)


def save_backhand_speed_cog_chart(
    output_path: Union[str, Path],
    time_seg: np.ndarray,
    v_head_clean: np.ndarray,
    body_center_y_seg: np.ndarray,
    t_contact: float,
    title: str,
) -> str:
    output_path = Path(output_path)
    time_seg = np.asarray(time_seg, dtype=np.float64)
    v_head_clean = np.asarray(v_head_clean, dtype=np.float64)
    body_center_y_seg = np.asarray(body_center_y_seg, dtype=np.float64)

    time_start = float(time_seg[0])
    time_end = float(time_seg[-1])
    if time_end < time_start:
        time_start, time_end = time_end, time_start

    t_green_start, t_green_end, tc = phase_background_times(time_start, time_end, t_contact)

    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    ax2 = ax1.twinx()

    ax1.axvspan(time_start, t_green_start, color="gold", alpha=0.15, label="准备/引拍", zorder=0)
    ax1.axvspan(t_green_start, t_green_end, color="limegreen", alpha=0.15, label="挥拍/击球", zorder=0)
    ax1.axvspan(t_green_end, time_end, color="skyblue", alpha=0.15, label="随挥", zorder=0)

    ax1.plot(
        time_seg,
        v_head_clean,
        color="crimson",
        linewidth=2,
        label="拍头瞬时速度",
        zorder=2,
    )
    ax2.plot(
        time_seg,
        body_center_y_seg,
        color="royalblue",
        linewidth=2,
        label="身体重心 Y",
        zorder=2,
    )

    if time_start <= tc <= time_end:
        ax1.axvline(tc, color="black", linestyle="--", linewidth=1.2, label="击球瞬间", zorder=3)

    ax2.invert_yaxis()

    ax1.set_title(title, fontsize=14)
    ax1.set_xlabel("时间 (秒)")
    ax1.set_ylabel("拍头瞬时速度 (pixel/s)")
    ax2.set_ylabel("身体重心 Y轴坐标")
    ax1.grid(True, alpha=0.35)
    ax1.set_xlim(time_start, time_end)
    ax1.margins(y=0.1)
    ax2.margins(y=0.1)

    h1, lab1 = ax1.get_legend_handles_labels()
    h2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, lab1 + lab2, loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)
    return str(output_path)
