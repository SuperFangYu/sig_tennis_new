"""
正手动力链与发力特征图：Matplotlib 绘图与落盘（与切分循环解耦，便于测试与维护）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def compute_contact_index(v: np.ndarray, p: int, end_idx: int) -> int:
    """在 [p, end_idx] 闭区间上取 v 最大处的索引；退化时返回 p。"""
    if p > end_idx:
        return p
    seg = v[p : end_idx + 1]
    if seg.size == 0:
        return p
    return p + int(np.argmax(seg))


def phase_background_times(time_start: float, time_end: float, time_contact: float) -> Tuple[float, float, float]:
    """
    以击球时刻为中心的自适应「核心挥拍区」（绿区）宽度，返回 (t_green_start, t_green_end, time_contact_clipped)。

    - clip_duration = time_end - time_start
    - hit_window = clip(clip_duration * 0.20, 0.2, 0.4)
    - 绿区：击球前 65% + 击球后 35%，并限制在 [time_start, time_end] 内
    """
    ts = float(min(time_start, time_end))
    te = float(max(time_start, time_end))
    tc = float(np.clip(time_contact, ts, te))
    clip_duration = te - ts
    if clip_duration <= 0:
        hit_window = 0.2
    else:
        base_window = clip_duration * 0.20
        hit_window = float(np.clip(base_window, 0.2, 0.4))
    t_green_start = max(ts, tc - hit_window * 0.65)
    t_green_end = min(te, tc + hit_window * 0.35)
    if t_green_end < t_green_start:
        t_green_end = t_green_start
    return t_green_start, t_green_end, tc


def save_forehand_kinetic_chart(
    output_path: Union[str, Path],
    time_seg: np.ndarray,
    knee_deg: np.ndarray,
    shoulder_hip_deg: np.ndarray,
    elbow_deg: np.ndarray,
    t_contact: float,
    title: str,
    y_label: str = "角度 (°)",
) -> str:
    """
    绘制单段正手动力链折线图并保存 PNG，保存后关闭 figure。

    横轴为切片时间 ``time_seg``（即 time_start .. time_end）。三阶段底色：
    - 准备（gold）：[time_start, t_green_start]
    - 挥拍（limegreen）：[t_green_start, t_green_end]，宽度由击球锚点自适应
    - 随挥（skyblue）：[t_green_end, time_end]
    """
    output_path = Path(output_path)
    time_seg = np.asarray(time_seg, dtype=np.float64)
    time_start = float(time_seg[0])
    time_end = float(time_seg[-1])
    if time_end < time_start:
        time_start, time_end = time_end, time_start

    t_green_start, t_green_end, tc = phase_background_times(time_start, time_end, t_contact)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.axvspan(time_start, t_green_start, color="gold", alpha=0.15, label="准备/引拍")
    ax.axvspan(t_green_start, t_green_end, color="limegreen", alpha=0.15, label="挥拍/击球")
    ax.axvspan(t_green_end, time_end, color="skyblue", alpha=0.15, label="随挥")

    ax.plot(time_seg, knee_deg, color="#e6550d", linewidth=1.8, label="膝角")
    ax.plot(time_seg, shoulder_hip_deg, color="#756bb1", linewidth=1.8, label="肩髋夹角")
    ax.plot(time_seg, elbow_deg, color="#3182bd", linewidth=1.8, label="肘角")

    # 击球瞬间竖虚线：暂时不绘制
    # if time_start <= tc <= time_end:
    #     ax.axvline(tc, color="black", linestyle="--", linewidth=1.2, label="击球瞬间")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(time_start, time_end)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)
    return str(output_path)


def save_forehand_speed_cog_chart(
    output_path: Union[str, Path],
    time_seg: np.ndarray,
    v_head_clean: np.ndarray,
    body_center_y_seg: np.ndarray,
    t_contact: float,
    title: str,
) -> str:
    """
    拍头瞬时速度（左轴）与身体重心 Y（右轴，Y 轴翻转以符合「向上蹬伸」视觉）。
    三色相位底色与动力链图一致；击球时刻黑色虚线。
    """
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

    ax1.axvspan(time_start, t_green_start, color="gold", alpha=0.15, label="准备/引拍")
    ax1.axvspan(t_green_start, t_green_end, color="limegreen", alpha=0.15, label="挥拍/击球")
    ax1.axvspan(t_green_end, time_end, color="skyblue", alpha=0.15, label="随挥")

    ax1.plot(
        time_seg,
        v_head_clean,
        color="crimson",
        linewidth=2,
        label="拍头瞬时速度",
    )
    ax2.plot(
        time_seg,
        body_center_y_seg,
        color="royalblue",
        linewidth=2,
        label="身体重心 Y",
    )

    if time_start <= tc <= time_end:
        ax1.axvline(tc, color="black", linestyle="--", linewidth=1.2, label="击球瞬间")

    ax2.invert_yaxis()

    ax1.set_title(title, fontsize=14)
    ax1.set_xlabel("时间 (秒)")
    ax1.set_ylabel("拍头瞬时速度 (pixel/s)")
    ax2.set_ylabel("身体重心 Y轴坐标")
    ax1.grid(True, alpha=0.35)
    ax1.set_xlim(time_start, time_end)

    h1, lab1 = ax1.get_legend_handles_labels()
    h2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, lab1 + lab2, loc="upper right", fontsize=9)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)
    return str(output_path)
