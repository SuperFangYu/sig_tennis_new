"""
2D 图像平面运动学曲线图：上肢/下肢/躯干/球拍 per-segment 可视化。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _smooth_series(y: np.ndarray, window: int = 5) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return y.copy()
    if np.any(np.isnan(y)):
        y = pd.Series(y).interpolate(limit_direction="both").bfill().ffill().to_numpy(dtype=np.float64)
    if y.size < window:
        return y.copy()
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def _plot_series(ax, time_seg: np.ndarray, values: np.ndarray, label: str, **kwargs) -> None:
    v = np.asarray(values, dtype=np.float64)
    if not np.any(np.isfinite(v)):
        return
    ax.plot(time_seg, _smooth_series(v), label=label, **kwargs)


def _add_markers(
    ax,
    time_start: float,
    time_end: float,
    time_contact: float,
    event_times: Optional[Dict[str, float]] = None,
    highlight_contact: bool = False,
) -> None:
    event_times = event_times or {}
    ax.axvline(time_start, color="gray", linestyle=":", linewidth=1.2, label="起始")
    ax.axvline(time_contact, color="black", linestyle="--", linewidth=1.5, label="击球")
    ax.axvline(time_end, color="gray", linestyle="-.", linewidth=1.2, label="结束")
    if "toss_time" in event_times:
        ax.axvline(event_times["toss_time"], color="orange", linestyle="--", linewidth=1.2, label="抛球")
    if "drop_time" in event_times:
        ax.axvline(event_times["drop_time"], color="purple", linestyle="--", linewidth=1.2, label="挠背")
    if "hit_time" in event_times:
        ax.axvline(event_times["hit_time"], color="red", linestyle="--", linewidth=1.5, label="击球")
    if highlight_contact:
        hw = 0.15
        ax.axvspan(time_contact - hw * 0.5, time_contact + hw * 0.5, alpha=0.15, color="limegreen", zorder=0)


def _segment_slice(df: pd.DataFrame, time_axis: np.ndarray, start_idx: int, end_idx: int) -> Tuple[np.ndarray, pd.DataFrame]:
    i0 = max(0, min(start_idx, end_idx))
    i1 = min(len(df) - 1, max(start_idx, end_idx))
    sl = slice(i0, i1 + 1)
    return np.asarray(time_axis[sl], dtype=np.float64), df.iloc[sl]


def plot_upper_limb_angles(
    output_path: Union[str, Path],
    df: pd.DataFrame,
    side_cols: Dict[str, str],
    time_axis: np.ndarray,
    start_idx: int,
    end_idx: int,
    contact_idx: int,
    title: str,
    event_times: Optional[Dict[str, float]] = None,
    highlight_contact: bool = False,
) -> str:
    """上肢角度曲线图（2D 图像平面角度）。"""
    time_seg, seg_df = _segment_slice(df, time_axis, start_idx, end_idx)
    if time_seg.size == 0:
        return str(Path(output_path).resolve())

    fig, ax = plt.subplots(figsize=(12, 5))
    t_start, t_end = float(time_seg[0]), float(time_seg[-1])
    ic = int(np.clip(contact_idx, start_idx, end_idx))
    t_contact = float(time_axis[ic])

    shoulder_col = side_cols.get("racket_shoulder_angle", "right_shoulder_angle")
    elbow_col = side_cols.get("racket_elbow_angle", "right_elbow_angle")
    off_elbow = "left_elbow_angle" if elbow_col.startswith("right") else "right_elbow_angle"

    _plot_series(ax, time_seg, seg_df[shoulder_col].to_numpy() if shoulder_col in seg_df.columns else np.full(len(seg_df), np.nan),
                 "持拍侧肩角", color="#d95319", lw=2)
    _plot_series(ax, time_seg, seg_df[elbow_col].to_numpy() if elbow_col in seg_df.columns else np.full(len(seg_df), np.nan),
                 "持拍侧肘角", color="#0072bd", lw=2)
    _plot_series(ax, time_seg, seg_df[off_elbow].to_numpy() if off_elbow in seg_df.columns else np.full(len(seg_df), np.nan),
                 "非持拍侧肘角", color="#77ac30", lw=1.5, linestyle="--")

    _add_markers(ax, t_start, t_end, t_contact, event_times, highlight_contact)
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("角度 (°)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return str(output_path.resolve())


def plot_lower_limb_angles(
    output_path: Union[str, Path],
    df: pd.DataFrame,
    side_cols: Dict[str, str],
    time_axis: np.ndarray,
    start_idx: int,
    end_idx: int,
    contact_idx: int,
    title: str,
    event_times: Optional[Dict[str, float]] = None,
    highlight_contact: bool = False,
) -> str:
    """下肢角度曲线图。"""
    time_seg, seg_df = _segment_slice(df, time_axis, start_idx, end_idx)
    if time_seg.size == 0:
        return str(Path(output_path).resolve())

    fig, ax = plt.subplots(figsize=(12, 5))
    t_start, t_end = float(time_seg[0]), float(time_seg[-1])
    ic = int(np.clip(contact_idx, start_idx, end_idx))
    t_contact = float(time_axis[ic])

    hip_col = side_cols.get("racket_hip_angle", "right_hip_angle")
    knee_col = side_cols.get("racket_knee_angle", "right_knee_angle")
    support_col = side_cols.get("support_knee_angle", "left_knee_angle")

    _plot_series(ax, time_seg, seg_df[hip_col].to_numpy() if hip_col in seg_df.columns else np.full(len(seg_df), np.nan),
                 "持拍侧髋角", color="#7e2f8e", lw=2)
    _plot_series(ax, time_seg, seg_df[knee_col].to_numpy() if knee_col in seg_df.columns else np.full(len(seg_df), np.nan),
                 "持拍侧膝角", color="#d95319", lw=2)
    _plot_series(ax, time_seg, seg_df[support_col].to_numpy() if support_col in seg_df.columns else np.full(len(seg_df), np.nan),
                 "支撑腿膝角", color="#0072bd", lw=2)

    _add_markers(ax, t_start, t_end, t_contact, event_times, highlight_contact)
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("角度 (°)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return str(output_path.resolve())


def plot_trunk_rotation(
    output_path: Union[str, Path],
    df: pd.DataFrame,
    time_axis: np.ndarray,
    start_idx: int,
    end_idx: int,
    contact_idx: int,
    title: str,
    event_times: Optional[Dict[str, float]] = None,
    highlight_contact: bool = False,
) -> str:
    """肩髋分离/躯干姿态曲线图。"""
    time_seg, seg_df = _segment_slice(df, time_axis, start_idx, end_idx)
    if time_seg.size == 0:
        return str(Path(output_path).resolve())

    fig, ax = plt.subplots(figsize=(12, 5))
    t_start, t_end = float(time_seg[0]), float(time_seg[-1])
    ic = int(np.clip(contact_idx, start_idx, end_idx))
    t_contact = float(time_axis[ic])

    for col, label, color in (
        ("shoulder_hip_angle", "肩髋分离角", "#7e2f8e"),
        ("trunk_angle", "躯干角", "#d95319"),
        ("shoulder_line_angle", "肩线角", "#0072bd"),
    ):
        _plot_series(
            ax, time_seg,
            seg_df[col].to_numpy() if col in seg_df.columns else np.full(len(seg_df), np.nan),
            label, color=color, lw=2,
        )

    _add_markers(ax, t_start, t_end, t_contact, event_times, highlight_contact)
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("角度 (°)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return str(output_path.resolve())


def plot_racket_kinematics(
    output_path: Union[str, Path],
    df: pd.DataFrame,
    time_axis: np.ndarray,
    speed: np.ndarray,
    start_idx: int,
    end_idx: int,
    contact_idx: int,
    title: str,
    event_times: Optional[Dict[str, float]] = None,
    highlight_contact: bool = False,
) -> str:
    """球拍运动学曲线图。"""
    i0 = max(0, min(start_idx, end_idx))
    i1 = min(len(df) - 1, max(start_idx, end_idx))
    sl = slice(i0, i1 + 1)
    time_seg = np.asarray(time_axis[sl], dtype=np.float64)
    if time_seg.size == 0:
        return str(Path(output_path).resolve())

    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    ax2 = ax1.twinx()
    t_start, t_end = float(time_seg[0]), float(time_seg[-1])
    ic = int(np.clip(contact_idx, i0, i1))
    t_contact = float(time_axis[ic])

    speed_seg = np.asarray(speed[sl], dtype=np.float64)
    _plot_series(ax1, time_seg, speed_seg, "拍头速度", color="crimson", lw=2)

    if "racket_head_acc" in df.columns:
        acc_seg = df["racket_head_acc"].iloc[sl].to_numpy(dtype=np.float64)
        _plot_series(ax1, time_seg, acc_seg, "拍头加速度", color="darkred", lw=1.5, linestyle="--")

    if "racket_long_axis_angle" in df.columns:
        axis_seg = df["racket_long_axis_angle"].iloc[sl].to_numpy(dtype=np.float64)
        _plot_series(ax2, time_seg, axis_seg, "球拍长轴角", color="#0072bd", lw=1.5)

    if "racket_head_to_wrist_dist" in df.columns:
        dist_seg = df["racket_head_to_wrist_dist"].iloc[sl].to_numpy(dtype=np.float64)
        _plot_series(ax2, time_seg, dist_seg, "拍头-手腕距离", color="#77ac30", lw=1.5, linestyle=":")

    if "racket_head_wrist_y_diff" in df.columns:
        ydiff_seg = df["racket_head_wrist_y_diff"].iloc[sl].to_numpy(dtype=np.float64)
        _plot_series(ax2, time_seg, ydiff_seg, "拍头相对手腕高度", color="#7e2f8e", lw=1.5, linestyle="-.")

    _add_markers(ax1, t_start, t_end, t_contact, event_times, highlight_contact)
    ax1.set_xlabel("时间 (秒)")
    ax1.set_ylabel("速度/加速度")
    ax2.set_ylabel("角度/距离/像素")
    ax1.set_title(title)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return str(output_path.resolve())


def generate_segment_kinematic_charts(
    output_dir: Union[str, Path],
    video_name: str,
    seg_n: int,
    action_label: str,
    df: pd.DataFrame,
    side_cols: Dict[str, str],
    time_axis: np.ndarray,
    speed: np.ndarray,
    start_idx: int,
    end_idx: int,
    contact_idx: int,
    event_times: Optional[Dict[str, float]] = None,
    highlight_contact: bool = False,
) -> Dict[str, str]:
    """生成单个片段的四类运动学图表，返回路径字典。"""
    output_dir = Path(output_dir)
    base = f"{video_name}"
    suffix = f"_{seg_n}.png"
    title_suffix = f" — {video_name} 片段{seg_n}"

    paths = {
        "upper_limb": plot_upper_limb_angles(
            output_dir / f"{base}_upper_limb_angles{suffix}",
            df, side_cols, time_axis, start_idx, end_idx, contact_idx,
            f"{action_label}上肢角度（2D 图像平面）{title_suffix}",
            event_times, highlight_contact,
        ),
        "lower_limb": plot_lower_limb_angles(
            output_dir / f"{base}_lower_limb_angles{suffix}",
            df, side_cols, time_axis, start_idx, end_idx, contact_idx,
            f"{action_label}下肢角度（2D 图像平面）{title_suffix}",
            event_times, highlight_contact,
        ),
        "trunk": plot_trunk_rotation(
            output_dir / f"{base}_trunk_rotation{suffix}",
            df, time_axis, start_idx, end_idx, contact_idx,
            f"{action_label}躯干旋转（2D 图像平面）{title_suffix}",
            event_times, highlight_contact,
        ),
        "racket": plot_racket_kinematics(
            output_dir / f"{base}_racket_kinematics{suffix}",
            df, time_axis, speed, start_idx, end_idx, contact_idx,
            f"{action_label}球拍运动学{title_suffix}",
            event_times, highlight_contact,
        ),
    }
    return paths
