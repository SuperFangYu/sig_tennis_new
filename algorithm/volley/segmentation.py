"""
截击无监督切分：速度突刺 + 轨迹紧凑 + 拍头高度稳定 + 小角度变化评分。
使用 2D 图像平面运动学代理指标。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from moviepy import VideoFileClip
from scipy.signal import find_peaks, savgol_filter

from algorithm.common.kinematic_plots import generate_segment_kinematic_charts
from algorithm.common.segmentation_helpers import (
    build_kinematic_phase_summary,
    build_kinematic_summary,
    is_left_handed,
    pick_side_columns,
    prepare_segmentation_df,
    range_in_window,
    robust_percentile_threshold,
    score_angle_ranges,
    write_kinematic_phase_summary_csv,
    write_kinematic_summary_csv,
)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

SPEED_PROMINENCE = 400.0
MIN_DISTANCE_SEC = 1.5
FALLBACK_HALF_WINDOW = 0.8
MIN_DURATION = 0.6
MAX_DURATION = 1.0
MIN_SCORE_THRESHOLD = 0.1


def _smooth_series(y: np.ndarray, window: int = 11) -> np.ndarray:
    n = len(y)
    if n < 4:
        return y.copy()
    wl = min(window, n if n % 2 == 1 else n - 1)
    if wl < 3:
        return y.copy()
    try:
        return np.asarray(savgol_filter(y, wl, 3, mode="interp"), dtype=np.float64)
    except ValueError:
        return y.copy()


def _compact_thresholds(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x_valid = x[np.isfinite(x)]
    y_valid = y[np.isfinite(y)]
    x_range = float(np.max(x_valid) - np.min(x_valid)) if x_valid.size else 200.0
    y_range = float(np.max(y_valid) - np.min(y_valid)) if y_valid.size else 200.0
    return max(120.0, x_range * 0.08), max(150.0, y_range * 0.10)


def _search_volley_boundaries(
    contact_idx: int,
    speed: np.ndarray,
    time_axis: np.ndarray,
    fps: int,
) -> Tuple[int, int, float, float]:
    """从 contact 前后搜索速度低谷，限制总时长 0.6–1.0s。"""
    half_min = int(MIN_DURATION * 0.5 * fps)
    half_max = int(MAX_DURATION * 0.5 * fps)
    speed_thr = robust_percentile_threshold(speed, 30, 80.0)

    start_idx = contact_idx
    for i in range(contact_idx, max(0, contact_idx - half_max), -1):
        if np.isfinite(speed[i]) and speed[i] <= speed_thr:
            start_idx = i
            break
    else:
        start_idx = max(0, contact_idx - half_min)

    end_idx = contact_idx
    for i in range(contact_idx, min(len(speed) - 1, contact_idx + half_max)):
        if np.isfinite(speed[i]) and speed[i] <= speed_thr:
            end_idx = i
            break
    else:
        end_idx = min(len(time_axis) - 1, contact_idx + half_min)

    duration = float(time_axis[end_idx] - time_axis[start_idx])
    if duration < MIN_DURATION or duration > MAX_DURATION + 0.15:
        tc = float(time_axis[contact_idx])
        start_t = max(0.0, tc - FALLBACK_HALF_WINDOW)
        end_t = min(float(time_axis[-1]), tc + FALLBACK_HALF_WINDOW)
        start_idx = int(np.searchsorted(time_axis, start_t, side="left"))
        end_idx = int(np.searchsorted(time_axis, end_t, side="right")) - 1
        return start_idx, end_idx, start_t, end_t

    return start_idx, end_idx, float(time_axis[start_idx]), float(time_axis[end_idx])


def _evaluate_volley_candidate(
    df: pd.DataFrame,
    p: int,
    start_idx: int,
    end_idx: int,
    x: np.ndarray,
    y: np.ndarray,
    speed: np.ndarray,
    acc: np.ndarray,
    fps: int,
    side_cols: Dict[str, str],
    has_enhanced: bool,
) -> Tuple[bool, float, int, int, float, float, str]:
    """评估截击候选，返回 (ok, score, idx_start, idx_end, start_t, end_t, reason)。"""
    contact_idx = p
    win_half = int(0.4 * fps)
    w0 = max(0, contact_idx - win_half)
    w1 = min(len(df) - 1, contact_idx + win_half)

    _, y_thr = _compact_thresholds(x, y)
    y_rng = range_in_window(df, "y_clean", w0, w1)
    x_rng = range_in_window(df, "x_clean", w0, w1)

    if np.isfinite(y_rng) and y_rng > y_thr:
        return False, 0.0, 0, 0, 0.0, 0.0, f"Y轴浮动过大({y_rng:.0f}px)"
    # 截击既可能是短促挡击，也可能带有明显前送；X 轴位移只保留为摘要指标，
    # 不再作为硬过滤条件，避免因拍摄视角或动作幅度差异误删真实截击。

    wrist_col = side_cols.get("racket_wrist_y", "right_wrist_y")
    if "racket_head_wrist_y_diff" in df.columns:
        ydiff = float(df["racket_head_wrist_y_diff"].iloc[contact_idx])
    elif wrist_col in df.columns:
        ydiff = float(y[contact_idx] - df[wrist_col].iloc[contact_idx])
    else:
        ydiff = float("nan")

    ydiff_thr = 60.0
    if np.isfinite(ydiff) and ydiff > ydiff_thr:
        return False, 0.0, 0, 0, 0.0, 0.0, "拍头低于手腕(掉拍头)"

    if "racket_head_to_wrist_dist" in df.columns:
        dist_rng = range_in_window(df, "racket_head_to_wrist_dist", w0, w1)
        dist_thr = robust_percentile_threshold(
            df["racket_head_to_wrist_dist"].to_numpy(dtype=np.float64), 60, 40.0
        )
        if np.isfinite(dist_rng) and dist_rng > dist_thr * 0.5:
            return False, 0.0, 0, 0, 0.0, 0.0, "拍头-手腕距离变化过大"

    acc_bonus = 0.0
    if has_enhanced and np.any(np.isfinite(acc)):
        acc_win = acc[max(0, contact_idx - int(0.15 * fps)) : contact_idx + 1]
        if np.any(np.isfinite(acc_win)) and float(np.nanmax(acc_win)) > robust_percentile_threshold(acc, 70, 200.0):
            acc_bonus = 0.15

    idx_start, idx_end, start_t, end_t = _search_volley_boundaries(contact_idx, speed, df["time"].to_numpy(dtype=np.float64), fps)

    angle_weights = {
        "racket_elbow_angle": 1.0,
        "shoulder_hip_angle": 0.9,
        "racket_shoulder_angle": 0.5,
        "trunk_angle": 0.4,
        "support_knee_angle": 0.4,
    }
    score, _ = score_angle_ranges(
        df,
        side_cols,
        idx_start,
        contact_idx,
        idx_end,
        angle_weights,
        invert_small=("racket_elbow_angle", "shoulder_hip_angle", "racket_shoulder_angle", "trunk_angle"),
    )
    score += acc_bonus

    if score < MIN_SCORE_THRESHOLD and has_enhanced:
        return False, score, idx_start, idx_end, start_t, end_t, f"角度评分过低({score:.2f})"

    reason = f"ok (X轴窗口位移仅记录: {x_rng:.0f}px)" if np.isfinite(x_rng) else "ok"
    return True, score, idx_start, idx_end, start_t, end_t, reason


def run_segmentation(
    video_path: Union[str, Path],
    racket_csv: Union[str, Path],
    body_csv: Union[str, Path],
    output_dir: Union[str, Path],
    handedness: str = "right",
) -> Dict[str, Any]:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(str(video_path)))[0]

    print("🚀 正在加载并融合多模态数据...")
    df, fps, has_enhanced = prepare_segmentation_df(racket_csv, body_csv, handedness)

    time_axis = df["time"].to_numpy(dtype=np.float64)
    x_clean = df["x_clean"].to_numpy(dtype=np.float64)
    y_clean = df["y_clean"].to_numpy(dtype=np.float64)
    speed_raw = df["racket_head_speed"].to_numpy(dtype=np.float64)
    speed = _smooth_series(speed_raw)
    acc = df["racket_head_acc"].to_numpy(dtype=np.float64) if "racket_head_acc" in df.columns else np.full(len(df), np.nan)

    side_cols = pick_side_columns(handedness)
    hand_label = "左手" if is_left_handed(handedness) else "右手"

    knee_col = side_cols["support_knee_angle"]
    elbow_col = side_cols["racket_elbow_angle"]
    knee_angle = df[knee_col].to_numpy(dtype=np.float64) if knee_col in df.columns else np.full(len(df), np.nan)
    elbow_angle = df[elbow_col].to_numpy(dtype=np.float64) if elbow_col in df.columns else np.full(len(df), np.nan)
    shoulder_hip_angle = (
        df["shoulder_hip_angle"].to_numpy(dtype=np.float64)
        if "shoulder_hip_angle" in df.columns
        else np.full(len(df), np.nan)
    )

    peaks, properties = find_peaks(
        speed,
        prominence=SPEED_PROMINENCE,
        width=max(1, int(0.1 * fps)),
        distance=int(MIN_DISTANCE_SEC * fps),
    )
    left_bases = properties.get("left_ips", peaks - int(0.3 * fps))
    right_bases = properties.get("right_ips", peaks + int(0.3 * fps))
    if not isinstance(left_bases, np.ndarray):
        left_bases = np.asarray(left_bases)
    if not isinstance(right_bases, np.ndarray):
        right_bases = np.asarray(right_bases)

    print(f"🔍 截击速度突刺分析（持拍手: {hand_label}），共 {len(peaks)} 个候选...")

    accepted: List[Dict[str, Any]] = []
    for i, p in enumerate(peaks):
        start_idx = max(0, int(left_bases[i]) - int(0.4 * fps))
        end_idx = min(len(time_axis) - 1, int(right_bases[i]) + int(0.4 * fps))
        tc = float(time_axis[p])

        ok, score, idx_start, idx_end, start_t, end_t, reason = _evaluate_volley_candidate(
            df, p, start_idx, end_idx, x_clean, y_clean, speed, acc, fps, side_cols, has_enhanced
        )
        if not ok:
            print(f"  [过滤] 击球点: {tc:.2f}s | 状态: 排除 ({reason})")
            continue

        accepted.append(
            {
                "contact_idx": p,
                "score": score,
                "idx_start": idx_start,
                "idx_end": idx_end,
                "start_t": start_t,
                "end_t": end_t,
            }
        )
        print(f"  [保留] 击球点: {tc:.2f}s | 评分: {score:.2f} | 窗 [{start_t:.2f}s–{end_t:.2f}s]")

    accepted.sort(key=lambda x: x["score"], reverse=True)

    volley_intervals: List[Tuple[float, float]] = []
    artifacts_trace: List[str] = []
    artifacts_kinetic: List[str] = []
    upper_limb_charts: List[str] = []
    lower_limb_charts: List[str] = []
    trunk_rotation_charts: List[str] = []
    racket_kinematic_charts: List[str] = []
    kinematic_summaries: List[Dict[str, Any]] = []
    kinematic_phase_rows: List[Dict[str, Any]] = []

    for seg_n, item in enumerate(accepted, start=1):
        contact_idx = item["contact_idx"]
        start_t, end_t = item["start_t"], item["end_t"]
        tc = float(time_axis[contact_idx])
        volley_intervals.append((start_t, end_t))

        sl = slice(
            max(0, contact_idx - int(1.2 * fps)),
            min(len(time_axis), contact_idx + int(1.2 * fps)),
        )
        seg_t = time_axis[sl]
        hit_window = 0.3
        tg_start, tg_end = tc - hit_window * 0.5, tc + hit_window * 0.5
        tg_start, tg_end = tc - hit_window * 0.5, tc + hit_window * 0.5

        idx_start = item["idx_start"]
        idx_end = item["idx_end"]

        chart_paths = generate_segment_kinematic_charts(
            output_dir,
            video_name,
            seg_n,
            "截击",
            df,
            side_cols,
            time_axis,
            speed,
            idx_start,
            idx_end,
            contact_idx,
            highlight_contact=True,
        )
        upper_limb_charts.append(chart_paths["upper_limb"])
        lower_limb_charts.append(chart_paths["lower_limb"])
        trunk_rotation_charts.append(chart_paths["trunk"])
        racket_kinematic_charts.append(chart_paths["racket"])

        elbow_col = side_cols.get("racket_elbow_angle", "right_elbow_angle")
        elbow_rng = range_in_window(df, elbow_col, idx_start, idx_end)
        elbow_stability = float(1.0 / (1.0 + elbow_rng / 15.0)) if np.isfinite(elbow_rng) else float("nan")

        summary = build_kinematic_summary(
            "volley",
            df,
            side_cols,
            idx_start,
            contact_idx,
            idx_end,
            time_axis,
            speed,
            score=item["score"],
            segment_id=seg_n,
            extra_fields={
                "y_range_window": range_in_window(df, "y_clean", idx_start, idx_end),
                "x_range_window": range_in_window(df, "x_clean", idx_start, idx_end),
                "elbow_stability_score": elbow_stability,
            },
        )
        kinematic_summaries.append(summary)
        kinematic_phase_rows.extend(
            build_kinematic_phase_summary(
                "volley",
                seg_n,
                df,
                side_cols,
                idx_start,
                contact_idx,
                idx_end,
                time_axis,
                speed,
            )
        )

        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = ax1.twinx()
        ax1.axvspan(seg_t[0], tg_start, color="gold", alpha=0.15, zorder=0)
        ax1.axvspan(tg_start, tg_end, color="limegreen", alpha=0.15, zorder=0)
        ax1.axvspan(tg_end, seg_t[-1], color="skyblue", alpha=0.15, zorder=0)
        ax1.axvline(tc, color="black", linestyle="--", linewidth=1.5, zorder=4)
        ax1.plot(seg_t, speed[sl], color="crimson", lw=2, label="拍头绝对速度(撞击力)", zorder=3)
        ax2.plot(seg_t, y_clean[sl], color="royalblue", lw=2, label="球拍Y轴轨迹(平稳验证)", zorder=3)
        ax1.set_ylabel("速度 (pixel/s)")
        ax2.set_ylabel("Y轴像素 (向下为正)")
        ax2.invert_yaxis()
        ax1.set_title(f"截击速度突刺与轨迹 - {video_name} (片段{seg_n})")
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
        trace_path = str(output_dir / f"{video_name}_volley_trace_chart_{seg_n}.png")
        fig1.tight_layout()
        fig1.savefig(trace_path, dpi=300)
        plt.close(fig1)
        artifacts_trace.append(trace_path)

        fig2, ax_k = plt.subplots(figsize=(12, 5))
        ax_k.axvspan(seg_t[0], tg_start, color="gold", alpha=0.15, zorder=0)
        ax_k.axvspan(tg_start, tg_end, color="limegreen", alpha=0.15, zorder=0)
        ax_k.axvspan(tg_end, seg_t[-1], color="skyblue", alpha=0.15, zorder=0)
        ax_k.axvline(tc, color="black", linestyle="--", linewidth=1.5, zorder=4)
        s_knee = _smooth_series(knee_angle[sl])
        s_elbow = _smooth_series(elbow_angle[sl])
        s_hip = _smooth_series(shoulder_hip_angle[sl])
        ax_k.plot(seg_t, s_knee, color="#d95319", lw=2, label="支撑腿膝角(垫步降重心)", zorder=3)
        ax_k.plot(seg_t, s_elbow, color="#0072bd", lw=2, label="持拍肘角(稳固V型)", zorder=3)
        ax_k.plot(seg_t, s_hip, color="#7e2f8e", lw=2, label="肩髋夹角(平稳微转)", zorder=3)
        ax_k.set_ylabel("角度 (°)")
        ax_k.set_title(f"截击稳定结构动力链 - {video_name} (片段{seg_n})")
        ax_k.legend(loc="upper right")
        kinetic_path = str(output_dir / f"{video_name}_volley_kinetic_chart_{seg_n}.png")
        fig2.tight_layout()
        fig2.savefig(kinetic_path, dpi=300)
        plt.close(fig2)
        artifacts_kinetic.append(kinetic_path)

    print(f"🎯 确认 {len(volley_intervals)} 次标准截击。")

    summary_csv = write_kinematic_summary_csv(
        kinematic_summaries,
        output_dir / f"{video_name}_volley_kinematic_summary.csv",
    )
    phase_csv = write_kinematic_phase_summary_csv(
        kinematic_phase_rows,
        output_dir / f"{video_name}_volley_kinematic_phase_summary.csv",
    )

    clips: List[str] = []
    if volley_intervals:
        video = VideoFileClip(str(video_path))
        for i, (st, et) in enumerate(volley_intervals):
            out_file = str(output_dir / f"{video_name}_Volley_v2_{i + 1}.mp4")
            try:
                clip = video.subclipped(st, et)
            except AttributeError:
                clip = video.subclip(st, et)
            clip.write_videofile(out_file, codec="libx264", audio=False, logger=None)
            clips.append(out_file)
        video.close()

    return {
        "intervals": volley_intervals,
        "clips": clips,
        "volley_trace_charts": artifacts_trace,
        "volley_kinetic_charts": artifacts_kinetic,
        "racket_csv": str(racket_csv),
        "body_csv": str(body_csv),
        "kinematic_summary_csv": summary_csv,
        "kinematic_phase_summary_csv": phase_csv,
        "upper_limb_charts": upper_limb_charts,
        "lower_limb_charts": lower_limb_charts,
        "trunk_rotation_charts": trunk_rotation_charts,
        "racket_kinematic_charts": racket_kinematic_charts,
    }
