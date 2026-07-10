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
    check_prep_direction_legacy,
    check_prep_direction_rel_body,
    clamp_interval_neighbors,
    compute_swing_width_simple,
    is_left_handed,
    pick_side_columns,
    prepare_segmentation_df,
    refine_contact_by_speed,
    score_angle_ranges,
    search_end_idx,
    search_start_idx,
    write_kinematic_phase_summary_csv,
    write_kinematic_summary_csv,
)
from algorithm.backhand.plot_kinetic_chain import (
    save_backhand_kinetic_chart,
    save_backhand_speed_cog_chart,
)
from algorithm.forehand.plot_kinetic_chain import compute_contact_index

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PROMINENCE_THRESHOLD = 45.0
MIN_DISTANCE_SEC = 1.6
PREP_EXTEND_RATIO = 1.0
PREP_MIN_BUFFER = 1.0
PREP_MAX_BUFFER = 1.5
FOLLOW_EXTEND_RATIO = 0.3
FOLLOW_MIN_BUFFER = 0.2
FOLLOW_MAX_BUFFER = 0.4
MIN_SWING_WIDTH = 200
BODY_TOLERANCE = 150
MIN_SCORE_THRESHOLD = 0.12


def _smooth_angle_segment(y: np.ndarray, fps: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return y.copy()
    if np.any(np.isnan(y)):
        y = pd.Series(y).interpolate(limit_direction="both").bfill().ffill().to_numpy(dtype=np.float64)
    n = int(y.size)
    if n < 4:
        return y.copy()
    wl_target = 15 if fps >= 26 else 11
    wl = min(wl_target, n if n % 2 == 1 else n - 1)
    if wl % 2 == 0:
        wl -= 1
    wl = max(3, wl)
    if wl > n:
        wl = n if n % 2 == 1 else n - 1
    if wl < 3:
        return y.copy()
    polyorder = min(3, wl - 1)
    if polyorder < 1:
        return y.copy()
    try:
        return np.asarray(savgol_filter(y, wl, polyorder, mode="interp"), dtype=np.float64)
    except ValueError:
        return y.copy()


def _swing_width_threshold(x: np.ndarray) -> float:
    valid = x[np.isfinite(x)]
    x_range = float(np.max(valid) - np.min(valid)) if valid.size else 0.0
    return max(MIN_SWING_WIDTH, x_range * 0.15)


def _pick_backhand_chart_series(df: pd.DataFrame, handedness: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """反手图表：支撑侧膝角 + 持拍侧肘角 + 肩髋角。"""
    side = pick_side_columns(handedness)
    n = len(df)
    knee = (
        df[side["support_knee_angle"]].to_numpy(dtype=np.float64)
        if side["support_knee_angle"] in df.columns
        else np.full(n, np.nan)
    )
    elbow = (
        df[side["racket_elbow_angle"]].to_numpy(dtype=np.float64)
        if side["racket_elbow_angle"] in df.columns
        else np.full(n, np.nan)
    )
    sh = (
        df["shoulder_hip_angle"].to_numpy(dtype=np.float64)
        if "shoulder_hip_angle" in df.columns
        else np.full(n, np.nan)
    )
    return knee, sh, elbow


def _evaluate_backhand_candidate(
    df: pd.DataFrame,
    p: int,
    start_idx: int,
    end_idx: int,
    handedness: str,
    has_enhanced: bool,
    x: np.ndarray,
    body_x: np.ndarray,
    speed: np.ndarray,
    fps: int,
) -> Tuple[bool, int, float, float, float, int, int, float, str]:
    time_axis = df["time"].to_numpy(dtype=np.float64)
    contact_idx = refine_contact_by_speed(speed, p, start_idx, end_idx)
    if not np.any(np.isfinite(speed[start_idx : end_idx + 1])):
        y = df["y_clean"].to_numpy(dtype=np.float64)
        dx = np.gradient(x)
        dy = np.gradient(y)
        v_racket = np.sqrt(dx * dx + dy * dy)
        contact_idx = compute_contact_index(v_racket, p, end_idx)

    swing_width = compute_swing_width_simple(x, start_idx, contact_idx, end_idx)
    if swing_width <= _swing_width_threshold(x):
        return False, contact_idx, 0.0, 0.0, 0.0, 0, 0, 0.0, f"横移不足({swing_width:.0f}px)"

    if has_enhanced and "racket_head_rel_body_x" in df.columns:
        prep_ok = check_prep_direction_rel_body(df, start_idx, contact_idx, handedness, forehand=False)
    else:
        prep_ok = check_prep_direction_legacy(
            x, body_x, start_idx, contact_idx, handedness, forehand=False, body_tolerance=BODY_TOLERANCE
        )
    if not prep_ok:
        return False, contact_idx, 0.0, 0.0, 0.0, 0, 0, 0.0, "引拍方向不符"

    side_cols = pick_side_columns(handedness)
    idx_start = search_start_idx(
        speed,
        x,
        df["racket_head_to_wrist_dist"].to_numpy(dtype=np.float64)
        if "racket_head_to_wrist_dist" in df.columns
        else None,
        contact_idx,
        fps,
        max_back_sec=1.8,
    )
    idx_start = min(idx_start, start_idx)
    idx_end = search_end_idx(speed, contact_idx, fps, max_fwd_sec=1.0)
    idx_end = max(idx_end, end_idx)

    original_duration = time_axis[end_idx] - time_axis[start_idx]
    prep_buffer = float(np.clip(original_duration * PREP_EXTEND_RATIO, PREP_MIN_BUFFER, PREP_MAX_BUFFER))
    follow_buffer = float(np.clip(original_duration * FOLLOW_EXTEND_RATIO, FOLLOW_MIN_BUFFER, FOLLOW_MAX_BUFFER))

    start_t = time_axis[idx_start] - prep_buffer
    end_t = time_axis[idx_end] + follow_buffer
    start_t = min(start_t, float(time_axis[contact_idx]) - 0.2)

    angle_weights = {
        "shoulder_hip_angle": 1.2,
        "racket_shoulder_angle": 0.8,
        "racket_elbow_angle": 0.3,
        "support_knee_angle": 0.6,
        "racket_hip_angle": 0.5,
    }
    if "shoulder_turn_x_diff" in df.columns:
        angle_weights["shoulder_turn_x_diff"] = 0.5

    score, _ = score_angle_ranges(df, side_cols, idx_start, contact_idx, idx_end, angle_weights)

    idx_start_clip = int(np.clip(idx_start, 0, len(time_axis) - 1))
    idx_end_clip = int(np.clip(idx_end, idx_start_clip, len(time_axis) - 1))

    return True, contact_idx, start_t, end_t, score, idx_start_clip, idx_end_clip, swing_width, "ok"


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

    TARGET_VIDEO = str(video_path)
    video_name = os.path.splitext(os.path.basename(TARGET_VIDEO))[0]

    print("🚀 正在加载并融合多模态数据...")
    df, fps, has_enhanced = prepare_segmentation_df(racket_csv, body_csv, handedness)

    y_clean = df["y_clean"].to_numpy(dtype=np.float64)
    x_clean = df["x_clean"].to_numpy(dtype=np.float64)
    body_x = df["body_center_x"].to_numpy(dtype=np.float64)
    time_axis = df["time"].to_numpy(dtype=np.float64)
    speed = df["racket_head_speed"].to_numpy(dtype=np.float64)

    body_cog_y = (
        df["body_center_y"].to_numpy(dtype=np.float64)
        if "body_center_y" in df.columns
        else np.full(len(df), np.nan)
    )

    dx_dt = np.gradient(x_clean, time_axis)
    dy_dt = np.gradient(y_clean, time_axis)
    v_head = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)

    knee_series, shoulder_hip_series, elbow_series = _pick_backhand_chart_series(df, handedness)

    peaks, properties = find_peaks(y_clean, prominence=PROMINENCE_THRESHOLD, distance=int(MIN_DISTANCE_SEC * fps))
    left_bases = properties["left_bases"]
    right_bases = properties["right_bases"]
    print(f"✅ 基础识别完成：共发现 {len(peaks)} 个潜在动作波动。")

    use_left = is_left_handed(handedness)
    hand_label = "左手" if use_left else "右手"
    print(f"🔍 反手多维度筛选（持拍手: {hand_label}，handedness-aware 引拍）...")

    accepted: List[Dict[str, Any]] = []
    other_peaks: List[int] = []

    for i, p in enumerate(peaks):
        start_idx = int(left_bases[i])
        end_idx = int(right_bases[i])
        hit_time = float(time_axis[p])

        ok, contact_idx, start_t, end_t, score, idx_start, idx_end, swing_width, reason = _evaluate_backhand_candidate(
            df, p, start_idx, end_idx, handedness, has_enhanced, x_clean, body_x, speed, fps
        )
        if not ok:
            other_peaks.append(p)
            print(f"  [过滤] {hit_time:5.2f}s | 状态: 排除 ({reason})")
            continue

        if score < MIN_SCORE_THRESHOLD and has_enhanced:
            other_peaks.append(p)
            print(f"  [过滤] {hit_time:5.2f}s | 状态: 角度评分过低 ({score:.2f})")
            continue

        prev_hit = float(time_axis[peaks[i - 1]]) if i > 0 else None
        next_hit = float(time_axis[peaks[i + 1]]) if i < len(peaks) - 1 else None
        start_t, end_t = clamp_interval_neighbors(start_t, end_t, hit_time, time_axis, prev_hit, next_hit)

        accepted.append(
            {
                "peak": p,
                "contact_idx": contact_idx,
                "start_t": start_t,
                "end_t": end_t,
                "score": score,
                "idx_start": int(np.searchsorted(time_axis, start_t, side="left")),
                "idx_end": int(np.searchsorted(time_axis, end_t, side="right")) - 1,
                "swing_width": swing_width,
            }
        )

    accepted.sort(key=lambda x: x["score"], reverse=True)

    backhand_intervals: List[Tuple[float, float]] = []
    backhand_peaks: List[int] = []
    kinetic_charts: List[str] = []
    speed_cog_charts: List[str] = []
    upper_limb_charts: List[str] = []
    lower_limb_charts: List[str] = []
    trunk_rotation_charts: List[str] = []
    racket_kinematic_charts: List[str] = []
    kinematic_summaries: List[Dict[str, Any]] = []
    kinematic_phase_rows: List[Dict[str, Any]] = []
    side_cols = pick_side_columns(handedness)

    for seg_n, item in enumerate(accepted, start=1):
        p = item["peak"]
        contact_idx = item["contact_idx"]
        start_t, end_t = item["start_t"], item["end_t"]
        hit_time = float(time_axis[p])

        idx_start = int(np.clip(item["idx_start"], 0, len(time_axis) - 1))
        idx_end = int(np.clip(item["idx_end"], idx_start, len(time_axis) - 1))
        sl_clip = slice(idx_start, idx_end + 1)

        segment_time = np.asarray(time_axis[sl_clip], dtype=np.float64)
        knee_s = _smooth_angle_segment(np.asarray(knee_series[sl_clip], dtype=np.float64), fps)
        sh_s = _smooth_angle_segment(np.asarray(shoulder_hip_series[sl_clip], dtype=np.float64), fps)
        elbow_s = _smooth_angle_segment(np.asarray(elbow_series[sl_clip], dtype=np.float64), fps)

        t_lo, t_hi = float(segment_time[0]), float(segment_time[-1])
        t_contact_c = float(np.clip(float(time_axis[contact_idx]), t_lo, t_hi))

        kc_path = output_dir / f"{video_name}_kinetic_chain_{seg_n}.png"
        save_backhand_kinetic_chart(
            kc_path,
            segment_time,
            knee_s,
            sh_s,
            elbow_s,
            t_contact_c,
            title=f"反手动力链与发力特征 — {video_name} 片段{seg_n}（{hand_label}）",
        )
        kinetic_charts.append(str(kc_path))

        v_seg = np.asarray(v_head[sl_clip], dtype=np.float64)
        v_head_clean = _smooth_angle_segment(v_seg, fps)
        body_seg = np.asarray(body_cog_y[sl_clip], dtype=np.float64)
        if np.any(np.isnan(body_seg)):
            body_seg = pd.Series(body_seg).interpolate(limit_direction="both").bfill().ffill().to_numpy(dtype=np.float64)
        body_seg_clean = _smooth_angle_segment(body_seg, fps)

        sc_path = output_dir / f"{video_name}_speed_cog_{seg_n}.png"
        save_backhand_speed_cog_chart(
            sc_path,
            segment_time,
            v_head_clean,
            body_seg_clean,
            t_contact_c,
            title=f"拍头速度与身体重心 — {video_name} 片段{seg_n}（{hand_label}）",
        )
        speed_cog_charts.append(str(sc_path))

        chart_paths = generate_segment_kinematic_charts(
            output_dir,
            video_name,
            seg_n,
            "反手",
            df,
            side_cols,
            time_axis,
            speed,
            idx_start,
            idx_end,
            contact_idx,
        )
        upper_limb_charts.append(chart_paths["upper_limb"])
        lower_limb_charts.append(chart_paths["lower_limb"])
        trunk_rotation_charts.append(chart_paths["trunk"])
        racket_kinematic_charts.append(chart_paths["racket"])

        summary = build_kinematic_summary(
            "backhand",
            df,
            side_cols,
            idx_start,
            contact_idx,
            idx_end,
            time_axis,
            speed,
            score=item["score"],
            segment_id=seg_n,
        )
        kinematic_summaries.append(summary)
        kinematic_phase_rows.extend(
            build_kinematic_phase_summary(
                "backhand",
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

        print(
            f"  [保留] {hit_time:5.2f}s | 位移: {item['swing_width']:4.0f}px | "
            f"评分: {item['score']:.2f} | 击球 t={t_contact_c:.2f}s | 标准反手 ({hand_label})"
        )
        backhand_peaks.append(p)
        backhand_intervals.append((start_t, end_t))

    print(f"🎯 最终结果：确认 {len(backhand_intervals)} 次标准反手。")

    summary_csv = write_kinematic_summary_csv(
        kinematic_summaries,
        output_dir / f"{video_name}_backhand_kinematic_summary.csv",
    )
    phase_csv = write_kinematic_phase_summary_csv(
        kinematic_phase_rows,
        output_dir / f"{video_name}_backhand_kinematic_phase_summary.csv",
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 11), gridspec_kw={"height_ratios": [2, 1]}, sharex=True
    )
    ax1.plot(time_axis, y_clean, color="#1f77b4", alpha=0.8, linewidth=1.5, label="球拍Y轴轨迹")
    if backhand_peaks:
        ax1.plot(time_axis[backhand_peaks], y_clean[backhand_peaks], "r*", markersize=18, markeredgecolor="black", label="确认为反手")
    if other_peaks:
        ax1.plot(time_axis[other_peaks], y_clean[other_peaks], "kX", markersize=12, alpha=0.4, label="已排除波动")
    for i, (st, et) in enumerate(backhand_intervals):
        ax1.axvspan(st, et, alpha=0.2, color="mediumpurple", label="非对称扩张切分区间" if i == 0 else "")
    ax1.set_title(f"多模态轨迹特征与自适应分割 - 反手 [{video_name}]", fontsize=16)
    ax1.set_ylabel("Y轴坐标 (像素)"), ax1.invert_yaxis(), ax1.legend(loc="lower right"), ax1.grid(True, alpha=0.3)

    ax2.plot(time_axis, x_clean, color="blue", linewidth=1.2, label="球拍 X 轴")
    ax2.plot(time_axis, body_x, color="darkorange", linestyle="--", linewidth=2, label="身体动态中心线")
    for i, (st, et) in enumerate(backhand_intervals):
        ax2.axvspan(st, et, alpha=0.15, color="magenta", label="反手有效挥动区" if i == 0 else "")
    ax2.set_xlabel("视频运行时间 (秒)"), ax2.set_ylabel("X轴坐标 (像素)"), ax2.legend(loc="upper right"), ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = str(output_dir / f"{video_name}_final_analysis.png")
    plt.savefig(chart_path, dpi=300)
    plt.close(fig)

    clip_paths: List[str] = []
    if backhand_intervals:
        video = VideoFileClip(TARGET_VIDEO)
        for i, (start_cut, end_cut) in enumerate(backhand_intervals):
            start_cut = max(0, start_cut)
            end_cut = min(video.duration, end_cut)
            output_filename = str(output_dir / f"{video_name}_Backhand_Prep_v2_{i + 1}.mp4")
            try:
                clip = video.subclipped(start_cut, end_cut)
            except AttributeError:
                clip = video.subclip(start_cut, end_cut)
            clip.write_videofile(output_filename, codec="libx264", audio=False, logger=None)
            clip_paths.append(output_filename)
        video.close()

    return {
        "chart": chart_path,
        "clips": clip_paths,
        "kinetic_charts": kinetic_charts,
        "speed_cog_charts": speed_cog_charts,
        "intervals": [(float(a), float(b)) for a, b in backhand_intervals],
        "racket_csv": str(racket_csv),
        "body_csv": str(body_csv),
        "kinematic_summary_csv": summary_csv,
        "kinematic_phase_summary_csv": phase_csv,
        "upper_limb_charts": upper_limb_charts,
        "lower_limb_charts": lower_limb_charts,
        "trunk_rotation_charts": trunk_rotation_charts,
        "racket_kinematic_charts": racket_kinematic_charts,
    }
