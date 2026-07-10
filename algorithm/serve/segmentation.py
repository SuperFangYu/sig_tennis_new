"""
发球无监督切分 + 双独立分析图（空间溯源 twinx、动力链单轴）。
使用 2D 图像平面运动学代理指标进行 toss-drop-hit-follow 事件链检测与角度软评分。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from moviepy import VideoFileClip
from scipy.signal import find_peaks

from algorithm.common.kinematic_plots import generate_segment_kinematic_charts
from algorithm.common.segmentation_helpers import (
    build_kinematic_phase_summary,
    build_kinematic_summary,
    normalize_handedness,
    pick_side_columns,
    prepare_segmentation_df,
    range_in_window,
    robust_percentile_threshold,
    score_angle_ranges,
    search_end_idx,
    write_kinematic_phase_summary_csv,
    write_kinematic_summary_csv,
)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

PROMINENCE_THRESHOLD = 80.0
MIN_DISTANCE_SEC = 2.5
PRE_TOSS_BUFFER = 1.8
POST_HIT_BUFFER = 1.5
TOSS_HEIGHT_THRESHOLD = 180
RACKET_DROP_DEPTH = 150
MIN_SCORE_THRESHOLD = 0.1

COLOR_PREP = "#cfe8ff"
COLOR_TOSS_SWING = "#fff2b2"
COLOR_POST_HIT = "#ffd4cc"


def _ensure_columns(df: pd.DataFrame, names: Tuple[str, ...]) -> None:
    for name in names:
        if name not in df.columns:
            df[name] = np.nan


def _toss_height_threshold(df: pd.DataFrame, body_y: np.ndarray, toss_y: np.ndarray) -> float:
    diffs = body_y - toss_y
    valid = diffs[np.isfinite(diffs)]
    if valid.size == 0:
        return float(TOSS_HEIGHT_THRESHOLD)
    return max(TOSS_HEIGHT_THRESHOLD, robust_percentile_threshold(valid, 40, TOSS_HEIGHT_THRESHOLD) * 0.6)


def _drop_depth_threshold(racket_y: np.ndarray) -> float:
    valid = racket_y[np.isfinite(racket_y)]
    if valid.size == 0:
        return float(RACKET_DROP_DEPTH)
    y_range = float(np.max(valid) - np.min(valid))
    return max(RACKET_DROP_DEPTH, y_range * 0.12)


def _search_serve_start_idx(
    df: pd.DataFrame,
    side_cols: Dict[str, str],
    toss_idx: int,
    fps: int,
    speed: np.ndarray,
) -> int:
    """从 toss_idx 向前搜索准备起点（膝角/髋角/拍头运动）。"""
    back = int(2.0 * fps)
    i0 = max(0, toss_idx - back)
    knee_col = side_cols.get("racket_knee_angle", "right_knee_angle")
    hip_col = side_cols.get("racket_hip_angle", "right_hip_angle")

    speed_thr = robust_percentile_threshold(speed[i0:toss_idx], 25, 30.0) if toss_idx > i0 else 30.0
    for i in range(toss_idx, i0, -1):
        if np.isfinite(speed[i]) and speed[i] > speed_thr * 1.5:
            return max(i0, i - int(0.15 * fps))

    if knee_col in df.columns:
        knee = df[knee_col].iloc[i0:toss_idx].to_numpy(dtype=np.float64)
        if np.any(np.isfinite(knee)):
            baseline = float(np.nanpercentile(knee, 70))
            for i in range(toss_idx, i0, -1):
                val = df[knee_col].iloc[i]
                if np.isfinite(val) and val < baseline - 5.0:
                    return max(i0, i - int(0.1 * fps))

    return i0


def _detect_event_chain(
    p: int,
    toss_wrist_y: np.ndarray,
    racket_y: np.ndarray,
    speed: np.ndarray,
    time_axis: np.ndarray,
    fps: int,
) -> Optional[Dict[str, int]]:
    """
    检测 toss -> drop -> hit -> follow 事件链索引。
    hit 候选 p 为 y 最小（击球高点）；enhanced 模式下可再精化。
    """
    search_start = max(0, p - int(2.5 * fps))
    wrist_seg = toss_wrist_y[search_start:p]
    if wrist_seg.size < 5:
        return None

    rel_toss = int(np.nanargmin(wrist_seg))
    toss_idx = search_start + rel_toss

    between = racket_y[toss_idx:p]
    if between.size < 3:
        return None
    rel_drop = int(np.nanargmax(between))
    drop_idx = toss_idx + rel_drop

    hit_idx = p
    if toss_idx < p:
        seg_speed = speed[toss_idx : p + 1]
        if np.any(np.isfinite(seg_speed)):
            hit_idx = toss_idx + int(np.nanargmax(seg_speed))

    follow_end_idx = search_end_idx(speed, hit_idx, fps, max_fwd_sec=1.5, speed_pct=30.0)
    follow_end_idx = max(follow_end_idx, hit_idx + int(0.2 * fps))

    if not (toss_idx < drop_idx < hit_idx < follow_end_idx):
        return None

    return {
        "toss_idx": toss_idx,
        "drop_idx": drop_idx,
        "hit_idx": hit_idx,
        "follow_end_idx": follow_end_idx,
    }


def _evaluate_serve_candidate(
    df: pd.DataFrame,
    p: int,
    toss_wrist_y: np.ndarray,
    racket_y: np.ndarray,
    body_center_y: np.ndarray,
    speed: np.ndarray,
    time_axis: np.ndarray,
    fps: int,
    side_cols: Dict[str, str],
    enhanced: bool,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """评估单个发球候选，返回 (accepted, event_dict, reason)。"""
    toss_thr = _toss_height_threshold(df, body_center_y, toss_wrist_y)
    drop_thr = _drop_depth_threshold(racket_y)

    if enhanced:
        chain = _detect_event_chain(p, toss_wrist_y, racket_y, speed, time_axis, fps)
        if chain is None:
            return False, None, "事件链不完整"

        toss_idx = chain["toss_idx"]
        hit_idx = chain["hit_idx"]
        drop_idx = chain["drop_idx"]
        follow_end_idx = chain["follow_end_idx"]

        toss_height_diff = float(body_center_y[toss_idx] - toss_wrist_y[toss_idx])
        if not np.isfinite(toss_height_diff) or toss_height_diff < toss_thr:
            return False, None, "抛球过低"

        drop_depth = float(np.nanmax(racket_y[toss_idx:hit_idx]) - racket_y[hit_idx])
        if not np.isfinite(drop_depth) or drop_depth < drop_thr:
            return False, None, "无明显挠背"

        chain_bonus = 1.0
    else:
        search_start = max(0, p - int(2.5 * fps))
        wrist_seg = toss_wrist_y[search_start:p]
        if wrist_seg.size < 5:
            return False, None, "抛球窗口不足"
        toss_idx = search_start + int(np.nanargmin(wrist_seg))
        hit_idx = p
        drop_idx = toss_idx + int(np.nanargmax(racket_y[toss_idx:hit_idx])) if hit_idx > toss_idx else toss_idx
        follow_end_idx = min(len(time_axis) - 1, hit_idx + int(POST_HIT_BUFFER * fps))

        toss_height_diff = float(body_center_y[toss_idx] - toss_wrist_y[toss_idx])
        drop_depth = float(np.nanmax(racket_y[toss_idx:hit_idx]) - racket_y[hit_idx]) if hit_idx > toss_idx else 0.0

        if not np.isfinite(toss_height_diff) or toss_height_diff < toss_thr:
            return False, None, "抛球过低"
        if not np.isfinite(drop_depth) or drop_depth < drop_thr:
            return False, None, "无明显挠背"
        chain_bonus = 0.5

    start_idx = _search_serve_start_idx(df, side_cols, toss_idx, fps, speed)
    start_t = float(time_axis[start_idx])
    if start_t > float(time_axis[toss_idx]) - 0.3:
        start_t = max(0.0, float(time_axis[toss_idx]) - PRE_TOSS_BUFFER)

    end_t = float(time_axis[follow_end_idx]) + 0.2

    angle_weights = {
        "racket_knee_angle": 0.9,
        "racket_hip_angle": 0.7,
        "racket_shoulder_angle": 0.8,
        "racket_elbow_angle": 0.7,
        "shoulder_hip_angle": 0.8,
    }
    score, _ = score_angle_ranges(df, side_cols, start_idx, hit_idx, follow_end_idx, angle_weights)

    if "shoulder_tilt_y" in df.columns:
        tilt_rng = range_in_window(df, "shoulder_tilt_y", start_idx, follow_end_idx)
        if np.isfinite(tilt_rng):
            score += min(tilt_rng / 40.0, 1.0) * 0.6
    if "trunk_angle" in df.columns:
        trunk_rng = range_in_window(df, "trunk_angle", start_idx, follow_end_idx)
        if np.isfinite(trunk_rng):
            score += min(trunk_rng / 35.0, 1.0) * 0.5
    if "racket_long_axis_angle" in df.columns:
        axis_rng = range_in_window(df, "racket_long_axis_angle", drop_idx, hit_idx)
        if np.isfinite(axis_rng):
            score += min(axis_rng / 45.0, 1.0) * 0.5

    score *= chain_bonus

    event = {
        "start_t": start_t,
        "end_t": end_t,
        "toss_time": float(time_axis[toss_idx]),
        "hit_time": float(time_axis[hit_idx]),
        "drop_time": float(time_axis[drop_idx]),
        "toss_idx": toss_idx,
        "hit_idx": hit_idx,
        "drop_idx": drop_idx,
        "follow_end_idx": follow_end_idx,
        "start_idx": start_idx,
        "score": score,
        "toss_height_diff": toss_height_diff,
        "drop_depth": drop_depth,
        "peak": p,
    }
    return True, event, "ok"


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
    video_name = video_path.stem

    hand = normalize_handedness(handedness)
    side_cols = pick_side_columns(handedness)

    if hand == "right":
        toss_wrist_label = "左腕(抛球手)"
        knee_label = "右膝角(蓄力)"
        elbow_label = "右肘角(持拍手)"
    else:
        toss_wrist_label = "右腕(抛球手)"
        knee_label = "左膝角(蓄力)"
        elbow_label = "左肘角(持拍手)"

    print("🚀 正在加载并融合多模态数据...")
    df, fps, has_enhanced = prepare_segmentation_df(racket_csv, body_csv, handedness)

    _ensure_columns(
        df,
        (
            "left_elbow_angle",
            "right_elbow_angle",
            "left_knee_angle",
            "right_knee_angle",
            "shoulder_hip_angle",
            "body_center_y",
            "hip_center_y",
            "left_wrist_y",
            "right_wrist_y",
            "shoulder_tilt_y",
            "trunk_angle",
            "racket_long_axis_angle",
        ),
    )

    racket_y = df.get("t_y_clean", df["y_clean"]).to_numpy(dtype=np.float64)
    toss_col = side_cols["toss_wrist_y"]
    toss_wrist_y = df[toss_col].to_numpy(dtype=np.float64) if toss_col in df.columns else np.full(len(df), np.nan)

    body_center_y = df["body_center_y"].to_numpy(dtype=np.float64)
    if "hip_center_y" in df.columns:
        ref_y = np.where(np.isfinite(body_center_y), body_center_y, df["hip_center_y"].to_numpy(dtype=np.float64))
    else:
        ref_y = body_center_y

    time_axis = df["time"].to_numpy(dtype=np.float64)
    speed = df["racket_head_speed"].to_numpy(dtype=np.float64)

    knee_col = side_cols["racket_knee_angle"]
    elbow_col = side_cols["racket_elbow_angle"]
    knee_series = df[knee_col].to_numpy(dtype=np.float64) if knee_col in df.columns else np.full(len(df), np.nan)
    elbow_series = df[elbow_col].to_numpy(dtype=np.float64) if elbow_col in df.columns else np.full(len(df), np.nan)
    shoulder_hip = (
        df["shoulder_hip_angle"].to_numpy(dtype=np.float64)
        if "shoulder_hip_angle" in df.columns
        else np.full(len(df), np.nan)
    )

    inverted_T = np.max(racket_y) - racket_y
    peaks, _ = find_peaks(inverted_T, prominence=PROMINENCE_THRESHOLD, distance=int(MIN_DISTANCE_SEC * fps))
    print(f"✅ 发球峰值检测：共 {len(peaks)} 个候选击球高点。")

    mode_label = "增强" if has_enhanced else "兼容"
    print(f"🔍 toss-drop-hit-follow 事件链检测（模式: {mode_label}）...")

    candidates: List[Dict[str, Any]] = []
    serve_hit_peaks: List[int] = []
    other_peaks: List[int] = []

    for i, p in enumerate(peaks):
        ok, event, reason = _evaluate_serve_candidate(
            df, p, toss_wrist_y, racket_y, ref_y, speed, time_axis, fps, side_cols, has_enhanced
        )
        if not ok or event is None:
            other_peaks.append(p)
            print(f"  [过滤] 击球: {float(time_axis[p]):5.2f}s | {reason}")
            continue

        if event["score"] < MIN_SCORE_THRESHOLD and has_enhanced:
            other_peaks.append(p)
            print(f"  [过滤] 击球: {float(time_axis[p]):5.2f}s | 角度评分过低 ({event['score']:.2f})")
            continue

        if i > 0:
            prev_p = peaks[i - 1]
            event["start_t"] = max(event["start_t"], (float(time_axis[prev_p]) + event["hit_time"]) / 2.0)
        else:
            event["start_t"] = max(event["start_t"], 0.0)

        candidates.append(event)
        serve_hit_peaks.append(p)
        print(
            f"  [保留] 击球: {event['hit_time']:5.2f}s | 抛球高差: {event['toss_height_diff']:4.0f}px | "
            f"挠背深度: {event['drop_depth']:4.0f}px | 评分: {event['score']:.2f}"
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    serve_events = [
        {
            "start_t": ev["start_t"],
            "end_t": ev["end_t"],
            "toss_time": ev["toss_time"],
            "hit_time": ev["hit_time"],
        }
        for ev in candidates
    ]

    print(f"🎯 确认 {len(serve_events)} 次标准发球。")

    kinematic_summaries: List[Dict[str, Any]] = []
    kinematic_phase_rows: List[Dict[str, Any]] = []
    upper_limb_charts: List[str] = []
    lower_limb_charts: List[str] = []
    trunk_rotation_charts: List[str] = []
    racket_kinematic_charts: List[str] = []

    for seg_n, ev in enumerate(candidates, start=1):
        event_times = {
            "toss_time": ev["toss_time"],
            "drop_time": ev["drop_time"],
            "hit_time": ev["hit_time"],
        }
        chart_paths = generate_segment_kinematic_charts(
            output_dir,
            video_name,
            seg_n,
            "发球",
            df,
            side_cols,
            time_axis,
            speed,
            ev["start_idx"],
            ev["follow_end_idx"],
            ev["hit_idx"],
            event_times=event_times,
        )
        upper_limb_charts.append(chart_paths["upper_limb"])
        lower_limb_charts.append(chart_paths["lower_limb"])
        trunk_rotation_charts.append(chart_paths["trunk"])
        racket_kinematic_charts.append(chart_paths["racket"])

        summary = build_kinematic_summary(
            "serve",
            df,
            side_cols,
            ev["start_idx"],
            ev["hit_idx"],
            ev["follow_end_idx"],
            time_axis,
            speed,
            score=ev["score"],
            segment_id=seg_n,
            extra_fields={
                "toss_time": ev["toss_time"],
                "racket_drop_time": ev["drop_time"],
                "toss_height_diff": ev["toss_height_diff"],
                "drop_depth": ev["drop_depth"],
            },
        )
        kinematic_summaries.append(summary)
        kinematic_phase_rows.extend(
            build_kinematic_phase_summary(
                "serve",
                seg_n,
                df,
                side_cols,
                ev["start_idx"],
                ev["hit_idx"],
                ev["follow_end_idx"],
                time_axis,
                speed,
                event_times=event_times,
            )
        )

    summary_csv = write_kinematic_summary_csv(
        kinematic_summaries,
        output_dir / f"{video_name}_serve_kinematic_summary.csv",
    )
    phase_csv = write_kinematic_phase_summary_csv(
        kinematic_phase_rows,
        output_dir / f"{video_name}_serve_kinematic_phase_summary.csv",
    )

    trace_path = output_dir / f"{video_name}_serve_trace_chart.png"
    kinetic_path = output_dir / f"{video_name}_serve_kinetic_chart.png"

    _save_trace_chart(
        trace_path,
        video_name,
        time_axis,
        racket_y,
        toss_wrist_y,
        body_center_y,
        toss_wrist_label,
        serve_hit_peaks,
        other_peaks,
        serve_events,
    )
    _save_kinetic_chart(
        kinetic_path,
        video_name,
        time_axis,
        shoulder_hip,
        knee_series,
        elbow_series,
        serve_hit_peaks,
        other_peaks,
        serve_events,
        knee_label,
        elbow_label,
    )

    clip_paths: List[str] = []
    intervals: List[Tuple[float, float]] = []

    if serve_events:
        print("✂️ MoviePy 发球片段导出...")
        video = VideoFileClip(str(video_path))
        for idx, ev in enumerate(serve_events):
            start_cut = max(0.0, float(ev["start_t"]))
            end_cut = min(float(video.duration), float(ev["end_t"]))
            intervals.append((start_cut, end_cut))
            out_mp4 = output_dir / f"{video_name}_Serve_v2_{idx + 1}.mp4"
            try:
                clip = video.subclipped(start_cut, end_cut)
            except AttributeError:
                clip = video.subclip(start_cut, end_cut)
            clip.write_videofile(str(out_mp4), codec="libx264", audio=False, logger=None)
            clip.close()
            clip_paths.append(str(out_mp4.resolve()))
            print(f"  -> 片段 {idx + 1}: [{start_cut:.2f}s – {end_cut:.2f}s]")
        video.close()

    return {
        "trace_chart": str(trace_path.resolve()),
        "kinetic_chart": str(kinetic_path.resolve()),
        "clips": clip_paths,
        "intervals": intervals,
        "racket_csv": str(Path(racket_csv).resolve()),
        "body_csv": str(Path(body_csv).resolve()),
        "kinematic_summary_csv": summary_csv,
        "kinematic_phase_summary_csv": phase_csv,
        "upper_limb_charts": upper_limb_charts,
        "lower_limb_charts": lower_limb_charts,
        "trunk_rotation_charts": trunk_rotation_charts,
        "racket_kinematic_charts": racket_kinematic_charts,
    }


def _paint_phase_spans(ax: plt.Axes, events: List[Dict[str, float]]) -> None:
    for j, ev in enumerate(events):
        st, tt, ht, et = ev["start_t"], ev["toss_time"], ev["hit_time"], ev["end_t"]
        ax.axvspan(st, tt, alpha=0.35, color=COLOR_PREP, label="准备阶段" if j == 0 else "")
        ax.axvspan(tt, ht, alpha=0.4, color=COLOR_TOSS_SWING, label="抛球与挠背" if j == 0 else "")
        ax.axvspan(ht, et, alpha=0.35, color=COLOR_POST_HIT, label="击球后" if j == 0 else "")


def _paint_hit_lines(ax: plt.Axes, events: List[Dict[str, float]]) -> None:
    for j, ev in enumerate(events):
        ax.axvline(
            ev["hit_time"],
            color="crimson",
            linestyle="--",
            linewidth=1.6,
            alpha=0.9,
            label="击球锚点" if j == 0 else "",
        )


def _save_trace_chart(
    path: Path,
    video_name: str,
    time_axis: np.ndarray,
    T: np.ndarray,
    toss_wrist_y: np.ndarray,
    body_center_y: np.ndarray,
    toss_label: str,
    serve_hit_peaks: List[int],
    other_peaks: List[int],
    serve_events: List[Dict[str, float]],
) -> None:
    fig, ax1 = plt.subplots(figsize=(14, 7))

    if serve_events:
        _paint_phase_spans(ax1, serve_events)
        _paint_hit_lines(ax1, serve_events)

    ax1.plot(time_axis, T, color="#1f77b4", alpha=0.9, linewidth=1.6, label="球拍 Y 轨迹")
    ax1.plot(time_axis, toss_wrist_y, color="#2ca02c", alpha=0.85, linewidth=1.4, label=f"{toss_label} Y")

    if serve_hit_peaks:
        ax1.plot(
            time_axis[serve_hit_peaks],
            T[serve_hit_peaks],
            "r*",
            markersize=16,
            markeredgecolor="black",
            label="确认击球最高点",
        )
    if other_peaks:
        ax1.plot(
            time_axis[other_peaks],
            T[other_peaks],
            "kx",
            markersize=10,
            alpha=0.45,
            label="已排除高点",
        )

    ax1.set_title(f"发球空间溯源与起跳 — {video_name}", fontsize=15)
    ax1.set_xlabel("时间 (s)")
    ax1.set_ylabel("主轴：像素 Y（越高越靠上）")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(time_axis, body_center_y, color="#ff7f0e", linestyle="--", linewidth=2.0, label="身体重心 Y")
    ax2.set_ylabel("副轴：重心 Y（像素）")
    ax2.invert_yaxis()

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(str(path), dpi=300)
    plt.close(fig)
    print(f"📈 溯源图: {path}")


def _save_kinetic_chart(
    path: Path,
    video_name: str,
    time_axis: np.ndarray,
    shoulder_hip: np.ndarray,
    knee: np.ndarray,
    elbow: np.ndarray,
    serve_hit_peaks: List[int],
    other_peaks: List[int],
    serve_events: List[Dict[str, float]],
    knee_label: str,
    elbow_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))

    if serve_events:
        _paint_phase_spans(ax, serve_events)
        _paint_hit_lines(ax, serve_events)

    ax.plot(time_axis, shoulder_hip, color="#9467bd", linewidth=1.8, label="肩-髋背弓角 (°)")
    ax.plot(time_axis, knee, color="#ff7f0e", linewidth=1.6, label=knee_label + " (°)")
    ax.plot(time_axis, elbow, color="#17becf", linewidth=1.6, label=elbow_label + " (°)")

    if serve_hit_peaks:
        ax.scatter(
            time_axis[serve_hit_peaks],
            shoulder_hip[serve_hit_peaks],
            c="red",
            s=120,
            zorder=5,
            edgecolors="black",
            label="击球帧",
        )

    ax.set_title(f"发球动力链与蓄力特征 — {video_name}", fontsize=15)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("角度 (°)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(path), dpi=300)
    plt.close(fig)
    print(f"📈 动力链图: {path}")
