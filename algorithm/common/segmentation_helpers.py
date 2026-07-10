"""
切分算法公共工具：2D 图像平面运动学代理指标的计算、评分与摘要。

供 forehand / backhand / serve / volley segmentation 复用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from algorithm.common.kinematic_fusion import build_fused_kinematic_df


def normalize_handedness(handedness: str) -> str:
    h = (handedness or "right").strip().lower()
    return "left" if h in ("left", "l", "左手") else "right"


def is_left_handed(handedness: str) -> bool:
    return normalize_handedness(handedness) == "left"


def pick_side_columns(handedness: str) -> Dict[str, str]:
    """
    根据持拍手返回持拍侧/支撑侧列名映射（2D 图像平面运动学代理指标）。
    """
    if is_left_handed(handedness):
        return {
            "racket_shoulder_angle": "left_shoulder_angle",
            "racket_elbow_angle": "left_elbow_angle",
            "racket_hip_angle": "left_hip_angle",
            "racket_knee_angle": "left_knee_angle",
            "support_knee_angle": "right_knee_angle",
            "racket_wrist_x": "left_wrist_x",
            "racket_wrist_y": "left_wrist_y",
            "toss_wrist_y": "right_wrist_y",
        }
    return {
        "racket_shoulder_angle": "right_shoulder_angle",
        "racket_elbow_angle": "right_elbow_angle",
        "racket_hip_angle": "right_hip_angle",
        "racket_knee_angle": "right_knee_angle",
        "support_knee_angle": "left_knee_angle",
        "racket_wrist_x": "right_wrist_x",
        "racket_wrist_y": "right_wrist_y",
        "toss_wrist_y": "left_wrist_y",
    }


def _series_values(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.array([], dtype=np.float64)
    return df[column].to_numpy(dtype=np.float64)


def range_in_window(df: pd.DataFrame, column: str, start_idx: int, end_idx: int) -> float:
    """窗口内某列变化幅度（max - min），NaN 安全。"""
    if column not in df.columns:
        return float("nan")
    i0 = int(max(0, min(start_idx, end_idx)))
    i1 = int(min(len(df) - 1, max(start_idx, end_idx)))
    if i1 < i0:
        return float("nan")
    seg = _series_values(df, column)[i0 : i1 + 1]
    valid = seg[np.isfinite(seg)]
    if valid.size < 2:
        return float("nan")
    return float(np.max(valid) - np.min(valid))


def delta_between_windows(
    df: pd.DataFrame,
    column: str,
    win_a: Tuple[int, int],
    win_b: Tuple[int, int],
) -> float:
    """比较两窗口均值差。"""
    if column not in df.columns:
        return float("nan")

    def _mean(win: Tuple[int, int]) -> float:
        i0, i1 = int(win[0]), int(win[1])
        if i1 < i0:
            i0, i1 = i1, i0
        i0 = max(0, i0)
        i1 = min(len(df) - 1, i1)
        seg = _series_values(df, column)[i0 : i1 + 1]
        valid = seg[np.isfinite(seg)]
        if valid.size == 0:
            return float("nan")
        return float(np.mean(valid))

    ma, mb = _mean(win_a), _mean(win_b)
    if not np.isfinite(ma) or not np.isfinite(mb):
        return float("nan")
    return float(abs(mb - ma))


def robust_percentile_threshold(
    series: Union[np.ndarray, pd.Series],
    percentile: float = 50.0,
    min_value: float = 0.0,
) -> float:
    """速度、幅度等自适应阈值。"""
    arr = np.asarray(series, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return float(min_value)
    pct_val = float(np.percentile(valid, percentile))
    return float(max(min_value, pct_val))


def ensure_head_columns(df: pd.DataFrame) -> pd.DataFrame:
    """拍头 t 列缺失时回退 x_clean/y_clean。"""
    if "t_x_clean" not in df.columns and "x_clean" in df.columns:
        df["t_x_clean"] = df["x_clean"]
    if "t_y_clean" not in df.columns and "y_clean" in df.columns:
        df["t_y_clean"] = df["y_clean"]
    if "x_clean" not in df.columns and "t_x_clean" in df.columns:
        df["x_clean"] = df["t_x_clean"]
    if "y_clean" not in df.columns and "t_y_clean" in df.columns:
        df["y_clean"] = df["t_y_clean"]
    return df


def ensure_speed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """racket_head_speed 缺失时由拍头坐标梯度估算。"""
    if "racket_head_speed" in df.columns and df["racket_head_speed"].notna().any():
        return df
    time_axis = df["time"].to_numpy(dtype=np.float64)
    tx = df.get("t_x_clean", df.get("x_clean", pd.Series(np.nan, index=df.index))).to_numpy(dtype=np.float64)
    ty = df.get("t_y_clean", df.get("y_clean", pd.Series(np.nan, index=df.index))).to_numpy(dtype=np.float64)
    if len(time_axis) > 1 and np.all(np.isfinite(time_axis)):
        dx_dt = np.gradient(tx, time_axis)
        dy_dt = np.gradient(ty, time_axis)
        speed = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
    else:
        speed = np.full(len(df), np.nan, dtype=np.float64)
    df["racket_head_speed"] = speed
    if "racket_head_acc" not in df.columns or not df["racket_head_acc"].notna().any():
        if len(time_axis) > 1 and np.all(np.isfinite(time_axis)):
            acc = np.abs(np.gradient(speed, time_axis))
        else:
            acc = np.full(len(df), np.nan, dtype=np.float64)
        df["racket_head_acc"] = acc
    return df


def has_enhanced_features(df: pd.DataFrame) -> bool:
    """是否具备角度或人拍相对特征（用于选择 enhanced 路径）。"""
    angle_cols = (
        "left_shoulder_angle",
        "right_shoulder_angle",
        "shoulder_hip_angle",
    )
    rel_cols = ("racket_head_rel_body_x", "racket_head_to_wrist_dist")
    has_angle = any(c in df.columns and df[c].notna().any() for c in angle_cols)
    has_rel = any(c in df.columns and df[c].notna().any() for c in rel_cols)
    return has_angle or has_rel


def prepare_segmentation_df(
    racket_csv: Union[str, Path],
    body_csv: Union[str, Path],
    handedness: str = "right",
) -> Tuple[pd.DataFrame, int, bool]:
    """
    加载融合 DataFrame，补齐拍头/速度列，返回 (df, fps, has_enhanced)。
    """
    df = build_fused_kinematic_df(racket_csv, body_csv, handedness)
    if len(df) == 0:
        raise ValueError("数据对齐后为空！请检查 CSV 文件是否匹配。")

    df = ensure_head_columns(df)
    df = ensure_speed_columns(df)

    time_axis = df["time"].to_numpy(dtype=np.float64)
    fps = int(round(len(df) / time_axis[-1])) if time_axis[-1] > 0 else 30

    for col in ("t_x_clean", "t_y_clean", "x_clean", "y_clean"):
        if col in df.columns:
            arr = df[col].to_numpy(dtype=np.float64)
            if np.any(np.isnan(arr)):
                df[col] = (
                    pd.Series(arr)
                    .interpolate(limit_direction="both")
                    .bfill()
                    .ffill()
                    .to_numpy(dtype=np.float64)
                )

    return df, fps, has_enhanced_features(df)


def refine_contact_by_speed(
    speed: np.ndarray,
    peak_idx: int,
    start_idx: int,
    end_idx: int,
) -> int:
    """在候选窗口内取拍头速度最大点作为 contact。"""
    i0 = int(max(0, min(start_idx, end_idx, peak_idx)))
    i1 = int(min(len(speed) - 1, max(start_idx, end_idx, peak_idx)))
    if i1 <= i0:
        return int(peak_idx)
    seg = speed[i0 : i1 + 1]
    if seg.size == 0 or not np.any(np.isfinite(seg)):
        return int(peak_idx)
    return i0 + int(np.nanargmax(seg))


def compute_swing_width_simple(
    x: np.ndarray,
    prep_start: int,
    contact_idx: int,
    follow_end: int,
) -> float:
    """prep 段与 follow 段 x 极值跨度。"""
    prep = x[prep_start : contact_idx + 1]
    follow = x[contact_idx : follow_end + 1]
    prep_valid = prep[np.isfinite(prep)]
    follow_valid = follow[np.isfinite(follow)]
    if prep_valid.size == 0 or follow_valid.size == 0:
        return 0.0
    return float(max(np.max(prep_valid), np.max(follow_valid)) - min(np.min(prep_valid), np.min(follow_valid)))


def adaptive_rel_body_threshold(df: pd.DataFrame, percentile: float = 25.0, min_value: float = 30.0) -> float:
    """引拍方向自适应阈值。"""
    if "racket_head_rel_body_x" not in df.columns:
        return float(min_value)
    rel = np.abs(df["racket_head_rel_body_x"].to_numpy(dtype=np.float64))
    return robust_percentile_threshold(rel, percentile, min_value)


def check_prep_direction_rel_body(
    df: pd.DataFrame,
    prep_start: int,
    contact_idx: int,
    handedness: str,
    forehand: bool = True,
) -> bool:
    """
    引拍方向硬过滤：正手持拍侧引拍，反手对侧引拍（2D 图像平面代理）。
    """
    if "racket_head_rel_body_x" not in df.columns:
        return True
    seg = df["racket_head_rel_body_x"].iloc[prep_start : contact_idx + 1].to_numpy(dtype=np.float64)
    valid = seg[np.isfinite(seg)]
    if valid.size == 0:
        return True
    median_rel = float(np.median(valid))
    thr = adaptive_rel_body_threshold(df) * 0.35
    use_left = is_left_handed(handedness)
    if forehand:
        if use_left:
            return median_rel < -thr
        return median_rel > thr
    # backhand: opposite side
    if use_left:
        return median_rel > thr
    return median_rel < -thr


def check_prep_direction_legacy(
    x: np.ndarray,
    body_x: np.ndarray,
    prep_start: int,
    contact_idx: int,
    handedness: str,
    forehand: bool = True,
    body_tolerance: float = 150.0,
) -> bool:
    """旧逻辑 fallback：基于 body_center_x 与拍头 x。"""
    avg_body = float(np.nanmean(body_x[prep_start : contact_idx + 1]))
    prep_x = x[prep_start : contact_idx + 1]
    if not np.any(np.isfinite(prep_x)):
        return True
    use_left = is_left_handed(handedness)
    if forehand:
        if use_left:
            return float(np.min(prep_x)) < avg_body + body_tolerance
        return float(np.max(prep_x)) > avg_body - body_tolerance
    if use_left:
        return float(np.max(prep_x)) > avg_body - body_tolerance
    return float(np.min(prep_x)) < avg_body + body_tolerance


def angle_range_stats(df: pd.DataFrame, column: str, start_idx: int, end_idx: int) -> Dict[str, float]:
    """单角度窗口 min/max/range。"""
    if column not in df.columns:
        return {"min": float("nan"), "max": float("nan"), "range": float("nan")}
    i0 = max(0, min(start_idx, end_idx))
    i1 = min(len(df) - 1, max(start_idx, end_idx))
    seg = df[column].iloc[i0 : i1 + 1].to_numpy(dtype=np.float64)
    valid = seg[np.isfinite(seg)]
    if valid.size == 0:
        return {"min": float("nan"), "max": float("nan"), "range": float("nan")}
    lo, hi = float(np.min(valid)), float(np.max(valid))
    return {"min": lo, "max": hi, "range": hi - lo}


def score_angle_ranges(
    df: pd.DataFrame,
    side_cols: Dict[str, str],
    start_idx: int,
    contact_idx: int,
    end_idx: int,
    weights: Dict[str, float],
    invert_small: Optional[Sequence[str]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    角度软评分：range 越大加分（invert_small 中的列 range 越小加分）。
    返回 (总分, 各项 range 明细)。
    """
    invert = set(invert_small or ())
    details: Dict[str, float] = {}
    total = 0.0
    for key, weight in weights.items():
        col = side_cols.get(key, key)
        if col not in df.columns and key not in df.columns:
            details[key] = float("nan")
            continue
        use_col = col if col in df.columns else key
        rng = angle_range_stats(df, use_col, start_idx, end_idx)["range"]
        details[key] = rng
        if not np.isfinite(rng):
            continue
        if key in invert:
            # 变化小加分：用 1 / (1 + range/scale)
            total += weight * (1.0 / (1.0 + rng / 15.0))
        else:
            total += weight * min(rng / 30.0, 1.0)
    return total, details


def search_start_idx(
    speed: np.ndarray,
    x: np.ndarray,
    dist: Optional[np.ndarray],
    contact_idx: int,
    fps: int,
    max_back_sec: float = 1.5,
    speed_pct: float = 20.0,
) -> int:
    """从 contact 向前搜索准备起点。"""
    back = int(max_back_sec * fps)
    i0 = max(0, contact_idx - back)
    speed_thr = robust_percentile_threshold(speed[i0:contact_idx], speed_pct, 50.0) if contact_idx > i0 else 0.0

    for i in range(contact_idx, i0, -1):
        low_speed = speed[i] <= speed_thr * 1.2 if np.isfinite(speed[i]) else False
        if low_speed:
            return i
    if dist is not None and contact_idx > i0:
        dseg = dist[i0:contact_idx]
        if np.any(np.isfinite(dseg)):
            baseline = float(np.nanpercentile(dseg, 20))
            for i in range(contact_idx, i0, -1):
                if np.isfinite(dist[i]) and dist[i] > baseline * 1.15:
                    return max(i0, i - int(0.1 * fps))
    return i0


def search_end_idx(
    speed: np.ndarray,
    contact_idx: int,
    fps: int,
    max_fwd_sec: float = 1.2,
    speed_pct: float = 25.0,
) -> int:
    """从 contact 向后搜索随挥结束。"""
    fwd = int(max_fwd_sec * fps)
    i1 = min(len(speed) - 1, contact_idx + fwd)
    if i1 <= contact_idx:
        return contact_idx
    speed_thr = robust_percentile_threshold(speed[contact_idx:i1], speed_pct, 80.0)
    for i in range(contact_idx, i1):
        if np.isfinite(speed[i]) and speed[i] <= speed_thr:
            return i
    return i1


def clamp_interval_neighbors(
    start_t: float,
    end_t: float,
    hit_time: float,
    time_axis: np.ndarray,
    prev_hit: Optional[float],
    next_hit: Optional[float],
) -> Tuple[float, float]:
    """邻峰中点约束。"""
    if prev_hit is not None:
        start_t = max(start_t, (prev_hit + hit_time) / 2.0)
    else:
        start_t = max(start_t, 0.0)
    if next_hit is not None:
        end_t = min(end_t, (hit_time + next_hit) / 2.0)
    return start_t, end_t


def _value_at_nearest_valid(df: pd.DataFrame, column: str, idx: int) -> float:
    """取 idx 附近最近有效值。"""
    if column not in df.columns:
        return float("nan")
    n = len(df)
    if n == 0:
        return float("nan")
    idx = int(np.clip(idx, 0, n - 1))
    val = df[column].iloc[idx]
    if np.isfinite(val):
        return float(val)
    arr = df[column].to_numpy(dtype=np.float64)
    for delta in range(1, n):
        for j in (idx - delta, idx + delta):
            if 0 <= j < n and np.isfinite(arr[j]):
                return float(arr[j])
    return float("nan")


def _window_mean(df: pd.DataFrame, column: str, start_idx: int, end_idx: int) -> float:
    if column not in df.columns:
        return float("nan")
    i0 = max(0, min(start_idx, end_idx))
    i1 = min(len(df) - 1, max(start_idx, end_idx))
    seg = df[column].iloc[i0 : i1 + 1].to_numpy(dtype=np.float64)
    valid = seg[np.isfinite(seg)]
    return float(np.mean(valid)) if valid.size else float("nan")


def _phase_background_times(time_start: float, time_end: float, time_contact: float) -> Tuple[float, float, float]:
    """以击球时刻为中心的核心挥拍区边界。"""
    ts = float(min(time_start, time_end))
    te = float(max(time_start, time_end))
    tc = float(np.clip(time_contact, ts, te))
    clip_duration = te - ts
    if clip_duration <= 0:
        hit_window = 0.2
    else:
        hit_window = float(np.clip(clip_duration * 0.20, 0.2, 0.4))
    t_green_start = max(ts, tc - hit_window * 0.65)
    t_green_end = min(te, tc + hit_window * 0.35)
    if t_green_end < t_green_start:
        t_green_end = t_green_start
    return t_green_start, t_green_end, tc


def _time_to_idx(time_axis: np.ndarray, t: float) -> int:
    return int(np.clip(np.searchsorted(time_axis, t, side="left"), 0, len(time_axis) - 1))


def _phase_definitions(
    action_type: str,
    time_start: float,
    time_end: float,
    time_contact: float,
    event_times: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, float, float]]:
    """返回 (phase_name, phase_start_time, phase_end_time) 列表。"""
    event_times = event_times or {}
    if action_type in ("forehand", "backhand"):
        t_green_start, _, tc = _phase_background_times(time_start, time_end, time_contact)
        hit_window = max((time_end - time_start) * 0.05, 0.05)
        contact_start = tc - hit_window * 0.5
        contact_end = tc + hit_window * 0.5
        return [
            ("preparation", time_start, t_green_start),
            ("acceleration", t_green_start, contact_start),
            ("contact", contact_start, contact_end),
            ("follow_through", contact_end, time_end),
        ]
    if action_type == "serve":
        toss_t = event_times.get("toss_time", time_start)
        drop_t = event_times.get("drop_time", toss_t)
        hit_t = event_times.get("hit_time", time_contact)
        return [
            ("preparation", time_start, toss_t),
            ("toss", toss_t, drop_t),
            ("racket_drop", drop_t, hit_t),
            ("upward_swing_contact", hit_t, min(hit_t + 0.15, time_end)),
            ("follow_through", min(hit_t + 0.15, time_end), time_end),
        ]
    # volley
    hit_window = 0.15
    contact_start = time_contact - hit_window * 0.5
    contact_end = time_contact + hit_window * 0.5
    return [
        ("preparation", time_start, contact_start),
        ("contact", contact_start, contact_end),
        ("recovery", contact_end, time_end),
    ]


def build_kinematic_summary(
    action_type: str,
    df: pd.DataFrame,
    side_cols: Dict[str, str],
    start_idx: int,
    contact_idx: int,
    end_idx: int,
    time_axis: np.ndarray,
    speed: np.ndarray,
    score: float = 0.0,
    segment_id: int = 1,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """片段运动学摘要（2D 图像平面运动学代理指标）。"""
    i0 = max(0, min(start_idx, end_idx))
    i1 = min(len(df) - 1, max(start_idx, end_idx, contact_idx))
    ic = int(np.clip(contact_idx, i0, i1))

    speed_seg = speed[i0 : i1 + 1]
    valid_speed = speed_seg[np.isfinite(speed_seg)]
    peak_speed = float(np.max(valid_speed)) if valid_speed.size else float("nan")
    peak_local = int(np.nanargmax(speed_seg)) if valid_speed.size else 0
    peak_time = float(time_axis[i0 + peak_local])

    acc = df["racket_head_acc"].to_numpy(dtype=np.float64) if "racket_head_acc" in df.columns else np.full(len(df), np.nan)
    acc_seg = acc[i0 : i1 + 1]
    valid_acc = acc_seg[np.isfinite(acc_seg)]
    peak_acc = float(np.max(valid_acc)) if valid_acc.size else float("nan")

    summary: Dict[str, Any] = {
        "segment_id": int(segment_id),
        "action_type": action_type,
        "start_time": float(time_axis[i0]),
        "contact_time": float(time_axis[ic]),
        "end_time": float(time_axis[i1]),
        "duration": float(time_axis[i1] - time_axis[i0]),
        "start_frame": int(i0),
        "contact_frame": int(ic),
        "end_frame": int(i1),
        "score": float(score),
        "quality_score": float(score),
        "reject_reason": "",
        "racket_head_speed_peak": peak_speed,
        "racket_head_speed_peak_time": peak_time,
        "racket_head_acc_peak": peak_acc,
    }

    if "racket_head_to_wrist_dist" in df.columns:
        d = df["racket_head_to_wrist_dist"].iloc[i0 : i1 + 1].to_numpy(dtype=np.float64)
        summary["racket_head_to_wrist_dist_peak"] = float(np.nanmax(d)) if np.any(np.isfinite(d)) else float("nan")
        summary["racket_head_to_wrist_dist_mean"] = _window_mean(df, "racket_head_to_wrist_dist", i0, i1)
    else:
        summary["racket_head_to_wrist_dist_peak"] = float("nan")
        summary["racket_head_to_wrist_dist_mean"] = float("nan")

    summary["racket_head_wrist_y_diff_at_contact"] = _value_at_nearest_valid(df, "racket_head_wrist_y_diff", ic)
    summary["racket_head_rel_body_x_at_contact"] = _value_at_nearest_valid(df, "racket_head_rel_body_x", ic)
    summary["racket_head_rel_body_y_at_contact"] = _value_at_nearest_valid(df, "racket_head_rel_body_y", ic)
    summary["racket_long_axis_angle_at_contact"] = _value_at_nearest_valid(df, "racket_long_axis_angle", ic)
    summary["racket_width_axis_angle_at_contact"] = _value_at_nearest_valid(df, "racket_width_axis_angle", ic)
    summary["racket_valid_kpt_count_mean"] = _window_mean(df, "racket_valid_kpt_count", i0, i1)
    summary["racket_mean_conf_mean"] = _window_mean(df, "racket_mean_conf", i0, i1)

    angle_keys = {
        "racket_shoulder_angle": side_cols.get("racket_shoulder_angle", "right_shoulder_angle"),
        "racket_elbow_angle": side_cols.get("racket_elbow_angle", "right_elbow_angle"),
        "racket_hip_angle": side_cols.get("racket_hip_angle", "right_hip_angle"),
        "racket_knee_angle": side_cols.get("racket_knee_angle", "right_knee_angle"),
        "support_knee_angle": side_cols.get("support_knee_angle", "left_knee_angle"),
        "shoulder_hip_angle": "shoulder_hip_angle",
        "trunk_angle": "trunk_angle",
    }
    for prefix, col in angle_keys.items():
        stats = angle_range_stats(df, col, i0, i1)
        summary[f"{prefix}_min"] = stats["min"]
        summary[f"{prefix}_max"] = stats["max"]
        summary[f"{prefix}_range"] = stats["range"]

    if extra_fields:
        summary.update(extra_fields)

    return summary


def build_kinematic_phase_summary(
    action_type: str,
    segment_id: int,
    df: pd.DataFrame,
    side_cols: Dict[str, str],
    start_idx: int,
    contact_idx: int,
    end_idx: int,
    time_axis: np.ndarray,
    speed: np.ndarray,
    event_times: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """片段内各阶段运动学摘要行。"""
    i0 = max(0, min(start_idx, end_idx))
    i1 = min(len(df) - 1, max(start_idx, end_idx, contact_idx))
    ic = int(np.clip(contact_idx, i0, i1))
    time_start = float(time_axis[i0])
    time_end = float(time_axis[i1])
    time_contact = float(time_axis[ic])

    phases = _phase_definitions(action_type, time_start, time_end, time_contact, event_times)
    elbow_col = side_cols.get("racket_elbow_angle", "right_elbow_angle")
    knee_col = side_cols.get("racket_knee_angle", "right_knee_angle")

    rows: List[Dict[str, Any]] = []
    for phase_name, pt_start, pt_end in phases:
        if pt_end <= pt_start:
            continue
        pi0 = _time_to_idx(time_axis, pt_start)
        pi1 = _time_to_idx(time_axis, pt_end)
        pi0 = max(i0, min(pi0, pi1))
        pi1 = min(i1, max(pi0, pi1))

        speed_seg = speed[pi0 : pi1 + 1]
        valid_speed = speed_seg[np.isfinite(speed_seg)]
        peak_speed = float(np.max(valid_speed)) if valid_speed.size else float("nan")

        rows.append(
            {
                "segment_id": int(segment_id),
                "action_type": action_type,
                "phase_name": phase_name,
                "phase_start_time": float(pt_start),
                "phase_end_time": float(pt_end),
                "phase_duration": float(pt_end - pt_start),
                "racket_head_speed_peak": peak_speed,
                "racket_elbow_angle_range": angle_range_stats(df, elbow_col, pi0, pi1)["range"],
                "racket_knee_angle_range": angle_range_stats(df, knee_col, pi0, pi1)["range"],
                "shoulder_hip_angle_range": angle_range_stats(df, "shoulder_hip_angle", pi0, pi1)["range"],
            }
        )
    return rows


def write_kinematic_summary_csv(summaries: List[Dict[str, Any]], output_path: Union[str, Path]) -> str:
    """写入运动学摘要 CSV。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not summaries:
        pd.DataFrame().to_csv(str(path), index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(summaries).to_csv(str(path), index=False, encoding="utf-8-sig")
    return str(path.resolve())


def write_kinematic_phase_summary_csv(rows: List[Dict[str, Any]], output_path: Union[str, Path]) -> str:
    """写入运动学阶段明细 CSV。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(str(path), index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(rows).to_csv(str(path), index=False, encoding="utf-8-sig")
    return str(path.resolve())
