"""用视频与 QTM 的拍头速度峰自动匹配五次重复动作。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from qualisys.core.action_profiles import ActionProfile
from qualisys.core.io import ValidationError


@dataclass(frozen=True)
class PreparedSignal:
    frame: pd.DataFrame
    candidate_indices: np.ndarray
    candidate_prominences: np.ndarray
    valid_speed_ratio: float


@dataclass(frozen=True)
class AlignmentResult:
    slope: float
    intercept: float
    rmse_seconds: float
    max_abs_residual_seconds: float
    video_peak_indices: np.ndarray
    qtm_peak_indices: np.ndarray
    anchors: pd.DataFrame


def _odd_window(sample_count: int, requested: int) -> int:
    window = min(sample_count if sample_count % 2 else sample_count - 1, requested)
    if window % 2 == 0:
        window -= 1
    return max(window, 3)


def prepare_speed_signal(
    source: pd.DataFrame,
    *,
    profile: ActionProfile,
    expected_repetitions: int,
    time_column: str = "time",
    speed_column: str = "racket_head_speed",
) -> PreparedSignal:
    """短缺口插值、平滑、鲁棒归一化并生成候选运动峰。"""
    if not {time_column, speed_column}.issubset(source.columns):
        raise ValueError(f"速度数据必须包含 {time_column} 和 {speed_column}")
    time = pd.to_numeric(source[time_column], errors="coerce").to_numpy(dtype=np.float64)
    speed = pd.to_numeric(source[speed_column], errors="coerce").to_numpy(dtype=np.float64)
    if len(time) < 5 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValidationError("对齐速度信号的时间列无效或数据太短")

    finite_speed = np.isfinite(speed)
    valid_ratio = float(np.mean(finite_speed))
    if np.count_nonzero(finite_speed) < 5:
        raise ValidationError("拍头速度的有效点少于 5 个，无法自动对齐")

    dt = float(np.median(np.diff(time)))
    max_gap = max(1, int(round(0.25 / dt)))
    filled = pd.Series(speed).interpolate(
        method="linear", limit=max_gap, limit_area="inside", limit_direction="both"
    )
    # 平滑需要完整数组；仅为峰值定位对剩余长缺口使用线性连接，但有效率仍单独质控。
    working = filled.interpolate(method="linear", limit_direction="both").to_numpy(dtype=float)
    requested_window = max(5, int(round(profile.smooth_window_s / dt)) | 1)
    window = _odd_window(len(working), requested_window)
    polyorder = min(3, window - 1)
    smoothed = savgol_filter(working, window, polyorder, mode="interp")
    smoothed = np.maximum(smoothed, 0.0)

    low = float(np.nanpercentile(smoothed, 10))
    high = float(np.nanpercentile(smoothed, 95))
    scale = high - low
    if not np.isfinite(scale) or scale <= 1e-9:
        raise ValidationError("拍头速度几乎不变，无法识别重复动作")
    normalized = np.maximum((smoothed - low) / scale, 0.0)

    duration = float(time[-1] - time[0])
    # 对很短的视频适当放宽间距，但绝不低于 0.35 秒，避免同一次挥拍的双峰被当两次。
    adaptive_distance = min(
        profile.min_peak_distance_s,
        max(0.35, duration / max(expected_repetitions * 2.2, 1.0)),
    )
    distance_samples = max(1, int(round(adaptive_distance / dt)))
    peaks, properties = find_peaks(
        normalized,
        distance=distance_samples,
        prominence=profile.min_prominence,
        height=0.18,
    )
    if len(peaks) < expected_repetitions:
        raise ValidationError(
            f"只检测到 {len(peaks)} 个可靠拍头速度峰，预期 {expected_repetitions} 个；"
            "请检查视频/QTM 是否包含完整动作以及球拍点识别质量"
        )

    # 限制组合搜索规模，同时保留最显著的候选峰。
    prominences = np.asarray(properties["prominences"], dtype=np.float64)
    heights = normalized[peaks]
    score = prominences + 0.25 * heights
    keep_count = min(len(peaks), max(expected_repetitions + 4, expected_repetitions))
    keep = np.argsort(score)[-keep_count:]
    candidate_indices = peaks[keep]
    candidate_prominences = prominences[keep]
    order = np.argsort(candidate_indices)

    frame = pd.DataFrame(
        {
            "time": time,
            "speed_raw": speed,
            "speed_smoothed": smoothed,
            "speed_normalized": normalized,
        }
    )
    return PreparedSignal(
        frame=frame,
        candidate_indices=candidate_indices[order],
        candidate_prominences=candidate_prominences[order],
        valid_speed_ratio=valid_ratio,
    )


def match_repetition_peaks(
    video: PreparedSignal,
    qtm: PreparedSignal,
    *,
    expected_repetitions: int,
    min_slope: float = 0.85,
    max_slope: float = 1.15,
    max_rmse_seconds: float = 0.20,
    max_residual_seconds: float = 0.35,
) -> AlignmentResult:
    """按时间顺序搜索两侧候选峰组合，并拟合一个全局仿射时间映射。"""
    video_time = video.frame["time"].to_numpy(dtype=float)
    qtm_time = qtm.frame["time"].to_numpy(dtype=float)
    best: tuple[float, np.ndarray, np.ndarray, float, float, np.ndarray] | None = None

    for video_combo in combinations(video.candidate_indices.tolist(), expected_repetitions):
        vt = video_time[np.asarray(video_combo, dtype=int)]
        for qtm_combo in combinations(qtm.candidate_indices.tolist(), expected_repetitions):
            qt = qtm_time[np.asarray(qtm_combo, dtype=int)]
            slope, intercept = np.polyfit(vt, qt, 1)
            if not min_slope <= slope <= max_slope:
                continue
            residual = qt - (slope * vt + intercept)
            rmse = float(np.sqrt(np.mean(np.square(residual))))
            # 很轻的斜率惩罚只用于多个组合 RMSE 近似相同时优先选择真实时钟比例。
            objective = rmse + 0.01 * abs(float(slope) - 1.0)
            if best is None or objective < best[0]:
                best = (
                    objective,
                    np.asarray(video_combo, dtype=int),
                    np.asarray(qtm_combo, dtype=int),
                    float(slope),
                    float(intercept),
                    residual,
                )

    if best is None:
        raise ValidationError(
            "候选动作峰无法得到合理的时间比例（允许 0.85–1.15）；"
            "请确认视频和 TSV 确为同一次采集且动作顺序一致"
        )

    _, video_indices, qtm_indices, slope, intercept, residual = best
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    max_residual = float(np.max(np.abs(residual)))
    if rmse > max_rmse_seconds or max_residual > max_residual_seconds:
        raise ValidationError(
            f"自动对齐未通过质控：锚点 RMSE={rmse:.3f}s，"
            f"最大残差={max_residual:.3f}s；请检查五次动作是否均被正确识别"
        )

    vt = video_time[video_indices]
    qt = qtm_time[qtm_indices]
    predicted = slope * vt + intercept
    anchors = pd.DataFrame(
        {
            "repetition": np.arange(1, expected_repetitions + 1),
            "video_anchor_time": vt,
            "qualisys_anchor_time": qt,
            "qualisys_predicted_time": predicted,
            "anchor_residual_seconds": qt - predicted,
        }
    )
    return AlignmentResult(
        slope=slope,
        intercept=intercept,
        rmse_seconds=rmse,
        max_abs_residual_seconds=max_residual,
        video_peak_indices=video_indices,
        qtm_peak_indices=qtm_indices,
        anchors=anchors,
    )


def derive_repetition_windows(
    video_signal: PreparedSignal,
    alignment: AlignmentResult,
    *,
    profile: ActionProfile,
) -> pd.DataFrame:
    """以视频拍头峰为中心，按速度回落点和动作上限确定五个不重叠窗口。"""
    frame = video_signal.frame
    time = frame["time"].to_numpy(dtype=float)
    normalized = frame["speed_normalized"].to_numpy(dtype=float)
    peaks = alignment.video_peak_indices
    rows: list[dict[str, float | int]] = []

    for order, peak in enumerate(peaks):
        peak_time = float(time[peak])
        threshold = max(0.08, profile.boundary_fraction * float(normalized[peak]))
        left_limit_time = peak_time - profile.max_pre_s
        right_limit_time = peak_time + profile.max_post_s
        if order > 0:
            left_limit_time = max(left_limit_time, float((time[peaks[order - 1]] + peak_time) / 2))
        if order + 1 < len(peaks):
            right_limit_time = min(
                right_limit_time, float((peak_time + time[peaks[order + 1]]) / 2)
            )

        left = int(peak)
        while left > 0 and time[left] > left_limit_time and normalized[left] > threshold:
            left -= 1
        right = int(peak)
        while (
            right < len(time) - 1
            and time[right] < right_limit_time
            and normalized[right] > threshold
        ):
            right += 1

        start = max(float(time[left]), left_limit_time, float(time[0]))
        end = min(float(time[right]), right_limit_time, float(time[-1]))
        if end - start < 0.20:
            raise ValidationError(f"第 {order + 1} 次动作窗口短于 0.20 秒，自动切分不可靠")
        rows.append(
            {
                "repetition": order + 1,
                "video_start_time": start,
                "video_anchor_time": peak_time,
                "video_end_time": end,
                "qualisys_start_time": alignment.slope * start + alignment.intercept,
                "qualisys_anchor_time_mapped": alignment.slope * peak_time + alignment.intercept,
                "qualisys_end_time": alignment.slope * end + alignment.intercept,
            }
        )
    return pd.DataFrame(rows)


def add_peak_labels(
    prepared: PreparedSignal,
    selected_indices: Iterable[int],
) -> pd.DataFrame:
    """给保存到 CSV 的速度序列增加候选峰/最终锚点标记。"""
    result = prepared.frame.copy()
    result["candidate_peak"] = 0
    result["selected_anchor"] = 0
    result.loc[prepared.candidate_indices, "candidate_peak"] = 1
    result.loc[np.asarray(list(selected_indices), dtype=int), "selected_anchor"] = 1
    return result


__all__ = [
    "AlignmentResult",
    "PreparedSignal",
    "add_peak_labels",
    "derive_repetition_windows",
    "match_repetition_peaks",
    "prepare_speed_signal",
]
