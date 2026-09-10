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
    """五个对应球拍事件及其连续分段线性时间映射。"""

    slope: float
    intercept: float
    rmse_seconds: float
    max_abs_residual_seconds: float
    video_peak_indices: np.ndarray
    qtm_peak_indices: np.ndarray
    anchors: pd.DataFrame
    segments: pd.DataFrame


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
    fixed_video_peak_indices: Iterable[int] | None = None,
    min_slope: float = 0.70,
    max_slope: float = 1.40,
) -> AlignmentResult:
    """匹配顺序一致的球拍峰，并建立通过五个锚点的分段线性时间映射。"""
    video_time = video.frame["time"].to_numpy(dtype=float)
    qtm_time = qtm.frame["time"].to_numpy(dtype=float)
    best: tuple[float, np.ndarray, np.ndarray, float, float, np.ndarray] | None = None

    if fixed_video_peak_indices is None:
        video_combos: Iterable[tuple[int, ...]] = combinations(
            video.candidate_indices.tolist(), expected_repetitions
        )
    else:
        fixed = np.asarray(list(fixed_video_peak_indices), dtype=int)
        if len(fixed) != expected_repetitions:
            raise ValueError(
                "手动视频峰数量必须与 expected_repetitions 一致："
                f"收到 {len(fixed)} 个，预期 {expected_repetitions} 个"
            )
        if np.any(fixed < 0) or np.any(fixed >= len(video_time)):
            raise ValueError("手动视频峰索引超出速度序列范围")
        if np.any(np.diff(fixed) <= 0):
            raise ValueError("手动视频峰必须按时间严格递增且不能重复")
        video_combos = [tuple(int(value) for value in fixed)]

    for video_combo in video_combos:
        vt = video_time[np.asarray(video_combo, dtype=int)]
        for qtm_combo in combinations(qtm.candidate_indices.tolist(), expected_repetitions):
            qt = qtm_time[np.asarray(qtm_combo, dtype=int)]
            local_slopes = np.diff(qt) / np.diff(vt)
            if np.any(~np.isfinite(local_slopes)) or np.any(
                (local_slopes < min_slope) | (local_slopes > max_slope)
            ):
                continue
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
            f"候选动作峰无法得到单调且合理的分段时间比例（允许 {min_slope:.2f}–{max_slope:.2f}）；"
            "请确认视频和 TSV 确为同一次采集且动作顺序一致"
        )

    _, video_indices, qtm_indices, slope, intercept, residual = best
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    max_residual = float(np.max(np.abs(residual)))
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
            "piecewise_mapped_time": qt,
            "piecewise_residual_seconds": np.zeros(len(qt), dtype=float),
        }
    )
    local_slopes = np.diff(qt) / np.diff(vt)
    segments = pd.DataFrame(
        {
            "segment": np.arange(1, len(vt)),
            "video_start_anchor_time": vt[:-1],
            "video_end_anchor_time": vt[1:],
            "qualisys_start_anchor_time": qt[:-1],
            "qualisys_end_anchor_time": qt[1:],
            "local_slope": local_slopes,
            "local_intercept_seconds": qt[:-1] - local_slopes * vt[:-1],
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
        segments=segments,
    )


def map_video_times_to_qualisys(
    video_times: Iterable[float] | np.ndarray,
    alignment: AlignmentResult,
) -> np.ndarray:
    """用锚点间线性插值映射时间，首尾使用相邻分段斜率外推。"""
    target = np.asarray(list(video_times), dtype=np.float64)
    video_anchors = alignment.anchors["video_anchor_time"].to_numpy(dtype=float)
    qtm_anchors = alignment.anchors["qualisys_anchor_time"].to_numpy(dtype=float)
    if len(video_anchors) < 2 or len(video_anchors) != len(qtm_anchors):
        raise ValueError("分段时间映射至少需要两个数量相同的对应锚点")
    if np.any(np.diff(video_anchors) <= 0) or np.any(np.diff(qtm_anchors) <= 0):
        raise ValueError("分段时间映射锚点必须严格递增")

    mapped = np.interp(target, video_anchors, qtm_anchors)
    local_slopes = np.diff(qtm_anchors) / np.diff(video_anchors)
    before = target < video_anchors[0]
    after = target > video_anchors[-1]
    mapped[before] = qtm_anchors[0] + local_slopes[0] * (
        target[before] - video_anchors[0]
    )
    mapped[after] = qtm_anchors[-1] + local_slopes[-1] * (
        target[after] - video_anchors[-1]
    )
    return mapped


def select_repetition_peaks_in_ranges(
    prepared: PreparedSignal,
    repetition_ranges: Iterable[tuple[float, float]],
) -> np.ndarray:
    """在人工指定的每个完整动作时间段内选择一个拍头速度最高点。"""
    time = prepared.frame["time"].to_numpy(dtype=float)
    speed = prepared.frame["speed_normalized"].to_numpy(dtype=float)
    selected: list[int] = []

    for repetition, (start, end) in enumerate(repetition_ranges, start=1):
        indices = np.flatnonzero((time >= float(start)) & (time <= float(end)))
        if len(indices) < 3:
            raise ValidationError(
                f"第 {repetition} 次手动动作范围 {start:.3f}–{end:.3f}s 内有效速度点不足 3 个"
            )
        local_speed = speed[indices]
        if not np.any(np.isfinite(local_speed)):
            raise ValidationError(f"第 {repetition} 次手动动作范围内没有有效拍头速度")
        peak = int(indices[int(np.nanargmax(local_speed))])
        if peak == int(indices[0]) or peak == int(indices[-1]):
            raise ValidationError(
                f"第 {repetition} 次动作的速度最高点落在手动范围边界；"
                "请适当扩大该动作的开始/结束时间"
            )
        selected.append(peak)

    result = np.asarray(selected, dtype=int)
    if np.any(np.diff(result) <= 0):
        raise ValueError("手动动作范围必须按视频时间严格递增且互不重叠")
    return result


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
        mapped = map_video_times_to_qualisys(
            np.asarray([start, peak_time, end], dtype=float),
            alignment,
        )
        rows.append(
            {
                "repetition": order + 1,
                "video_start_time": start,
                "video_anchor_time": peak_time,
                "video_end_time": end,
                "qualisys_start_time": float(mapped[0]),
                "qualisys_anchor_time_mapped": float(mapped[1]),
                "qualisys_end_time": float(mapped[2]),
            }
        )
    return pd.DataFrame(rows)


def derive_manual_repetition_windows(
    repetition_ranges: Iterable[tuple[float, float]],
    alignment: AlignmentResult,
) -> pd.DataFrame:
    """使用人工填写的完整动作边界，并用统一时间映射生成 Qualisys 边界。"""
    ranges = list(repetition_ranges)
    if len(ranges) != len(alignment.video_peak_indices):
        raise ValueError("手动动作范围数量必须与已对齐的视频峰数量一致")

    video_time = alignment.anchors["video_anchor_time"].to_numpy(dtype=float)
    rows: list[dict[str, float | int]] = []
    for repetition, ((start, end), anchor_time) in enumerate(
        zip(ranges, video_time, strict=True), start=1
    ):
        if not float(start) < float(anchor_time) < float(end):
            raise ValidationError(
                f"第 {repetition} 次视频锚点 {anchor_time:.3f}s 不在手动动作范围 "
                f"{start:.3f}–{end:.3f}s 内"
            )
        mapped = map_video_times_to_qualisys(
            np.asarray([start, anchor_time, end], dtype=float),
            alignment,
        )
        rows.append(
            {
                "repetition": repetition,
                "video_start_time": float(start),
                "video_anchor_time": float(anchor_time),
                "video_end_time": float(end),
                "qualisys_start_time": float(mapped[0]),
                "qualisys_anchor_time_mapped": float(mapped[1]),
                "qualisys_end_time": float(mapped[2]),
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
    "derive_manual_repetition_windows",
    "derive_repetition_windows",
    "match_repetition_peaks",
    "map_video_times_to_qualisys",
    "prepare_speed_signal",
    "select_repetition_peaks_in_ranges",
]
