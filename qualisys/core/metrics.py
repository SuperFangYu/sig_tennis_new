"""把 Qualisys 插值到视频时刻，并计算八角一致性指标。"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from qualisys.config import ANGLE_COLUMNS
from qualisys.core.alignment import AlignmentResult, map_video_times_to_qualisys
from qualisys.core.io import ValidationError


def interpolate_column(
    reference: pd.DataFrame,
    column: str,
    target_time: np.ndarray,
) -> np.ndarray:
    source_time = reference["time"].to_numpy(dtype=np.float64)
    source_value = pd.to_numeric(reference[column], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(source_time) & np.isfinite(source_value)
    output = np.full(len(target_time), np.nan, dtype=np.float64)
    if np.count_nonzero(valid) < 2:
        return output
    x = source_time[valid]
    y = source_value[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique = np.concatenate(([True], np.diff(x) > 0))
    x = x[unique]
    y = y[unique]
    if len(x) < 2:
        return output
    inside = (target_time >= x[0]) & (target_time <= x[-1])
    output[inside] = np.interp(target_time[inside], x, y)
    return output


def align_angle_curves(
    rtmpose: pd.DataFrame,
    qualisys: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    alignment: AlignmentResult,
) -> pd.DataFrame:
    """五次动作和八个角严格共用同一个连续分段时间映射。"""
    q_min = float(qualisys["time"].min())
    q_max = float(qualisys["time"].max())
    if (
        float(windows["qualisys_start_time"].min()) < q_min - 0.05
        or float(windows["qualisys_end_time"].max()) > q_max + 0.05
    ):
        raise ValidationError("映射后的动作窗口超出 Qualisys 记录范围，请检查峰值配对")

    parts: list[pd.DataFrame] = []
    for row in windows.itertuples(index=False):
        segment = rtmpose[
            (rtmpose["time"] >= row.video_start_time)
            & (rtmpose["time"] <= row.video_end_time)
        ].copy()
        if len(segment) < 2:
            raise ValidationError(f"第 {row.repetition} 次动作内的 RTMPose 数据不足")
        video_time = segment["time"].to_numpy(dtype=float)
        qtm_time = map_video_times_to_qualisys(video_time, alignment)
        aligned = pd.DataFrame(
            {
                "repetition": int(row.repetition),
                "video_frame": segment["frame"].to_numpy(),
                "video_time": video_time,
                "qualisys_time": qtm_time,
            }
        )
        for angle in ANGLE_COLUMNS:
            estimate = pd.to_numeric(segment[angle], errors="coerce").to_numpy(dtype=float)
            reference = interpolate_column(qualisys, angle, qtm_time)
            aligned[f"{angle}_rtmpose"] = estimate
            aligned[f"{angle}_qualisys"] = reference
            aligned[f"{angle}_error"] = estimate - reference
        parts.append(aligned)
    return pd.concat(parts, ignore_index=True)


def _metric_row(
    repetition: str | int,
    angle: str,
    estimate: Iterable[float],
    reference: Iterable[float],
) -> dict[str, object]:
    estimate_array = np.asarray(list(estimate), dtype=np.float64)
    reference_array = np.asarray(list(reference), dtype=np.float64)
    valid = np.isfinite(estimate_array) & np.isfinite(reference_array)
    x = estimate_array[valid]
    y = reference_array[valid]
    row: dict[str, object] = {
        "repetition": repetition,
        "angle": angle,
        "n": int(len(x)),
        "valid_percent": float(np.mean(valid) * 100.0) if len(valid) else 0.0,
    }
    if len(x) == 0:
        return row

    error = x - y
    bias = float(np.mean(error))
    sd_error = float(np.std(error, ddof=1)) if len(error) > 1 else float("nan")
    row.update(
        {
            "mae_deg": float(np.mean(np.abs(error))),
            "rmse_deg": float(np.sqrt(np.mean(np.square(error)))),
            "bias_deg": bias,
            "sd_error_deg": sd_error,
            "loa_lower_deg": bias - 1.96 * sd_error,
            "loa_upper_deg": bias + 1.96 * sd_error,
            "within_5deg_percent": float(np.mean(np.abs(error) <= 5.0) * 100.0),
            "within_10deg_percent": float(np.mean(np.abs(error) <= 10.0) * 100.0),
        }
    )
    if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
        row["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
        covariance = float(np.mean((x - np.mean(x)) * (y - np.mean(y))))
        denominator = float(np.var(x) + np.var(y) + (np.mean(x) - np.mean(y)) ** 2)
        row["ccc"] = float(2.0 * covariance / denominator) if denominator > 0 else float("nan")
    else:
        row["pearson_r"] = float("nan")
        row["ccc"] = float("nan")
    return row


def calculate_angle_metrics(aligned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    repetitions: list[str | int] = sorted(aligned["repetition"].unique().tolist())
    for repetition in [*repetitions, "all"]:
        part = aligned if repetition == "all" else aligned[aligned["repetition"] == repetition]
        for angle in ANGLE_COLUMNS:
            rows.append(
                _metric_row(
                    repetition,
                    angle,
                    part[f"{angle}_rtmpose"],
                    part[f"{angle}_qualisys"],
                )
            )
    return pd.DataFrame(rows)


def calculate_anchor_angle_errors(
    rtmpose: pd.DataFrame,
    qualisys: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    alignment: AlignmentResult,
) -> pd.DataFrame:
    """输出拍头速度锚点处的角度误差；锚点不是触球帧。"""
    rows: list[dict[str, object]] = []
    video_time = rtmpose["time"].to_numpy(dtype=float)
    for anchor in anchors.itertuples(index=False):
        index = int(np.argmin(np.abs(video_time - anchor.video_anchor_time)))
        mapped_time = float(
            map_video_times_to_qualisys(
                np.asarray([video_time[index]], dtype=float),
                alignment,
            )[0]
        )
        for angle in ANGLE_COLUMNS:
            estimate = float(rtmpose.iloc[index][angle])
            reference = float(interpolate_column(qualisys, angle, np.array([mapped_time]))[0])
            rows.append(
                {
                    "repetition": int(anchor.repetition),
                    "anchor_type": "racket_speed_peak",
                    "angle": angle,
                    "video_time": float(video_time[index]),
                    "qualisys_time": mapped_time,
                    "rtmpose_angle_deg": estimate,
                    "qualisys_angle_deg": reference,
                    "error_deg": estimate - reference,
                    "absolute_error_deg": abs(estimate - reference),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "align_angle_curves",
    "calculate_anchor_angle_errors",
    "calculate_angle_metrics",
    "interpolate_column",
]
