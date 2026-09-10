"""单个动作验证任务的完整编排；五个入口只负责提供路径和动作类型。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import traceback
from typing import Any

import numpy as np
import pandas as pd

from qualisys.config import (
    ACTION_LABELS,
    ANGLE_COLUMNS,
    OUTPUT_ROOT,
    RACKET_MODEL_PATH,
    SUPPORTED_ACTIONS,
)
from qualisys.core.action_profiles import get_action_profile
from qualisys.core.alignment import (
    add_peak_labels,
    derive_manual_repetition_windows,
    derive_repetition_windows,
    match_repetition_peaks,
    prepare_speed_signal,
    select_repetition_peaks_in_ranges,
)
from qualisys.core.io import (
    ValidationError,
    create_run_directory,
    read_angle_csv,
    validate_input_file,
    write_json,
)
from qualisys.core.metrics import (
    align_angle_curves,
    calculate_anchor_angle_errors,
    calculate_angle_metrics,
)
from qualisys.core.plotting import save_alignment_qc_plot
from qualisys.core.qtm import build_qtm_racket_signal, build_qualisys_angles, read_qtm_3d_tsv


def _csv(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(path, index=False, encoding="utf-8-sig")


def _run_video_racket_tracking(video_path: Path, output_dir: Path) -> pd.DataFrame:
    """延迟导入 YOLO，避免仅检查脚本时加载模型与 CUDA。"""
    if not RACKET_MODEL_PATH.is_file():
        raise FileNotFoundError(f"找不到球拍 YOLO 权重: {RACKET_MODEL_PATH}")
    from algorithm.common.point_track_core import run_point_track

    result = run_point_track(
        video_path,
        output_dir,
        RACKET_MODEL_PATH,
        save_chart=False,
        csv_filename="video_racket_alignment.csv",
    )
    data = pd.read_csv(result["csv"])
    required = {
        "frame",
        "time",
        "t_x_raw",
        "t_y_raw",
        "t_conf",
        "racket_head_speed",
        "racket_valid_kpt_count",
        "racket_mean_conf",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"球拍追踪 CSV 缺少列: {', '.join(sorted(missing))}")
    if len(data) < 2:
        raise ValidationError("视频球拍追踪结果不足两帧")
    data["time"] = pd.to_numeric(data["time"], errors="coerce")
    data["time"] = data["time"] - float(data["time"].iloc[0])
    return data


def _angle_valid_percent(data: pd.DataFrame) -> float:
    values = data[list(ANGLE_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return float(np.mean(np.isfinite(values)) * 100.0)


def _select_video_time_range(
    data: pd.DataFrame,
    video_time_range: tuple[float, float] | None,
) -> pd.DataFrame:
    """仅限制视频侧候选动作搜索范围；保留原视频绝对秒数。"""
    if video_time_range is None:
        return data.copy().reset_index(drop=True)
    if not isinstance(video_time_range, (tuple, list)) or len(video_time_range) != 2:
        raise ValueError("VIDEO_TIME_RANGE 必须是 (开始秒, 结束秒)，例如 (12.0, 22.2)")
    start, end = (float(value) for value in video_time_range)
    duration = float(data["time"].iloc[-1])
    if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
        raise ValueError("VIDEO_TIME_RANGE 必须满足 0 <= 开始秒 < 结束秒")
    if start >= duration or end > duration + 1e-6:
        raise ValueError(
            f"VIDEO_TIME_RANGE={video_time_range} 超出视频时长 {duration:.3f} 秒"
        )
    selected = data[(data["time"] >= start) & (data["time"] <= end)].copy()
    if len(selected) < 5:
        raise ValidationError("VIDEO_TIME_RANGE 内有效视频帧不足 5 帧")
    return selected.reset_index(drop=True)


def _validate_video_repetition_ranges(
    data: pd.DataFrame,
    video_repetition_ranges: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    *,
    expected_repetitions: int,
) -> tuple[tuple[float, float], ...]:
    """校验人工填写的逐次视频动作范围，返回规范化的秒数元组。"""
    if not isinstance(video_repetition_ranges, (tuple, list)):
        raise ValueError("VIDEO_REPETITION_RANGES 必须是五个 (开始秒, 结束秒)")
    if len(video_repetition_ranges) != expected_repetitions:
        raise ValueError(
            "VIDEO_REPETITION_RANGES 数量必须与 EXPECTED_REPETITIONS 一致："
            f"收到 {len(video_repetition_ranges)} 段，预期 {expected_repetitions} 段"
        )

    duration = float(data["time"].iloc[-1])
    normalized: list[tuple[float, float]] = []
    previous_end = -np.inf
    for repetition, value in enumerate(video_repetition_ranges, start=1):
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(f"第 {repetition} 次动作必须填写为 (开始秒, 结束秒)")
        start, end = (float(item) for item in value)
        if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"第 {repetition} 次动作必须满足 0 <= 开始秒 < 结束秒")
        if end - start < 0.20:
            raise ValueError(f"第 {repetition} 次动作范围短于 0.20 秒")
        if start < previous_end:
            raise ValueError("VIDEO_REPETITION_RANGES 必须按时间递增且不能重叠")
        if end > duration + 1e-6:
            raise ValueError(
                f"第 {repetition} 次动作结束时间 {end:.3f}s 超出视频时长 {duration:.3f}s"
            )
        if len(data[(data["time"] >= start) & (data["time"] <= end)]) < 5:
            raise ValidationError(f"第 {repetition} 次手动动作范围内视频帧不足 5 帧")
        normalized.append((start, end))
        previous_end = end
    return tuple(normalized)


def run_validation_job(
    *,
    action: str,
    video_path: Path | str,
    rtmpose_csv_path: Path | str,
    qualisys_tsv_path: Path | str,
    video_time_range: tuple[float, float] | None = None,
    video_repetition_ranges: tuple[tuple[float, float], ...]
    | list[tuple[float, float]]
    | None = None,
    expected_repetitions: int = 5,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, str]:
    """建立新目录，按自动或手动逐次模式对齐八角并生成描述指标。"""
    if action not in SUPPORTED_ACTIONS:
        raise ValueError(f"不支持的动作: {action}")
    if expected_repetitions < 2:
        raise ValueError("expected_repetitions 至少为 2；本实验默认且建议保持 5")
    if video_time_range is not None and video_repetition_ranges is not None:
        raise ValueError(
            "VIDEO_TIME_RANGE 与 VIDEO_REPETITION_RANGES 不能同时填写；"
            "手动逐次模式请将 VIDEO_TIME_RANGE 设为 None"
        )

    video_path = validate_input_file(video_path, "视频")
    rtmpose_csv_path = validate_input_file(rtmpose_csv_path, "RTMPose 八角 CSV")
    qualisys_tsv_path = validate_input_file(qualisys_tsv_path, "Qualisys 3D TSV")
    profile = get_action_profile(action)
    output_dir = create_run_directory(
        action,
        video_path.stem,
        output_root=output_root,
    )
    config_payload: dict[str, Any] = {
        "status": "running",
        "action": action,
        "action_label": ACTION_LABELS[action],
        "video_path": str(video_path.resolve()),
        "rtmpose_csv_path": str(rtmpose_csv_path.resolve()),
        "qualisys_tsv_path": str(qualisys_tsv_path.resolve()),
        "video_time_range_seconds": video_time_range,
        "video_repetition_ranges_seconds": video_repetition_ranges,
        "repetition_selection_mode": (
            "manual_video_ranges" if video_repetition_ranges is not None else "automatic"
        ),
        "expected_repetitions": expected_repetitions,
        "output_dir": str(output_dir.resolve()),
        "alignment_anchor": "five racket_speed_peaks (not ball contact)",
        "time_map": "piecewise linear through five paired racket-speed anchors",
        "alignment_method": "piecewise_linear_racket_anchors",
        "action_profile": asdict(profile),
    }
    write_json(output_dir / "run_status.json", config_payload)

    try:
        rtmpose = read_angle_csv(rtmpose_csv_path, "RTMPose")
        metadata, qtm_raw = read_qtm_3d_tsv(qualisys_tsv_path)
        qualisys = build_qualisys_angles(qtm_raw)
        qtm_racket = build_qtm_racket_signal(qtm_raw)
        video_racket = _run_video_racket_tracking(video_path, output_dir)
        normalized_repetition_ranges = None
        if video_repetition_ranges is not None:
            normalized_repetition_ranges = _validate_video_repetition_ranges(
                video_racket,
                video_repetition_ranges,
                expected_repetitions=expected_repetitions,
            )
            video_racket = _select_video_time_range(
                video_racket,
                (normalized_repetition_ranges[0][0], normalized_repetition_ranges[-1][1]),
            )
            print(
                f"🎯 发球使用 {expected_repetitions} 个手动视频动作范围；"
                "每段内自动选择拍头速度最高点"
            )
        else:
            video_racket = _select_video_time_range(video_racket, video_time_range)
        if video_time_range is not None:
            print(
                f"🎯 自动对齐仅使用视频 {video_time_range[0]:.3f}–"
                f"{video_time_range[1]:.3f} 秒；Qualisys 使用全部长度"
            )

        raw_video_valid = np.isfinite(
            video_racket[["t_x_raw", "t_y_raw"]].to_numpy(dtype=float)
        ).all(axis=1)
        video_tracking_valid_percent = float(np.mean(raw_video_valid) * 100.0)
        qtm_tracking_valid_percent = float(
            pd.to_numeric(qtm_racket["valid_position"], errors="coerce").mean() * 100.0
        )
        if video_tracking_valid_percent < 20.0:
            raise ValidationError(
                f"视频拍头原始有效率仅 {video_tracking_valid_percent:.1f}%，低于 20%；"
                "自动动作对齐不可靠"
            )
        if qtm_tracking_valid_percent < 80.0:
            raise ValidationError(
                f"QTM Racket_top 有效率仅 {qtm_tracking_valid_percent:.1f}%，低于 80%"
            )

        video_prepared = prepare_speed_signal(
            video_racket,
            profile=profile,
            expected_repetitions=expected_repetitions,
        )
        qtm_prepared = prepare_speed_signal(
            qtm_racket,
            profile=profile,
            expected_repetitions=expected_repetitions,
        )
        fixed_video_peak_indices = None
        if normalized_repetition_ranges is not None:
            fixed_video_peak_indices = select_repetition_peaks_in_ranges(
                video_prepared,
                normalized_repetition_ranges,
            )
            config_payload["video_anchor_times_seconds"] = video_prepared.frame.loc[
                fixed_video_peak_indices, "time"
            ].tolist()
            write_json(output_dir / "run_status.json", config_payload)
        alignment = match_repetition_peaks(
            video_prepared,
            qtm_prepared,
            expected_repetitions=expected_repetitions,
            fixed_video_peak_indices=fixed_video_peak_indices,
        )
        if normalized_repetition_ranges is None:
            windows = derive_repetition_windows(video_prepared, alignment, profile=profile)
        else:
            windows = derive_manual_repetition_windows(
                normalized_repetition_ranges,
                alignment,
            )
        aligned = align_angle_curves(
            rtmpose,
            qualisys,
            windows,
            alignment=alignment,
        )
        metrics = calculate_angle_metrics(aligned)
        anchor_angles = calculate_anchor_angle_errors(
            rtmpose,
            qualisys,
            alignment.anchors,
            alignment=alignment,
        )

        video_signal = add_peak_labels(video_prepared, alignment.video_peak_indices)
        qtm_signal = add_peak_labels(qtm_prepared, alignment.qtm_peak_indices)
        compact_video = video_racket[
            [
                "frame",
                "time",
                "t_x_raw",
                "t_y_raw",
                "t_conf",
                "racket_valid_kpt_count",
                "racket_mean_conf",
            ]
        ].copy()
        compact_video = pd.concat(
            [compact_video.reset_index(drop=True), video_signal.drop(columns="time")], axis=1
        )
        compact_qtm = pd.concat(
            [
                qtm_racket.drop(columns="racket_head_speed").reset_index(drop=True),
                qtm_signal.drop(columns="time").reset_index(drop=True),
            ],
            axis=1,
        )

        _csv(compact_video, output_dir / "video_racket_alignment.csv")
        _csv(compact_qtm, output_dir / "qualisys_racket_alignment.csv")
        _csv(qualisys, output_dir / "qualisys_8_angles.csv")
        _csv(alignment.anchors, output_dir / "alignment_anchors.csv")
        _csv(alignment.segments, output_dir / "alignment_segments.csv")
        _csv(windows, output_dir / "repetition_windows.csv")
        _csv(aligned, output_dir / "aligned_angles.csv")
        _csv(metrics, output_dir / "angle_metrics.csv")
        _csv(anchor_angles, output_dir / "anchor_angle_errors.csv")

        qc = pd.DataFrame(
            [
                {
                    "status": "pass",
                    "action": action,
                    "alignment_method": "piecewise_linear_racket_anchors",
                    "repetition_selection_mode": (
                        "manual_video_ranges"
                        if normalized_repetition_ranges is not None
                        else "automatic"
                    ),
                    "expected_repetitions": expected_repetitions,
                    "video_candidate_peaks": len(video_prepared.candidate_indices),
                    "qualisys_candidate_peaks": len(qtm_prepared.candidate_indices),
                    "global_affine_only_diagnostic": True,
                    "slope": alignment.slope,
                    "time_scale_warning": bool(
                        np.any(
                            (alignment.segments["local_slope"].to_numpy(float) < 0.90)
                            | (alignment.segments["local_slope"].to_numpy(float) > 1.10)
                        )
                    ),
                    "intercept_seconds": alignment.intercept,
                    "anchor_rmse_seconds": alignment.rmse_seconds,
                    "anchor_max_abs_residual_seconds": alignment.max_abs_residual_seconds,
                    "global_affine_fit_warning": (
                        alignment.rmse_seconds > 0.20
                        or alignment.max_abs_residual_seconds > 0.35
                    ),
                    "local_slope_min": float(alignment.segments["local_slope"].min()),
                    "local_slope_max": float(alignment.segments["local_slope"].max()),
                    "video_racket_raw_valid_percent": video_tracking_valid_percent,
                    "qualisys_racket_valid_percent": qtm_tracking_valid_percent,
                    "rtmpose_angle_valid_percent": _angle_valid_percent(rtmpose),
                    "qualisys_angle_valid_percent": _angle_valid_percent(qualisys),
                    "qtm_frequency_hz": metadata.get("FREQUENCY", ""),
                    "aligned_video_frames": len(aligned),
                }
            ]
        )
        _csv(qc, output_dir / "alignment_qc.csv")
        save_alignment_qc_plot(
            video_signal,
            qtm_signal,
            output_dir / "alignment_qc.png",
        )

        config_payload.update(
            {
                "status": "complete",
                "slope": alignment.slope,
                "intercept_seconds": alignment.intercept,
                "anchor_rmse_seconds": alignment.rmse_seconds,
                "anchor_max_abs_residual_seconds": alignment.max_abs_residual_seconds,
                "global_affine_only_diagnostic": True,
                "global_affine_fit_warning": (
                    alignment.rmse_seconds > 0.20
                    or alignment.max_abs_residual_seconds > 0.35
                ),
                "local_slopes": alignment.segments["local_slope"].tolist(),
                "video_anchor_times_seconds": alignment.anchors[
                    "video_anchor_time"
                ].tolist(),
                "qualisys_anchor_times_seconds": alignment.anchors[
                    "qualisys_anchor_time"
                ].tolist(),
            }
        )
        write_json(output_dir / "run_status.json", config_payload)
    except Exception as exc:
        config_payload.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(output_dir / "run_status.json", config_payload)
        raise

    return {
        "output_dir": str(output_dir.resolve()),
        "alignment_qc": str((output_dir / "alignment_qc.csv").resolve()),
        "angle_metrics": str((output_dir / "angle_metrics.csv").resolve()),
        "aligned_angles": str((output_dir / "aligned_angles.csv").resolve()),
    }


__all__ = [
    "_select_video_time_range",
    "_validate_video_repetition_ranges",
    "run_validation_job",
]
