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
    derive_repetition_windows,
    match_repetition_peaks,
    prepare_speed_signal,
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


def run_validation_job(
    *,
    action: str,
    video_path: Path | str,
    rtmpose_csv_path: Path | str,
    qualisys_tsv_path: Path | str,
    expected_repetitions: int = 5,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, str]:
    """自动建立新目录、识别五次动作、对齐八角并生成信效度描述指标。"""
    if action not in SUPPORTED_ACTIONS:
        raise ValueError(f"不支持的动作: {action}")
    if expected_repetitions < 2:
        raise ValueError("expected_repetitions 至少为 2；本实验默认且建议保持 5")

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
        "expected_repetitions": expected_repetitions,
        "output_dir": str(output_dir.resolve()),
        "alignment_anchor": "racket_speed_peak (not ball contact)",
        "time_map": "t_qualisys = slope * t_video + intercept",
        "action_profile": asdict(profile),
    }
    write_json(output_dir / "run_status.json", config_payload)

    try:
        rtmpose = read_angle_csv(rtmpose_csv_path, "RTMPose")
        metadata, qtm_raw = read_qtm_3d_tsv(qualisys_tsv_path)
        qualisys = build_qualisys_angles(qtm_raw)
        qtm_racket = build_qtm_racket_signal(qtm_raw)
        video_racket = _run_video_racket_tracking(video_path, output_dir)

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
        alignment = match_repetition_peaks(
            video_prepared,
            qtm_prepared,
            expected_repetitions=expected_repetitions,
        )
        windows = derive_repetition_windows(video_prepared, alignment, profile=profile)
        aligned = align_angle_curves(
            rtmpose,
            qualisys,
            windows,
            slope=alignment.slope,
            intercept=alignment.intercept,
        )
        metrics = calculate_angle_metrics(aligned)
        anchor_angles = calculate_anchor_angle_errors(
            rtmpose,
            qualisys,
            alignment.anchors,
            slope=alignment.slope,
            intercept=alignment.intercept,
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
        _csv(windows, output_dir / "repetition_windows.csv")
        _csv(aligned, output_dir / "aligned_angles.csv")
        _csv(metrics, output_dir / "angle_metrics.csv")
        _csv(anchor_angles, output_dir / "anchor_angle_errors.csv")

        qc = pd.DataFrame(
            [
                {
                    "status": "pass",
                    "action": action,
                    "expected_repetitions": expected_repetitions,
                    "video_candidate_peaks": len(video_prepared.candidate_indices),
                    "qualisys_candidate_peaks": len(qtm_prepared.candidate_indices),
                    "slope": alignment.slope,
                    "intercept_seconds": alignment.intercept,
                    "anchor_rmse_seconds": alignment.rmse_seconds,
                    "anchor_max_abs_residual_seconds": alignment.max_abs_residual_seconds,
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


__all__ = ["run_validation_job"]
