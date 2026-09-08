"""Web 分析流水线编排；四类动作共享同一套执行骨架。"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, List

from algorithm.common.kinematic_fusion import write_fused_debug_csv

REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_ACTIONS = ("forehand", "backhand", "serve", "volley")

Runner = Callable[..., Dict[str, Any]]


def _write_fused_debug(
    out_dir: Path,
    video_stem: str,
    racket_csv: str,
    body_csv: str,
    handedness: str,
) -> None:
    """切分前写入融合 debug CSV（不进入 API artifacts）。"""
    debug_path = out_dir / f"{video_stem}_fused_kinematic_debug.csv"
    write_fused_debug_csv(racket_csv, body_csv, debug_path, handedness=handedness)


def _rel_outputs(path: Path) -> str:
    """返回相对于 data/outputs 的路径（posix 风格）。"""
    resolved = path.resolve()
    base = (REPO_ROOT / "data" / "outputs").resolve()
    return resolved.relative_to(base).as_posix()


def build_artifact_list(result: Dict[str, Any]) -> List[Dict[str, str]]:
    """把流水线产物转换为前端使用的稳定 API 列表。"""
    artifacts = result["artifacts"]
    items: List[Dict[str, str]] = []

    def add(kind: str, value: str) -> None:
        if not value:
            return
        path_obj = Path(value)
        items.append(
            {
                "kind": kind,
                "filename": path_obj.name,
                "relative_path": _rel_outputs(path_obj),
            }
        )

    # 顺序与重构前 API 完全一致，避免前端或外部调用方依赖列表顺序时发生变化。
    ordered_paths = (
        ("single", "racket_csv", "racket_csv"),
        ("single", "body_csv", "body_csv"),
        ("single", "racket_chart", "racket_chart"),
        ("single", "final_chart", "final_chart"),
        ("list", "clip", "clips"),
        ("list", "kinetic_chart", "kinetic_charts"),
        ("list", "speed_cog_chart", "speed_cog_charts"),
        ("single", "serve_trace_chart", "serve_trace_chart"),
        ("single", "serve_kinetic_chart", "serve_kinetic_chart"),
        ("list", "volley_trace_chart", "volley_trace_charts"),
        ("list", "volley_kinetic_chart", "volley_kinetic_charts"),
        ("single", "kinematic_summary_csv", "kinematic_summary_csv"),
        ("single", "kinematic_phase_summary_csv", "kinematic_phase_summary_csv"),
        ("list", "upper_limb_angle_chart", "upper_limb_charts"),
        ("list", "lower_limb_angle_chart", "lower_limb_charts"),
        ("list", "trunk_rotation_chart", "trunk_rotation_charts"),
        ("list", "racket_kinematic_chart", "racket_kinematic_charts"),
    )
    for cardinality, kind, key in ordered_paths:
        value = artifacts.get(key)
        if cardinality == "single":
            add(kind, value or "")
        else:
            for path_value in value or []:
                add(kind, path_value)
    return items


def _load_action_runners(action: str) -> tuple[Runner, Runner, Runner]:
    """延迟加载模型相关模块，保证仅打开 Web 首页时不初始化 CUDA/模型。"""
    point_track = getattr(import_module(f"algorithm.{action}.point_track"), "run_point_track")
    angle_export = getattr(import_module(f"algorithm.{action}.angles_csv"), "run_angles_csv")
    segmentation = getattr(import_module(f"algorithm.{action}.segmentation"), "run_segmentation")
    return point_track, angle_export, segmentation


def _collect_action_artifacts(
    action: str,
    point_result: Dict[str, Any],
    pose_result: Dict[str, Any],
    segmentation_result: Dict[str, Any],
) -> Dict[str, Any]:
    """保留现有 API kind 和文件名，只统一组装过程。"""
    artifacts: Dict[str, Any] = {
        "racket_csv": point_result["csv"],
        "racket_chart": point_result["chart"],
        "body_csv": pose_result["csv"],
        "clips": segmentation_result["clips"],
        "kinematic_summary_csv": segmentation_result.get("kinematic_summary_csv", ""),
        "kinematic_phase_summary_csv": segmentation_result.get(
            "kinematic_phase_summary_csv", ""
        ),
        "upper_limb_charts": segmentation_result.get("upper_limb_charts") or [],
        "lower_limb_charts": segmentation_result.get("lower_limb_charts") or [],
        "trunk_rotation_charts": segmentation_result.get("trunk_rotation_charts") or [],
        "racket_kinematic_charts": segmentation_result.get("racket_kinematic_charts") or [],
    }

    if action in {"forehand", "backhand"}:
        artifacts.update(
            {
                "final_chart": segmentation_result["chart"],
                "kinetic_charts": segmentation_result.get("kinetic_charts") or [],
                "speed_cog_charts": segmentation_result.get("speed_cog_charts") or [],
            }
        )
    elif action == "serve":
        artifacts.update(
            {
                "serve_trace_chart": segmentation_result["trace_chart"],
                "serve_kinetic_chart": segmentation_result["kinetic_chart"],
            }
        )
    else:
        artifacts.update(
            {
                "volley_trace_charts": segmentation_result.get("volley_trace_charts") or [],
                "volley_kinetic_charts": segmentation_result.get("volley_kinetic_charts") or [],
            }
        )
    return artifacts


def run_action_pipeline(
    action: str,
    video_path: Path,
    run_id: str,
    handedness: str = "right",
) -> Dict[str, Any]:
    """运行指定动作的球拍追踪、角度CSV导出、融合和切分。"""
    normalized = str(action).strip().lower()
    if normalized not in SUPPORTED_ACTIONS:
        raise ValueError(f"不支持的动作类型: {action!r}")

    resolved_video = Path(video_path).resolve()
    out_dir = REPO_ROOT / "data" / "outputs" / run_id / resolved_video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    run_point_track, run_angles_csv, run_segmentation = _load_action_runners(normalized)
    point_result = run_point_track(resolved_video, out_dir)
    pose_result = run_angles_csv(resolved_video, out_dir)
    _write_fused_debug(
        out_dir,
        resolved_video.stem,
        point_result["csv"],
        pose_result["csv"],
        handedness,
    )
    segmentation_result = run_segmentation(
        resolved_video,
        point_result["csv"],
        pose_result["csv"],
        out_dir,
        handedness=handedness,
    )

    return {
        "run_id": run_id,
        "video_name": resolved_video.stem,
        "artifacts": _collect_action_artifacts(
            normalized,
            point_result,
            pose_result,
            segmentation_result,
        ),
        "intervals": [
            [float(start), float(end)] for start, end in segmentation_result["intervals"]
        ],
    }


def run_forehand_pipeline(video_path: Path, run_id: str, handedness: str = "right") -> Dict[str, Any]:
    return run_action_pipeline("forehand", video_path, run_id, handedness)


def run_backhand_pipeline(video_path: Path, run_id: str, handedness: str = "right") -> Dict[str, Any]:
    return run_action_pipeline("backhand", video_path, run_id, handedness)


def run_serve_pipeline(video_path: Path, run_id: str, handedness: str = "right") -> Dict[str, Any]:
    return run_action_pipeline("serve", video_path, run_id, handedness)


def run_volley_pipeline(video_path: Path, run_id: str, handedness: str = "right") -> Dict[str, Any]:
    return run_action_pipeline("volley", video_path, run_id, handedness)
