from pathlib import Path
from typing import Any, Dict, List

from algorithm.common.kinematic_fusion import write_fused_debug_csv

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    p = path.resolve()
    base = (REPO_ROOT / "data" / "outputs").resolve()
    rel = p.relative_to(base)
    return rel.as_posix()


def build_artifact_list(result: Dict[str, Any]) -> List[Dict[str, str]]:
    """构建 API artifacts 列表（含运动学摘要与角度曲线图）。"""
    run_id = result["run_id"]
    video_name = result["video_name"]
    arts = result["artifacts"]
    items: List[Dict[str, str]] = []

    def add(kind: str, p: str) -> None:
        if not p:
            return
        path_obj = Path(p)
        items.append(
            {
                "kind": kind,
                "filename": path_obj.name,
                "relative_path": _rel_outputs(path_obj),
            }
        )

    add("racket_csv", arts.get("racket_csv", ""))
    add("body_csv", arts.get("body_csv", ""))
    add("racket_chart", arts.get("racket_chart", ""))
    add("final_chart", arts.get("final_chart", ""))
    for clip in arts.get("clips") or []:
        add("clip", clip)
    for kc in arts.get("kinetic_charts") or []:
        add("kinetic_chart", kc)
    for sc in arts.get("speed_cog_charts") or []:
        add("speed_cog_chart", sc)
    add("serve_trace_chart", arts.get("serve_trace_chart", ""))
    add("serve_kinetic_chart", arts.get("serve_kinetic_chart", ""))
    for vtc in arts.get("volley_trace_charts") or []:
        add("volley_trace_chart", vtc)
    for vkc in arts.get("volley_kinetic_charts") or []:
        add("volley_kinetic_chart", vkc)
    add("kinematic_summary_csv", arts.get("kinematic_summary_csv", ""))
    add("kinematic_phase_summary_csv", arts.get("kinematic_phase_summary_csv", ""))
    for ul in arts.get("upper_limb_charts") or []:
        add("upper_limb_angle_chart", ul)
    for ll in arts.get("lower_limb_charts") or []:
        add("lower_limb_angle_chart", ll)
    for tr in arts.get("trunk_rotation_charts") or []:
        add("trunk_rotation_chart", tr)
    for rk in arts.get("racket_kinematic_charts") or []:
        add("racket_kinematic_chart", rk)
    return items


def run_forehand_pipeline(video_path: Path, run_id: str, handedness: str = "right") -> Dict[str, Any]:
    from algorithm.forehand.point_track import run_point_track
    from algorithm.forehand.rtmpose_csv import run_rtmpose_csv
    from algorithm.forehand.segmentation import run_segmentation

    video_path = video_path.resolve()
    out_dir = REPO_ROOT / "data" / "outputs" / run_id / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    pt = run_point_track(video_path, out_dir)
    rt = run_rtmpose_csv(video_path, out_dir)
    _write_fused_debug(out_dir, video_path.stem, pt["csv"], rt["csv"], handedness)
    seg = run_segmentation(video_path, pt["csv"], rt["csv"], out_dir, handedness=handedness)

    return {
        "run_id": run_id,
        "video_name": video_path.stem,
        "artifacts": {
            "racket_csv": pt["csv"],
            "racket_chart": pt["chart"],
            "body_csv": rt["csv"],
            "final_chart": seg["chart"],
            "clips": seg["clips"],
            "kinetic_charts": seg.get("kinetic_charts") or [],
            "speed_cog_charts": seg.get("speed_cog_charts") or [],
            "kinematic_summary_csv": seg.get("kinematic_summary_csv", ""),
            "kinematic_phase_summary_csv": seg.get("kinematic_phase_summary_csv", ""),
            "upper_limb_charts": seg.get("upper_limb_charts") or [],
            "lower_limb_charts": seg.get("lower_limb_charts") or [],
            "trunk_rotation_charts": seg.get("trunk_rotation_charts") or [],
            "racket_kinematic_charts": seg.get("racket_kinematic_charts") or [],
        },
        "intervals": [[float(a), float(b)] for a, b in seg["intervals"]],
    }


def run_backhand_pipeline(video_path: Path, run_id: str, handedness: str = "right") -> Dict[str, Any]:
    from algorithm.backhand.point_track import run_point_track
    from algorithm.backhand.rtmpose_csv import run_rtmpose_csv
    from algorithm.backhand.segmentation import run_segmentation

    video_path = video_path.resolve()
    out_dir = REPO_ROOT / "data" / "outputs" / run_id / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    pt = run_point_track(video_path, out_dir)
    rt = run_rtmpose_csv(video_path, out_dir)
    _write_fused_debug(out_dir, video_path.stem, pt["csv"], rt["csv"], handedness)
    seg = run_segmentation(video_path, pt["csv"], rt["csv"], out_dir, handedness=handedness)

    return {
        "run_id": run_id,
        "video_name": video_path.stem,
        "artifacts": {
            "racket_csv": pt["csv"],
            "racket_chart": pt["chart"],
            "body_csv": rt["csv"],
            "final_chart": seg["chart"],
            "clips": seg["clips"],
            "kinetic_charts": seg.get("kinetic_charts") or [],
            "speed_cog_charts": seg.get("speed_cog_charts") or [],
            "kinematic_summary_csv": seg.get("kinematic_summary_csv", ""),
            "kinematic_phase_summary_csv": seg.get("kinematic_phase_summary_csv", ""),
            "upper_limb_charts": seg.get("upper_limb_charts") or [],
            "lower_limb_charts": seg.get("lower_limb_charts") or [],
            "trunk_rotation_charts": seg.get("trunk_rotation_charts") or [],
            "racket_kinematic_charts": seg.get("racket_kinematic_charts") or [],
        },
        "intervals": [[float(a), float(b)] for a, b in seg["intervals"]],
    }


def run_serve_pipeline(video_path: Path, run_id: str, handedness: str = "right") -> Dict[str, Any]:
    from algorithm.serve.point_track import run_point_track
    from algorithm.serve.rtmpose_serve_csv import run_rtmpose_csv
    from algorithm.serve.segmentation import run_segmentation

    video_path = video_path.resolve()
    out_dir = REPO_ROOT / "data" / "outputs" / run_id / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    pt = run_point_track(video_path, out_dir)
    rt = run_rtmpose_csv(video_path, out_dir)
    _write_fused_debug(out_dir, video_path.stem, pt["csv"], rt["csv"], handedness)
    seg = run_segmentation(video_path, pt["csv"], rt["csv"], out_dir, handedness=handedness)

    return {
        "run_id": run_id,
        "video_name": video_path.stem,
        "artifacts": {
            "racket_csv": pt["csv"],
            "racket_chart": pt["chart"],
            "body_csv": rt["csv"],
            "serve_trace_chart": seg["trace_chart"],
            "serve_kinetic_chart": seg["kinetic_chart"],
            "clips": seg["clips"],
            "kinematic_summary_csv": seg.get("kinematic_summary_csv", ""),
            "kinematic_phase_summary_csv": seg.get("kinematic_phase_summary_csv", ""),
            "upper_limb_charts": seg.get("upper_limb_charts") or [],
            "lower_limb_charts": seg.get("lower_limb_charts") or [],
            "trunk_rotation_charts": seg.get("trunk_rotation_charts") or [],
            "racket_kinematic_charts": seg.get("racket_kinematic_charts") or [],
        },
        "intervals": [[float(a), float(b)] for a, b in seg["intervals"]],
    }


def run_volley_pipeline(video_path: Path, run_id: str, handedness: str = "right") -> Dict[str, Any]:
    from algorithm.volley.point_track import run_point_track
    from algorithm.volley.rtmpose_volley_csv import run_rtmpose_csv
    from algorithm.volley.segmentation import run_segmentation

    video_path = video_path.resolve()
    out_dir = REPO_ROOT / "data" / "outputs" / run_id / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    pt = run_point_track(video_path, out_dir)
    rt = run_rtmpose_csv(video_path, out_dir)
    _write_fused_debug(out_dir, video_path.stem, pt["csv"], rt["csv"], handedness)
    seg = run_segmentation(video_path, pt["csv"], rt["csv"], out_dir, handedness=handedness)

    return {
        "run_id": run_id,
        "video_name": video_path.stem,
        "artifacts": {
            "racket_csv": pt["csv"],
            "racket_chart": pt["chart"],
            "body_csv": rt["csv"],
            "clips": seg["clips"],
            "volley_trace_charts": seg.get("volley_trace_charts") or [],
            "volley_kinetic_charts": seg.get("volley_kinetic_charts") or [],
            "kinematic_summary_csv": seg.get("kinematic_summary_csv", ""),
            "kinematic_phase_summary_csv": seg.get("kinematic_phase_summary_csv", ""),
            "upper_limb_charts": seg.get("upper_limb_charts") or [],
            "lower_limb_charts": seg.get("lower_limb_charts") or [],
            "trunk_rotation_charts": seg.get("trunk_rotation_charts") or [],
            "racket_kinematic_charts": seg.get("racket_kinematic_charts") or [],
        },
        "intervals": [[float(a), float(b)] for a, b in seg["intervals"]],
    }
