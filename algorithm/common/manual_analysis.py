"""不启动前后端的四动作离线实验流水线。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from algorithm.common.thresholds import HUMAN_KPT_CONF_THRESH

PathLike = Union[str, Path]
SUPPORTED_ACTIONS = ("forehand", "backhand", "serve", "volley")


def normalize_action(action: str) -> str:
    normalized = str(action).strip().lower()
    if normalized not in SUPPORTED_ACTIONS:
        raise ValueError(f"不支持的动作类型: {action!r}，可选: {', '.join(SUPPORTED_ACTIONS)}")
    return normalized


def run_manual_analysis(
    action: str,
    video_path: PathLike,
    output_root: PathLike,
    *,
    device: str = "cuda:0",
    handedness: str = "right",
    yolo_conf: float = 0.25,
    kpt_thr: float = HUMAN_KPT_CONF_THRESH,
    pose_config_path: Optional[PathLike] = None,
    pose_checkpoint_path: Optional[PathLike] = None,
    human_yolo_model_path: Optional[PathLike] = None,
    racket_model_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """运行人体、球拍、融合与叠加视频四阶段，返回全部本地产物路径。"""
    normalized = normalize_action(action)
    video_path = Path(video_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(
            f"输入视频不存在: {video_path}\n请修改独立脚本顶部的 INPUT_VIDEO。"
        )

    output_dir = output_root / normalized / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    # 延迟导入，PyCharm 打开或单元测试导入脚本时不会提前加载模型。
    from algorithm.common.action_angles import run_action_angles_csv
    from algorithm.common.analysis_overlay import render_pose_racket_video
    from algorithm.common.kinematic_fusion import write_fused_debug_csv
    from algorithm.common.point_track_core import run_point_track

    print(f"\n[1/4] 提取 {normalized} 人体关键点与二维关节角")
    body = run_action_angles_csv(
        normalized,
        video_path,
        output_dir,
        device=device,
        yolo_conf=yolo_conf,
        kpt_thr=kpt_thr,
        config_path=pose_config_path,
        checkpoint_path=pose_checkpoint_path,
        yolo_model_path=human_yolo_model_path,
        csv_filename=f"{stem}_body_angles.csv",
    )

    print("\n[2/4] 追踪球拍 t/l/r/b/h 五个关键点")
    racket = run_point_track(
        video_path,
        output_dir,
        racket_model_path,
        save_chart=False,
        csv_filename=f"{stem}_racket_keypoints.csv",
    )

    print("\n[3/4] 按帧合并人体与球拍数据")
    combined_path = write_fused_debug_csv(
        racket["csv"],
        body["csv"],
        output_dir / f"{stem}_combined.csv",
        handedness=handedness,
    )

    print("\n[4/4] 回绘人体骨架与球拍轮廓视频")
    overlay_path = render_pose_racket_video(
        video_path,
        body["csv"],
        racket["csv"],
        output_dir / f"{stem}_pose_racket_overlay.mp4",
        action=normalized,
        body_kpt_thr=kpt_thr,
    )

    artifacts = {
        "action": normalized,
        "input_video": str(video_path),
        "output_dir": str(output_dir),
        "body_csv": str(Path(body["csv"]).resolve()),
        "racket_csv": str(Path(racket["csv"]).resolve()),
        "combined_csv": combined_path,
        "overlay_video": overlay_path,
    }
    print("\n✅ 离线分析完成：")
    for key in ("body_csv", "racket_csv", "combined_csv", "overlay_video"):
        print(f"  {key}: {artifacts[key]}")
    return artifacts


__all__ = ["SUPPORTED_ACTIONS", "normalize_action", "run_manual_analysis"]
