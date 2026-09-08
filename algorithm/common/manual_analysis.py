"""不启动前后端的五动作离线实验流水线；正/反手截击复用 volley 底层算法。"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, Dict, Optional, Union

from algorithm.common.pose_features import EIGHT_ANGLE_COLUMNS
from algorithm.common.thresholds import HUMAN_KPT_CONF_THRESH

PathLike = Union[str, Path]
SUPPORTED_ACTIONS = (
    "forehand",
    "backhand",
    "forehand_volley",
    "backhand_volley",
    "serve",
)
_PIPELINE_ACTION = {
    "forehand": "forehand",
    "backhand": "backhand",
    "forehand_volley": "volley",
    "backhand_volley": "volley",
    "serve": "serve",
}


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
    """运行人体角度、球拍追踪与叠加视频三阶段，返回两个正式产物路径。"""
    normalized = normalize_action(action)
    pipeline_action = _PIPELINE_ACTION[normalized]
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
    from algorithm.common.point_track_core import run_point_track

    print(f"\n[1/3] 提取 {normalized} 人体关键点与二维关节角")
    body = run_action_angles_csv(
        pipeline_action,
        video_path,
        output_dir,
        device=device,
        yolo_conf=yolo_conf,
        kpt_thr=kpt_thr,
        config_path=pose_config_path,
        checkpoint_path=pose_checkpoint_path,
        yolo_model_path=human_yolo_model_path,
        csv_filename=f"{stem}_body_angles.csv",
        csv_columns=("frame", "time", *EIGHT_ANGLE_COLUMNS),
        collect_halpe26=True,
    )

    # 球拍完整数据只为本次视频绘制服务，临时 CSV 会在绘制结束后自动清理。
    with tempfile.TemporaryDirectory(prefix=".racket_visual_", dir=str(output_dir)) as temp_dir:
        print("\n[2/3] 追踪球拍 t/l/r/b/h 五个关键点")
        racket = run_point_track(
            video_path,
            temp_dir,
            racket_model_path,
            save_chart=False,
            csv_filename=f"{stem}_racket_keypoints.csv",
        )

        print("\n[3/3] 回绘 Halpe26 简化面部骨架与球拍轮廓视频")
        overlay_path = render_pose_racket_video(
            video_path,
            body["csv"],
            racket["csv"],
            output_dir / f"{stem}_pose_racket_overlay.mp4",
            action=normalized,
            body_kpt_thr=kpt_thr,
            pose_frames=body["pose_frames"],
        )

    # 清理由上一版离线入口生成的两个固定名称调试 CSV，避免与“仅一个角度 CSV”混淆。
    for obsolete_name in (
        f"{stem}_racket_keypoints.csv",
        f"{stem}_combined.csv",
    ):
        obsolete_path = output_dir / obsolete_name
        if obsolete_path.is_file():
            obsolete_path.unlink()

    artifacts = {
        "action": normalized,
        "input_video": str(video_path),
        "output_dir": str(output_dir),
        "body_csv": str(Path(body["csv"]).resolve()),
        "overlay_video": overlay_path,
    }
    print("\n✅ 离线分析完成：")
    for key in ("body_csv", "overlay_video"):
        print(f"  {key}: {artifacts[key]}")
    return artifacts


__all__ = ["SUPPORTED_ACTIONS", "normalize_action", "run_manual_analysis"]
