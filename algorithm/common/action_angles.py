"""四类网球动作的人体 2D 角度 CSV 统一入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from algorithm.common.thresholds import HUMAN_KPT_CONF_THRESH

PathLike = Union[str, Path]

_CSV_SUFFIXES = {
    "forehand": "forehand_rtmpose",
    "backhand": "backhand_rtmpose",
    "serve": "serve_rtmpose",
    "volley": "volley_rtmpose",
}


def _backhand_extra(row: Dict[str, float]) -> Dict[str, float]:
    """保留反手切分依赖的肩部水平差特征。"""
    right_x = row.get("right_shoulder_x", float("nan"))
    left_x = row.get("left_shoulder_x", float("nan"))
    if np.isfinite(right_x) and np.isfinite(left_x):
        return {"shoulder_turn_x_diff": float(right_x - left_x)}
    return {"shoulder_turn_x_diff": float("nan")}


def run_action_angles_csv(
    action: str,
    video_path: PathLike,
    output_dir: PathLike,
    *,
    device: str = "cuda:0",
    yolo_conf: float = 0.25,
    kpt_thr: float = HUMAN_KPT_CONF_THRESH,
    config_path: Optional[PathLike] = None,
    checkpoint_path: Optional[PathLike] = None,
    yolo_model_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """按动作类型导出与 Web 流水线完全相同的人体关键点和角度 CSV。"""
    normalized = str(action).strip().lower()
    if normalized not in _CSV_SUFFIXES:
        supported = ", ".join(sorted(_CSV_SUFFIXES))
        raise ValueError(f"不支持的动作类型: {action!r}，可选: {supported}")

    # 延迟导入，查询 --help 或启动 Web 时不初始化 Ultralytics/MMPose。
    from algorithm.common.pose_csv_core import run_pose_csv

    return run_pose_csv(
        video_path,
        output_dir,
        _CSV_SUFFIXES[normalized],
        device=device,
        kpt_thr=kpt_thr,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        yolo_model_path=yolo_model_path,
        yolo_conf=yolo_conf,
        extra_frame_fn=_backhand_extra if normalized == "backhand" else None,
    )


def run_action_angles_cli(action: str) -> None:
    """四个动作独立脚本共用的命令行参数。"""
    parser = argparse.ArgumentParser(
        description=f"导出 {action} 动作的 RTMPose 人体关键点与 2D 关节角 CSV（不画图）。"
    )
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--output-dir", default="result_analysis", help="CSV 输出目录")
    parser.add_argument("--device", default="cuda:0", help="推理设备，例如 cuda:0 或 cpu")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="人体检测置信度阈值")
    parser.add_argument(
        "--kpt-thr",
        type=float,
        default=HUMAN_KPT_CONF_THRESH,
        help="人体关键点置信度阈值",
    )
    parser.add_argument("--config", dest="config_path", help="可选 RTMPose 配置文件")
    parser.add_argument("--checkpoint", dest="checkpoint_path", help="可选 RTMPose 权重")
    parser.add_argument("--yolo-model", dest="yolo_model_path", help="可选人体检测权重")
    args = parser.parse_args()

    result = run_action_angles_csv(
        action,
        args.video,
        args.output_dir,
        device=args.device,
        yolo_conf=args.yolo_conf,
        kpt_thr=args.kpt_thr,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        yolo_model_path=args.yolo_model_path,
    )
    print(result["csv"])
