"""人体姿态 CSV 导出核心（RTMPose Halpe-26）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import cv2
import numpy as np
import pandas as pd

from algorithm.common.pose_features import (
    build_frame_pose_features,
    empty_frame_pose_row,
)
from algorithm.common.thresholds import HUMAN_KPT_CONF_THRESH
from algorithm.common.inference.detector import TennisDetector
from algorithm.common.inference.pose_estimator import PoseEstimator

_REPO_ROOT = Path(__file__).resolve().parents[2]

ExtraFrameFn = Callable[[Dict[str, float]], Dict[str, float]]


def run_pose_csv(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    csv_suffix: str,
    *,
    device: str = "cuda:0",
    kpt_thr: float = HUMAN_KPT_CONF_THRESH,
    config_path: Optional[Union[str, Path]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    yolo_model_path: Optional[Union[str, Path]] = None,
    extra_frame_fn: Optional[ExtraFrameFn] = None,
    yolo_conf: float = 0.25,
) -> Dict[str, Any]:
    """
    逐帧提取人体 2D kinematic features 并写入 CSV。

    csv_suffix: 输出文件名 {video}_body_{csv_suffix}.csv
    extra_frame_fn: 在基础特征上追加动作特有列（如 shoulder_turn_x_diff）
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"无法打开视频文件: {video_path}")

    cfg = Path(config_path or (_REPO_ROOT / "configs" / "rtmpose_m_halpe26.py"))
    ckpt = Path(checkpoint_path or (_REPO_ROOT / "weights" / "rtmpose_m_halpe26.pth"))
    yolo = Path(yolo_model_path or (_REPO_ROOT / "weights" / "yolov8m.pt"))

    for p, label in ((cfg, "配置文件"), (ckpt, "权重"), (yolo, "YOLO")):
        if not p.exists():
            raise FileNotFoundError(f"RTMPose {label}不存在: {p}")

    video_name = video_path.stem
    print(f"🚀 开始分析视频: [{video_name}]")
    print("加载 AI 引擎中...")

    detector = TennisDetector(model_path=str(yolo), confidence=yolo_conf)
    pose_estimator = PoseEstimator(
        config_path=str(cfg),
        checkpoint_path=str(ckpt),
        device=device,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        fps = 30.0

    data_list: list[Dict[str, float]] = []
    frame_cnt = 0
    print("🎬 引擎加载完毕，开始逐帧提取身体 2D 运动学特征...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_cnt += 1

        frame_data = empty_frame_pose_row(float(frame_cnt), float(frame_cnt / fps))

        results = detector.detect(frame)
        person_bboxes: list[np.ndarray] = []
        if results is not None and hasattr(results, "boxes") and results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            for i, cls_id in enumerate(classes):
                if int(cls_id) == 0:
                    person_bboxes.append(boxes[i])

        person_bboxes_arr = np.array(person_bboxes, dtype=np.float32)
        if len(person_bboxes_arr) > 0:
            keypoints, scores = pose_estimator.estimate(frame, person_bboxes_arr)
            if keypoints is not None and scores is not None:
                frame_data.update(
                    build_frame_pose_features(keypoints[0], scores[0], kpt_thr=kpt_thr)
                )

        if extra_frame_fn is not None:
            frame_data.update(extra_frame_fn(frame_data))

        data_list.append(frame_data)

        if frame_cnt % 30 == 0 and total_frames > 0:
            progress = (frame_cnt / total_frames) * 100
            print(f"⏳ 处理进度: {frame_cnt}/{total_frames} 帧 ({progress:.1f}%)...", end="\r")

    cap.release()

    print("\n🛠️ 正在清洗数据并保存 CSV...")
    df = pd.DataFrame(data_list)
    conf_cols = [c for c in df.columns if c.endswith("_conf")]
    interp_cols = [c for c in df.columns if c not in conf_cols]
    df[interp_cols] = df[interp_cols].interpolate(method="linear", limit_direction="both")
    df[interp_cols] = df[interp_cols].bfill().ffill()

    csv_save_path = output_dir / f"{video_name}_body_{csv_suffix}.csv"
    df.to_csv(str(csv_save_path), index=False, encoding="utf-8-sig")
    print(f"✅ 身体 CSV 已保存: {csv_save_path}")

    return {"csv": str(csv_save_path), "video_name": video_name}
