"""
rtmpose-csv.py

逐帧读取视频 -> YOLO 检测人体(取最大/过滤 person) -> RTMPose 估计 Halpe-26 关键点 ->
计算若干运动学特征 -> 导出 CSV
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd

# 允许脚本直接运行时也能找到 `algorithm` 包（与 backend/main.py 的做法一致）
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.processors.detector import TennisDetector  # noqa: E402
from algorithm.processors.pose_estimator import PoseEstimator  # noqa: E402


# ================================================================
# 1. Halpe-26 格式的关键点索引字典（与 algorithm/utils/pose_metrics.py 一致）
# ================================================================
HALPE_IDX = {
    "l_shoulder": 5,
    "r_shoulder": 6,
    "l_elbow": 7,
    "r_elbow": 8,
    "l_wrist": 9,
    "r_wrist": 10,
    "l_hip": 11,
    "r_hip": 12,
    "l_knee": 13,
    "r_knee": 14,
    "l_ankle": 15,
    "r_ankle": 16,
}


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    计算三个点之间的夹角（以 b 为顶点），返回角度（deg）。
    任一向量为 NaN 或长度为 0 时返回 NaN。
    """
    if (
        a is None
        or b is None
        or c is None
        or np.any(np.isnan(a))
        or np.any(np.isnan(b))
        or np.any(np.isnan(c))
    ):
        return float("nan")

    ba = np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)
    bc = np.array(c, dtype=np.float64) - np.array(b, dtype=np.float64)

    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))
    if norm_ba < 1e-9 or norm_bc < 1e-9:
        return float("nan")

    cosine_angle = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cosine_angle = float(np.clip(cosine_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine_angle)))


def safe_mean(xs: list[float]) -> float:
    xs_arr = np.array(xs, dtype=np.float64)
    if xs_arr.size == 0 or np.any(np.isnan(xs_arr)):
        return float("nan")
    return float(xs_arr.mean())


def main():
    parser = argparse.ArgumentParser(description="Extract RTMPose features and export to CSV.")
    parser.add_argument(
        "--video",
        type=str,
        required=False,
        default=r"D:\FY\mediapipe_demo\视频\个人\dark1.mp4",
        help="输入视频路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default="result_analysis",
        help="输出目录（相对根目录）",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="推理设备：cuda:0 或 cpu")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO 置信度阈值（内部固定也可用默认）")
    parser.add_argument("--kpt-thr", type=float, default=0.3, help="关键点置信度阈值")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"无法打开视频文件: {video_path}")

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    video_name = video_path.stem
    print(f"🚀 开始分析视频: [{video_name}]")
    print("加载 AI 引擎中...")

    # ================================================================
    # 2. 显式定位项目内的配置与权重文件
    # ================================================================
    config_path = PROJECT_ROOT / "algorithm" / "configs" / "rtmpose_m_halpe26.py"
    checkpoint_path = PROJECT_ROOT / "algorithm" / "weights" / "rtmpose_m_halpe26.pth"
    yolo_model_path = PROJECT_ROOT / "algorithm" / "weights" / "yolov8m.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"RTMPose 配置文件不存在: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"RTMPose 权重文件不存在: {checkpoint_path}")
    if not yolo_model_path.exists():
        raise FileNotFoundError(f"YOLO 权重文件不存在: {yolo_model_path}")

    detector = TennisDetector(model_path=str(yolo_model_path))
    pose_estimator = PoseEstimator(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device=args.device,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0:
        fps = 30.0  # 容错：有些编码器会读不到 FPS

    data_list: list[Dict[str, float]] = []
    frame_cnt = 0

    print("🎬 引擎加载完毕，开始逐帧提取身体运动学特征...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_cnt += 1

        frame_data: Dict[str, float] = {
            "frame": float(frame_cnt),
            "time": float(frame_cnt / fps),
            "body_center_x": float("nan"),
            "body_center_y": float("nan"),
            "right_elbow_angle": float("nan"),
            "right_knee_angle": float("nan"),
            "shoulder_hip_angle": float("nan"),
        }

        # ---------------------------------------------------------
        # A. 目标检测 (找人：YOLO person=class 0)
        # ---------------------------------------------------------
        results = detector.detect(frame)
        person_bboxes = []

        if results is not None and hasattr(results, "boxes") and results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # (N, 4)
            classes = results.boxes.cls.cpu().numpy()  # (N,)

            for i, cls_id in enumerate(classes):
                if int(cls_id) == 0:
                    person_bboxes.append(boxes[i])

        person_bboxes = np.array(person_bboxes, dtype=np.float32)

        # ---------------------------------------------------------
        # B. 姿态估计与特征计算
        # ---------------------------------------------------------
        if len(person_bboxes) > 0:
            keypoints, scores = pose_estimator.estimate(frame, person_bboxes)
            if keypoints is not None and scores is not None:
                # PoseEstimator 会在内部选择面积最大的人；这里按你的模板取第 0 个实例
                kpts = keypoints[0]  # (26, 2)
                sc = scores[0]  # (26,)

                def kpt(i: int) -> np.ndarray:
                    return kpts[i]

                def kpt_valid(i: int) -> bool:
                    return float(sc[i]) >= args.kpt_thr

                # 关键点读取：不满足阈值则置为 NaN
                r_shoulder = kpt(HALPE_IDX["r_shoulder"]) if kpt_valid(HALPE_IDX["r_shoulder"]) else np.array([np.nan, np.nan])
                l_shoulder = kpt(HALPE_IDX["l_shoulder"]) if kpt_valid(HALPE_IDX["l_shoulder"]) else np.array([np.nan, np.nan])
                r_elbow = kpt(HALPE_IDX["r_elbow"]) if kpt_valid(HALPE_IDX["r_elbow"]) else np.array([np.nan, np.nan])
                r_wrist = kpt(HALPE_IDX["r_wrist"]) if kpt_valid(HALPE_IDX["r_wrist"]) else np.array([np.nan, np.nan])
                r_hip = kpt(HALPE_IDX["r_hip"]) if kpt_valid(HALPE_IDX["r_hip"]) else np.array([np.nan, np.nan])
                l_hip = kpt(HALPE_IDX["l_hip"]) if kpt_valid(HALPE_IDX["l_hip"]) else np.array([np.nan, np.nan])
                r_knee = kpt(HALPE_IDX["r_knee"]) if kpt_valid(HALPE_IDX["r_knee"]) else np.array([np.nan, np.nan])
                r_ankle = kpt(HALPE_IDX["r_ankle"]) if kpt_valid(HALPE_IDX["r_ankle"]) else np.array([np.nan, np.nan])

                # 特征 1：身体中线 X（通过双肩中心）
                if kpt_valid(HALPE_IDX["r_shoulder"]) and kpt_valid(HALPE_IDX["l_shoulder"]):
                    frame_data["body_center_x"] = float((r_shoulder[0] + l_shoulder[0]) / 2.0)

                if kpt_valid(HALPE_IDX["r_hip"]) and kpt_valid(HALPE_IDX["l_hip"]):
                    frame_data["body_center_y"] = float((float(r_hip[1]) + float(l_hip[1])) / 2.0)

                # 特征 2：右肘角度（发力特征）
                frame_data["right_elbow_angle"] = calculate_angle(r_shoulder, r_elbow, r_wrist)

                # 特征 3：右膝角度（发球蓄力特征）
                frame_data["right_knee_angle"] = calculate_angle(r_hip, r_knee, r_ankle)

                # 特征 4：肩髋向量夹角（躯干扭转蓄力特征）
                if (
                    kpt_valid(HALPE_IDX["r_shoulder"])
                    and kpt_valid(HALPE_IDX["l_shoulder"])
                    and kpt_valid(HALPE_IDX["r_hip"])
                    and kpt_valid(HALPE_IDX["l_hip"])
                ):
                    shoulder_vec = np.array(r_shoulder, dtype=np.float64) - np.array(l_shoulder, dtype=np.float64)
                    hip_vec = np.array(r_hip, dtype=np.float64) - np.array(l_hip, dtype=np.float64)
                    norm_sh = float(np.linalg.norm(shoulder_vec))
                    norm_hip = float(np.linalg.norm(hip_vec))
                    if norm_sh >= 1e-9 and norm_hip >= 1e-9:
                        cos_sh = float(np.dot(shoulder_vec, hip_vec) / (norm_sh * norm_hip))
                        cos_sh = float(np.clip(cos_sh, -1.0, 1.0))
                        frame_data["shoulder_hip_angle"] = float(np.degrees(np.arccos(cos_sh)))

        data_list.append(frame_data)

        if frame_cnt % 30 == 0 and total_frames > 0:
            progress = (frame_cnt / total_frames) * 100
            print(f"⏳ 处理进度: {frame_cnt}/{total_frames} 帧 ({progress:.1f}%)...", end="\r")

    cap.release()

    # ---------------------------------------------------------
    # C. 数据清洗与保存
    # ---------------------------------------------------------
    print("\n🛠️ 正在清洗数据并保存 CSV...")
    df = pd.DataFrame(data_list)

    # 用线性插值填补未检测到的空白帧（角度曲线更平滑）
    df = df.interpolate(method="linear").bfill().ffill()

    csv_save_path = output_dir / f"{video_name}_body_forehand_rtmpose.csv"
    df.to_csv(str(csv_save_path), index=False, encoding="utf-8-sig")

    print(f"✅ 完美收工！身体高维运动学特征已保存至: {csv_save_path}")


if __name__ == "__main__":
    main()

