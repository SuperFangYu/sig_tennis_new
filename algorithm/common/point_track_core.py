"""球拍五点追踪核心实现（研究二扩展）。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

from algorithm.common.racket_features import (
    RACKET_POINT_NAMES,
    add_legacy_aliases,
    compute_racket_derived_features,
    interpolate_point_columns,
    smooth_point_columns,
)
from algorithm.common.thresholds import (
    RACKET_BOX_CONF_THRESH,
    RACKET_DETECT_CONF_THRESH,
    RACKET_KPT_VALID_THRESH,
)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL = _REPO_ROOT / "weights" / "bestnew.pt"

KEYPOINT_NAMES = list(RACKET_POINT_NAMES)
MIN_AREA_RATIO = 0.0
MIN_KEYPOINTS = 0


def _empty_point_frame() -> Dict[str, float]:
    row: Dict[str, float] = {}
    for p in RACKET_POINT_NAMES:
        row[f"{p}_x_raw"] = float("nan")
        row[f"{p}_y_raw"] = float("nan")
        row[f"{p}_conf"] = float("nan")
    return row


def run_point_track(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    追踪球拍五点并输出扩展 CSV 与轨迹图（含旧字段兼容别名）。

    Returns:
        dict with keys: csv, chart, fps, video_name
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if model_path is None:
        model_path = _DEFAULT_MODEL
    else:
        model_path = Path(model_path)

    video_name = video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_pixels = max(width * height, 1)

    rows: List[Dict[str, float]] = []
    print(f"🚀 开始处理视频 [{video_name}]，追踪球拍五点 t/l/r/b/h")

    frame_cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_cnt += 1

        results = model(frame, verbose=False, conf=RACKET_DETECT_CONF_THRESH)
        candidates: List[Dict[str, Any]] = []

        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            kpts = r.keypoints.data.cpu().numpy() if r.keypoints is not None else []

            for box, conf, kpt_set in zip(boxes, confs, kpts):
                if conf < RACKET_BOX_CONF_THRESH:
                    continue
                x1, y1, x2, y2 = box
                area_ratio = ((x2 - x1) * (y2 - y1)) / total_pixels
                if area_ratio < MIN_AREA_RATIO:
                    continue
                valid_kpts_count = sum(1 for _, _, kconf in kpt_set if kconf > 0)
                if valid_kpts_count < MIN_KEYPOINTS:
                    continue
                candidates.append({"conf": float(conf), "kpts": kpt_set})

        row: Dict[str, float] = {
            "frame": float(frame_cnt),
            "time": float(frame_cnt / fps),
        }
        row.update(_empty_point_frame())

        if candidates:
            best = max(candidates, key=lambda x: x["conf"])
            kpt_set = best["kpts"]
            for idx, pname in enumerate(KEYPOINT_NAMES):
                if idx >= len(kpt_set):
                    continue
                kx, ky, kconf = kpt_set[idx]
                kconf = float(kconf)
                row[f"{pname}_conf"] = kconf
                if kconf >= RACKET_KPT_VALID_THRESH:
                    row[f"{pname}_x_raw"] = float(kx)
                    row[f"{pname}_y_raw"] = float(ky)

        rows.append(row)
        if frame_cnt % 50 == 0:
            print(f"⏳ 已处理: {frame_cnt} 帧...", end="\r")

    cap.release()
    print("\n🛠️ 正在进行五点插值、平滑与 2D 运动学特征计算...")

    df = pd.DataFrame(rows)
    valid_masks: Dict[str, np.ndarray] = {}
    for p in RACKET_POINT_NAMES:
        conf_col = f"{p}_conf"
        valid_masks[p] = df[conf_col].to_numpy(dtype=np.float64) >= RACKET_KPT_VALID_THRESH
        interpolate_point_columns(df, p)
        smooth_point_columns(df, p, fps, valid_masks[p])

    add_legacy_aliases(df)
    compute_racket_derived_features(df, fps)

    csv_save_path = str(output_dir / f"{video_name}_kpt_t_backend.csv")
    df.to_csv(csv_save_path, index=False)
    print(f"💾 球拍五点 CSV 已保存: {csv_save_path}")

    chart_save_path = str(output_dir / f"{video_name}_kpt_t_plot.png")
    _save_trajectory_chart(df, video_name, chart_save_path)
    print(f"📈 轨迹图已保存: {chart_save_path}")

    return {
        "csv": csv_save_path,
        "chart": chart_save_path,
        "fps": float(fps),
        "video_name": video_name,
    }


def _save_trajectory_chart(df: pd.DataFrame, video_name: str, path: str) -> None:
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.scatter(df["time"], df["t_x_raw"], color="black", s=12, zorder=3, label="拍头 raw")
    plt.plot(df["time"], df["t_x_interp"], color="lightblue", linestyle="--", linewidth=1, label="插值")
    plt.plot(df["time"], df["t_x_clean"], color="blue", linewidth=2, label="拍头 clean X")
    plt.title(f"视频 [{video_name}] - 球拍五点追踪（拍头 t）", fontsize=14)
    plt.ylabel("X")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.scatter(df["time"], df["t_y_raw"], color="black", s=12, zorder=3, label="拍头 raw")
    plt.plot(df["time"], df["t_y_interp"], color="lightpink", linestyle="--", linewidth=1, label="插值")
    plt.plot(df["time"], df["t_y_clean"], color="red", linewidth=2, label="拍头 clean Y")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m algorithm.common.point_track_core <video> <output_dir> [model]")
        sys.exit(1)
    mp = sys.argv[3] if len(sys.argv) > 3 else None
    run_point_track(sys.argv[1], sys.argv[2], mp)
