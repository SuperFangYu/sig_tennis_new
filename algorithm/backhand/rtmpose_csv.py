"""反手人体姿态 CSV 导出。"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Union
from pathlib import Path

import numpy as np

from algorithm.common.pose_csv_core import run_pose_csv
from algorithm.common.thresholds import HUMAN_KPT_CONF_THRESH

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _backhand_extra(row: Dict[str, float]) -> Dict[str, float]:
    rx = row.get("right_shoulder_x", float("nan"))
    lx = row.get("left_shoulder_x", float("nan"))
    if np.isfinite(rx) and np.isfinite(lx):
        return {"shoulder_turn_x_diff": float(rx - lx)}
    return {"shoulder_turn_x_diff": float("nan")}


def run_rtmpose_csv(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    device: str = "cuda:0",
    yolo_conf: float = 0.25,
    kpt_thr: float = HUMAN_KPT_CONF_THRESH,
    config_path: Optional[Union[str, Path]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    yolo_model_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    return run_pose_csv(
        video_path,
        output_dir,
        "backhand_rtmpose",
        device=device,
        kpt_thr=kpt_thr,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        yolo_model_path=yolo_model_path,
        yolo_conf=yolo_conf,
        extra_frame_fn=_backhand_extra,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rtmpose_csv.py <video> [output_dir]")
        sys.exit(1)
    run_rtmpose_csv(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "result_analysis")
