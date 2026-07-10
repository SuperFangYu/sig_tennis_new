"""发球人体姿态 CSV 导出。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Union

from algorithm.common.pose_csv_core import run_pose_csv
from algorithm.common.thresholds import HUMAN_KPT_CONF_THRESH

_REPO_ROOT = Path(__file__).resolve().parents[2]


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
    """含 left_wrist_y / right_wrist_y / shoulder_tilt_y / body_center_x（统一基线）。"""
    return run_pose_csv(
        video_path,
        output_dir,
        "serve_rtmpose",
        device=device,
        kpt_thr=kpt_thr,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        yolo_model_path=yolo_model_path,
        yolo_conf=yolo_conf,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="result_analysis")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    run_rtmpose_csv(args.video, args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
