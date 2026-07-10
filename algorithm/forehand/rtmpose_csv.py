"""正手人体姿态 CSV 导出。"""

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
    return run_pose_csv(
        video_path,
        output_dir,
        "forehand_rtmpose",
        device=device,
        kpt_thr=kpt_thr,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        yolo_model_path=yolo_model_path,
        yolo_conf=yolo_conf,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract RTMPose features and export to CSV.")
    parser.add_argument("--video", type=str, required=False)
    parser.add_argument("--output-dir", type=str, default="result_analysis")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--kpt-thr", type=float, default=HUMAN_KPT_CONF_THRESH)
    args = parser.parse_args()
    if not args.video:
        parser.error("--video is required when running as __main__")
    run_rtmpose_csv(
        args.video,
        args.output_dir,
        device=args.device,
        yolo_conf=args.yolo_conf,
        kpt_thr=args.kpt_thr,
    )


if __name__ == "__main__":
    main()
