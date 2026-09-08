"""为清单中的成对视频独立生成 RTMPose 八角 CSV。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from algorithm.common.action_angles import run_action_angles_csv
from algorithm.common.pose_features import EIGHT_ANGLE_COLUMNS
from qualisys.config import DEFAULT_MANIFEST
from qualisys.trials import Trial, load_trials, select_trials


def run_rtmpose_trial(trial: Trial, *, device: str = "cuda:0") -> dict[str, Any]:
    """运行单个视频，输出 `intermediate/{trial}/rtmpose_8_angles.csv`。"""
    trial.validate_video()
    trial.ensure_output_dirs()
    result = run_action_angles_csv(
        trial.action,
        trial.video_path,
        trial.intermediate_dir,
        device=device,
        csv_filename="rtmpose_8_angles.csv",
        csv_columns=("frame", "time", *EIGHT_ANGLE_COLUMNS),
    )

    # 统一把首帧时刻归零，消除现有视觉导出从 1/fps 起计造成的固定偏移。
    csv_path = Path(result["csv"])
    frame = pd.read_csv(csv_path)
    if not frame.empty:
        frame["time"] = frame["time"] - float(frame["time"].iloc[0])
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return {**result, "trial_id": trial.trial_id, "csv": str(csv_path.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description="按试次清单生成 RTMPose 八角 CSV。")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--trial", action="append", dest="trial_ids", help="只处理指定 trial_id，可重复")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    trials = select_trials(load_trials(args.manifest, validate_files=False), args.trial_ids)
    for trial in trials:
        result = run_rtmpose_trial(trial, device=args.device)
        print(f"[{trial.trial_id}] {result['csv']}")


if __name__ == "__main__":
    main()
