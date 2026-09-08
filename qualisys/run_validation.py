"""按阶段运行同一试次的 RTMPose、Qualisys 与对比流程。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qualisys.compare_angles import compare_trial, write_event_templates
from qualisys.config import DEFAULT_MANIFEST
from qualisys.run_qualisys import run_qualisys_trial
from qualisys.run_rtmpose import run_rtmpose_trial
from qualisys.trials import load_trials, select_trials

STAGES = ("rtmpose", "qualisys", "compare")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 RTMPose–Qualisys 验证流水线。")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--trial", action="append", dest="trial_ids", help="只处理指定 trial_id，可重复")
    parser.add_argument("--stage", action="append", choices=STAGES, dest="stages", help="指定阶段，可重复")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--prepare-events",
        action="store_true",
        help="仅创建事件模板；适合两套八角 CSV 已生成后的第一次操作",
    )
    args = parser.parse_args()

    trials = select_trials(load_trials(args.manifest, validate_files=False), args.trial_ids)
    if args.prepare_events:
        for trial in trials:
            write_event_templates(trial)
            print(f"[{trial.trial_id}] 已准备人工事件模板")
        return

    stages = tuple(args.stages or STAGES)
    for trial in trials:
        print(f"\n=== {trial.trial_id} ({trial.action}) ===")
        if "rtmpose" in stages:
            run_rtmpose_trial(trial, device=args.device)
        if "qualisys" in stages:
            run_qualisys_trial(trial)
        if "compare" in stages:
            compare_trial(trial)
        print(f"[{trial.trial_id}] 已完成阶段: {', '.join(stages)}")


if __name__ == "__main__":
    main()
