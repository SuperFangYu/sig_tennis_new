"""PyCharm 总入口：依次把 RUN_MODE 设为 extract、prepare_events、compare。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qualisys.compare_angles import compare_trial, write_event_templates
from qualisys.config import ACTION_LABELS
from qualisys.run_qualisys import run_qualisys_trial
from qualisys.run_rtmpose import run_rtmpose_trial
from qualisys.trials import load_trials, select_trials

# ===== PyCharm 直接运行时只需要修改这里 =====
# extract：生成两套八角；prepare_events：生成待填写的时间表；compare：完成对齐与指标。
RUN_MODE = "extract"
TRIAL_ID = ""  # 留空处理所有同名配对；例如只处理 fy_zs_1 时填 "fy_zs_1"
DEVICE = "cuda:0"
# ========================================


def main() -> None:
    parser = argparse.ArgumentParser(description="三步运行 RTMPose–Qualisys 验证。")
    parser.add_argument(
        "--mode",
        choices=("extract", "prepare_events", "compare"),
        default=RUN_MODE,
        help="extract=提取角度，prepare_events=建立时间表，compare=对齐比较",
    )
    parser.add_argument("--trial", default=TRIAL_ID, help="试次主文件名；留空处理全部配对")
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    requested = [args.trial] if args.trial else None
    trials = select_trials(load_trials(validate_files=False), requested)
    if args.mode == "prepare_events":
        for trial in trials:
            write_event_templates(trial)
            print(f"[{trial.trial_id}] 已准备人工事件模板")
        return

    for trial in trials:
        print(f"\n=== {trial.trial_id}（{ACTION_LABELS[trial.action]}）===")
        if args.mode == "extract":
            run_rtmpose_trial(trial, device=args.device)
            run_qualisys_trial(trial)
        elif args.mode == "compare":
            compare_trial(trial)
        print(f"[{trial.trial_id}] 已完成: {args.mode}")


if __name__ == "__main__":
    main()
