"""发球视频独立人体关键点与 2D 关节角 CSV 导出。"""

from algorithm.common.action_angles import run_action_angles_cli, run_action_angles_csv


def run_angles_csv(video_path, output_dir, **kwargs):
    return run_action_angles_csv("serve", video_path, output_dir, **kwargs)


run_rtmpose_csv = run_angles_csv


if __name__ == "__main__":
    run_action_angles_cli("serve")
