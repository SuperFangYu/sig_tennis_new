"""正手视频独立人体关键点与 2D 关节角 CSV 导出。"""

from algorithm.common.action_angles import run_action_angles_cli, run_action_angles_csv


def run_angles_csv(video_path, output_dir, **kwargs):
    return run_action_angles_csv("forehand", video_path, output_dir, **kwargs)


# 保留 Python 调用层兼容名称；Web 与新代码统一使用 run_angles_csv。
run_rtmpose_csv = run_angles_csv


if __name__ == "__main__":
    run_action_angles_cli("forehand")
