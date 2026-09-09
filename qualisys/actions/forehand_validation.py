"""正手：修改下方三条输入路径后，在 PyCharm 直接运行本文件。"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qualisys.core.runner import run_validation_job

# ===== 修改三条输入路径和一个视频有效范围；三个文件必须来自同一次采集 =====
VIDEO_PATH = REPO_ROOT / "qualisys/data/input/video/fy_zs_1.avi"
RTMPOSE_CSV_PATH = REPO_ROOT / "qualisys/data/input/rtmpose/fy_zs_1.csv"
QUALISYS_TSV_PATH = REPO_ROOT / "qualisys/data/input/qtm/fy_zs_1.tsv"
# 原视频中只包含五次正式动作的范围；例如 (12.0, 22.2)。不限制时填 None。
VIDEO_TIME_RANGE = None
# ===========================================================
EXPECTED_REPETITIONS = 5


if __name__ == "__main__":
    result = run_validation_job(
        action="forehand",
        video_path=VIDEO_PATH,
        rtmpose_csv_path=RTMPOSE_CSV_PATH,
        qualisys_tsv_path=QUALISYS_TSV_PATH,
        video_time_range=VIDEO_TIME_RANGE,
        expected_repetitions=EXPECTED_REPETITIONS,
    )
    print(f"完成，结果目录：{result['output_dir']}")
