"""发球：修改下方三条输入路径后，在 PyCharm 直接运行本文件。"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qualisys.core.runner import run_validation_job

# ===== 修改三条输入路径和五次发球的视频时间段；三个文件必须来自同一次采集 =====
VIDEO_PATH = REPO_ROOT / "qualisys/data/input/video/fy_fq.avi"
RTMPOSE_CSV_PATH = REPO_ROOT / "qualisys/data/input/rtmpose/fy_fq_1.csv"
QUALISYS_TSV_PATH = REPO_ROOT / "qualisys/data/input/qtm/fy_fq_1.tsv"
# 依次填写五次完整发球的 (开始秒, 结束秒)，必须按时间递增且不能重叠。
# 手动时间段只用于确定动作边界；每段内的拍头速度峰仍由程序自动选择。
VIDEO_REPETITION_RANGES = (
    (15.0, 18),
    (23.0, 25),
    (29.5, 32.5),
    (36.0, 39.5),
    (42.0, 44.8),
)
# 使用上面的五段手动模式时保持 None；自动模式才填写一个总范围。
VIDEO_TIME_RANGE = None
# ===========================================================
EXPECTED_REPETITIONS = 5


if __name__ == "__main__":
    result = run_validation_job(
        action="serve",
        video_path=VIDEO_PATH,
        rtmpose_csv_path=RTMPOSE_CSV_PATH,
        qualisys_tsv_path=QUALISYS_TSV_PATH,
        video_time_range=VIDEO_TIME_RANGE,
        video_repetition_ranges=VIDEO_REPETITION_RANGES,
        expected_repetitions=EXPECTED_REPETITIONS,
    )
    print(f"完成，结果目录：{result['output_dir']}")
