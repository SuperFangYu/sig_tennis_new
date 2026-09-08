"""正手截击：修改下方参数后，在 PyCharm 中直接运行本文件。"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithm.common.manual_analysis import run_manual_analysis

# ===== 通常只需要修改这里 =====
INPUT_VIDEO = REPO_ROOT / "data" / "manual" / "input" / "forehand_volley.mp4"
OUTPUT_ROOT = REPO_ROOT / "data" / "manual" / "output"
DEVICE = "cuda:0"
HANDEDNESS = "right"  # 右手持拍填 right，左手持拍填 left
# ============================


if __name__ == "__main__":
    run_manual_analysis(
        "forehand_volley",
        INPUT_VIDEO,
        OUTPUT_ROOT,
        device=DEVICE,
        handedness=HANDEDNESS,
    )
