# 兼容旧脚本路径：实现已迁移至 rtmpose_serve_csv.py
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from algorithm.serve.rtmpose_serve_csv import main, run_rtmpose_csv  # noqa: E402,F401

if __name__ == "__main__":
    main()
