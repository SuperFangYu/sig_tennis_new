"""项目入口：启动 FastAPI 并打开浏览器。"""
import sys
import threading
import webbrowser
from pathlib import Path

# 须在 ultralytics / mmpose 等库之前完成初始化，避免 pandas 循环导入
import numpy  # noqa: F401
import pandas  # noqa: F401

import uvicorn

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app import app  # noqa: E402

if __name__ == "__main__":
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
