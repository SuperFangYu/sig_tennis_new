"""Qualisys 验证流水线的路径、列名与默认点位映射。"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(__file__).resolve().parent / "data"
INPUT_ROOT = DATA_ROOT / "input"
INTERMEDIATE_ROOT = DATA_ROOT / "intermediate"
OUTPUT_ROOT = DATA_ROOT / "output"
DEFAULT_MANIFEST = INPUT_ROOT / "manifest.csv"

SUPPORTED_ACTIONS = ("forehand", "backhand", "serve", "volley")
SUPPORTED_PLANES = ("ZY",)

ANGLE_COLUMNS = (
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_knee_angle",
    "right_knee_angle",
)

# 每个关节可以由一个或多个反光点求均值。髋中心只是当前实验的临时近似，正式分析前
# 应优先替换为 QTM Skeleton 求解后的解剖学髋关节中心。
DEFAULT_MARKER_MAP = {
    "left_shoulder": ("L.shoulder",),
    "right_shoulder": ("R.shoulder",),
    "left_elbow": ("L.elbow.lat", "L.elbow.med"),
    "right_elbow": ("R.elbow.lat", "R.elbow.med"),
    "left_wrist": ("L.wrist.lat", "L.wrist.med"),
    "right_wrist": ("R.wrist.lat", "R.wrist.med"),
    "left_hip": ("L.ASIS", "L.PSIS"),
    "right_hip": ("R.ASIS", "R.PSIS"),
    "left_knee": ("L.knee.lat", "L.knee.med"),
    "right_knee": ("R.knee.lat", "R.knee.med"),
    "left_ankle": ("L.ankle.lat", "L.ankle.med"),
    "right_ankle": ("R.ankle.lat", "R.ankle.med"),
}

EVENT_NAMES = ("start", "contact", "end")


def ensure_data_directories() -> None:
    """创建本地输入、中间结果和最终结果目录。"""
    for path in (INPUT_ROOT, INTERMEDIATE_ROOT, OUTPUT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "ANGLE_COLUMNS",
    "DATA_ROOT",
    "DEFAULT_MANIFEST",
    "DEFAULT_MARKER_MAP",
    "EVENT_NAMES",
    "INPUT_ROOT",
    "INTERMEDIATE_ROOT",
    "OUTPUT_ROOT",
    "REPO_ROOT",
    "SUPPORTED_ACTIONS",
    "SUPPORTED_PLANES",
    "ensure_data_directories",
]
