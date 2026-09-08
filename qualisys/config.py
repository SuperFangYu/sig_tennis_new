"""Qualisys–RTMPose 验证模块的稳定常量与默认点位映射。"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(__file__).resolve().parent / "data"
INPUT_ROOT = DATA_ROOT / "input"
VIDEO_INPUT_ROOT = INPUT_ROOT / "video"
RTMPOSE_INPUT_ROOT = INPUT_ROOT / "rtmpose"
QTM_INPUT_ROOT = INPUT_ROOT / "qtm"
OUTPUT_ROOT = DATA_ROOT / "output"
RACKET_MODEL_PATH = REPO_ROOT / "weights" / "bestnew.pt"

# Web 仍是四动作；这里将截击分成正手和反手，只用于效度验证分组。
SUPPORTED_ACTIONS = (
    "forehand",
    "backhand",
    "forehand_volley",
    "backhand_volley",
    "serve",
)

ACTION_LABELS = {
    "forehand": "正手",
    "backhand": "反手",
    "forehand_volley": "正手截击",
    "backhand_volley": "反手截击",
    "serve": "发球",
}

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

# 一个关节可由多个 marker 的 Z/Y 坐标均值表示。髋中心仍是当前实验的临时近似，
# 正式论文分析时应优先替换成 QTM Skeleton 求解得到的解剖学髋关节中心。
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

QTM_RACKET_HEAD_MARKER = "Racket_top"


def ensure_data_directories() -> None:
    """创建可提交说明文件之外的本地数据目录。"""
    for path in (VIDEO_INPUT_ROOT, RTMPOSE_INPUT_ROOT, QTM_INPUT_ROOT, OUTPUT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


__all__ = [
    "ACTION_LABELS",
    "ANGLE_COLUMNS",
    "DATA_ROOT",
    "DEFAULT_MARKER_MAP",
    "INPUT_ROOT",
    "OUTPUT_ROOT",
    "QTM_INPUT_ROOT",
    "QTM_RACKET_HEAD_MARKER",
    "RACKET_MODEL_PATH",
    "REPO_ROOT",
    "RTMPOSE_INPUT_ROOT",
    "SUPPORTED_ACTIONS",
    "VIDEO_INPUT_ROOT",
    "ensure_data_directories",
]
