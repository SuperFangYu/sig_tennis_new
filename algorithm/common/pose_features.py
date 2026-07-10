"""人体 Halpe-26 核心关键点与 2D kinematic feature 计算。"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from algorithm.common.thresholds import HUMAN_KPT_CONF_THRESH

HALPE_IDX = {
    "l_shoulder": 5,
    "r_shoulder": 6,
    "l_elbow": 7,
    "r_elbow": 8,
    "l_wrist": 9,
    "r_wrist": 10,
    "l_hip": 11,
    "r_hip": 12,
    "l_knee": 13,
    "r_knee": 14,
    "l_ankle": 15,
    "r_ankle": 16,
}

# CSV 列前缀（left/right）
JOINT_CSV_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

_HALPE_TO_CSV = {
    "l_shoulder": "left_shoulder",
    "r_shoulder": "right_shoulder",
    "l_elbow": "left_elbow",
    "r_elbow": "right_elbow",
    "l_wrist": "left_wrist",
    "r_wrist": "right_wrist",
    "l_hip": "left_hip",
    "r_hip": "right_hip",
    "l_knee": "left_knee",
    "r_knee": "right_knee",
    "l_ankle": "left_ankle",
    "r_ankle": "right_ankle",
}


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """三点夹角（顶点 b），单位度；任一点无效返回 NaN。"""
    if a is None or b is None or c is None:
        return float("nan")
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return float("nan")
    ba = a - b
    bc = c - b
    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))
    if norm_ba < 1e-9 or norm_bc < 1e-9:
        return float("nan")
    cos_angle = float(np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _nan_point() -> np.ndarray:
    return np.array([np.nan, np.nan], dtype=np.float64)


def extract_joint_points(
    kpts: np.ndarray,
    scores: np.ndarray,
    kpt_thr: float = HUMAN_KPT_CONF_THRESH,
) -> Dict[str, np.ndarray]:
    """提取 12 核心关节 xy；低于阈值则为 NaN。"""
    out: Dict[str, np.ndarray] = {}
    for halpe_name, csv_name in _HALPE_TO_CSV.items():
        idx = HALPE_IDX[halpe_name]
        if float(scores[idx]) >= kpt_thr:
            out[csv_name] = np.asarray(kpts[idx][:2], dtype=np.float64)
        else:
            out[csv_name] = _nan_point()
    return out


def joint_coord_columns(
    joints: Dict[str, np.ndarray],
    scores: np.ndarray,
    kpt_thr: float = HUMAN_KPT_CONF_THRESH,
) -> Dict[str, float]:
    """导出 12 关节 x/y/conf 列。"""
    cols: Dict[str, float] = {}
    for halpe_name, csv_name in _HALPE_TO_CSV.items():
        idx = HALPE_IDX[halpe_name]
        pt = joints[csv_name]
        conf = float(scores[idx])
        cols[f"{csv_name}_x"] = float(pt[0]) if conf >= kpt_thr and np.isfinite(pt[0]) else float("nan")
        cols[f"{csv_name}_y"] = float(pt[1]) if conf >= kpt_thr and np.isfinite(pt[1]) else float("nan")
        cols[f"{csv_name}_conf"] = conf if np.isfinite(conf) else float("nan")
    return cols


def compute_nine_angles(joints: Dict[str, np.ndarray]) -> Dict[str, float]:
    """9 个核心 2D 关节角 + 肩髋分离角。"""
    ls = joints["left_shoulder"]
    rs = joints["right_shoulder"]
    le = joints["left_elbow"]
    re = joints["right_elbow"]
    lw = joints["left_wrist"]
    rw = joints["right_wrist"]
    lh = joints["left_hip"]
    rh = joints["right_hip"]
    lk = joints["left_knee"]
    rk = joints["right_knee"]
    la = joints["left_ankle"]
    ra = joints["right_ankle"]

    angles = {
        "left_shoulder_angle": calculate_angle(le, ls, lh),
        "right_shoulder_angle": calculate_angle(re, rs, rh),
        "left_elbow_angle": calculate_angle(ls, le, lw),
        "right_elbow_angle": calculate_angle(rs, re, rw),
        "left_hip_angle": calculate_angle(ls, lh, lk),
        "right_hip_angle": calculate_angle(rs, rh, rk),
        "left_knee_angle": calculate_angle(lh, lk, la),
        "right_knee_angle": calculate_angle(rh, rk, ra),
        "shoulder_hip_angle": float("nan"),
    }

    if not (
        (np.any(np.isnan(ls)) or np.any(np.isnan(rs)) or np.any(np.isnan(lh)) or np.any(np.isnan(rh)))
    ):
        shoulder_vec = rs - ls
        hip_vec = rh - lh
        norm_sh = float(np.linalg.norm(shoulder_vec))
        norm_hip = float(np.linalg.norm(hip_vec))
        if norm_sh >= 1e-9 and norm_hip >= 1e-9:
            cos_sh = float(np.clip(np.dot(shoulder_vec, hip_vec) / (norm_sh * norm_hip), -1.0, 1.0))
            angles["shoulder_hip_angle"] = float(np.degrees(np.arccos(cos_sh)))

    return angles


def compute_position_features(joints: Dict[str, np.ndarray]) -> Dict[str, float]:
    """肩/髋中心、兼容 body_center、倾斜与躯干倾角（2D kinematic proxy）。"""
    ls = joints["left_shoulder"]
    rs = joints["right_shoulder"]
    lh = joints["left_hip"]
    rh = joints["right_hip"]

    pos: Dict[str, float] = {
        "shoulder_center_x": float("nan"),
        "shoulder_center_y": float("nan"),
        "hip_center_x": float("nan"),
        "hip_center_y": float("nan"),
        "body_center_x": float("nan"),
        "body_center_y": float("nan"),
        "shoulder_tilt_y": float("nan"),
        "shoulder_line_angle": float("nan"),
        "hip_tilt_y": float("nan"),
        "hip_line_angle": float("nan"),
        "trunk_angle": float("nan"),
    }

    if not ((np.any(np.isnan(ls)) or np.any(np.isnan(rs)))):
        pos["shoulder_center_x"] = float((ls[0] + rs[0]) / 2.0)
        pos["shoulder_center_y"] = float((ls[1] + rs[1]) / 2.0)
        pos["shoulder_tilt_y"] = float(rs[1] - ls[1])
        pos["shoulder_line_angle"] = float(math.degrees(math.atan2(rs[1] - ls[1], rs[0] - ls[0])))

    if not ((np.any(np.isnan(lh)) or np.any(np.isnan(rh)))):
        pos["hip_center_x"] = float((lh[0] + rh[0]) / 2.0)
        pos["hip_center_y"] = float((lh[1] + rh[1]) / 2.0)
        pos["hip_tilt_y"] = float(rh[1] - lh[1])
        pos["hip_line_angle"] = float(math.degrees(math.atan2(rh[1] - lh[1], rh[0] - lh[0])))

    if np.isfinite(pos["shoulder_center_x"]):
        pos["body_center_x"] = pos["shoulder_center_x"]
    if np.isfinite(pos["hip_center_y"]):
        pos["body_center_y"] = pos["hip_center_y"]

    if np.isfinite(pos["shoulder_center_x"]) and np.isfinite(pos["hip_center_x"]):
        pos["trunk_angle"] = float(
            math.degrees(
                math.atan2(
                    pos["shoulder_center_y"] - pos["hip_center_y"],
                    pos["shoulder_center_x"] - pos["hip_center_x"],
                )
            )
        )

    return pos


def build_frame_pose_features(
    kpts: np.ndarray,
    scores: np.ndarray,
    kpt_thr: float = HUMAN_KPT_CONF_THRESH,
) -> Dict[str, float]:
    """单帧：坐标 + 9 角 + 位置特征。"""
    joints = extract_joint_points(kpts, scores, kpt_thr)
    out: Dict[str, float] = {}
    out.update(joint_coord_columns(joints, scores, kpt_thr))
    out.update(compute_nine_angles(joints))
    out.update(compute_position_features(joints))
    return out


def empty_frame_pose_row(frame: float, time: float) -> Dict[str, float]:
    """无人检测时的空行骨架。"""
    row: Dict[str, float] = {"frame": frame, "time": time}
    for name in JOINT_CSV_NAMES:
        row[f"{name}_x"] = float("nan")
        row[f"{name}_y"] = float("nan")
        row[f"{name}_conf"] = float("nan")
    for key in (
        "left_shoulder_angle",
        "right_shoulder_angle",
        "left_elbow_angle",
        "right_elbow_angle",
        "left_hip_angle",
        "right_hip_angle",
        "left_knee_angle",
        "right_knee_angle",
        "shoulder_hip_angle",
        "shoulder_center_x",
        "shoulder_center_y",
        "hip_center_x",
        "hip_center_y",
        "body_center_x",
        "body_center_y",
        "shoulder_tilt_y",
        "shoulder_line_angle",
        "hip_tilt_y",
        "hip_line_angle",
        "trunk_angle",
    ):
        row[key] = float("nan")
    return row
