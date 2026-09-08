"""读取 Qualisys 3D TSV，并在 ZY 平面生成八角与球拍速度信号。"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from algorithm.common.pose_features import calculate_angle
from qualisys.config import ANGLE_COLUMNS, DEFAULT_MARKER_MAP, QTM_RACKET_HEAD_MARKER

JOINTS = tuple(DEFAULT_MARKER_MAP)


def read_qtm_3d_tsv(path: Path | str) -> tuple[dict[str, str], pd.DataFrame]:
    """读取 QTM TSV 元数据和以 ``Frame/Time`` 开始的 3D 数据表。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 Qualisys TSV: {path}")

    metadata: dict[str, str] = {}
    header_index: int | None = None
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for index, raw_line in enumerate(handle):
            cells = raw_line.rstrip("\r\n").split("\t")
            if len(cells) >= 2 and cells[0] == "Frame" and cells[1] == "Time":
                header_index = index
                break
            if cells and cells[0]:
                metadata[cells[0]] = "\t".join(cells[1:]).strip()
    if header_index is None:
        raise ValueError(f"没有找到 Frame/Time 数据表头: {path}")
    if metadata.get("DATA_INCLUDED", "").upper() != "3D":
        raise ValueError(f"文件不是 Qualisys 3D 轨迹导出: {path}")

    data = pd.read_csv(path, sep="\t", skiprows=header_index, encoding="utf-8-sig")
    data = data.dropna(axis=1, how="all")
    if not {"Frame", "Time"}.issubset(data.columns):
        raise ValueError(f"Qualisys 数据缺少 Frame/Time 列: {path}")
    return metadata, data


def _marker_center_zy(data: pd.DataFrame, markers: Sequence[str]) -> np.ndarray:
    """取一个或多个 marker 的 Z/Y 坐标均值；某帧缺任一点则返回 NaN。"""
    arrays: list[np.ndarray] = []
    missing: list[str] = []
    for marker in markers:
        columns = [f"{marker} Z", f"{marker} Y"]
        absent = [column for column in columns if column not in data.columns]
        if absent:
            missing.extend(absent)
        else:
            arrays.append(data[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    if missing:
        raise ValueError(f"Qualisys TSV 缺少点位列: {', '.join(sorted(set(missing)))}")

    stacked = np.stack(arrays, axis=0)
    valid = np.all(np.isfinite(stacked), axis=(0, 2))
    center = np.full((len(data), 2), np.nan, dtype=np.float64)
    center[valid] = np.mean(stacked[:, valid, :], axis=0)
    return center


def _normalized_time(data: pd.DataFrame) -> np.ndarray:
    time = pd.to_numeric(data["Time"], errors="coerce").to_numpy(dtype=np.float64)
    if len(time) == 0 or not np.isfinite(time[0]):
        raise ValueError("Qualisys 首帧 Time 不是有效数字")
    time = time - time[0]
    valid = np.isfinite(time)
    if np.count_nonzero(valid) < 2 or np.any(np.diff(time[valid]) <= 0):
        raise ValueError("Qualisys Time 必须严格递增")
    return time


def build_qualisys_angles(
    data: pd.DataFrame,
    marker_map: Mapping[str, Sequence[str]] = DEFAULT_MARKER_MAP,
) -> pd.DataFrame:
    """按 RTMPose 同定义计算 ZY 平面 0–180° 无符号三点内角。"""
    joints = {joint: _marker_center_zy(data, marker_map[joint]) for joint in JOINTS}
    triples = {
        "left_shoulder_angle": ("left_elbow", "left_shoulder", "left_hip"),
        "right_shoulder_angle": ("right_elbow", "right_shoulder", "right_hip"),
        "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
        "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
        "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
        "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
        "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
        "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    }
    result = pd.DataFrame(
        {
            "frame": pd.to_numeric(data["Frame"], errors="coerce"),
            "time": _normalized_time(data),
        }
    )
    for column, (a, b, c) in triples.items():
        result[column] = [
            calculate_angle(joints[a][index], joints[b][index], joints[c][index])
            for index in range(len(data))
        ]
    return result[["frame", "time", *ANGLE_COLUMNS]]


def build_qtm_racket_signal(
    data: pd.DataFrame,
    marker: str = QTM_RACKET_HEAD_MARKER,
) -> pd.DataFrame:
    """用 QTM ``Racket_top`` 的 ZY 速度构造独立于关节角的对齐信号。"""
    position = _marker_center_zy(data, (marker,))
    time = _normalized_time(data)
    valid_position = np.all(np.isfinite(position), axis=1)

    if len(time) > 1:
        median_dt = float(np.nanmedian(np.diff(time)))
        max_gap = max(1, int(round(0.15 / median_dt))) if median_dt > 0 else 1
        working = pd.DataFrame(position, columns=["z", "y"])
        working = working.interpolate(
            method="linear", limit=max_gap, limit_area="inside", limit_direction="both"
        )
        z = working["z"].to_numpy(dtype=np.float64)
        y = working["y"].to_numpy(dtype=np.float64)
        speed = np.hypot(np.gradient(z, time), np.gradient(y, time))
    else:
        speed = np.full(len(time), np.nan, dtype=np.float64)

    return pd.DataFrame(
        {
            "frame": pd.to_numeric(data["Frame"], errors="coerce"),
            "time": time,
            "racket_head_z": position[:, 0],
            "racket_head_y": position[:, 1],
            "racket_head_speed": speed,
            "valid_position": valid_position.astype(int),
        }
    )


__all__ = [
    "build_qtm_racket_signal",
    "build_qualisys_angles",
    "read_qtm_3d_tsv",
]
