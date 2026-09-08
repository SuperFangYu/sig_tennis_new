"""阶段 2：读取一个试次的 Qualisys 3D TSV，生成 ZY 平面八角 CSV。

本文件不运行 RTMPose、不做时间对齐，也不读取测力台数据。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from algorithm.common.pose_features import calculate_angle
from qualisys.config import ANGLE_COLUMNS, DEFAULT_MANIFEST, DEFAULT_MARKER_MAP
from qualisys.trials import Trial, load_trials, select_trials

JOINTS = tuple(DEFAULT_MARKER_MAP)


def read_qtm_3d_tsv(path: Path | str) -> tuple[dict[str, str], pd.DataFrame]:
    """读取 QTM TSV 元数据和以 `Frame/Time` 开始的数据表。"""
    path = Path(path)
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


def load_marker_map(path: Path | None) -> dict[str, tuple[str, ...]]:
    """加载可选 JSON 点位映射；未提供时使用当前实验默认映射。"""
    if path is None:
        return dict(DEFAULT_MARKER_MAP)
    with path.open("r", encoding="utf-8-sig") as handle:
        raw = json.load(handle)
    missing = [joint for joint in JOINTS if joint not in raw]
    if missing:
        raise ValueError(f"点位映射缺少关节: {', '.join(missing)}")
    mapping: dict[str, tuple[str, ...]] = {}
    for joint in JOINTS:
        markers = raw[joint]
        if isinstance(markers, str):
            markers = [markers]
        if not isinstance(markers, list) or not markers or not all(isinstance(item, str) for item in markers):
            raise ValueError(f"点位映射 {joint} 必须是非空字符串列表")
        mapping[joint] = tuple(markers)
    return mapping


def _joint_zy(data: pd.DataFrame, markers: Sequence[str]) -> np.ndarray:
    """由一个或多个 marker 的 Z/Y 坐标构造关节中心；任一 marker 缺失则该帧无效。"""
    arrays: list[np.ndarray] = []
    missing_columns: list[str] = []
    for marker in markers:
        columns = [f"{marker} Z", f"{marker} Y"]
        missing_columns.extend(column for column in columns if column not in data.columns)
        if not missing_columns:
            arrays.append(data[columns].to_numpy(dtype=np.float64))
    if missing_columns:
        raise ValueError(f"Qualisys TSV 缺少点位列: {', '.join(sorted(set(missing_columns)))}")
    stacked = np.stack(arrays, axis=0)
    valid = np.all(np.isfinite(stacked), axis=(0, 2))
    center = np.full((len(data), 2), np.nan, dtype=np.float64)
    center[valid] = np.mean(stacked[:, valid, :], axis=0)
    return center


def build_qualisys_angles(
    data: pd.DataFrame,
    marker_map: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """按与 RTMPose 完全相同的三点定义计算 ZY 平面无符号内角。"""
    joints = {joint: _joint_zy(data, marker_map[joint]) for joint in JOINTS}
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
            "time": pd.to_numeric(data["Time"], errors="coerce"),
        }
    )
    if not result.empty:
        first_time = float(result["time"].iloc[0])
        if not np.isfinite(first_time):
            raise ValueError("Qualisys 首帧 Time 不是有效数字")
        result["time"] = result["time"] - first_time
    for column, (a, b, c) in triples.items():
        result[column] = [
            calculate_angle(joints[a][index], joints[b][index], joints[c][index])
            for index in range(len(data))
        ]
    return result[["frame", "time", *ANGLE_COLUMNS]]


def run_qualisys_trial(trial: Trial) -> dict[str, object]:
    """处理一个试次并输出 `intermediate/{trial}/qualisys_8_angles.csv`。"""
    trial.validate_qualisys()
    trial.ensure_output_dirs()
    metadata, data = read_qtm_3d_tsv(trial.qualisys_3d_path)
    marker_map = load_marker_map(trial.marker_map_path)
    angles = build_qualisys_angles(data, marker_map)
    output_path = trial.intermediate_dir / "qualisys_8_angles.csv"
    angles.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "trial_id": trial.trial_id,
        "csv": str(output_path.resolve()),
        "frequency_hz": metadata.get("FREQUENCY", ""),
        "rows": len(angles),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="按自动配对或可选清单生成 Qualisys ZY 平面八角 CSV。")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="可选 manifest.csv；不存在时自动扫描 input/video 与 input/qtm",
    )
    parser.add_argument("--trial", action="append", dest="trial_ids", help="只处理指定 trial_id，可重复")
    args = parser.parse_args()

    trials = select_trials(load_trials(args.manifest, validate_files=False), args.trial_ids)
    for trial in trials:
        result = run_qualisys_trial(trial)
        print(f"[{trial.trial_id}] {result['csv']}")


if __name__ == "__main__":
    main()
