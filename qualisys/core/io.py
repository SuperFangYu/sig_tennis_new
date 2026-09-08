"""验证任务的输入校验、CSV 读取与逐次输出目录管理。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from qualisys.config import ANGLE_COLUMNS, OUTPUT_ROOT


class ValidationError(RuntimeError):
    """数据或自动对齐未达到预设质控要求。"""


def validate_input_file(path: Path | str, label: str) -> Path:
    result = Path(path).expanduser()
    if not result.is_absolute():
        result = result.resolve()
    if not result.is_file():
        raise FileNotFoundError(f"找不到{label}: {result}")
    return result


def read_angle_csv(path: Path | str, label: str = "RTMPose") -> pd.DataFrame:
    """读取 ``frame/time + 8 angles``，排序、去重并把时间归零。"""
    path = validate_input_file(path, f"{label} 角度 CSV")
    data = pd.read_csv(path)
    required = {"frame", "time", *ANGLE_COLUMNS}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{label} 角度 CSV 缺少列: {', '.join(sorted(missing))}")

    result = data[["frame", "time", *ANGLE_COLUMNS]].copy()
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result[np.isfinite(result["time"])].sort_values("time")
    result = result.drop_duplicates("time", keep="first").reset_index(drop=True)
    if len(result) < 2:
        raise ValueError(f"{label} 角度 CSV 至少需要两个有效时间点: {path}")
    result["time"] = result["time"] - float(result["time"].iloc[0])
    if np.any(np.diff(result["time"].to_numpy(dtype=float)) <= 0):
        raise ValueError(f"{label} 角度 CSV 的 time 必须严格递增: {path}")
    return result


def create_run_directory(
    action: str,
    sample_stem: str,
    *,
    output_root: Path = OUTPUT_ROOT,
    now: datetime | None = None,
) -> Path:
    """创建 ``output/动作/样本/时间戳``，永不复用旧运行目录。"""
    safe_stem = re.sub(r"[^0-9A-Za-z._-]+", "_", sample_stem).strip("._") or "sample"
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    parent = Path(output_root) / action / safe_stem
    candidate = parent / timestamp
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


__all__ = [
    "ValidationError",
    "create_run_directory",
    "read_angle_csv",
    "validate_input_file",
    "write_json",
]
