"""成对试次清单的读取、校验与目录管理。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import DEFAULT_MANIFEST, INPUT_ROOT, INTERMEDIATE_ROOT, OUTPUT_ROOT, SUPPORTED_ACTIONS

REQUIRED_COLUMNS = (
    "trial_id",
    "participant_id",
    "action",
    "handedness",
    "view_plane",
    "video_file",
    "qualisys_3d_file",
)


@dataclass(frozen=True)
class Trial:
    """清单中的一个同步采集试次。"""

    trial_id: str
    participant_id: str
    action: str
    handedness: str
    view_plane: str
    video_path: Path
    qualisys_3d_path: Path
    marker_map_path: Path | None = None
    notes: str = ""

    @property
    def intermediate_dir(self) -> Path:
        return INTERMEDIATE_ROOT / self.trial_id

    @property
    def output_dir(self) -> Path:
        return OUTPUT_ROOT / self.trial_id

    def ensure_output_dirs(self) -> None:
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate_files(self) -> None:
        missing = [path for path in (self.video_path, self.qualisys_3d_path) if not path.is_file()]
        if self.marker_map_path is not None and not self.marker_map_path.is_file():
            missing.append(self.marker_map_path)
        if missing:
            details = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"试次 {self.trial_id} 缺少文件：\n{details}")

    def validate_video(self) -> None:
        if not self.video_path.is_file():
            raise FileNotFoundError(f"试次 {self.trial_id} 缺少视频: {self.video_path}")

    def validate_qualisys(self) -> None:
        missing = [path for path in (self.qualisys_3d_path, self.marker_map_path) if path is not None and not path.is_file()]
        if missing:
            details = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"试次 {self.trial_id} 缺少 Qualisys 输入：\n{details}")


def _resolve_input_path(value: str, input_root: Path) -> Path:
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError("manifest.csv 的视频或 Qualisys 文件路径不能为空")
    path = Path(normalized).expanduser()
    return path.resolve() if path.is_absolute() else (input_root / path).resolve()


def load_trials(
    manifest_path: Path | str = DEFAULT_MANIFEST,
    *,
    validate_files: bool = True,
) -> list[Trial]:
    """读取 manifest.csv；文件路径相对 `qualisys/data/input/` 解析。"""
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"找不到试次清单: {manifest_path}\n"
            "请复制 qualisys/data/input/manifest.example.csv 为 manifest.csv 后填写。"
        )

    input_root = manifest_path.parent
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = tuple(reader.fieldnames or ())
        missing_headers = [name for name in REQUIRED_COLUMNS if name not in headers]
        if missing_headers:
            raise ValueError(f"manifest.csv 缺少列: {', '.join(missing_headers)}")
        rows = list(reader)

    trials: list[Trial] = []
    seen: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        trial_id = (row.get("trial_id") or "").strip()
        if not trial_id:
            raise ValueError(f"manifest.csv 第 {line_number} 行 trial_id 为空")
        if trial_id in seen:
            raise ValueError(f"manifest.csv trial_id 重复: {trial_id}")
        if Path(trial_id).name != trial_id or trial_id in {".", ".."}:
            raise ValueError(f"trial_id 只能是单层目录名: {trial_id!r}")
        seen.add(trial_id)

        action = (row.get("action") or "").strip().lower()
        if action not in SUPPORTED_ACTIONS:
            raise ValueError(f"试次 {trial_id} 的 action 无效: {action!r}")
        handedness = (row.get("handedness") or "").strip().lower()
        if handedness not in {"left", "right"}:
            raise ValueError(f"试次 {trial_id} 的 handedness 必须是 left 或 right")
        view_plane = (row.get("view_plane") or "ZY").strip().upper()
        if view_plane != "ZY":
            raise ValueError(f"当前框架只支持 ZY 平面，试次 {trial_id} 填写的是 {view_plane!r}")

        marker_value = (row.get("marker_map_file") or "").strip()
        trial = Trial(
            trial_id=trial_id,
            participant_id=(row.get("participant_id") or "").strip(),
            action=action,
            handedness=handedness,
            view_plane=view_plane,
            video_path=_resolve_input_path(row.get("video_file") or "", input_root),
            qualisys_3d_path=_resolve_input_path(row.get("qualisys_3d_file") or "", input_root),
            marker_map_path=_resolve_input_path(marker_value, input_root) if marker_value else None,
            notes=(row.get("notes") or "").strip(),
        )
        if validate_files:
            trial.validate_files()
        trials.append(trial)

    if not trials:
        raise ValueError("manifest.csv 没有试次记录")
    return trials


def select_trials(trials: Iterable[Trial], trial_ids: Iterable[str] | None = None) -> list[Trial]:
    """选择指定试次；不传 ID 时返回全部试次。"""
    all_trials = list(trials)
    requested = {value.strip() for value in (trial_ids or ()) if value.strip()}
    if not requested:
        return all_trials
    selected = [trial for trial in all_trials if trial.trial_id in requested]
    missing = requested - {trial.trial_id for trial in selected}
    if missing:
        raise ValueError(f"manifest.csv 中不存在试次: {', '.join(sorted(missing))}")
    return selected


__all__ = ["REQUIRED_COLUMNS", "Trial", "load_trials", "select_trials"]
