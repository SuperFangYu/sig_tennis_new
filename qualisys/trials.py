"""发现或读取成对试次，并把输入文件转换为统一的 Trial 对象。

正常使用不需要写 manifest：video/ 与 qtm/ 中同名文件会自动配对。只有需要覆盖持拍手、
点位映射或备注时，才复制 manifest.example.csv 为 manifest.csv。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import (
    ACTION_FILENAME_ALIASES,
    DEFAULT_MANIFEST,
    INTERMEDIATE_ROOT,
    OUTPUT_ROOT,
    QTM_INPUT_ROOT,
    SUPPORTED_ACTIONS,
    VIDEO_INPUT_ROOT,
)

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

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


def normalize_experiment_action(value: str) -> str:
    """把中英文动作名称统一为五类实验动作代码。"""
    normalized = str(value).strip().lower()
    alias_lookup = {key.lower(): action for key, action in ACTION_FILENAME_ALIASES.items()}
    action = alias_lookup.get(normalized, normalized)
    if action not in SUPPORTED_ACTIONS:
        supported = "、".join(SUPPORTED_ACTIONS)
        raise ValueError(f"无法识别动作 {value!r}，支持: {supported}")
    return action


def parse_trial_stem(stem: str) -> tuple[str, str]:
    """解析 `人名_动作` 文件名，返回 `(人名, 五类动作代码)`。"""
    aliases = sorted(ACTION_FILENAME_ALIASES, key=len, reverse=True)
    lower_stem = stem.lower()
    for alias in aliases:
        suffix = f"_{alias.lower()}"
        if lower_stem.endswith(suffix):
            participant = stem[: -len(suffix)].strip("_")
            if not participant:
                raise ValueError(f"文件名缺少人名: {stem!r}")
            return participant, ACTION_FILENAME_ALIASES[alias]
    expected = "、".join(f"人名_{alias}" for alias in ("正手", "反手", "正手截击", "反手截击", "发球"))
    raise ValueError(f"文件名不符合五动作规则: {stem!r}；示例: {expected}")


def _unique_files_by_stem(root: Path, extensions: tuple[str, ...]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    if not root.exists():
        return files
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.stem in files:
            raise ValueError(f"目录中存在同名的多个文件: {files[path.stem]} / {path}")
        files[path.stem] = path.resolve()
    return files


def discover_trials(
    *,
    validate_pairs: bool = True,
    video_root: Path = VIDEO_INPUT_ROOT,
    qtm_root: Path = QTM_INPUT_ROOT,
) -> list[Trial]:
    """从 `input/video` 与 `input/qtm` 自动发现同名成对文件。

    自动模式默认按右手持拍处理；左手受试者请使用 manifest.csv 明确填写 handedness。
    """
    videos = _unique_files_by_stem(video_root, VIDEO_EXTENSIONS)
    qtm_files = _unique_files_by_stem(qtm_root, (".tsv",))
    video_only = sorted(set(videos) - set(qtm_files))
    qtm_only = sorted(set(qtm_files) - set(videos))
    if validate_pairs and (video_only or qtm_only):
        details = []
        if video_only:
            details.append(f"缺少同名 QTM: {', '.join(video_only)}")
        if qtm_only:
            details.append(f"缺少同名视频: {', '.join(qtm_only)}")
        raise FileNotFoundError("自动配对不完整：\n- " + "\n- ".join(details))

    trials: list[Trial] = []
    for stem in sorted(set(videos) & set(qtm_files)):
        participant, action = parse_trial_stem(stem)
        trials.append(
            Trial(
                trial_id=stem,
                participant_id=participant,
                action=action,
                handedness="right",
                view_plane="ZY",
                video_path=videos[stem],
                qualisys_3d_path=qtm_files[stem],
            )
        )
    if not trials:
        raise FileNotFoundError(
            f"没有发现成对试次；请把同名文件分别放入 {video_root} 和 {qtm_root}"
        )
    return trials


def load_trials(
    manifest_path: Path | str = DEFAULT_MANIFEST,
    *,
    validate_files: bool = True,
) -> list[Trial]:
    """优先读取 manifest.csv；文件不存在时按 `人名_动作` 自动配对。"""
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        return discover_trials(validate_pairs=validate_files)

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

        action = normalize_experiment_action(row.get("action") or "")
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


__all__ = [
    "REQUIRED_COLUMNS",
    "Trial",
    "discover_trials",
    "load_trials",
    "normalize_experiment_action",
    "parse_trial_stem",
    "select_trials",
]
