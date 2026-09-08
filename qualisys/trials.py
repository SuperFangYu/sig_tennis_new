"""扫描 video/ 与 qtm/，按 `英文名_英文动作_编号` 自动建立成对试次。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from .config import (
    ACTION_FILENAME_ALIASES,
    INTERMEDIATE_ROOT,
    OUTPUT_ROOT,
    QTM_INPUT_ROOT,
    VIDEO_INPUT_ROOT,
)

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


@dataclass(frozen=True)
class Trial:
    """由一对同名视频/TSV 构成的同步采集试次。"""

    trial_id: str
    participant_id: str
    action: str
    trial_number: int
    view_plane: str
    video_path: Path
    qualisys_3d_path: Path

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
        if missing:
            details = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"试次 {self.trial_id} 缺少文件：\n{details}")

    def validate_video(self) -> None:
        if not self.video_path.is_file():
            raise FileNotFoundError(f"试次 {self.trial_id} 缺少视频: {self.video_path}")

    def validate_qualisys(self) -> None:
        if not self.qualisys_3d_path.is_file():
            raise FileNotFoundError(
                f"试次 {self.trial_id} 缺少 Qualisys 3D TSV: {self.qualisys_3d_path}"
            )


def parse_trial_stem(stem: str) -> tuple[str, str, int]:
    """解析 `英文名_英文动作_编号`，返回 `(姓名, 五类动作代码, 编号)`。"""
    aliases = sorted(ACTION_FILENAME_ALIASES, key=len, reverse=True)
    for alias in aliases:
        pattern = rf"^(?P<participant>.+)_{re.escape(alias)}_(?P<number>[1-9]\d*)$"
        matched = re.match(pattern, stem, flags=re.IGNORECASE)
        if matched:
            participant = matched.group("participant").strip("_")
            if participant:
                return participant, ACTION_FILENAME_ALIASES[alias], int(matched.group("number"))
    raise ValueError(
        f"文件名不符合 `英文名_英文动作_编号`: {stem!r}；"
        "例如 fy_zs_1、fy_jjfs_1、fy_serve_1"
    )


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
    """从 `input/video` 与 `input/qtm` 自动发现主文件名完全相同的文件。"""
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
        participant, action, trial_number = parse_trial_stem(stem)
        trials.append(
            Trial(
                trial_id=stem,
                participant_id=participant,
                action=action,
                trial_number=trial_number,
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


def load_trials(*, validate_files: bool = True) -> list[Trial]:
    """项目统一入口：直接扫描两个输入目录，不再读取 manifest。"""
    return discover_trials(validate_pairs=validate_files)


def select_trials(trials: Iterable[Trial], trial_ids: Iterable[str] | None = None) -> list[Trial]:
    """选择指定试次；不传 ID 时返回全部试次。"""
    all_trials = list(trials)
    requested = {value.strip() for value in (trial_ids or ()) if value.strip()}
    if not requested:
        return all_trials
    selected = [trial for trial in all_trials if trial.trial_id in requested]
    missing = requested - {trial.trial_id for trial in selected}
    if missing:
        raise ValueError(f"输入目录中不存在试次: {', '.join(sorted(missing))}")
    return selected


__all__ = [
    "Trial",
    "discover_trials",
    "load_trials",
    "parse_trial_stem",
    "select_trials",
]
