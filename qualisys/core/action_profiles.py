"""五类网球动作的自动峰值检测与动作窗口参数。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionProfile:
    """只描述动作节奏差异，不改变八个关节角的计算定义。"""

    min_peak_distance_s: float
    smooth_window_s: float
    min_prominence: float
    boundary_fraction: float
    max_pre_s: float
    max_post_s: float


ACTION_PROFILES = {
    "forehand": ActionProfile(1.20, 0.25, 0.12, 0.20, 1.40, 1.60),
    "backhand": ActionProfile(1.20, 0.25, 0.12, 0.20, 1.40, 1.60),
    "forehand_volley": ActionProfile(0.80, 0.20, 0.10, 0.22, 0.90, 1.00),
    "backhand_volley": ActionProfile(0.80, 0.20, 0.10, 0.22, 0.90, 1.00),
    "serve": ActionProfile(1.80, 0.30, 0.12, 0.18, 2.20, 1.80),
}


def get_action_profile(action: str) -> ActionProfile:
    try:
        return ACTION_PROFILES[action]
    except KeyError as exc:
        supported = ", ".join(ACTION_PROFILES)
        raise ValueError(f"不支持的动作 {action!r}；可选: {supported}") from exc


__all__ = ["ACTION_PROFILES", "ActionProfile", "get_action_profile"]
