"""
人体-球拍融合 2D kinematic features（图像平面运动学代理）。

四动作用途摘要（供后续切分与论文）：
- forehand: 持拍侧肩/肘/髋/膝角、shoulder_hip_angle、racket_head_speed、
  racket_head_rel_body_x、racket_head_to_wrist_dist
- backhand: 持拍侧或双手肩肘角、支撑膝角、shoulder_hip_angle、shoulder_turn_x_diff、
  racket_head_speed、racket_head_rel_body_x
- serve: 非持拍手腕 y、持拍侧肩肘髋膝角、shoulder_tilt_y、shoulder_hip_angle、
  racket_long_axis_angle、racket_head_speed
- volley: 持拍肘角、支撑膝角、shoulder_hip_angle、racket_head_speed、
  racket_head_wrist_y_diff、racket_head_to_wrist_dist、racket_long_axis_angle
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def _normalize_handedness(handedness: str) -> str:
    h = (handedness or "right").strip().lower()
    return "left" if h in ("left", "l", "左手") else "right"


def _robust_read_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        clean_text = f.read()
    clean_text = clean_text.replace("'", "").replace("'", "").replace("'", "")
    try:
        df = pd.read_csv(io.StringIO(clean_text), engine="python", on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(io.StringIO(clean_text), engine="python", error_bad_lines=False)
    for col in df.columns:
        if col not in ("frame",):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _pick_column(df: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(np.nan, index=df.index, dtype=np.float64)


def build_fused_kinematic_df(
    racket_csv: Union[str, Path],
    body_csv: Union[str, Path],
    handedness: str = "right",
) -> pd.DataFrame:
    """
    按 frame 内连接球拍与人体 CSV，追加人体-球拍相对 2D kinematic features。
    """
    df_r = _robust_read_csv(racket_csv)
    df_b = _robust_read_csv(body_csv)
    df_r["frame"] = df_r["frame"].astype(int)
    df_b["frame"] = df_b["frame"].astype(int)
    fused = pd.merge(df_r, df_b, on="frame", how="inner", suffixes=("_racket", "_body"))

    hand = _normalize_handedness(handedness)
    if hand == "right":
        wrist_x = _pick_column(fused, "right_wrist_x")
        wrist_y = _pick_column(fused, "right_wrist_y")
    else:
        wrist_x = _pick_column(fused, "left_wrist_x")
        wrist_y = _pick_column(fused, "left_wrist_y")

    tx = _pick_column(fused, "t_x_clean", "x_clean")
    ty = _pick_column(fused, "t_y_clean", "y_clean")
    bcx = _pick_column(fused, "body_center_x")
    bcy = _pick_column(fused, "body_center_y")

    fused["racket_wrist_x"] = wrist_x
    fused["racket_wrist_y"] = wrist_y

    dx = tx - wrist_x
    dy = ty - wrist_y
    fused["racket_head_to_wrist_dist"] = np.sqrt(dx * dx + dy * dy)
    fused["racket_head_wrist_y_diff"] = ty - wrist_y
    fused["racket_head_rel_body_x"] = tx - bcx
    fused["racket_head_rel_body_y"] = ty - bcy

    rcx = _pick_column(fused, "racket_center_x")
    rcy = _pick_column(fused, "racket_center_y")
    fused["racket_center_rel_body_x"] = rcx - bcx
    fused["racket_center_rel_body_y"] = rcy - bcy

    if "time" in fused.columns and fused["time"].notna().any():
        pass
    elif "time_racket" in fused.columns:
        fused["time"] = fused["time_racket"]
    elif "time_body" in fused.columns:
        fused["time"] = fused["time_body"]
    else:
        fused["time"] = np.nan

    fused["handedness"] = hand
    return fused


def write_fused_debug_csv(
    racket_csv: Union[str, Path],
    body_csv: Union[str, Path],
    output_path: Union[str, Path],
    handedness: str = "right",
) -> str:
    """写入融合 debug CSV，返回路径。"""
    df = build_fused_kinematic_df(racket_csv, body_csv, handedness)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(path), index=False, encoding="utf-8-sig")
    return str(path.resolve())
