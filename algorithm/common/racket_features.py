"""球拍侧 2D kinematic feature（图像平面运动学特征）派生计算。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

RACKET_POINT_NAMES = ("t", "l", "r", "b", "h")
CENTER_POINTS = ("t", "l", "r", "b")


def _savgol_series(y: np.ndarray, window_length: int, poly_order: int = 3) -> np.ndarray:
    n = len(y)
    if n < poly_order + 2:
        return y.copy()
    wl = min(window_length, n if n % 2 == 1 else n - 1)
    if wl % 2 == 0:
        wl -= 1
    wl = max(3, wl)
    if wl > n:
        wl = n if n % 2 == 1 else n - 1
    if wl < 3:
        return y.copy()
    po = min(poly_order, wl - 1)
    if po < 1:
        return y.copy()
    try:
        return np.asarray(savgol_filter(y, wl, po, mode="interp"), dtype=np.float64)
    except ValueError:
        return y.copy()


def interpolate_point_columns(df: pd.DataFrame, point: str) -> None:
    """对单点 raw 列做线性插值，生成 interp 列。"""
    x_raw = f"{point}_x_raw"
    y_raw = f"{point}_y_raw"
    x_interp = f"{point}_x_interp"
    y_interp = f"{point}_y_interp"
    df[x_interp] = df[x_raw].interpolate(method="linear", limit_direction="both")
    df[y_interp] = df[y_raw].interpolate(method="linear", limit_direction="both")


def smooth_point_columns(
    df: pd.DataFrame,
    point: str,
    fps: float,
    valid_mask: np.ndarray,
) -> None:
    """对 interp 列 Savitzky-Golay 平滑；低置信度帧 clean 保持 NaN。"""
    x_interp = f"{point}_x_interp"
    y_interp = f"{point}_y_interp"
    x_clean = f"{point}_x_clean"
    y_clean = f"{point}_y_clean"
    poly_order = 3
    n = len(df)
    if n < poly_order + 2:
        df[x_clean] = df[x_interp]
        df[y_clean] = df[y_interp]
    else:
        wl = int(fps // 2) | 1
        df[x_clean] = _savgol_series(df[x_interp].to_numpy(dtype=np.float64), wl, poly_order)
        df[y_clean] = _savgol_series(df[y_interp].to_numpy(dtype=np.float64), wl, poly_order)
    invalid = ~valid_mask
    df.loc[invalid, x_clean] = np.nan
    df.loc[invalid, y_clean] = np.nan


def add_legacy_aliases(df: pd.DataFrame) -> None:
    """旧字段别名 = 拍头 t 点。"""
    df["x_raw"] = df["t_x_raw"]
    df["y_raw"] = df["t_y_raw"]
    df["x_interp"] = df["t_x_interp"]
    df["y_interp"] = df["t_y_interp"]
    df["x_clean"] = df["t_x_clean"]
    df["y_clean"] = df["t_y_clean"]


def compute_racket_derived_features(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    基于 clean 坐标与 time 计算球拍 2D kinematic features。
    修改 df 并返回。
    """
    time_axis = df["time"].to_numpy(dtype=np.float64)
    tx = df["t_x_clean"].to_numpy(dtype=np.float64)
    ty = df["t_y_clean"].to_numpy(dtype=np.float64)
    hx = df["h_x_clean"].to_numpy(dtype=np.float64)
    hy = df["h_y_clean"].to_numpy(dtype=np.float64)
    lx = df["l_x_clean"].to_numpy(dtype=np.float64)
    ly = df["l_y_clean"].to_numpy(dtype=np.float64)
    rx = df["r_x_clean"].to_numpy(dtype=np.float64)
    ry = df["r_y_clean"].to_numpy(dtype=np.float64)

    if len(time_axis) > 1 and np.all(np.isfinite(time_axis)):
        dt = np.gradient(time_axis)
        dt = np.where(dt <= 0, np.nan, dt)
        vx = np.gradient(tx) / dt
        vy = np.gradient(ty) / dt
    else:
        vx = np.full(len(df), np.nan)
        vy = np.full(len(df), np.nan)

    speed = np.sqrt(vx * vx + vy * vy)
    if len(time_axis) > 1 and np.all(np.isfinite(time_axis)):
        ax = np.gradient(vx) / dt
        ay = np.gradient(vy) / dt
    else:
        ax = np.full(len(df), np.nan)
        ay = np.full(len(df), np.nan)
    acc = np.sqrt(ax * ax + ay * ay)

    long_angle = np.degrees(np.arctan2(ty - hy, tx - hx))
    width_angle = np.degrees(np.arctan2(ry - ly, rx - lx))

    center_x = []
    center_y = []
    valid_counts = []
    mean_confs = []
    conf_cols = [f"{p}_conf" for p in RACKET_POINT_NAMES]

    for i in range(len(df)):
        xs, ys = [], []
        confs = []
        for p in CENTER_POINTS:
            xc = df.at[i, f"{p}_x_clean"]
            yc = df.at[i, f"{p}_y_clean"]
            if np.isfinite(xc) and np.isfinite(yc):
                xs.append(xc)
                ys.append(yc)
        for ccol in conf_cols:
            cval = df.at[i, ccol]
            if np.isfinite(cval):
                confs.append(cval)
        center_x.append(float(np.mean(xs)) if xs else np.nan)
        center_y.append(float(np.mean(ys)) if ys else np.nan)
        valid_counts.append(
            sum(
                1
                for p in RACKET_POINT_NAMES
                if np.isfinite(df.at[i, f"{p}_x_raw"]) and np.isfinite(df.at[i, f"{p}_y_raw"])
            )
        )
        mean_confs.append(float(np.mean(confs)) if confs else np.nan)

    df["racket_head_vx"] = vx
    df["racket_head_vy"] = vy
    df["racket_head_speed"] = speed
    df["racket_head_ax"] = ax
    df["racket_head_ay"] = ay
    df["racket_head_acc"] = acc
    df["racket_long_axis_angle"] = long_angle
    df["racket_width_axis_angle"] = width_angle
    df["racket_center_x"] = center_x
    df["racket_center_y"] = center_y
    df["racket_valid_kpt_count"] = valid_counts
    df["racket_mean_conf"] = mean_confs
    return df
