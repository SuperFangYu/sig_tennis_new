"""把已导出的 RTMPose 人体关节与 YOLO 球拍五点回绘到原视频。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from algorithm.common.pose_features import JOINT_CSV_NAMES
from algorithm.common.racket_features import RACKET_POINT_NAMES
from algorithm.common.thresholds import HUMAN_KPT_CONF_THRESH, RACKET_KPT_VALID_THRESH

PathLike = Union[str, Path]
Point = Tuple[int, int]

BODY_CONNECTIONS = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)

# t=top、l=left、r=right、b=bottom 构成拍面，h=handle 与 bottom 相连。
RACKET_FACE_CONNECTIONS = (("t", "l"), ("l", "b"), ("b", "r"), ("r", "t"))


def _finite_number(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _row_by_frame(df: pd.DataFrame) -> Dict[int, pd.Series]:
    rows: Dict[int, pd.Series] = {}
    if "frame" not in df.columns:
        return rows
    for _, row in df.iterrows():
        if _finite_number(row.get("frame")):
            rows[int(round(float(row["frame"])))] = row
    return rows


def _body_points(row: Optional[pd.Series], threshold: float) -> Dict[str, Point]:
    points: Dict[str, Point] = {}
    if row is None:
        return points
    for name in JOINT_CSV_NAMES:
        x, y, conf = row.get(f"{name}_x"), row.get(f"{name}_y"), row.get(f"{name}_conf")
        if all(_finite_number(v) for v in (x, y, conf)) and float(conf) >= threshold:
            points[name] = (int(round(float(x))), int(round(float(y))))
    return points


def _first_finite(row: pd.Series, columns: Iterable[str]) -> Optional[float]:
    for column in columns:
        value = row.get(column)
        if _finite_number(value):
            return float(value)
    return None


def _racket_points(row: Optional[pd.Series], threshold: float) -> Dict[str, Point]:
    points: Dict[str, Point] = {}
    if row is None:
        return points
    for name in RACKET_POINT_NAMES:
        conf = row.get(f"{name}_conf")
        if not _finite_number(conf) or float(conf) < threshold:
            continue
        # clean 用于获得更稳定的连线；raw 为当前帧原始预测的兜底值。
        x = _first_finite(row, (f"{name}_x_clean", f"{name}_x_raw"))
        y = _first_finite(row, (f"{name}_y_clean", f"{name}_y_raw"))
        if x is not None and y is not None:
            points[name] = (int(round(x)), int(round(y)))
    return points


def _draw_connections(
    frame: np.ndarray,
    points: Dict[str, Point],
    connections: Iterable[Tuple[str, str]],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    for start, end in connections:
        if start in points and end in points:
            cv2.line(frame, points[start], points[end], color, thickness, cv2.LINE_AA)


def render_pose_racket_video(
    video_path: PathLike,
    body_csv: PathLike,
    racket_csv: PathLike,
    output_path: PathLike,
    *,
    action: str = "",
    body_kpt_thr: float = HUMAN_KPT_CONF_THRESH,
    racket_kpt_thr: float = RACKET_KPT_VALID_THRESH,
) -> str:
    """按 frame 读取两个 CSV，在原视频上绘制人体骨架和完整球拍轮廓。"""
    video_path = Path(video_path)
    body_csv = Path(body_csv)
    racket_csv = Path(racket_csv)
    output_path = Path(output_path)
    for path, label in ((video_path, "视频"), (body_csv, "人体 CSV"), (racket_csv, "球拍 CSV")):
        if not path.exists():
            raise FileNotFoundError(f"{label}不存在: {path}")

    body_rows = _row_by_frame(pd.read_csv(body_csv))
    racket_rows = _row_by_frame(pd.read_csv(racket_csv))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError(f"无法读取视频尺寸: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频: {output_path}")

    frame_number = 0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            body_points = _body_points(body_rows.get(frame_number), body_kpt_thr)
            racket_points = _racket_points(racket_rows.get(frame_number), racket_kpt_thr)

            # 人体：青色连线 + 黄色关节点。
            _draw_connections(frame, body_points, BODY_CONNECTIONS, (255, 210, 0), 3)
            for point in body_points.values():
                cv2.circle(frame, point, 4, (0, 230, 255), -1, cv2.LINE_AA)

            # 球拍：洋红拍面、橙色拍柄、绿色五点，与人体配色明确区分。
            _draw_connections(frame, racket_points, RACKET_FACE_CONNECTIONS, (255, 0, 220), 3)
            _draw_connections(frame, racket_points, (("b", "h"),), (0, 140, 255), 4)
            for name, point in racket_points.items():
                cv2.circle(frame, point, 5, (80, 255, 80), -1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    name,
                    (point[0] + 6, point[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (80, 255, 80),
                    1,
                    cv2.LINE_AA,
                )

            label = f"{action.upper()}  Frame {frame_number}" if action else f"Frame {frame_number}"
            cv2.putText(frame, label, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "BODY", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 210, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "RACKET", (88, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 220), 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    if frame_number == 0:
        raise ValueError(f"输入视频没有可读取的帧: {video_path}")
    return str(output_path.resolve())


__all__ = ["BODY_CONNECTIONS", "RACKET_FACE_CONNECTIONS", "render_pose_racket_video"]
