"""把已导出的 RTMPose 人体关节与 YOLO 球拍五点回绘到原视频。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pandas as pd

from algorithm.common.pose_features import (
    HALPE26_CONNECTIONS,
    HALPE26_NAMES,
    JOINT_CSV_NAMES,
)
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
RACKET_COLOR = (255, 0, 220)


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


def _point_cloud_size(points: Dict[str, Point]) -> float:
    """五点最大间距作为透视下的自适应球拍尺度，不依赖拍面宽度。"""
    coords = np.asarray(list(points.values()), dtype=np.float64)
    if len(coords) < 2:
        return float("nan")
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    return float(np.max(distances))


def _point_cloud_center(points: Dict[str, Point]) -> Optional[np.ndarray]:
    if len(points) < 2:
        return None
    return np.mean(np.asarray(list(points.values()), dtype=np.float64), axis=0)


def stabilize_racket_visual_points(
    df: pd.DataFrame,
    threshold: float = RACKET_KPT_VALID_THRESH,
    *,
    max_gap: int = 2,
) -> Dict[int, Dict[str, Point]]:
    """
    仅为视频绘制做宽松的时间/尺寸异常抑制，不回写球拍 CSV。

    侧视时拍面宽度可能接近 0，因此不检查凸性、面积或长宽比。尺度使用五点最大间距，
    只有相对近期稳定中位数缩小到 0.4 倍以下或放大到 2.5 倍以上才判为明显异常。
    """
    candidates = {
        frame: _racket_points(row, threshold)
        for frame, row in _row_by_frame(df).items()
    }
    stabilized: Dict[int, Dict[str, Point]] = {}
    recent_sizes: list[float] = []
    recent_centers: list[tuple[int, np.ndarray]] = []
    previous_points: Dict[str, Point] = {}

    for frame_number in sorted(candidates):
        points = dict(candidates[frame_number])
        size = _point_cloud_size(points)
        reference_size = (
            float(np.median(recent_sizes[-9:])) if recent_sizes else float("nan")
        )

        # 尺寸门限刻意宽松，只去掉明显整帧误检；不会因拍面侧切变窄而拒绝。
        if np.isfinite(size) and np.isfinite(reference_size) and reference_size > 1.0:
            ratio = size / reference_size
            if ratio < 0.40 or ratio > 2.50:
                points = {}

        center = _point_cloud_center(points)
        if center is not None and len(recent_centers) >= 2 and np.isfinite(reference_size):
            previous_frame, previous_center = recent_centers[-1]
            older_frame, older_center = recent_centers[-2]
            history_dt = max(previous_frame - older_frame, 1)
            predict_dt = max(frame_number - previous_frame, 1)
            predicted = previous_center + (previous_center - older_center) * (
                predict_dt / history_dt
            )
            # 允许一帧移动接近三个球拍长度，只拦截非常离谱的瞬移。
            if float(np.linalg.norm(center - predicted)) > max(60.0, 2.8 * reference_size):
                points = {}
                center = None

        # 去掉相对球拍整体运动明显脱离的单个点；允许高速平移和较大旋转。
        common = [name for name in points if name in previous_points]
        if len(common) >= 3 and np.isfinite(reference_size):
            deltas = np.asarray(
                [
                    np.asarray(points[name], dtype=np.float64)
                    - np.asarray(previous_points[name], dtype=np.float64)
                    for name in common
                ]
            )
            common_motion = np.median(deltas, axis=0)
            residual_limit = max(35.0, 1.15 * reference_size)
            for name, delta in zip(common, deltas):
                if float(np.linalg.norm(delta - common_motion)) > residual_limit:
                    points.pop(name, None)

        stabilized[frame_number] = points
        size = _point_cloud_size(points)
        center = _point_cloud_center(points)
        if len(points) >= 3 and np.isfinite(size) and size > 1.0 and center is not None:
            recent_sizes.append(size)
            recent_centers.append((frame_number, center))
            previous_points = dict(points)

    # 只补 1～2 帧的小缺口；更长缺失保持不画，避免制造虚假的球拍轨迹。
    if max_gap > 0:
        for name in RACKET_POINT_NAMES:
            valid_frames = [frame for frame, points in stabilized.items() if name in points]
            for start, end in zip(valid_frames, valid_frames[1:]):
                gap = end - start - 1
                if gap <= 0 or gap > max_gap:
                    continue
                start_point = np.asarray(stabilized[start][name], dtype=np.float64)
                end_point = np.asarray(stabilized[end][name], dtype=np.float64)
                for offset in range(1, gap + 1):
                    frame_number = start + offset
                    if frame_number not in stabilized or name in stabilized[frame_number]:
                        continue
                    alpha = offset / (gap + 1)
                    point = start_point * (1.0 - alpha) + end_point * alpha
                    stabilized[frame_number][name] = (
                        int(round(float(point[0]))),
                        int(round(float(point[1]))),
                    )
    return stabilized


def _halpe26_points(
    pose_frame: tuple[np.ndarray, np.ndarray],
    threshold: float,
) -> Dict[str, Point]:
    keypoints, scores = pose_frame
    points: Dict[str, Point] = {}
    count = min(len(HALPE26_NAMES), len(keypoints), len(scores))
    for index in range(count):
        x, y = keypoints[index][:2]
        score = scores[index]
        if all(_finite_number(v) for v in (x, y, score)) and float(score) >= threshold:
            points[HALPE26_NAMES[index]] = (int(round(float(x))), int(round(float(y))))
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
    pose_frames: Optional[Sequence[tuple[np.ndarray, np.ndarray]]] = None,
) -> str:
    """在原视频绘制完整 Halpe26 骨架和经过保守稳定的球拍轮廓。"""
    video_path = Path(video_path)
    body_csv = Path(body_csv)
    racket_csv = Path(racket_csv)
    output_path = Path(output_path)
    for path, label in ((video_path, "视频"), (body_csv, "人体 CSV"), (racket_csv, "球拍 CSV")):
        if not path.exists():
            raise FileNotFoundError(f"{label}不存在: {path}")

    body_rows = _row_by_frame(pd.read_csv(body_csv))
    racket_rows = stabilize_racket_visual_points(
        pd.read_csv(racket_csv), threshold=racket_kpt_thr
    )

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
            if pose_frames is not None and frame_number <= len(pose_frames):
                body_points = _halpe26_points(
                    pose_frames[frame_number - 1], body_kpt_thr
                )
                body_connections = HALPE26_CONNECTIONS
            else:
                body_points = _body_points(body_rows.get(frame_number), body_kpt_thr)
                body_connections = BODY_CONNECTIONS
            racket_points = racket_rows.get(frame_number, {})

            # 人体：青色连线 + 黄色关节点；离线入口会绘制完整 Halpe26。
            _draw_connections(frame, body_points, body_connections, (255, 210, 0), 3)
            for point in body_points.values():
                cv2.circle(frame, point, 3, (0, 230, 255), -1, cv2.LINE_AA)

            # 拍面和拍柄统一为洋红色，绿色五点与人体配色明确区分。
            _draw_connections(frame, racket_points, RACKET_FACE_CONNECTIONS, RACKET_COLOR, 3)
            _draw_connections(frame, racket_points, (("b", "h"),), RACKET_COLOR, 4)
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
            cv2.putText(frame, "RACKET", (88, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RACKET_COLOR, 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    if frame_number == 0:
        raise ValueError(f"输入视频没有可读取的帧: {video_path}")
    return str(output_path.resolve())


__all__ = [
    "BODY_CONNECTIONS",
    "RACKET_FACE_CONNECTIONS",
    "render_pose_racket_video",
    "stabilize_racket_visual_points",
]
