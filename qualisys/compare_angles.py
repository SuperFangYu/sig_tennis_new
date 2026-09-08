"""阶段 3：按人工事件锚点对齐两套八角曲线并计算初步一致性指标。

输入是阶段 1/2 的 CSV 与两张人工事件表；输出是对齐明细、角度指标、事件误差和质控表。
同一次重复动作的八个关节严格共用一个时间映射。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from qualisys.config import ANGLE_COLUMNS, EVENT_GUIDES, EVENT_NAMES
from qualisys.trials import Trial, load_trials, select_trials

EVENT_COLUMNS = ("repetition", "event", "time")


def write_event_templates(trial: Trial) -> tuple[Path, Path]:
    """在中间目录创建不覆盖现有内容的人工事件模板。"""
    trial.ensure_output_dirs()
    paths = (
        trial.intermediate_dir / "video_events.csv",
        trial.intermediate_dir / "qualisys_events.csv",
    )
    guide = EVENT_GUIDES[trial.action]
    template = pd.DataFrame(
        [(1, event, "", guide[event]) for event in EVENT_NAMES],
        columns=(*EVENT_COLUMNS, "definition"),
    )
    for path in paths:
        if not path.exists():
            template.to_csv(path, index=False, encoding="utf-8-sig")
    return paths


def _read_angles(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {label} 角度文件: {path}")
    data = pd.read_csv(path)
    required = {"frame", "time", *ANGLE_COLUMNS}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{label} 角度文件缺少列: {', '.join(sorted(missing))}")
    data["time"] = pd.to_numeric(data["time"], errors="coerce")
    return data


def _read_events(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {label} 事件文件: {path}")
    data = pd.read_csv(path)
    missing = set(EVENT_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"{label} 事件文件缺少列: {', '.join(sorted(missing))}")
    data["repetition"] = pd.to_numeric(data["repetition"], errors="coerce")
    data["time"] = pd.to_numeric(data["time"], errors="coerce")
    if data[["repetition", "time"]].isna().any().any():
        raise ValueError(f"{label} 事件时间尚未填写完整: {path}")
    repetition_values = data["repetition"].to_numpy(dtype=np.float64)
    if np.any(repetition_values < 1) or np.any(repetition_values != np.floor(repetition_values)):
        raise ValueError(f"{label} repetition 必须是从 1 开始的正整数")
    invalid = set(data["event"].astype(str)) - set(EVENT_NAMES)
    if invalid:
        raise ValueError(f"{label} 事件名无效: {', '.join(sorted(invalid))}")
    duplicates = data.duplicated(["repetition", "event"], keep=False)
    if duplicates.any():
        raise ValueError(f"{label} 同一重复动作存在重复事件: {path}")
    for repetition, part in data.groupby("repetition"):
        event_time = dict(zip(part["event"], part["time"]))
        if all(event in event_time for event in EVENT_NAMES):
            if not event_time["start"] < event_time["contact"] < event_time["end"]:
                raise ValueError(
                    f"{label} repetition {int(repetition)} 必须满足 start < contact < end"
                )
    return data


def _fit_time_map(video_events: pd.DataFrame, qtm_events: pd.DataFrame) -> dict[str, float]:
    merged = video_events.merge(qtm_events, on=["repetition", "event"], suffixes=("_video", "_qtm"))
    if len(merged) < 2:
        raise ValueError("每次动作至少需要两个共同事件才能拟合时间映射")
    video_time = merged["time_video"].to_numpy(dtype=np.float64)
    qtm_time = merged["time_qtm"].to_numpy(dtype=np.float64)
    if np.ptp(video_time) <= 0:
        raise ValueError("视频事件时间不能全部相同")
    slope, intercept = np.polyfit(video_time, qtm_time, 1)
    if slope <= 0:
        raise ValueError("事件配对得到非正时间斜率，请检查 repetition 和事件时间")
    predicted = slope * video_time + intercept
    residual = qtm_time - predicted
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "anchor_count": int(len(merged)),
        "anchor_rmse_seconds": float(np.sqrt(np.mean(np.square(residual)))),
    }


def _interpolate(reference: pd.DataFrame, column: str, target_time: np.ndarray) -> np.ndarray:
    source_time = reference["time"].to_numpy(dtype=np.float64)
    source_value = pd.to_numeric(reference[column], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(source_time) & np.isfinite(source_value)
    output = np.full(len(target_time), np.nan, dtype=np.float64)
    if np.count_nonzero(valid) < 2:
        return output
    x = source_time[valid]
    y = source_value[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    inside = (target_time >= x[0]) & (target_time <= x[-1])
    output[inside] = np.interp(target_time[inside], x, y)
    return output


def _metric_row(repetition: str, angle: str, estimate: Iterable[float], reference: Iterable[float]) -> dict[str, object]:
    estimate_array = np.asarray(list(estimate), dtype=np.float64)
    reference_array = np.asarray(list(reference), dtype=np.float64)
    valid = np.isfinite(estimate_array) & np.isfinite(reference_array)
    x = estimate_array[valid]
    y = reference_array[valid]
    row: dict[str, object] = {"repetition": repetition, "angle": angle, "n": int(len(x))}
    if len(x) == 0:
        return row
    error = x - y
    bias = float(np.mean(error))
    sd_error = float(np.std(error, ddof=1)) if len(error) > 1 else float("nan")
    row.update(
        {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "bias": bias,
            "sd_error": sd_error,
            "loa_lower": bias - 1.96 * sd_error,
            "loa_upper": bias + 1.96 * sd_error,
            "within_5deg_percent": float(np.mean(np.abs(error) <= 5.0) * 100.0),
            "within_10deg_percent": float(np.mean(np.abs(error) <= 10.0) * 100.0),
        }
    )
    if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
        row["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
        covariance = float(np.mean((x - np.mean(x)) * (y - np.mean(y))))
        denominator = float(np.var(x) + np.var(y) + (np.mean(x) - np.mean(y)) ** 2)
        row["ccc"] = float(2.0 * covariance / denominator) if denominator > 0 else float("nan")
    else:
        row["pearson_r"] = float("nan")
        row["ccc"] = float("nan")
    return row


def compare_trial(trial: Trial) -> dict[str, str]:
    """按每次动作的一组共享时间映射对齐全部八角并输出 CSV。"""
    trial.ensure_output_dirs()
    rtmpose = _read_angles(trial.intermediate_dir / "rtmpose_8_angles.csv", "RTMPose")
    qualisys = _read_angles(trial.intermediate_dir / "qualisys_8_angles.csv", "Qualisys")
    video_event_path, qtm_event_path = write_event_templates(trial)
    video_events = _read_events(video_event_path, "视频")
    qtm_events = _read_events(qtm_event_path, "Qualisys")

    common_repetitions = sorted(
        set(video_events["repetition"].astype(int)) & set(qtm_events["repetition"].astype(int))
    )
    if not common_repetitions:
        raise ValueError(f"试次 {trial.trial_id} 没有可配对的 repetition")

    aligned_parts: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for repetition in common_repetitions:
        ve = video_events[video_events["repetition"] == repetition]
        qe = qtm_events[qtm_events["repetition"] == repetition]
        time_map = _fit_time_map(ve, qe)
        video_lookup = dict(zip(ve["event"], ve["time"]))
        qtm_lookup = dict(zip(qe["event"], qe["time"]))
        if "start" not in video_lookup or "end" not in video_lookup:
            raise ValueError(f"repetition {repetition} 的视频事件必须包含 start 和 end")

        segment = rtmpose[
            (rtmpose["time"] >= video_lookup["start"])
            & (rtmpose["time"] <= video_lookup["end"])
        ].copy()
        mapped_time = (
            time_map["slope"] * segment["time"].to_numpy(dtype=np.float64)
            + time_map["intercept"]
        )
        aligned = pd.DataFrame(
            {
                "repetition": repetition,
                "video_frame": segment["frame"].to_numpy(),
                "video_time": segment["time"].to_numpy(),
                "qualisys_time": mapped_time,
            }
        )
        for angle in ANGLE_COLUMNS:
            aligned[f"{angle}_rtmpose"] = pd.to_numeric(segment[angle], errors="coerce").to_numpy()
            aligned[f"{angle}_qualisys"] = _interpolate(qualisys, angle, mapped_time)
            aligned[f"{angle}_error"] = aligned[f"{angle}_rtmpose"] - aligned[f"{angle}_qualisys"]
        aligned_parts.append(aligned)

        qc_rows.append(
            {
                "repetition": repetition,
                **time_map,
                "video_start": video_lookup["start"],
                "video_end": video_lookup["end"],
                "qualisys_start": qtm_lookup.get("start", np.nan),
                "qualisys_end": qtm_lookup.get("end", np.nan),
                "aligned_frame_count": len(aligned),
            }
        )
        common_events = sorted(set(video_lookup) & set(qtm_lookup), key=EVENT_NAMES.index)
        for event in common_events:
            event_row: dict[str, object] = {"repetition": repetition, "event": event}
            for angle in ANGLE_COLUMNS:
                video_value = _interpolate(rtmpose, angle, np.array([video_lookup[event]]))[0]
                qtm_value = _interpolate(qualisys, angle, np.array([qtm_lookup[event]]))[0]
                event_row[f"{angle}_error"] = video_value - qtm_value
            event_rows.append(event_row)

    aligned_all = pd.concat(aligned_parts, ignore_index=True)
    metric_rows: list[dict[str, object]] = []
    for repetition in common_repetitions:
        part = aligned_all[aligned_all["repetition"] == repetition]
        for angle in ANGLE_COLUMNS:
            metric_rows.append(
                _metric_row(
                    str(repetition),
                    angle,
                    part[f"{angle}_rtmpose"],
                    part[f"{angle}_qualisys"],
                )
            )
    for angle in ANGLE_COLUMNS:
        metric_rows.append(
            _metric_row(
                "all",
                angle,
                aligned_all[f"{angle}_rtmpose"],
                aligned_all[f"{angle}_qualisys"],
            )
        )

    outputs = {
        "aligned_angles": trial.output_dir / "aligned_angles.csv",
        "angle_metrics": trial.output_dir / "angle_metrics.csv",
        "event_metrics": trial.output_dir / "event_metrics.csv",
        "alignment_qc": trial.output_dir / "alignment_qc.csv",
    }
    aligned_all.to_csv(outputs["aligned_angles"], index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(outputs["angle_metrics"], index=False, encoding="utf-8-sig")
    pd.DataFrame(event_rows).to_csv(outputs["event_metrics"], index=False, encoding="utf-8-sig")
    pd.DataFrame(qc_rows).to_csv(outputs["alignment_qc"], index=False, encoding="utf-8-sig")
    return {key: str(path.resolve()) for key, path in outputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="按人工事件对齐并比较 RTMPose 与 Qualisys 八角。")
    parser.add_argument("--trial", action="append", dest="trial_ids", help="只处理指定 trial_id，可重复")
    parser.add_argument(
        "--prepare-events",
        action="store_true",
        help="只创建 video_events.csv 与 qualisys_events.csv 模板，不执行比较",
    )
    args = parser.parse_args()

    trials = select_trials(load_trials(validate_files=False), args.trial_ids)
    for trial in trials:
        if args.prepare_events:
            paths = write_event_templates(trial)
            print(f"[{trial.trial_id}] 事件模板: {paths[0]} / {paths[1]}")
            continue
        outputs = compare_trial(trial)
        print(f"[{trial.trial_id}] {outputs['angle_metrics']}")


if __name__ == "__main__":
    main()
