"""四类动作共享的上传、产物查询和安全下载路由。"""

from __future__ import annotations

import mimetypes
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.schemas import AnalyzeResponse, ArtifactItem, ArtifactsListResponse
from backend.services.pipeline import REPO_ROOT, build_artifact_list

PipelineRunner = Callable[[Path, str, str], Dict[str, Any]]

OUTPUTS_ROOT = (REPO_ROOT / "data" / "outputs").resolve()
INPUTS_ROOT = (REPO_ROOT / "data" / "inputs").resolve()


def _safe_output_file(relative_path: str) -> Path:
    if ".." in relative_path:
        raise HTTPException(status_code=400, detail="非法路径")
    relative = Path(relative_path.replace("\\", "/").lstrip("/"))
    if relative.is_absolute():
        raise HTTPException(status_code=400, detail="非法路径")
    full = (OUTPUTS_ROOT / relative).resolve()
    try:
        full.relative_to(OUTPUTS_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="路径必须在 data/outputs 下") from exc
    return full


def classify_artifact(action: str, path: Path) -> str:
    """按稳定文件命名规则识别历史运行目录中的产物类型。"""
    lower = path.name.lower()

    if action == "volley" and "_volley_trace_chart_" in lower:
        return "volley_trace_chart"
    if action == "volley" and "_volley_kinetic_chart_" in lower:
        return "volley_kinetic_chart"
    if lower.endswith("_serve_trace_chart.png"):
        return "serve_trace_chart"
    if lower.endswith("_serve_kinetic_chart.png"):
        return "serve_kinetic_chart"
    if lower.endswith("_backend.csv") and "kpt" in lower:
        return "racket_csv"
    if lower.endswith(f"_body_{action}_rtmpose.csv"):
        return "body_csv"
    if lower.endswith("_plot.png") and "kpt" in lower:
        return "racket_chart"
    if lower.endswith("_final_analysis.png"):
        return "final_chart"
    if lower.endswith(".mp4") and action in lower:
        return "clip"
    if "_kinetic_chain_" in lower and lower.endswith(".png"):
        return "kinetic_chart"
    if "_speed_cog_" in lower and lower.endswith(".png"):
        return "speed_cog_chart"
    if "kinematic_summary" in lower and lower.endswith(".csv") and "phase" not in lower:
        return "kinematic_summary_csv"
    if "kinematic_phase" in lower and lower.endswith(".csv"):
        return "kinematic_phase_summary_csv"
    if "upper_limb_angles" in lower and lower.endswith(".png"):
        return "upper_limb_angle_chart"
    if "lower_limb_angles" in lower and lower.endswith(".png"):
        return "lower_limb_angle_chart"
    if "trunk_rotation" in lower and lower.endswith(".png"):
        return "trunk_rotation_chart"
    if "racket_kinematics" in lower and lower.endswith(".png"):
        return "racket_kinematic_chart"
    return "other"


def _list_run_artifacts(action: str, run_id: str) -> ArtifactsListResponse:
    base = OUTPUTS_ROOT / run_id
    if not base.is_dir():
        raise HTTPException(404, "未找到该次运行")

    jobs: List[dict[str, Any]] = []
    for subdirectory in sorted(base.iterdir(), key=lambda value: value.name):
        if not subdirectory.is_dir():
            continue
        artifacts: List[dict[str, str]] = []
        for path in sorted(subdirectory.iterdir()):
            if not path.is_file():
                continue
            artifacts.append(
                {
                    "kind": classify_artifact(action, path),
                    "filename": path.name,
                    "relative_path": path.relative_to(OUTPUTS_ROOT).as_posix(),
                }
            )
        jobs.append({"video_name": subdirectory.name, "artifacts": artifacts})
    return ArtifactsListResponse(run_id=run_id, jobs=jobs)


def create_action_router(action: str, pipeline_runner: PipelineRunner) -> APIRouter:
    """创建动作路由，同时保持原有 URL 和响应结构不变。"""
    router = APIRouter(tags=[action])

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.post("/analyze", response_model=AnalyzeResponse)
    async def analyze(
        file: UploadFile = File(...),
        handedness: str = Form("right"),
    ) -> AnalyzeResponse:
        if not file.filename:
            raise HTTPException(400, "缺少文件名")

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_path = Path(file.filename)
        safe_stem = re.sub(r"[^\w\-_.\u4e00-\u9fff]", "_", source_path.stem) or "video"
        extension = source_path.suffix or ".mp4"
        input_directory = INPUTS_ROOT / run_id
        input_directory.mkdir(parents=True, exist_ok=True)
        destination = input_directory / f"{safe_stem}{extension}"
        destination.write_bytes(await file.read())

        try:
            result = pipeline_runner(destination, run_id, handedness)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        items = build_artifact_list(result)
        return AnalyzeResponse(
            run_id=result["run_id"],
            video_name=result["video_name"],
            artifacts=[ArtifactItem(**item) for item in items],
            intervals=result.get("intervals") or [],
        )

    @router.get("/artifacts/{run_id}", response_model=ArtifactsListResponse)
    def list_artifacts(run_id: str) -> ArtifactsListResponse:
        return _list_run_artifacts(action, run_id)

    @router.get("/file")
    def download_file(path: str) -> FileResponse:
        full = _safe_output_file(path)
        if not full.is_file():
            raise HTTPException(404, "文件不存在")
        media_type, _ = mimetypes.guess_type(full.name)
        return FileResponse(
            full,
            filename=full.name,
            media_type=media_type or "application/octet-stream",
        )

    return router
