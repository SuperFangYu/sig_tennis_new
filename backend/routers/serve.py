import mimetypes
import re
from datetime import datetime
from pathlib import Path
from typing import Any, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.schemas import AnalyzeResponse, ArtifactItem, ArtifactsListResponse
from backend.services.pipeline import REPO_ROOT, build_artifact_list, run_serve_pipeline

router = APIRouter(tags=["serve"])

OUTPUTS_ROOT = (REPO_ROOT / "data" / "outputs").resolve()
INPUTS_ROOT = (REPO_ROOT / "data" / "inputs").resolve()


def _safe_output_file(relative_path: str) -> Path:
    if ".." in relative_path:
        raise HTTPException(status_code=400, detail="非法路径")
    rel = Path(relative_path.replace("\\", "/").lstrip("/"))
    if rel.is_absolute():
        raise HTTPException(status_code=400, detail="非法路径")
    full = (OUTPUTS_ROOT / rel).resolve()
    try:
        full.relative_to(OUTPUTS_ROOT)
    except ValueError as e:
        raise HTTPException(status_code=403, detail="路径必须在 data/outputs 下") from e
    return full


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
    stem = Path(file.filename).stem
    safe_stem = re.sub(r"[^\w\-_.\u4e00-\u9fff]", "_", stem) or "video"
    ext = Path(file.filename).suffix or ".mp4"
    in_dir = INPUTS_ROOT / run_id
    in_dir.mkdir(parents=True, exist_ok=True)
    dest = in_dir / f"{safe_stem}{ext}"

    content = await file.read()
    dest.write_bytes(content)

    try:
        result = run_serve_pipeline(dest, run_id, handedness=handedness)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    items = build_artifact_list(result)
    return AnalyzeResponse(
        run_id=result["run_id"],
        video_name=result["video_name"],
        artifacts=[ArtifactItem(**x) for x in items],
        intervals=result.get("intervals") or [],
    )


@router.get("/artifacts/{run_id}", response_model=ArtifactsListResponse)
def list_artifacts(run_id: str) -> ArtifactsListResponse:
    base = OUTPUTS_ROOT / run_id
    if not base.is_dir():
        raise HTTPException(404, "未找到该次运行")

    jobs: List[dict[str, Any]] = []
    for sub in sorted(base.iterdir(), key=lambda p: p.name):
        if not sub.is_dir():
            continue
        video_name = sub.name
        arts: List[dict[str, str]] = []
        for f in sorted(sub.iterdir()):
            if not f.is_file():
                continue
            rel = f.relative_to(OUTPUTS_ROOT).as_posix()
            lower = f.name.lower()
            if lower.endswith("_backend.csv") and "kpt" in lower:
                kind = "racket_csv"
            elif lower.endswith("_body_serve_rtmpose.csv"):
                kind = "body_csv"
            elif lower.endswith("_plot.png") and "kpt" in lower:
                kind = "racket_chart"
            elif lower.endswith("_serve_trace_chart.png"):
                kind = "serve_trace_chart"
            elif lower.endswith("_serve_kinetic_chart.png"):
                kind = "serve_kinetic_chart"
            elif lower.endswith(".mp4") and "serve" in lower:
                kind = "clip"
            elif "kinematic_summary" in lower and lower.endswith(".csv") and "phase" not in lower:
                kind = "kinematic_summary_csv"
            elif "kinematic_phase" in lower and lower.endswith(".csv"):
                kind = "kinematic_phase_summary_csv"
            elif "upper_limb_angles" in lower and lower.endswith(".png"):
                kind = "upper_limb_angle_chart"
            elif "lower_limb_angles" in lower and lower.endswith(".png"):
                kind = "lower_limb_angle_chart"
            elif "trunk_rotation" in lower and lower.endswith(".png"):
                kind = "trunk_rotation_chart"
            elif "racket_kinematics" in lower and lower.endswith(".png"):
                kind = "racket_kinematic_chart"
            else:
                kind = "other"
            arts.append({"kind": kind, "filename": f.name, "relative_path": rel})
        jobs.append({"video_name": video_name, "artifacts": arts})

    return ArtifactsListResponse(run_id=run_id, jobs=jobs)


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
