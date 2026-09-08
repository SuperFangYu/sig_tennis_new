from typing import List

from pydantic import BaseModel, Field


class ArtifactItem(BaseModel):
    kind: str = Field(
        ...,
        description=(
            "产物类型：racket_csv | body_csv | racket_chart | final_chart | clip | "
            "kinetic_chart | speed_cog_chart | serve_trace_chart | serve_kinetic_chart | "
            "volley_trace_chart | volley_kinetic_chart | kinematic_summary_csv | "
            "kinematic_phase_summary_csv | upper_limb_angle_chart | lower_limb_angle_chart | "
            "trunk_rotation_chart | racket_kinematic_chart"
        ),
    )
    filename: str
    relative_path: str = Field(..., description="相对 data/outputs 的路径，用于 /file 下载")


class AnalyzeResponse(BaseModel):
    run_id: str
    video_name: str
    message: str = "ok"
    artifacts: List[ArtifactItem]
    intervals: List[List[float]] = Field(default_factory=list)


class ArtifactsListResponse(BaseModel):
    run_id: str
    jobs: List[dict]
