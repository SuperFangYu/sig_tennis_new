"""截击动作 API。"""

from backend.routers.action_router import create_action_router
from backend.services.pipeline import run_volley_pipeline

router = create_action_router("volley", run_volley_pipeline)
