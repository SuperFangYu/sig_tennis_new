"""发球动作 API。"""

from backend.routers.action_router import create_action_router
from backend.services.pipeline import run_serve_pipeline

router = create_action_router("serve", run_serve_pipeline)
