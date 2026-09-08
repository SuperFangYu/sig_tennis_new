"""正手动作 API。"""

from backend.routers.action_router import create_action_router
from backend.services.pipeline import run_forehand_pipeline

router = create_action_router("forehand", run_forehand_pipeline)
