from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routers import backhand, forehand, serve, volley

REPO_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="sig_tennis_new")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forehand.router, prefix="/api/forehand")
app.include_router(backhand.router, prefix="/api/backhand")
app.include_router(serve.router, prefix="/api/serve")
app.include_router(volley.router, prefix="/api/volley")
app.mount("/", StaticFiles(directory=str(REPO_ROOT / "Vue"), html=True), name="vue")
