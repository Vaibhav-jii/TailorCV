"""
StoreIQ Configuration — Centralized settings with environment variable support.

All thresholds, paths, and model configs are managed here.
Override via .env file or environment variables.
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Global application settings."""

    # ── Paths ──────────────────────────────────────────────
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    VIDEOS_DIR: Path = DATA_DIR / "videos"
    OUTPUT_DIR: Path = DATA_DIR / "output"
    DB_PATH: Path = DATA_DIR / "store_intelligence.db"
    ZONES_CONFIG: Path = PROJECT_ROOT / "config" / "zones.json"

    # ── YOLO Model ─────────────────────────────────────────
    YOLO_MODEL: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.3
    IOU_THRESHOLD: float = 0.5

    # ── ByteTrack Tracking ─────────────────────────────────
    TRACK_ACTIVATION_THRESHOLD: float = 0.25
    LOST_TRACK_BUFFER: int = 60       # frames before a lost track is removed
    MATCH_THRESHOLD: float = 0.4
    FRAME_RATE: int = 5               # tracking frame rate

    # ── Processing ─────────────────────────────────────────
    PROCESS_FPS: int = 5              # target processing FPS (skip frames to achieve)

    # ── Event Thresholds ───────────────────────────────────
    DWELL_THRESHOLD_SECONDS: float = 5.0     # minimum dwell to emit ZONE_DWELL event
    QUEUE_THRESHOLD_SECONDS: float = 30.0    # billing dwell to infer purchase
    REENTRY_WINDOW_SECONDS: float = 300.0    # 5 min window for re-entry matching
    LOST_TRACK_TIMEOUT_FRAMES: int = 30      # frames of absence before EXIT event

    # ── API ────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # ── Dashboard ──────────────────────────────────────────
    DASHBOARD_PORT: int = 8501

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton instance
settings = Settings()

# Ensure required directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
