from pathlib import Path
from typing import List

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Default repo root: works for both layouts.
#   Source: apps/api/app/config.py        → parents[3] is the repo root.
#   Docker: /app/app/config.py            → parents[1] is /app (repo root in
#           the container, where scripts/, models/, samples/ are copied).
# Overridable via REPO_ROOT env var. Don't raise IndexError on shallow paths.
def _detect_repo_root() -> Path:
    here = Path(__file__).resolve()
    for idx in (3, 1):
        if idx < len(here.parents):
            candidate = here.parents[idx]
            if (candidate / "scripts").exists() or (candidate / "models").exists():
                return candidate
    # Last resort: highest parent we have
    return here.parents[min(len(here.parents) - 1, 3)]


_DEFAULT_REPO_ROOT = _detect_repo_root()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "ANPR Inference API"
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: str = "INFO"

    REPO_ROOT: Path = _DEFAULT_REPO_ROOT
    MODEL_DIR: Path = _DEFAULT_REPO_ROOT / "models"
    SCRIPTS_DIR: Path = _DEFAULT_REPO_ROOT / "scripts"
    SAMPLES_DIR: Path = _DEFAULT_REPO_ROOT / "samples"

    PLATE_MODEL: str = "placa_detector_yolo11n.pt"
    BRAND_MODEL: str = "marca_detector_yolo11n.pt"
    VEHICLE_MODEL: str = "yolo11n.pt"
    COLOR_MODEL: str = "color_classifier_efficientnet.h5"
    COLOR_TFLITE: str = "tflite_exports/color_classifier_int8.tflite"

    USE_TFLITE_COLOR: bool = True
    INFERENCE_DEVICE: str = "cpu"
    CONFIDENCE_DEFAULT: float = 0.5
    OCR_MODEL: str = "cct-xs-v1-global-model"

    DATABASE_URL: str = ""

    FRONTEND_URL: str = "http://localhost:3000"
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    MAX_UPLOAD_MB: int = 8
    RATE_LIMIT_DETECT: str = "100/minute"

    @model_validator(mode="after")
    def _derive_paths(self) -> "Settings":
        # When REPO_ROOT is overridden (e.g. /app in Docker), re-derive
        # MODEL_DIR / SCRIPTS_DIR / SAMPLES_DIR unless explicitly set.
        if self.MODEL_DIR == _DEFAULT_REPO_ROOT / "models":
            self.MODEL_DIR = self.REPO_ROOT / "models"
        if self.SCRIPTS_DIR == _DEFAULT_REPO_ROOT / "scripts":
            self.SCRIPTS_DIR = self.REPO_ROOT / "scripts"
        if self.SAMPLES_DIR == _DEFAULT_REPO_ROOT / "samples":
            self.SAMPLES_DIR = self.REPO_ROOT / "samples"
        if not self.DATABASE_URL:
            self.DATABASE_URL = f"sqlite+aiosqlite:///{self.REPO_ROOT / 'apps' / 'api' / 'anpr.db'}"
        return self

    @field_validator("DATABASE_URL", mode="after")
    @classmethod
    def _coerce_async_driver(cls, value: str) -> str:
        # Railway exposes Postgres as `postgresql://...`; SQLAlchemy
        # needs `postgresql+asyncpg://...`. Translate transparently.
        if value.startswith("postgresql://"):
            return "postgresql+asyncpg://" + value[len("postgresql://"):]
        if value.startswith("postgres://"):
            return "postgresql+asyncpg://" + value[len("postgres://"):]
        return value


settings = Settings()
