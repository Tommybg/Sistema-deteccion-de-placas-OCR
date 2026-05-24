from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings
from app.schemas.detection import SampleInfo

router = APIRouter(prefix="/samples", tags=["samples"])

_ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_dir() -> List[Path]:
    if not settings.SAMPLES_DIR.exists():
        return []
    return sorted(
        p for p in settings.SAMPLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in _ALLOWED
    )


@router.get("", response_model=List[SampleInfo])
async def list_samples() -> List[SampleInfo]:
    return [
        SampleInfo(
            name=p.name,
            url=f"/api/samples/{p.name}",
            size_bytes=p.stat().st_size,
        )
        for p in _list_dir()
    ]


@router.get("/{name}")
async def get_sample(name: str):
    safe_name = Path(name).name  # strip any directory components
    file_path = settings.SAMPLES_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Sample not found")
    if file_path.suffix.lower() not in _ALLOWED:
        raise HTTPException(status_code=415, detail="Unsupported sample format")
    return FileResponse(file_path)
