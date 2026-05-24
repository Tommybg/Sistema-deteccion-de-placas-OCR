from __future__ import annotations

import hashlib
import io
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Detection
from app.db.session import get_session
from app.inference import pipeline
from app.schemas.detection import DetectionResult

router = APIRouter(prefix="/detect", tags=["detect"])

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp"}


async def _read_image(upload: UploadFile) -> tuple[np.ndarray, bytes]:
    if upload.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {upload.content_type}")

    raw = await upload.read()
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Image exceeds {settings.MAX_UPLOAD_MB} MB limit")

    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not decode image") from exc

    arr_rgb = np.array(pil)
    arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    return arr_bgr, raw


def _best(items, key: str) -> Optional[str]:
    best_val = None
    best_conf = -1.0
    for item in items:
        val = getattr(item, key, None)
        conf = getattr(item, f"{key}_confidence", None) or getattr(item, "confidence", None) or 0.0
        if val and (conf or 0) > best_conf:
            best_val = val
            best_conf = conf or 0.0
    return best_val


async def _persist(
    session: AsyncSession,
    result: DetectionResult,
    image_bytes: bytes,
) -> None:
    sha = hashlib.sha256(image_bytes).hexdigest()
    plate_text = next((p.text for p in result.plates if p.text), None)
    brand = _best(result.vehicles, "brand")
    color = _best(result.vehicles, "color")

    row = Detection(
        id=str(result.id),
        created_at=result.timestamp.replace(tzinfo=None),
        source=result.source,
        image_sha256=sha,
        image_width=result.image_width,
        image_height=result.image_height,
        latency_ms=result.latency_ms,
        confidence_used=result.confidence_used,
        plates=[p.model_dump(mode="json") for p in result.plates],
        vehicles=[v.model_dump(mode="json") for v in result.vehicles],
        plate_text=plate_text,
        brand=brand,
        color=color,
    )
    session.add(row)
    await session.commit()


@router.post("", response_model=DetectionResult)
async def detect(
    request: Request,
    image: UploadFile = File(...),
    confidence: float = Form(default=settings.CONFIDENCE_DEFAULT),
    source: str = Form(default="upload"),
    session: AsyncSession = Depends(get_session),
) -> DetectionResult:
    bundle = getattr(request.app.state, "models", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if not 0.05 <= confidence <= 0.99:
        raise HTTPException(status_code=400, detail="confidence must be between 0.05 and 0.99")

    image_bgr, raw = await _read_image(image)
    result = await pipeline.run(bundle, image_bgr, confidence, source=source)
    await _persist(session, result, raw)
    return result


@router.post("/batch", response_model=List[DetectionResult])
async def detect_batch(
    request: Request,
    images: List[UploadFile] = File(...),
    confidence: float = Form(default=settings.CONFIDENCE_DEFAULT),
    session: AsyncSession = Depends(get_session),
) -> List[DetectionResult]:
    bundle = getattr(request.app.state, "models", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if len(images) > 20:
        raise HTTPException(status_code=400, detail="Batch limited to 20 images")

    results: list[DetectionResult] = []
    for upload in images:
        image_bgr, raw = await _read_image(upload)
        result = await pipeline.run(bundle, image_bgr, confidence, source="batch")
        await _persist(session, result, raw)
        results.append(result)
    return results
