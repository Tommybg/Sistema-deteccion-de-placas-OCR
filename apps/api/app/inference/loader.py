"""Singleton loader for AI models. Called once at FastAPI startup.

Mirrors the @st.cache_resource pattern from app_demo.py:74-123 but works
correctly under concurrent requests because models are loaded into
process-global state, not per-session state.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config import settings

# Ensure the repo root is on sys.path so `scripts.*` imports resolve,
# both when running from source and inside the Docker image.
if str(settings.REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(settings.REPO_ROOT))

log = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    plate_model: object
    vehicle_detector: object
    brand_detector: Optional[object]
    color_classifier: Optional[object]
    ocr: Optional[object]
    plate_model_path: Path
    vehicle_model_path: Path
    brand_model_path: Optional[Path]
    color_model_path: Optional[Path]


def _resolve(model_dir: Path, filename: str) -> Optional[Path]:
    p = model_dir / filename
    return p if p.exists() else None


def load_models() -> ModelBundle:
    from ultralytics import YOLO
    from scripts.vehicle_detector import VehicleDetector
    from scripts.brand_detector import BrandDetector
    from scripts.color_classifier import ColorClassifier

    model_dir = settings.MODEL_DIR

    plate_path = _resolve(model_dir, settings.PLATE_MODEL)
    if plate_path is None:
        raise RuntimeError(
            f"Plate detector not found at {model_dir / settings.PLATE_MODEL}"
        )

    log.info("Loading plate detector from %s", plate_path)
    plate_model = YOLO(str(plate_path))

    color_classifier = None
    color_path = None
    if settings.USE_TFLITE_COLOR:
        color_path = _resolve(model_dir, settings.COLOR_TFLITE)
        if color_path is not None:
            try:
                log.info("Loading TFLite color classifier from %s", color_path)
                color_classifier = ColorClassifier(
                    model_path=str(color_path), use_tflite=True
                )
            except Exception as e:
                log.warning("TFLite color classifier failed to load: %s", e)
                color_classifier = None

    if color_classifier is None:
        color_path = _resolve(model_dir, settings.COLOR_MODEL)
        if color_path is not None:
            try:
                log.info("Loading Keras color classifier from %s", color_path)
                color_classifier = ColorClassifier(
                    model_path=str(color_path), use_tflite=False
                )
            except Exception as e:
                log.warning("Color classifier unavailable: %s", e)
                color_classifier = None

    vehicle_path = settings.SCRIPTS_DIR / settings.VEHICLE_MODEL
    log.info("Loading vehicle detector from %s", vehicle_path)
    vehicle_detector = VehicleDetector(
        model_path=str(vehicle_path),
        device=settings.INFERENCE_DEVICE,
        color_classifier=color_classifier,
    )

    brand_detector = None
    brand_path = _resolve(model_dir, settings.BRAND_MODEL)
    if brand_path is not None:
        log.info("Loading brand detector from %s", brand_path)
        brand_detector = BrandDetector(
            model_path=str(brand_path), device=settings.INFERENCE_DEVICE
        )

    ocr = None
    try:
        from fast_plate_ocr import LicensePlateRecognizer
        log.info("Loading OCR model %s", settings.OCR_MODEL)
        ocr = LicensePlateRecognizer(settings.OCR_MODEL)
    except Exception as e:
        log.warning("OCR unavailable: %s", e)

    return ModelBundle(
        plate_model=plate_model,
        vehicle_detector=vehicle_detector,
        brand_detector=brand_detector,
        color_classifier=color_classifier,
        ocr=ocr,
        plate_model_path=plate_path,
        vehicle_model_path=vehicle_path,
        brand_model_path=brand_path,
        color_model_path=color_path,
    )
