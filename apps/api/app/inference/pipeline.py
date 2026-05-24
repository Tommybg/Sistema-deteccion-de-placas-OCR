"""Inference pipeline adapter.

Calls the existing detector classes from scripts/* in the same order
as `detect_plates()` in app_demo.py:130-216, then returns a typed
Pydantic DetectionResult. No model logic is reimplemented here — this
is a thin adapter that turns mutated lists into immutable JSON.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.inference.loader import ModelBundle
from app.schemas.detection import DetectionResult, Plate, Vehicle


# Single-flight lock: YOLO/torch aren't thread-safe under the GIL when
# called concurrently against the same model instance. Serialize per
# bundle to keep behaviour deterministic.
_inference_lock = asyncio.Lock()


def _read_plate_text(ocr, plate_image: np.ndarray) -> Optional[str]:
    if ocr is None or plate_image is None or plate_image.size == 0:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cv2.imwrite(tmp_path, plate_image)
            result = ocr.run(tmp_path)
            if not result:
                return None
            value = result[0] if isinstance(result, (list, tuple)) else result
            text = str(value).strip()
            return text or None
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
    except Exception:
        return None


def _run_sync(
    bundle: ModelBundle,
    image_bgr: np.ndarray,
    confidence: float,
    source: str,
) -> DetectionResult:
    from scripts.vehicle_detector import VehicleDetector
    from scripts.brand_detector import BrandDetector

    start = time.perf_counter()
    height, width = image_bgr.shape[:2]

    raw_vehicles = bundle.vehicle_detector.detect(image_bgr)
    vehicle_index_by_id: dict[int, int] = {
        id(v): idx for idx, v in enumerate(raw_vehicles)
    }

    brand_detections = []
    vehicle_brands: dict[int, tuple[str, float]] = {}
    if bundle.brand_detector is not None:
        brand_detections = bundle.brand_detector.detect(image_bgr)
        for bd in brand_detections:
            matched = BrandDetector.associate_brand_to_vehicle(bd["bbox"], raw_vehicles)
            if matched is None:
                continue
            v_id = id(matched)
            existing = vehicle_brands.get(v_id)
            if existing is None or bd["confidence"] > existing[1]:
                vehicle_brands[v_id] = (bd["brand"], bd["confidence"])

    plate_results = bundle.plate_model(
        image_bgr, conf=confidence, verbose=False, device="cpu"
    )

    plates: list[Plate] = []
    for box in plate_results[0].boxes:
        coords = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        conf = float(box.conf[0])

        plate_crop = image_bgr[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
        plate_text = _read_plate_text(bundle.ocr, plate_crop)

        vehicle_id = None
        vehicle_type = None
        vehicle_color = None
        vehicle_color_confidence = None
        brand = None

        if raw_vehicles:
            match = VehicleDetector.associate_plate_to_vehicle((x1, y1, x2, y2), raw_vehicles)
            if match is not None:
                vehicle_id = vehicle_index_by_id[id(match)]
                vehicle_type = match.get("type")
                vehicle_color = match.get("color")
                vehicle_color_confidence = match.get("color_confidence")
                if id(match) in vehicle_brands:
                    brand = vehicle_brands[id(match)][0]

        plates.append(
            Plate(
                bbox=(x1, y1, x2, y2),
                text=plate_text,
                confidence=conf,
                vehicle_id=vehicle_id,
                vehicle_type=vehicle_type,
                vehicle_color=vehicle_color,
                vehicle_color_confidence=vehicle_color_confidence,
                brand=brand,
            )
        )

    vehicles_out: list[Vehicle] = []
    for idx, v in enumerate(raw_vehicles):
        brand_tuple = vehicle_brands.get(id(v))
        vehicles_out.append(
            Vehicle(
                id=idx,
                bbox=tuple(v["bbox"]),
                type=v["type"],
                type_en=v.get("type_en", ""),
                type_confidence=float(v["confidence"]),
                color=v.get("color"),
                color_en=None,
                color_confidence=v.get("color_confidence"),
                brand=brand_tuple[0] if brand_tuple else None,
                brand_confidence=brand_tuple[1] if brand_tuple else None,
            )
        )

    latency_ms = int((time.perf_counter() - start) * 1000)

    return DetectionResult(
        id=uuid.uuid4(),
        timestamp=datetime.now(timezone.utc),
        image_width=int(width),
        image_height=int(height),
        confidence_used=float(confidence),
        latency_ms=latency_ms,
        vehicles=vehicles_out,
        plates=plates,
        source=source,
    )


async def run(
    bundle: ModelBundle,
    image_bgr: np.ndarray,
    confidence: float,
    source: str = "upload",
) -> DetectionResult:
    """Run the inference pipeline; serialized via an asyncio lock.

    The lock keeps YOLO/torch model calls single-flight per process,
    preventing CUDA/CPU contention and matching the implicit
    serialization in Streamlit's single-script-execution model.
    """
    async with _inference_lock:
        return await asyncio.to_thread(_run_sync, bundle, image_bgr, confidence, source)
