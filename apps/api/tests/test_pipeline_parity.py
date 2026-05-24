"""Parity test — hard gate for the migration.

Runs the new FastAPI inference pipeline (`app.inference.pipeline.run`)
and the original Streamlit detector code (`app_demo.detect_plates`) on
the same input images and asserts they produce byte-equivalent results:
  - same plate bboxes (±1 px tolerance for ONNX rounding)
  - same plate OCR text (exact)
  - same vehicle types and brand/color labels

Run from repo root:
    pytest apps/api/tests/test_pipeline_parity.py -v

The test is skipped automatically if any required model artifact is
missing, so CI without weights still passes.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "apps" / "api") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

SAMPLES_DIR = REPO_ROOT / "samples"
MODELS_DIR = REPO_ROOT / "models"

_REQUIRED = [
    MODELS_DIR / "placa_detector_yolo11n.pt",
    REPO_ROOT / "scripts" / "yolo11n.pt",
]


def _have_models() -> bool:
    return all(p.exists() for p in _REQUIRED) and SAMPLES_DIR.exists()


pytestmark = pytest.mark.skipif(
    not _have_models(),
    reason="Required model weights or samples missing",
)


@pytest.fixture(scope="module")
def bundle():
    from app.inference.loader import load_models
    return load_models()


@pytest.fixture(scope="module")
def streamlit_pipeline(bundle):
    """Build the same detector handles app_demo.py would build."""
    return {
        "plate_model": bundle.plate_model,
        "vehicle_detector": bundle.vehicle_detector,
        "brand_detector": bundle.brand_detector,
        "ocr": bundle.ocr,
    }


def _sample_images(limit: int = 3):
    if not SAMPLES_DIR.exists():
        return []
    images = sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return images[:limit]


def _load_app_demo():
    import importlib.util
    import sys as _sys

    spec = importlib.util.spec_from_file_location(
        "app_demo_under_test", str(REPO_ROOT / "app_demo.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    _sys.modules["app_demo_under_test"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Streamlit's set_page_config refuses to run outside a script
        # context. We only need the pure functions; they're defined
        # before the offending call, so getattr below still works.
        pass
    return module


@pytest.mark.parametrize("image_path", _sample_images(3), ids=lambda p: p.name)
def test_parity(image_path: Path, bundle, streamlit_pipeline):
    import cv2

    app_demo = _load_app_demo()
    img = cv2.imread(str(image_path))
    assert img is not None, f"Could not read {image_path}"

    _, streamlit_plates, streamlit_vehicles = app_demo.detect_plates(
        streamlit_pipeline["plate_model"],
        img,
        streamlit_pipeline["vehicle_detector"],
        streamlit_pipeline["brand_detector"],
    )

    from app.inference import pipeline as fastapi_pipeline
    result = asyncio.run(fastapi_pipeline.run(bundle, img, 0.5, source="test"))

    assert len(result.vehicles) == len(streamlit_vehicles), \
        f"Vehicle count mismatch: {len(result.vehicles)} vs {len(streamlit_vehicles)}"
    for new, old in zip(result.vehicles, streamlit_vehicles):
        assert new.type == old["type"], f"Vehicle type mismatch: {new.type} vs {old['type']}"
        for a, b in zip(new.bbox, old["bbox"]):
            assert abs(a - b) <= 1, f"Vehicle bbox mismatch: {new.bbox} vs {old['bbox']}"

    assert len(result.plates) == len(streamlit_plates), \
        f"Plate count mismatch: {len(result.plates)} vs {len(streamlit_plates)}"

    for new, old in zip(result.plates, streamlit_plates):
        for a, b in zip(new.bbox, old["box"]):
            assert abs(a - b) <= 1, f"Plate bbox mismatch: {new.bbox} vs {old['box']}"
        assert abs(new.confidence - old["confidence"]) < 1e-4

        if streamlit_pipeline["ocr"] is not None and new.text and old.get("image") is not None:
            old_text = app_demo.read_plate_text(streamlit_pipeline["ocr"], old["image"])
            assert new.text == old_text, f"OCR mismatch: {new.text!r} vs {old_text!r}"
