from fastapi import APIRouter, Request

from app.config import settings
from app.schemas.detection import HealthResponse, ModelInfo, ModelsResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    bundle = getattr(request.app.state, "models", None)
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        models_ready=bundle is not None,
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models(request: Request) -> ModelsResponse:
    bundle = getattr(request.app.state, "models", None)
    models: list[ModelInfo] = []

    if bundle is None:
        return ModelsResponse(device=settings.INFERENCE_DEVICE, models=models)

    models.append(
        ModelInfo(
            name="Plate detector",
            version="YOLOv11n",
            type="object-detection",
            classes=1,
            file=bundle.plate_model_path.name,
            loaded=True,
        )
    )
    models.append(
        ModelInfo(
            name="Vehicle detector",
            version="YOLOv11n (COCO)",
            type="object-detection",
            classes=4,
            file=bundle.vehicle_model_path.name,
            loaded=True,
        )
    )
    if bundle.brand_detector is not None and bundle.brand_model_path is not None:
        models.append(
            ModelInfo(
                name="Brand detector",
                version="YOLOv11n",
                type="object-detection",
                classes=30,
                file=bundle.brand_model_path.name,
                loaded=True,
            )
        )
    if bundle.color_classifier is not None and bundle.color_model_path is not None:
        models.append(
            ModelInfo(
                name="Color classifier",
                version="EfficientNetB0" if not settings.USE_TFLITE_COLOR else "EfficientNetB0 INT8 TFLite",
                type="image-classification",
                classes=15,
                file=bundle.color_model_path.name,
                loaded=True,
            )
        )
    if bundle.ocr is not None:
        models.append(
            ModelInfo(
                name="OCR",
                version=settings.OCR_MODEL,
                type="ocr",
                classes=0,
                file=settings.OCR_MODEL,
                loaded=True,
            )
        )

    return ModelsResponse(device=settings.INFERENCE_DEVICE, models=models)
