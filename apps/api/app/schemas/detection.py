from datetime import datetime
from typing import List, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel, Field


BBox = Tuple[int, int, int, int]


class Vehicle(BaseModel):
    id: int = Field(..., description="Index within the detection result")
    bbox: BBox
    type: str
    type_en: str
    type_confidence: float
    color: Optional[str] = None
    color_en: Optional[str] = None
    color_confidence: Optional[float] = None
    brand: Optional[str] = None
    brand_confidence: Optional[float] = None


class Plate(BaseModel):
    bbox: BBox
    text: Optional[str] = None
    confidence: float
    vehicle_id: Optional[int] = None
    vehicle_type: Optional[str] = None
    vehicle_color: Optional[str] = None
    vehicle_color_confidence: Optional[float] = None
    brand: Optional[str] = None


class DetectionResult(BaseModel):
    id: UUID
    timestamp: datetime
    image_width: int
    image_height: int
    confidence_used: float
    latency_ms: int
    vehicles: List[Vehicle]
    plates: List[Plate]
    source: str = "upload"


class DetectionListItem(BaseModel):
    id: UUID
    created_at: datetime
    source: str
    plate_text: Optional[str] = None
    brand: Optional[str] = None
    color: Optional[str] = None
    plate_count: int
    vehicle_count: int
    latency_ms: int


class DetectionList(BaseModel):
    items: List[DetectionListItem]
    next_cursor: Optional[str] = None
    total: int


class ModelInfo(BaseModel):
    name: str
    version: str
    type: str
    classes: int
    file: str
    loaded: bool


class ModelsResponse(BaseModel):
    device: str
    models: List[ModelInfo]


class SampleInfo(BaseModel):
    name: str
    url: str
    size_bytes: int


class HealthResponse(BaseModel):
    status: str
    version: str
    models_ready: bool
