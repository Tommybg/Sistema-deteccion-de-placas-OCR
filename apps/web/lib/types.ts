// Frontend types — mirror the FastAPI Pydantic schemas in
// apps/api/app/schemas/detection.py. Regenerate from /api/openapi.json
// via `pnpm gen:types` when the backend contract changes.

export type BBox = [number, number, number, number];

export interface Vehicle {
  id: number;
  bbox: BBox;
  type: string;
  type_en: string;
  type_confidence: number;
  color: string | null;
  color_en: string | null;
  color_confidence: number | null;
  brand: string | null;
  brand_confidence: number | null;
}

export interface Plate {
  bbox: BBox;
  text: string | null;
  confidence: number;
  vehicle_id: number | null;
  vehicle_type: string | null;
  vehicle_color: string | null;
  vehicle_color_confidence: number | null;
  brand: string | null;
}

export interface DetectionResult {
  id: string;
  timestamp: string;
  image_width: number;
  image_height: number;
  confidence_used: number;
  latency_ms: number;
  vehicles: Vehicle[];
  plates: Plate[];
  source: "upload" | "webcam" | "sample" | "batch" | string;
}

export interface DetectionListItem {
  id: string;
  created_at: string;
  source: string;
  plate_text: string | null;
  brand: string | null;
  color: string | null;
  plate_count: number;
  vehicle_count: number;
  latency_ms: number;
}

export interface DetectionList {
  items: DetectionListItem[];
  next_cursor: string | null;
  total: number;
}

export interface ModelInfo {
  name: string;
  version: string;
  type: string;
  classes: number;
  file: string;
  loaded: boolean;
}

export interface ModelsResponse {
  device: string;
  models: ModelInfo[];
}

export interface SampleInfo {
  name: string;
  url: string;
  size_bytes: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  models_ready: boolean;
}

export type DetectionSource = "upload" | "webcam" | "sample" | "batch";
