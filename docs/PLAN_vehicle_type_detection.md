# Plan: Add Vehicle Type Detection to ANPR System

## Goal
Add vehicle type detection (car, motorcycle, bus, truck) to the existing ANPR system using the COCO-pretrained `yolo11n.pt` model already present in `scripts/yolo11n.pt`. No new training or datasets required. All changes must maintain Coral Edge TPU compatibility.

## Context
- **Current system:** YOLOv11n custom-trained on single class `placa` (license plate) → OCR reads plate text
- **What we're adding:** A second YOLOv11n model (COCO pretrained, already in `scripts/yolo11n.pt`) that detects vehicles and classifies them by type
- **COCO vehicle class IDs:** `2=car`, `3=motorcycle`, `5=bus`, `7=truck`
- **Architecture:** Dual-model pipeline — detect vehicles first, then detect plates, then associate each plate to its parent vehicle

## Files to Create

### 1. `scripts/vehicle_detector.py` (NEW)
Create a shared, reusable module for vehicle detection:

```python
# Key components:
# - VEHICLE_CLASSES dict: {2: "Automóvil", 3: "Motocicleta", 5: "Bus", 7: "Camión"}
# - VEHICLE_CLASSES_EN dict: {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# - VehicleDetector class:
#     __init__(self, model_path, confidence=0.4, device="cpu")
#       - Loads yolo11n.pt (COCO pretrained)
#     detect(self, frame) -> list[dict]
#       - Runs inference, filters to vehicle classes only
#       - Returns: [{"bbox": (x1,y1,x2,y2), "type": "Automóvil", "type_en": "car", "confidence": 0.87}, ...]
#     associate_plate_to_vehicle(self, plate_bbox, vehicles) -> dict|None
#       - Given a plate bounding box and list of detected vehicles,
#       - Find the vehicle whose bbox contains the plate (or has highest IoU overlap)
#       - Return the matching vehicle dict, or None if no match
```

The COCO pretrained model path should default to: `Path(__file__).parent / "yolo11n.pt"` (it already lives in `scripts/`).

## Files to Modify

### 2. `app_demo.py`
- Import `VehicleDetector` from `scripts/vehicle_detector.py`
- Add `@st.cache_resource` function `load_vehicle_detector()` that loads the COCO model
- Add a new function `detect_vehicles(model, image_np)` that returns vehicle detections
- In `main()`:
  - After loading detector and ocr, also load vehicle detector
  - When processing an image:
    1. Run vehicle detection first → get vehicle list with types
    2. Run plate detection (existing code)
    3. For each detected plate, call `associate_plate_to_vehicle()` to find its parent vehicle
  - In the annotated image (col2 "Detecciones"):
    - Draw **blue** bounding boxes around vehicles with type label (e.g., "Automóvil 92%")
    - Keep existing **green** boxes for plates
  - In the plate result cards section:
    - Below each plate text and confidence, add a line showing vehicle type: `🚗 Tipo: Automóvil`
  - Update statistics section: add a metric for "Vehículos Detectados"
  - Update sidebar model info to mention both models

### 3. `app_cloud.py`
Apply the exact same changes as `app_demo.py`. This file has the same structure — it's the Railway cloud version. The only difference is it uses `samples/` instead of `dataset_combinado/test/` for test images and doesn't have webcam support. The vehicle detection changes are identical.

### 4. `scripts/04_inferencia_tiempo_real.py`
- Import `VehicleDetector` from `vehicle_detector.py`
- Update `ANPRSystem.__init__()`:
  - Add optional `vehicle_detector: VehicleDetector = None` parameter
  - Store it as `self.vehicle_detector`
- Update `ANPRSystem.process_frame()`:
  - If vehicle_detector is present, run vehicle detection on the frame first
  - After plate detection, associate each plate to a vehicle
  - Include `vehicle_type` in the results dict for each detection
- Update `ANPRSystem._annotate_frame()`:
  - Add method or logic to draw **blue** vehicle bounding boxes with type labels
  - Keep **green** boxes for plates as-is
- Update `_init_log_file()` and `_log_plate()`:
  - Add `vehicle_type` column to CSV header and log entries
- Update `parse_args()`:
  - Add `--vehicle-model` flag (default: `scripts/yolo11n.pt`)
  - Add `--no-vehicle-detection` flag to disable vehicle detection
- Update `main()`:
  - Initialize VehicleDetector if vehicle detection is enabled
  - Pass it to ANPRSystem
- Add color constant: `COLOR_VEHICLE_BBOX = (255, 165, 0)` (orange/blue in BGR)

### 5. `scripts/03_exportar_tflite.py`
- Add `--coco` CLI flag
- When `--coco` is passed, also load and export `scripts/yolo11n.pt` to TFLite INT8
- Save the exported COCO model alongside the plate model in `models/tflite_exports/` with name like `yolo11n_coco_vehicle_int8.tflite`

### 6. `README.md`
- Add a new section "🚗 Detección de Tipo de Vehículo" explaining:
  - The dual-model pipeline (vehicle detection + plate detection + OCR)
  - Supported vehicle types: Automóvil, Motocicleta, Bus, Camión
  - That it uses the COCO pretrained model, no custom training needed
- Update the project structure tree to include `scripts/vehicle_detector.py`
- Add CLI usage examples showing `--vehicle-model` and `--no-vehicle-detection` flags

## Important Implementation Notes

1. **Model loading:** The COCO model is `scripts/yolo11n.pt`. Do NOT re-download it — it already exists in the project.
2. **Bounding box colors:** Vehicles = **blue** `(255, 0, 0)` in BGR. Plates stay **green** `(0, 255, 0)`.
3. **Association logic:** A plate belongs to a vehicle if the plate bbox center point falls inside the vehicle bbox. If multiple vehicles contain the plate, pick the smallest (tightest fit). If no vehicle contains the plate, try IoU > 0.1.
4. **Performance:** Run both models sequentially on each frame. On MPS/CUDA this adds negligible latency (~5ms). On CPU it may add ~30-50ms.
5. **No new pip dependencies.** Everything uses `ultralytics` which is already installed.
6. **Spanish labels** in the UI: Automóvil, Motocicleta, Bus, Camión.

## Verification
After implementing, run:
```bash
cd /Users/tommygoat/Desktop/claude_cowork/placas/anpr_project
source anpr_env/bin/activate
streamlit run app_demo.py
```
Then select a sample image from "📂 Dataset de prueba" and verify:
- Blue vehicle bounding boxes appear with type labels
- Green plate bounding boxes still appear
- Each plate card shows the associated vehicle type
- Stats section shows vehicle count
