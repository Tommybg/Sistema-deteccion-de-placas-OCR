"""
Vehicle type detection module using COCO-pretrained YOLOv11n.
Detects: Automóvil (car), Motocicleta (motorcycle), Bus, Camión (truck).
"""

from pathlib import Path
from ultralytics import YOLO

# COCO class IDs for vehicles
VEHICLE_CLASSES = {2: "Automóvil", 3: "Motocicleta", 5: "Bus", 7: "Camión"}
VEHICLE_CLASSES_EN = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASS_IDS = set(VEHICLE_CLASSES.keys())

DEFAULT_MODEL_PATH = Path(__file__).parent / "yolo11n.pt"


class VehicleDetector:
    def __init__(self, model_path=None, confidence=0.4, device="cpu", color_classifier=None):
        self.model_path = str(model_path or DEFAULT_MODEL_PATH)
        self.confidence = confidence
        self.device = device
        self.model = YOLO(self.model_path)
        self.color_classifier = color_classifier

    def detect(self, frame):
        """Run vehicle detection on a frame.

        Returns list of dicts with keys: bbox, type, type_en, confidence, class_id
        """
        results = self.model(frame, conf=self.confidence, device=self.device, verbose=False)
        vehicles = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in VEHICLE_CLASS_IDS:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    vehicle = {
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "type": VEHICLE_CLASSES[cls_id],
                        "type_en": VEHICLE_CLASSES_EN[cls_id],
                        "confidence": float(box.conf[0]),
                        "class_id": cls_id,
                        "color": None,
                        "color_confidence": None,
                    }
                    vehicles.append(vehicle)

        # Classify color for each vehicle if classifier is available
        if self.color_classifier is not None:
            h, w = frame.shape[:2]
            for v in vehicles:
                x1, y1, x2, y2 = v["bbox"]
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if crop.size > 0:
                    result = self.color_classifier.classify(crop)
                    v["color"] = result["color"]
                    v["color_confidence"] = result["confidence"]

        return vehicles

    @staticmethod
    def associate_plate_to_vehicle(plate_bbox, vehicles):
        """Find the vehicle that contains a plate bounding box.

        Uses plate center-point containment. If multiple vehicles contain the
        plate, picks the smallest (tightest fit). Falls back to IoU > 0.1.
        """
        if not vehicles:
            return None

        px1, py1, px2, py2 = plate_bbox
        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2

        # Try center-point containment
        containing = []
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v["bbox"]
            if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                area = (vx2 - vx1) * (vy2 - vy1)
                containing.append((area, v))

        if containing:
            containing.sort(key=lambda x: x[0])
            return containing[0][1]

        # Fallback: IoU
        plate_area = (px2 - px1) * (py2 - py1)
        best_iou = 0
        best_vehicle = None
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v["bbox"]
            ix1 = max(px1, vx1)
            iy1 = max(py1, vy1)
            ix2 = min(px2, vx2)
            iy2 = min(py2, vy2)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                vehicle_area = (vx2 - vx1) * (vy2 - vy1)
                union = plate_area + vehicle_area - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_vehicle = v

        return best_vehicle if best_iou > 0.1 else None
