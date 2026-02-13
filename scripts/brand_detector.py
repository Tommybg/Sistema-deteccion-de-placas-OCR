"""
Vehicle brand logo detection module using custom-trained YOLOv11n.
Detects 30 car brand logos: from Acura to Volvo.

Usage:
    from brand_detector import BrandDetector
    detector = BrandDetector()
    brands = detector.detect(frame)
"""

from pathlib import Path
from ultralytics import YOLO

# 30 brand classes (unified scheme, alphabetical order)
BRAND_CLASSES = {
    0: "Acura", 1: "Audi", 2: "BMW", 3: "Chevrolet", 4: "Citroen",
    5: "Dacia", 6: "Fiat", 7: "Ford", 8: "Honda", 9: "Hyundai",
    10: "Infiniti", 11: "KIA", 12: "Lamborghini", 13: "Lexus", 14: "Mazda",
    15: "MercedesBenz", 16: "Mitsubishi", 17: "Nissan", 18: "Opel",
    19: "Perodua", 20: "Peugeot", 21: "Porsche", 22: "Proton",
    23: "Renault", 24: "Seat", 25: "Suzuki", 26: "Tesla", 27: "Toyota",
    28: "Volkswagen", 29: "Volvo",
}

# Colombian-relevant brands (17 of 30)
COLOMBIAN_BRANDS = {
    1,   # Audi
    2,   # BMW
    3,   # Chevrolet
    6,   # Fiat
    7,   # Ford
    8,   # Honda
    9,   # Hyundai
    11,  # KIA
    14,  # Mazda
    15,  # MercedesBenz
    16,  # Mitsubishi
    17,  # Nissan
    23,  # Renault
    25,  # Suzuki
    27,  # Toyota
    28,  # Volkswagen
    29,  # Volvo
}

DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "marca_detector_yolo11n.pt"


class BrandDetector:
    def __init__(self, model_path=None, confidence=0.3, device="cpu",
                 colombian_only=False):
        """
        Args:
            model_path: Path to the brand logo YOLO model
            confidence: Detection confidence threshold
            device: Inference device
            colombian_only: If True, filter to Colombian-relevant brands only
        """
        self.model_path = str(model_path or DEFAULT_MODEL_PATH)
        self.confidence = confidence
        self.device = device
        self.colombian_only = colombian_only
        self.model = YOLO(self.model_path)

    def detect(self, frame):
        """Run brand logo detection on a frame.

        Returns list of dicts with keys:
            bbox: (x1, y1, x2, y2)
            brand: str (brand name)
            confidence: float
            class_id: int
        """
        results = self.model(frame, conf=self.confidence,
                             device=self.device, verbose=False)
        brands = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in BRAND_CLASSES:
                    continue
                if self.colombian_only and cls_id not in COLOMBIAN_BRANDS:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                brands.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "brand": BRAND_CLASSES[cls_id],
                    "confidence": float(box.conf[0]),
                    "class_id": cls_id,
                })
        return brands

    @staticmethod
    def associate_brand_to_vehicle(brand_bbox, vehicles):
        """Find the vehicle that contains a brand logo bounding box.

        Uses center-point containment. If multiple vehicles contain the
        logo, picks the smallest (tightest fit). Falls back to IoU > 0.05.
        """
        if not vehicles:
            return None

        bx1, by1, bx2, by2 = brand_bbox
        bcx = (bx1 + bx2) / 2
        bcy = (by1 + by2) / 2

        # Try center-point containment
        containing = []
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v["bbox"]
            if vx1 <= bcx <= vx2 and vy1 <= bcy <= vy2:
                area = (vx2 - vx1) * (vy2 - vy1)
                containing.append((area, v))

        if containing:
            containing.sort(key=lambda x: x[0])
            return containing[0][1]

        # Fallback: IoU (lower threshold since logos are small)
        brand_area = (bx2 - bx1) * (by2 - by1)
        best_iou = 0
        best_vehicle = None
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v["bbox"]
            ix1 = max(bx1, vx1)
            iy1 = max(by1, vy1)
            ix2 = min(bx2, vx2)
            iy2 = min(by2, vy2)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                vehicle_area = (vx2 - vx1) * (vy2 - vy1)
                union = brand_area + vehicle_area - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_vehicle = v

        return best_vehicle if best_iou > 0.05 else None
