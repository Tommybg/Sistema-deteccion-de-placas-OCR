#!/usr/bin/env python3
"""
app_coral_cli.py — Inferencia CLI nativa para hardware Google Coral Edge TPU.

Ejecuta el pipeline ANPR usando los modelos `_edgetpu.tflite` vía PyCoral 
y el OCR en CPU. Todo directo en terminal sin Streamlit.

Dependencias en el Coral:
    sudo apt-get install python3-pycoral
    pip install "fast-plate-ocr[onnx]" opencv-python-headless numpy

Uso:
    python3 app_coral_cli.py ruta/a/imagen.jpg
"""

import sys
import time
import argparse
from pathlib import Path

# --- OpenCV sin UI (headless) es más ligero en la terminal de la Pi/Coral ---
import cv2
import numpy as np

# Intentar importar PyCoral (API nativa para el TPU)
try:
    from pycoral.utils.edgetpu import make_interpreter
    from pycoral.adapters import common
except ImportError:
    print("❌ ERROR: No se encontró 'pycoral'. Asegúrate de estar en el Coral y tenerlo instalado.")
    print("   Instalar con: sudo apt-get install python3-pycoral")
    sys.exit(1)

# --- Rutas de modelos ---
# Asumimos que corres este script desde la carpeta raíz del proyecto anpr_project/
MODELS_DIR = Path(__file__).parent / "models" / "tflite_exports"

# --- Configuración de Modelos ---
# Base names de los modelos INT8
MODEL_BASENAMES = {
    "vehicle": "yolo11n_coco_vehicle_int8.tflite",
    "color":   "color_classifier_int8.tflite",
    "brand":   "marca_detector_yolo11n_int8.tflite",
    "plate":   "placa_detector_yolo11n_int8.tflite"
}

# --- Diccionarios de Clases ---
COLOR_CLASSES = sorted([
    "beige", "black", "blue", "brown", "gold", "green", "grey",
    "orange", "pink", "purple", "red", "silver", "tan", "white", "yellow",
])

BRAND_CLASSES = {
    0: "Acura", 1: "Audi", 2: "BMW", 3: "Chevrolet", 4: "Citroën",
    5: "Dacia", 6: "Fiat", 7: "Ford", 8: "Honda", 9: "Hyundai",
    10: "Infiniti", 11: "KIA", 12: "Lamborghini", 13: "Lexus", 14: "Mazda",
    15: "Mercedes-Benz", 16: "Mitsubishi", 17: "Nissan", 18: "Opel",
    19: "Perodua", 20: "Peugeot", 21: "Porsche", 22: "Proton",
    23: "Renault", 24: "Seat", 25: "Suzuki", 26: "Tesla", 27: "Toyota",
    28: "Volkswagen", 29: "Volvo",
}

VEHICLE_CLASSES = {
    2: "Auto", 3: "Moto", 5: "Bus", 7: "Camión", 1: "Bicicleta",
}

# --- Funciones Core ---
def load_models():
    """Carga los intérpretes. Prioriza archivos _edgetpu.tflite, sino usa .tflite normal."""
    interpreters = {}
    print("\n[INFO] buscando modelos en:", MODELS_DIR)
    
    for key, base_name in MODEL_BASENAMES.items():
        # Generar nombres posibles: el compilado y el crudo
        compiled_name = base_name.replace(".tflite", "_edgetpu.tflite")
        
        path_to_use = None
        is_tpu = False
        
        # 1. Buscar el compilado (Edge TPU)
        for p in [MODELS_DIR / compiled_name, Path(compiled_name)]:
            if p.exists():
                path_to_use = p
                is_tpu = True
                break
        
        # 2. Si no hay compilado, buscar el crudo (INT8 normal)
        if not path_to_use:
            for p in [MODELS_DIR / base_name, Path(base_name)]:
                if p.exists():
                    path_to_use = p
                    is_tpu = False
                    break
        
        if path_to_use:
            try:
                # `make_interpreter` detecta si es edgetpu o normal
                interpreter = make_interpreter(str(path_to_use))
                interpreter.allocate_tensors()
                interpreters[key] = interpreter
                status = "TPU 🚀" if is_tpu else "CPU 🐌"
                print(f"  ✅ {key.upper():<8} : {path_to_use.name} [{status}]")
            except Exception as e:
                print(f"  ❌ Error cargando {key}: {e}")
        else:
            print(f"  ❌ No se encontró el archivo para {key} (ni compilado ni crudo)")

    # Carga OCR
    try:
        from fast_plate_ocr import LicensePlateRecognizer
        interpreters["ocr"] = LicensePlateRecognizer("cct-xs-v1-global-model")
        print("  ✅ OCR cargado en CPU (ONNX)")
    except ImportError:
        print("  ❌ 'fast-plate-ocr' no está instalado.")
        interpreters["ocr"] = None

    return interpreters

def _decode_yolo(interpreter, img_w, img_h, conf_thresh=0.25):
    """
    Decodifica la detección de mayor confianza de un YOLOv11n
    exportado a INT8 TFLite. Identifica si viene crudo o transpuesto.
    """
    output_details = interpreter.get_output_details()[0]
    # Extraer el tensor (suele venir [1, 84, 8400] o transposicionado [1, 8400, 84])
    raw = interpreter.tensor(output_details['index'])()[0].copy()

    # Dequantizar si el output es INT8/UINT8 (necesario en Edge TPU)
    if output_details['dtype'] in (np.int8, np.uint8):
        quant = output_details['quantization']
        scale, zero_point = quant[0], quant[1]
        if scale != 0:
            raw = (raw.astype(np.float32) - zero_point) * scale

    # YOLO TFLite output suele venir transpuesto (features vs num_boxes)
    if raw.shape[0] < raw.shape[-1]:
        raw = raw.T # Ahora queda [num_boxes, 4+clases]
        
    if raw.shape[1] < 5:
        return None, 0.0, None # Formato inesperado
        
    scores = raw[:, 4:]
    max_scores = np.max(scores, axis=1)
    best_idx = int(np.argmax(max_scores))
    conf = float(max_scores[best_idx])
    
    if conf < conf_thresh:
        return None, 0.0, None
        
    class_id = int(np.argmax(scores[best_idx]))
    cx, cy, w, h = raw[best_idx, :4]
    
    # Denormalizar bbox (YOLO11 INT8 devuelve coords 0-1)
    # Verificamos si es float muy grande (> 1.5) asume píxeles directos (cálculo de seguridad)
    coords_max = float(max(abs(cx), abs(cy), abs(w), abs(h)))
    if coords_max <= 1.5:
        cx *= img_w; w *= img_w
        cy *= img_h; h *= img_h
    else:
        # En caso de que viniera en píxeles según el input_size del modelo
        model_size = interpreter.get_input_details()[0]['shape'][1]
        scale_x = img_w / model_size
        scale_y = img_h / model_size
        cx *= scale_x; w *= scale_x
        cy *= scale_y; h *= scale_y

    x1 = max(0, int((cx - w / 2)))
    y1 = max(0, int((cy - h / 2)))
    x2 = min(img_w, int((cx + w / 2)))
    y2 = min(img_h, int((cy + h / 2)))
    
    return class_id, conf, (x1, y1, x2, y2)

def process_frame(interpreters, img_bgr, frame_id=None):
    """Procesa un único frame y devuelve los resultados"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_rgb.shape[:2]
    
    res = {"vehicle": None, "color": None, "brand": None, "plate": None, "ocr": None}
    latencias = {}
    
    # ── 1. Vehículo ──
    vehicle_bbox = None
    if "vehicle" in interpreters:
        interp = interpreters["vehicle"]
        size = common.input_size(interp)
        resized = cv2.resize(img_rgb, size)
        common.set_input(interp, resized)
        
        t0 = time.perf_counter()
        interp.invoke()
        latencias["vehicle"] = (time.perf_counter() - t0) * 1000
        
        cls_id, conf, vehicle_bbox = _decode_yolo(interp, img_w, img_h, conf_thresh=0.25)
        if cls_id is not None:
            res["vehicle"] = (VEHICLE_CLASSES.get(cls_id, "Auto"), conf)

    # ── 2. Color ──
    if "color" in interpreters:
        interp = interpreters["color"]
        size = common.input_size(interp)
        crop = img_rgb[vehicle_bbox[1]:vehicle_bbox[3], vehicle_bbox[0]:vehicle_bbox[2]] if vehicle_bbox else img_rgb
        if crop.size > 0:
            resized = cv2.resize(crop, size)
            common.set_input(interp, resized)
            t0 = time.perf_counter()
            interp.invoke()
            latencias["color"] = (time.perf_counter() - t0) * 1000
            
            color_out_details = interp.get_output_details()[0]
            out_flat = interp.tensor(color_out_details['index'])()[0].flatten().copy()
            # Dequantizar si es INT8/UINT8
            if color_out_details['dtype'] in (np.int8, np.uint8):
                cq = color_out_details['quantization']
                if cq[0] != 0:
                    out_flat = (out_flat.astype(np.float32) - cq[1]) * cq[0]
            out_flat = out_flat.astype(np.float32)
            probs = np.exp(out_flat - np.max(out_flat)); probs /= probs.sum()
            top_idx = int(np.argmax(probs))
            res["color"] = (COLOR_CLASSES[top_idx], probs[top_idx])

    # ── 3. Marca ──
    if "brand" in interpreters:
        interp = interpreters["brand"]
        resized = cv2.resize(img_rgb, common.input_size(interp))
        common.set_input(interp, resized)
        t0 = time.perf_counter()
        interp.invoke()
        latencias["brand"] = (time.perf_counter() - t0) * 1000
        cls_id, conf, _ = _decode_yolo(interp, img_w, img_h, conf_thresh=0.2)
        if cls_id is not None:
            res["brand"] = (BRAND_CLASSES.get(cls_id, "Desconocido"), conf)

    # ── 4. Placa ──
    plate_bbox = None
    if "plate" in interpreters:
        interp = interpreters["plate"]
        resized = cv2.resize(img_rgb, common.input_size(interp))
        common.set_input(interp, resized)
        t0 = time.perf_counter()
        interp.invoke()
        latencias["plate"] = (time.perf_counter() - t0) * 1000
        _, conf, plate_bbox = _decode_yolo(interp, img_w, img_h, conf_thresh=0.25)
        if plate_bbox:
            res["plate"] = (plate_bbox, conf)

    # ── 5. OCR ──
    if interpreters.get("ocr"):
        crop = img_rgb[plate_bbox[1]:plate_bbox[3], plate_bbox[0]:plate_bbox[2]] if plate_bbox else \
               img_rgb[int(img_h*0.5):, int(img_w*0.2):int(img_w*0.8)]
        if crop.size > 0:
            t0 = time.perf_counter()
            txts, confs = interpreters["ocr"].run(crop, return_confidence=True)
            latencias["ocr"] = (time.perf_counter() - t0) * 1000
            if txts:
                res["ocr"] = (txts[0], float(confs.mean()))

    # Imprimir resumen de línea (compacto para video)
    v = f"{res['vehicle'][0]} {res['vehicle'][1]:.0%}" if res["vehicle"] else "---"
    c = f"{res['color'][0]}" if res["color"] else "---"
    b = f"{res['brand'][0]}" if res["brand"] else "---"
    p = f"{res['ocr'][0]}" if res["ocr"] else (f"BBox:{res['plate'][1]:.0%}" if res['plate'] else "---")
    
    total_ms = sum(latencias.values())
    hdr = f"Frame {frame_id:04d} | " if frame_id is not None else ""
    print(f"{hdr}V:{v} | C:{c:<7} | B:{b:<10} | P:{p:<8} | {total_ms:>5.1f}ms")
    
    return res, latencias

def run_pipeline(interpreters, source):
    """Detecta si es imagen, video o cámara y procesa"""
    # Intentar cargar como cámara (si es un número)
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        is_video = True
    else:
        path = Path(source)
        if not path.exists():
            print(f"❌ Error: {source} no existe.")
            return

        # Si es directorio, procesar todas las imágenes
        if path.is_dir():
            IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
            images = sorted([f for f in path.iterdir() if f.suffix.lower() in IMG_EXTS])
            if not images:
                print(f"❌ No se encontraron imágenes en {source}")
                return
            print(f"\n[INFO] Procesando DIRECTORIO: {source} ({len(images)} imágenes)")
            print("-" * 75)
            all_latencias = []
            for i, img_path in enumerate(images):
                img = cv2.imread(str(img_path))
                if img is not None:
                    _, lat = process_frame(interpreters, img, frame_id=i)
                    all_latencias.append(sum(lat.values()))
            if all_latencias:
                print("-" * 75)
                avg = sum(all_latencias) / len(all_latencias)
                print(f"[RESUMEN] {len(all_latencias)} imágenes | Promedio: {avg:.1f}ms | "
                      f"Min: {min(all_latencias):.1f}ms | Max: {max(all_latencias):.1f}ms")
            return

        # Check extensión
        if path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
            is_video = False
        else:
            cap = cv2.VideoCapture(source)
            is_video = True

    if is_video:
        if not cap.isOpened():
            print(f"❌ Error abriendo video o cámara: {source}")
            return
        
        print(f"\n[INFO] Iniciando procesamiento de VIDEO: {source}")
        print("-" * 75)
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                process_frame(interpreters, frame, frame_id=frame_count)
                frame_count += 1
                # En Raspberry Pi, puedes descomentar esto para ver el video si tienes monitor:
                # cv2.imshow('ANPR Coral', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'): break
        except KeyboardInterrupt:
            print("\n[INFO] Detenido por el usuario.")
        finally:
            cap.release()
            # cv2.destroyAllWindows()
    else:
        img = cv2.imread(source)
        if img is not None:
            print(f"\n[INFO] Procesando IMAGEN: {source}")
            print("-" * 75)
            process_frame(interpreters, img)
        else:
            print(f"❌ Error decodificando imagen: {source}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ANPR nativo para Google Coral (Imagen/Video/Cámara)")
    parser.add_argument("source", help="Ruta a imagen, video o índice de cámara (ej: 0)")
    args = parser.parse_args()
    
    interpreters = load_models()
    run_pipeline(interpreters, args.source)
