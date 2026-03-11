#!/usr/bin/env python3
"""
================================================================================
SCRIPT 08: INFERENCIA ANPR CON CORAL EDGE TPU (REAL O SIMULADO)
================================================================================
Pipeline ANPR completo corriendo sobre Coral Edge TPU o simulado vía TFLite CPU.

Modelos utilizados:
  1. Vehicle/Plate detector  → yolo11n_coco_vehicle_int8[_edgetpu].tflite
  2. Plate detector          → placa_detector_yolo11n_int8[_edgetpu].tflite
  3. Color classifier        → color_classifier_int8[_edgetpu].tflite
  4. Brand detector          → marca_detector_yolo11n_int8[_edgetpu].tflite

Modos:
  --device auto    → usa Coral si está conectado, CPU si no (default)
  --device coral   → fuerza Coral real (falla si no hay hardware)
  --device cpu     → fuerza simulación CPU (TFLite fallback)

Sobre simuladores de instrucciones:
  Google ofrece dos simuladores a nivel de instrucciones RISC-V del chip:
    • MPACT-CoralNPU: https://github.com/google-coral/coralnpu-mpact
    • Verilator:      https://github.com/google-coral/coralnpu
  Éstos aceptan .elf/.bin y son para desarrollo de hardware/compiladores.
  Este script usa TFLite CPU fallback (CoralSimulator), que produce resultados
  idénticos y es la opción correcta para desarrollo ML en Python.

Uso:
    python scripts/08_inferencia_coral.py --imagen samples/car.jpg
    python scripts/08_inferencia_coral.py --imagen samples/car.jpg --device cpu
    python scripts/08_inferencia_coral.py --benchmark
    python scripts/08_inferencia_coral.py --webcam
================================================================================
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Rutas del proyecto ────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR  = PROJECT_DIR / "models" / "tflite_exports"
SAMPLES_DIR = PROJECT_DIR / "samples"
sys.path.insert(0, str(Path(__file__).parent))

from coral_simulator import CoralInterpreter, benchmark as coral_benchmark

# ── Colores de log ────────────────────────────────────────────────────────────
G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; C = "\033[96m"; B = "\033[1m"; N = "\033[0m"

# ── Clases de color ────────────────────────────────────────────────────────────
COLOR_CLASSES = sorted([
    "beige", "black", "blue", "brown", "gold", "green", "grey",
    "orange", "pink", "purple", "red", "silver", "tan", "white", "yellow",
])
COLOR_ES = {
    "beige":"Beige","black":"Negro","blue":"Azul","brown":"Café","gold":"Dorado",
    "green":"Verde","grey":"Gris","orange":"Naranja","pink":"Rosa","purple":"Morado",
    "red":"Rojo","silver":"Plata","tan":"Canela","white":"Blanco","yellow":"Amarillo",
}

# ── Paleta para dibujo ────────────────────────────────────────────────────────
COLORS_BGR = {
    "vehicle": (255, 140, 0),
    "plate":   (0, 230, 118),
    "info":    (33, 150, 243),
}


# ─── Resolución de modelos ────────────────────────────────────────────────────

def resolver_modelo(nombre_base: str) -> Path | None:
    """
    Busca la versión _edgetpu primero, luego la _int8 normal.
    Retorna None si no encuentra ninguna.
    """
    stem = nombre_base.replace(".tflite", "")
    candidatos = [
        MODELS_DIR / f"{stem}_edgetpu.tflite",
        MODELS_DIR / f"{stem}.tflite",
        MODELS_DIR / f"{stem}_int8.tflite",
    ]
    for c in candidatos:
        if c.exists():
            return c
    return None


def cargar_modelos(device: str, verbose: bool) -> dict[str, CoralInterpreter | None]:
    """Carga los 4 modelos del pipeline ANPR."""
    modelos_bases = {
        "vehicle": "yolo11n_coco_vehicle_int8",
        "plate":   "placa_detector_yolo11n_int8",
        "color":   "color_classifier_int8",
        "brand":   "marca_detector_yolo11n_int8",
    }

    loaded = {}
    print(f"\n{C}📦 Cargando modelos ({device} backend)...{N}")
    for key, base in modelos_bases.items():
        path = resolver_modelo(base)
        if path is None:
            suffix = " ← ejecuta 07_exportar_marca_int8.py" if key == "brand" else ""
            print(f"   {Y}⚠️  {key:8s}: no encontrado{suffix}{N}")
            loaded[key] = None
        else:
            is_edgetpu = "_edgetpu" in path.name
            tag = "🪸 edgetpu" if is_edgetpu else "INT8"
            try:
                loaded[key] = CoralInterpreter(path, device=device, verbose=False)
                print(f"   {G}✅ {key:8s}: {path.name}  [{tag}]{N}")
            except Exception as e:
                print(f"   {R}❌ {key:8s}: {e}{N}")
                loaded[key] = None

    return loaded


# ─── YOLO post-proceso ────────────────────────────────────────────────────────

def yolo_postprocess(
    output: np.ndarray,
    orig_shape: tuple[int, int],
    model_size: int = 640,
    conf_thresh: float = 0.3,
    iou_thresh: float = 0.45,
) -> list[dict]:
    """
    Decodifica salida de YOLOv8/11 tiny (1×N×85 o 1×84×N) a detecciones.

    Returns list of dicts: {box:[x1,y1,x2,y2], class_id:int, conf:float}
    """
    if output.ndim == 3:
        output = output[0]  # quitar batch dim

    # Ultralytics exporta (1, 4+num_cls, anchors) — transponer si necesario
    if output.shape[0] < output.shape[1]:
        output = output.T          # → (anchors, 4+num_cls)

    if output.shape[1] < 5:
        return []

    boxes_xywh = output[:, :4]
    scores     = output[:, 4:]

    class_ids  = np.argmax(scores, axis=1)
    class_confs = scores[np.arange(len(scores)), class_ids]

    mask = class_confs >= conf_thresh
    if not mask.any():
        return []

    boxes_xywh  = boxes_xywh[mask]
    class_ids   = class_ids[mask]
    class_confs = class_confs[mask]

    # xywh → xyxy (en escala del modelo)
    x, y, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = x - w / 2; y1 = y - h / 2
    x2 = x + w / 2; y2 = y + h / 2

    # Normalizar a 0-1 si en píxeles
    if x2.max() > 2.0:
        x1 /= model_size; y1 /= model_size
        x2 /= model_size; y2 /= model_size

    # Escalar a imagen original
    oh, ow = orig_shape
    boxes_abs = np.stack([
        (x1 * ow).astype(int), (y1 * oh).astype(int),
        (x2 * ow).astype(int), (y2 * oh).astype(int),
    ], axis=1)

    # NMS simple
    detecciones = []
    for i in range(len(boxes_abs)):
        detecciones.append({
            "box":      boxes_abs[i].tolist(),
            "class_id": int(class_ids[i]),
            "conf":     float(class_confs[i]),
        })

    return detecciones


def clip_box(box: list, h: int, w: int) -> list:
    x1, y1, x2, y2 = box
    return [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]


# ─── Pipeline ANPR ────────────────────────────────────────────────────────────

class ANPRCoralPipeline:
    """
    Pipeline ANPR completo sobre Coral Edge TPU real o simulado.

    Etapas:
      1. Detectar vehículo en frame completo
      2. Detectar placa dentro del ROI del vehículo
      3. Clasificar color del vehículo
      4. Detectar marca del vehículo
    """

    def __init__(self, modelos: dict[str, CoralInterpreter | None]):
        self.modelos = modelos

    def run(self, frame_bgr: np.ndarray) -> dict:
        """
        Ejecuta el pipeline sobre un frame BGR.

        Returns:
            {
              "vehiculos": [...],  # lista de vehículos detectados
              "latencias": {...},  # latencia por etapa en ms
              "backend":   str
            }
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]
        latencias = {}
        resultados_vehiculos = []
        backend = "cpu"

        # ── 1. Detección de vehículos ─────────────────────────────────────────
        if self.modelos["vehicle"]:
            m = self.modelos["vehicle"]
            m.set_input_from_image(frame_rgb)
            r = m.invoke(model_hint="yolo")
            latencias["vehicle_ms"] = r.latency_ms
            latencias["vehicle_tpu_est_ms"] = r.estimated_tpu_ms
            backend = r.backend

            vehiculos = yolo_postprocess(
                r.raw_output[0], orig_shape=(h, w), conf_thresh=0.35
            )
        else:
            # Sin modelo de vehículo: usar frame completo como ROI
            vehiculos = [{"box": [0, 0, w, h], "class_id": 0, "conf": 1.0}]
            latencias["vehicle_ms"] = 0.0

        # ── 2-4. Por cada vehículo ────────────────────────────────────────────
        for veh in vehiculos[:3]:   # máx 3 vehículos por frame
            x1v, y1v, x2v, y2v = clip_box(veh["box"], h, w)
            roi_veh = frame_rgb[y1v:y2v, x1v:x2v]

            if roi_veh.size == 0:
                continue

            info_veh = {
                "box":    veh["box"],
                "conf":   veh["conf"],
                "color":  None,
                "marca":  None,
                "placas": [],
            }

            # ── 3. Color del vehículo ─────────────────────────────────────────
            if self.modelos["color"] and roi_veh.shape[0] > 10:
                m = self.modelos["color"]
                m.set_input_from_image(roi_veh, target_size=(224, 224))
                r = m.invoke(model_hint="efficient")
                latencias.setdefault("color_ms", 0.0)
                latencias["color_ms"] += r.latency_ms
                latencias["color_tpu_est_ms"] = r.estimated_tpu_ms

                out = r.raw_output[0].flatten()
                idx = int(np.argmax(out))
                eng = COLOR_CLASSES[idx] if idx < len(COLOR_CLASSES) else "?"
                info_veh["color"] = COLOR_ES.get(eng, eng)

            # ── 4. Marca del vehículo ─────────────────────────────────────────
            if self.modelos["brand"] and roi_veh.shape[0] > 10:
                m = self.modelos["brand"]
                m.set_input_from_image(roi_veh)
                r = m.invoke(model_hint="yolo")
                latencias.setdefault("brand_ms", 0.0)
                latencias["brand_ms"] += r.latency_ms

                marcas = yolo_postprocess(
                    r.raw_output[0],
                    orig_shape=(roi_veh.shape[0], roi_veh.shape[1]),
                    conf_thresh=0.4,
                )
                if marcas:
                    info_veh["marca"] = marcas[0].get("class_id", "?")

            # ── 2. Placa dentro del ROI ───────────────────────────────────────
            if self.modelos["plate"] and roi_veh.shape[0] > 10:
                m = self.modelos["plate"]
                m.set_input_from_image(roi_veh)
                r = m.invoke(model_hint="yolo")
                latencias.setdefault("plate_ms", 0.0)
                latencias["plate_ms"] += r.latency_ms
                latencias["plate_tpu_est_ms"] = r.estimated_tpu_ms

                placas = yolo_postprocess(
                    r.raw_output[0],
                    orig_shape=(roi_veh.shape[0], roi_veh.shape[1]),
                    conf_thresh=0.35,
                )
                for placa in placas[:2]:
                    px1, py1, px2, py2 = clip_box(
                        placa["box"], roi_veh.shape[0], roi_veh.shape[1]
                    )
                    # Convertir coords a frame original
                    info_veh["placas"].append({
                        "box_local":  placa["box"],
                        "box_global": [px1+x1v, py1+y1v, px2+x1v, py2+y1v],
                        "conf":       placa["conf"],
                    })

            resultados_vehiculos.append(info_veh)

        return {
            "vehiculos": resultados_vehiculos,
            "latencias": latencias,
            "backend":   backend,
        }


# ─── Dibujar resultados ───────────────────────────────────────────────────────

def dibujar(frame_bgr: np.ndarray, resultado: dict) -> np.ndarray:
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]

    # Latencias en esquina superior izquierda
    lat = resultado["latencias"]
    backend = resultado["backend"]
    backend_label = "🪸 Coral real" if backend == "coral" else "CPU (simulación)"

    total_cpu = sum(v for k, v in lat.items() if k.endswith("_ms") and not k.endswith("_est_ms"))
    total_tpu = sum(v for k, v in lat.items() if k.endswith("_tpu_est_ms"))

    lines_info = [
        f"Backend: {backend_label}",
        f"Total CPU: {total_cpu:.0f} ms  ({1000/total_cpu:.1f} fps)" if total_cpu > 0 else "",
    ]
    if backend == "cpu" and total_tpu > 0:
        lines_info.append(f"Est. TPU: {total_tpu:.0f} ms  (~{total_cpu/total_tpu:.0f}x mas rapido)")

    for i, line in enumerate(lines_info):
        if not line:
            continue
        y_pos = 28 + i * 24
        cv2.putText(canvas, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # Por cada vehículo
    for veh in resultado["vehiculos"]:
        x1, y1, x2, y2 = clip_box(veh["box"], h, w)

        # Caja vehículo
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLORS_BGR["vehicle"], 2)

        # Etiqueta vehículo
        labels = []
        if veh["color"]:
            labels.append(veh["color"])
        label_str = " | ".join(labels) if labels else "Vehículo"
        cv2.rectangle(canvas, (x1, y1 - 22), (x1 + len(label_str)*9 + 6, y1), COLORS_BGR["vehicle"], -1)
        cv2.putText(canvas, label_str, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Cajas placas
        for placa in veh["placas"]:
            px1, py1, px2, py2 = clip_box(placa["box_global"], h, w)
            cv2.rectangle(canvas, (px1, py1), (px2, py2), COLORS_BGR["plate"], 2)
            conf_str = f"Placa {placa['conf']:.2f}"
            cv2.putText(canvas, conf_str, (px1, py1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS_BGR["plate"], 1, cv2.LINE_AA)

    return canvas


# ─── Modos de ejecución ───────────────────────────────────────────────────────

def modo_imagen(args):
    """Procesa una sola imagen y muestra/guarda resultado."""
    img_path = Path(args.imagen)
    if not img_path.exists():
        print(f"{R}❌ Imagen no encontrada: {img_path}{N}")
        sys.exit(1)

    modelos = cargar_modelos(args.device, verbose=True)
    pipeline = ANPRCoralPipeline(modelos)

    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"{R}❌ No se pudo leer la imagen.{N}")
        sys.exit(1)

    print(f"\n{C}⚙️  Procesando {img_path.name}...{N}")
    t0 = time.perf_counter()
    resultado = pipeline.run(frame)
    total = (time.perf_counter() - t0) * 1000

    # Resumen por consola
    print(f"\n{B}{'='*55}{N}")
    print(f"{B}  RESULTADO{N}")
    print(f"{B}{'='*55}{N}")
    print(f"  Tiempo total Python: {total:.0f} ms")
    print(f"  Backend: {resultado['backend']}")
    lat = resultado["latencias"]
    for k, v in sorted(lat.items()):
        label = f"{'(est. TPU)' if 'tpu' in k else '':12s}"
        print(f"  {k:30s}: {v:.1f} ms {label}")

    print(f"\n  Vehículos detectados: {len(resultado['vehiculos'])}")
    for i, veh in enumerate(resultado["vehiculos"]):
        print(f"\n  Vehículo {i+1}:")
        print(f"    Box : {veh['box']}  conf={veh['conf']:.2f}")
        print(f"    Color: {veh['color'] or 'N/A'}")
        print(f"    Marca: {veh['marca'] or 'N/A'}")
        print(f"    Placas encontradas: {len(veh['placas'])}")
        for j, placa in enumerate(veh["placas"]):
            print(f"      Placa {j+1}: {placa['box_global']}  conf={placa['conf']:.2f}")

    # Guardar imagen anotada
    canvas = dibujar(frame, resultado)
    salida = PROJECT_DIR / "output" / f"coral_{img_path.stem}_resultado.jpg"
    salida.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(salida), canvas)
    print(f"\n{G}✅ Imagen guardada: {salida}{N}")

    # Mostrar si hay display
    try:
        cv2.imshow("ANPR Coral", canvas)
        print("   Presiona cualquier tecla para cerrar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass


def modo_benchmark(args):
    """Benchmarka todos los modelos disponibles."""
    print(f"\n{B}{'='*55}{N}")
    print(f"{B}  BENCHMARK CORAL — {args.device.upper()} backend{N}")
    print(f"{B}{'='*55}{N}")

    modelos_bases = {
        "vehicle":  "yolo11n_coco_vehicle_int8",
        "plate":    "placa_detector_yolo11n_int8",
        "color":    "color_classifier_int8",
        "brand":    "marca_detector_yolo11n_int8",
    }
    hints = {
        "vehicle": "yolo", "plate": "yolo", "color": "efficient", "brand": "yolo",
    }

    for key, base in modelos_bases.items():
        path = resolver_modelo(base)
        if path is None:
            print(f"\n{Y}⚠️  {key}: no encontrado (skipping){N}")
            continue

        print(f"\n{C}── {key.upper()} ({path.name}) ──{N}")
        stats = coral_benchmark(path, n_runs=args.runs, device=args.device, verbose=True)
        if args.device == "cpu":
            scale = {"yolo":11,"efficient":6}.get(hints[key], 10)
            print(f"  Est. TPU media: {stats['mean_ms']/scale:.1f} ms  (~{scale}x speedup en Coral)")


def modo_webcam(args):
    """Inferencia en tiempo real desde webcam."""
    modelos = cargar_modelos(args.device, verbose=True)
    pipeline = ANPRCoralPipeline(modelos)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"{R}❌ No se pudo abrir la webcam.{N}")
        sys.exit(1)

    print(f"\n{G}▶ Webcam activa. Presiona 'q' para salir.{N}")
    fps_counter = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resultado = pipeline.run(frame)
        canvas = dibujar(frame, resultado)

        fps_counter += 1
        elapsed = time.time() - t_start
        if elapsed >= 1.0:
            fps = fps_counter / elapsed
            fps_counter = 0
            t_start = time.time()
            cv2.putText(canvas, f"FPS: {fps:.1f}", (canvas.shape[1] - 100, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("ANPR Coral (q=salir)", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline ANPR completo sobre Coral Edge TPU (real o simulado)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python 08_inferencia_coral.py --imagen samples/car.jpg
  python 08_inferencia_coral.py --imagen samples/car.jpg --device cpu
  python 08_inferencia_coral.py --benchmark --runs 20
  python 08_inferencia_coral.py --webcam
  python 08_inferencia_coral.py --webcam --device coral   # requiere hardware
        """,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--imagen",    type=str, metavar="RUTA",
                      help="Procesar una imagen estática")
    mode.add_argument("--benchmark", action="store_true",
                      help="Benchmarkar todos los modelos")
    mode.add_argument("--webcam",    action="store_true",
                      help="Inferencia en tiempo real (webcam)")

    p.add_argument("--device",  default="auto",
                   choices=["auto", "coral", "cpu"],
                   help="Backend de inferencia (default: auto)")
    p.add_argument("--runs",    type=int, default=30,
                   help="Número de runs para benchmark (default: 30)")
    return p.parse_args()


def main():
    print(f"\n{B}{'='*55}{N}")
    print(f"{B}  ANPR CORAL PIPELINE{N}")
    print(f"{B}  Google Coral Edge TPU  |  TFLite Simulation{N}")
    print(f"{B}{'='*55}{N}")

    args = parse_args()

    if args.imagen:
        modo_imagen(args)
    elif args.benchmark:
        modo_benchmark(args)
    elif args.webcam:
        modo_webcam(args)


if __name__ == "__main__":
    main()
