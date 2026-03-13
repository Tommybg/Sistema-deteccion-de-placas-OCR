#!/usr/bin/env python3
"""
================================================================================
SCRIPT 09b: EXPORTACIÓN COLOR CLASSIFIER YOLO11n-cls A TFLITE INT8 (CORAL)
================================================================================
Exporta color_classifier_yolo11n.pt a TFLite INT8 con cuantización completa
para Coral Edge TPU. Reemplaza la versión anterior basada en EfficientNetB0.

Uso:
    python scripts/09_exportar_color_yolo_int8.py
    python scripts/09_exportar_color_yolo_int8.py --validar

Salida:
    models/tflite_exports/color_classifier_int8.tflite
================================================================================
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "models"
EXPORTS_DIR = MODELS_DIR / "tflite_exports"
DATASET_DIR = PROJECT_DIR / "datasets" / "vehicle_colors"

# ─── Colores para terminal ────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def log(msg, color=RESET):
    print(f"{color}{msg}{RESET}")


# ─── Verificar requisitos ─────────────────────────────────────────────────────

def verificar_requisitos() -> bool:
    log("\n🔍 Verificando requisitos...", CYAN)
    ok = True
    for pkg, name in [
        ("ultralytics", "ultralytics"),
        ("tensorflow", "tensorflow"),
    ]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            log(f"   ✅ {name} {ver}", GREEN)
        except ImportError:
            log(f"   ❌ {name} no instalado  →  pip install {pkg}", RED)
            ok = False
    return ok


# ─── Dataset de calibración ──────────────────────────────────────────────────

def _crear_generador_calibracion(data_dir: Path, imgsz: int = 224, num_samples: int = 200):
    """
    Genera imágenes de calibración desde el dataset para cuantización INT8.

    Yields arrays float32 con shape [1, imgsz, imgsz, 3] en rango [0, 1].
    """
    import numpy as np

    try:
        from PIL import Image
    except ImportError:
        log("   ❌ Pillow no instalado → pip install Pillow", RED)
        return

    train_dir = data_dir / "train"
    if not train_dir.exists():
        train_dir = data_dir  # fallback

    imgs = list(train_dir.rglob("*.jpg")) + list(train_dir.rglob("*.png"))
    if not imgs:
        log(f"   ⚠️  No se encontraron imágenes en {train_dir}", YELLOW)
        return

    import random
    random.shuffle(imgs)
    imgs = imgs[:num_samples]

    log(f"   📸 Usando {len(imgs)} imágenes para calibración", CYAN)

    for img_path in imgs:
        img = Image.open(img_path).convert("RGB").resize((imgsz, imgsz))
        arr = np.array(img, dtype=np.float32) / 255.0
        yield [arr[np.newaxis, ...]]


# ─── Exportación INT8 completa (Coral-ready) ────────────────────────────────

def exportar_int8(model_path: Path, data_dir: Path, nombre_salida: str) -> Path | None:
    """
    Exporta el modelo YOLO-cls .pt a TFLite con cuantización INT8 completa.

    Proceso en dos pasos:
      1. YOLO .pt → SavedModel (vía ultralytics)
      2. SavedModel → TFLite INT8 completo (vía tf.lite.TFLiteConverter)

    Args:
        model_path   : ruta al .pt del clasificador de color
        data_dir     : directorio del dataset para calibración
        nombre_salida: nombre del archivo .tflite de destino

    Returns:
        Path al archivo .tflite en EXPORTS_DIR o None si falló
    """
    from ultralytics import YOLO
    import tensorflow as tf
    import torch

    # Parche para PyTorch 2.6+ y versiones antiguas de ultralytics
    # que fallan por weights_only=True al cargar pesos custom.
    torch.serialization.add_safe_globals(["ultralytics.nn.tasks.ClassificationModel"])
    import builtins
    original_load = torch.load
    def safe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = safe_load

    log(f"\n📦 Exportando {model_path.name} → INT8 TFLite (full integer)...", CYAN)
    log(f"   Calibración: {data_dir}", YELLOW)

    model = YOLO(str(model_path))

    try:
        # Exportar a TFLite INT8
        log("\n   Exportando YOLO → TFLite INT8 (EdgeTPU compatible)...", CYAN)
        resultado = model.export(
            format="tflite",
            imgsz=224,
            int8=True,
            data=str(data_dir) # Se le pasa el dataset para la calibración
        )
        if resultado is None:
            log("   ❌ La exportación a TFLite devolvió None.", RED)
            return None

        # Ultralytics exporta a la carpeta del modelo original
        export_dir = model_path.parent
        tflite_path = export_dir / "color_classifier_yolo11n_full_integer_quant.tflite"

        if not tflite_path.exists():
             tflite_path = export_dir / "color_classifier_yolo11n_int8.tflite"

        if not tflite_path.exists():
            log(f"   ❌ No se encontró el archivo tflite generado en {export_dir}", RED)
            return None

        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        destino = EXPORTS_DIR / nombre_salida
        destino.write_bytes(tflite_model)

        size_mb = destino.stat().st_size / 1024 / 1024
        log(f"   ✅ Guardado: {destino.name} ({size_mb:.2f} MB)", GREEN)
        return destino

    except Exception as e:
        log(f"   ❌ Error durante exportación: {e}", RED)
        import traceback
        traceback.print_exc()
        return None


# ─── Validación TFLite ────────────────────────────────────────────────────────

def validar_tflite(tflite_path: Path) -> bool:
    """Carga el modelo y hace una inferencia dummy para verificar integridad."""
    import tensorflow as tf
    import numpy as np

    log(f"\n🔍 Validando {tflite_path.name}...", CYAN)
    try:
        interp = tf.lite.Interpreter(model_path=str(tflite_path))
        interp.allocate_tensors()

        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]

        log(f"   📥 Entrada : dtype={inp['dtype'].__name__}  shape={inp['shape'].tolist()}")
        log(f"   📤 Salida  : dtype={out['dtype'].__name__}  shape={out['shape'].tolist()}")

        # Verificar dimensiones esperadas para clasificador de color
        expected_input = [1, 224, 224, 3]
        if inp["shape"].tolist() == expected_input:
            log(f"   ✅ Shape de entrada correcta: {expected_input}", GREEN)
        else:
            log(f"   ⚠️  Shape de entrada inesperada (esperado {expected_input})", YELLOW)

        # Verificar 15 clases en la salida
        out_shape = out["shape"].tolist()
        if len(out_shape) == 2 and out_shape[1] == 15:
            log(f"   ✅ Shape de salida correcta: {out_shape} (15 clases)", GREEN)
        else:
            log(f"   ⚠️  Shape de salida inesperada: {out_shape} (esperado [1, 15])", YELLOW)

        # Inferencia dummy
        dummy = np.zeros(inp["shape"], dtype=inp["dtype"])
        interp.set_tensor(inp["index"], dummy)
        interp.invoke()
        resultado = interp.get_tensor(out["index"])
        log(f"   ✅ Inferencia OK  |  output shape: {resultado.shape}", GREEN)

        # ¿Es Coral-ready? (dtype INT8/UINT8 en entrada y salida)
        coral_ready = inp["dtype"] in (np.int8, np.uint8) and out["dtype"] in (np.int8, np.uint8)
        if coral_ready:
            log("   🪸 Coral-ready: entrada y salida son INT8/UINT8 ✅", GREEN)
        else:
            log("   ⚠️  No totalmente cuantizado — no es Coral-ready (entrada/salida float32)", YELLOW)

        # Tamaño del archivo
        size_mb = tflite_path.stat().st_size / 1024 / 1024
        if size_mb < 3.0:
            log(f"   ✅ Tamaño: {size_mb:.2f} MB (menor que EfficientNetB0 ~4.9 MB)", GREEN)
        else:
            log(f"   ℹ️  Tamaño: {size_mb:.2f} MB", CYAN)

        return True

    except Exception as e:
        log(f"   ❌ Validación fallida: {e}", RED)
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Exportar clasificador de color YOLO11n-cls a TFLite INT8")
    p.add_argument("--validar", action="store_true",
                   help="Valida el modelo INT8 exportado")
    p.add_argument("--model", type=str, default=None,
                   help="Ruta alternativa al .pt del clasificador de color")
    p.add_argument("--data", type=str, default=None,
                   help="Ruta alternativa al dataset para calibración")
    return p.parse_args()


def main():
    log(f"\n{'='*60}", BOLD)
    log(f"  EXPORTACIÓN COLOR CLASSIFIER YOLO11n-cls → TFLITE INT8", BOLD)
    log(f"{'='*60}", BOLD)

    args = parse_args()

    if not verificar_requisitos():
        sys.exit(1)

    # Resolver rutas
    model_path = Path(args.model) if args.model else MODELS_DIR / "color_classifier_yolo11n.pt"
    data_dir = Path(args.data) if args.data else DATASET_DIR

    if not model_path.exists():
        log(f"\n❌ No se encontró el modelo: {model_path}", RED)
        log("   Ejecuta primero: python scripts/09_entrenar_color_yolo.py", YELLOW)
        sys.exit(1)

    if not data_dir.exists():
        log(f"\n❌ Dataset no encontrado: {data_dir}", RED)
        sys.exit(1)

    # Exportar
    resultado = exportar_int8(
        model_path=model_path,
        data_dir=data_dir,
        nombre_salida="color_classifier_int8.tflite",
    )

    # Validar
    if resultado and args.validar:
        validar_tflite(resultado)
    elif args.validar:
        # Validar existente
        existente = EXPORTS_DIR / "color_classifier_int8.tflite"
        if existente.exists():
            validar_tflite(existente)
        else:
            log(f"\n❌ No hay modelo para validar: {existente}", RED)

    # Resumen
    log(f"\n{'='*60}", BOLD)
    if resultado:
        log("  ✅ EXPORTACIÓN COMPLETADA", GREEN)
        log(f"  → {resultado}", CYAN)
        log(f"\n  Siguiente paso: compilar para Edge TPU con edgetpu_compiler", CYAN)
    else:
        log("  ❌ EXPORTACIÓN FALLIDA", RED)
    log(f"{'='*60}", BOLD)


if __name__ == "__main__":
    main()
