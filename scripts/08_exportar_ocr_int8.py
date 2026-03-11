#!/usr/bin/env python3
"""
================================================================================
SCRIPT 08_exportar_ocr_int8.py: OCR (CCT-XS) → TFLite INT8 para Coral
================================================================================
Convierte el modelo fast-plate-ocr (CCT-XS, ONNX) a TFLite INT8 completo,
listo para compilar con el Edge TPU Compiler.

Arquitectura:
  CCT-XS (Compact Convolutional Transformer, extra-small)
    Input : [1, 64, 128, 3]  uint8 / float32
    Output: [1, 9, 37]       softmax por cada carácter (9 posiciones × 37 clases)
  ~0.51M params — ideal para Coral (muy pequeño)

Pipeline de conversión:
  ONNX → TensorFlow SavedModel (via onnx2tf) → TFLite INT8

Calibración:
  Usa recortes de placa del dataset para calibrar la cuantización.
  Si no hay recortes disponibles, genera datos sintéticos.

Uso:
  python scripts/08_exportar_ocr_int8.py
  python scripts/08_exportar_ocr_int8.py --calibration datasets/plates_crops/
  python scripts/08_exportar_ocr_int8.py --validar
================================================================================
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).parent.parent
ONNX_MODEL  = Path.home() / ".cache/fast-plate-ocr/cct-xs-v1-global-model/cct_xs_v1_global.onnx"
EXPORTS_DIR = PROJECT_DIR / "models" / "tflite_exports"
DATASET_DIR = PROJECT_DIR / "dataset_combinado"

# Input specs del modelo
IMG_H, IMG_W, IMG_C = 64, 128, 3        # input shape
NUM_CHARS, NUM_CLASSES = 9, 37           # output: 9 posiciones × 37 caracteres

G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; C = "\033[96m"; B = "\033[1m"; N = "\033[0m"


# ─── Calibration dataset ──────────────────────────────────────────────────────

def calibration_generator(calibration_dir: Path | None, n_samples: int = 200):
    """
    Genera imágenes de calibración para la cuantización INT8.

    Si existe un directorio con recortes de placas, los usa.
    Si no, genera imágenes sintéticas (bordes, gradientes, texto simulado).
    """
    import cv2

    samples_generated = 0

    if calibration_dir and calibration_dir.exists():
        # Buscar imágenes en el directorio
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        images = []
        for ext in exts:
            images.extend(list(calibration_dir.rglob(ext)))

        print(f"   Usando {len(images)} imágenes de calibración de {calibration_dir}")

        for img_path in images[:n_samples]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_W, IMG_H))
            img = img.astype(np.float32) / 255.0
            yield [np.expand_dims(img, 0)]
            samples_generated += 1

    if samples_generated < n_samples:
        # Complementar con datos sintéticos si no hay suficientes imágenes reales
        needed = n_samples - samples_generated
        print(f"   Generando {needed} imágenes sintéticas de calibración...")

        rng = np.random.default_rng(42)
        for _ in range(needed):
            # Fondo (blanco/amarillo — colores típicos de placas colombianas)
            bg_color = rng.choice([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
            img = np.full((IMG_H, IMG_W, IMG_C), bg_color, dtype=np.float32)

            # Texto sintético (barras negras/blancas para simular letras)
            n_chars = rng.integers(6, 9)
            char_w = IMG_W // (n_chars + 2)
            for i in range(n_chars):
                x = char_w + i * char_w + rng.integers(-3, 3)
                h_bar = rng.integers(IMG_H // 3, int(IMG_H * 0.8))
                y = (IMG_H - h_bar) // 2
                color = 0.0 if img[IMG_H//2, min(x, IMG_W-1), 0] > 0.5 else 1.0
                img[y:y+h_bar, max(0, x):min(IMG_W, x+max(2, char_w-4))] = color

            # Ruido ligero
            img += rng.normal(0, 0.02, img.shape)
            img = np.clip(img, 0.0, 1.0)

            yield [np.expand_dims(img, 0)]


# ─── Conversión ONNX → SavedModel ─────────────────────────────────────────────

def convert_onnx_to_savedmodel(onnx_path: Path, savedmodel_dir: Path) -> bool:
    """Convierte el modelo ONNX a TF SavedModel usando onnx2tf."""
    print(f"\n{C}1️⃣  ONNX → TF SavedModel...{N}")
    try:
        import onnx2tf
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(savedmodel_dir),
            non_verbose=True,
        )
        print(f"   {G}✅ SavedModel generado en {savedmodel_dir}{N}")
        return True
    except ImportError:
        print(f"   {R}❌ onnx2tf no instalado: pip install onnx2tf{N}")
        return False
    except Exception as e:
        print(f"   {R}❌ Error en conversión ONNX→SavedModel: {e}{N}")
        return False


# ─── SavedModel → TFLite INT8 ─────────────────────────────────────────────────

def convert_to_tflite_int8(
    savedmodel_dir: Path,
    output_path: Path,
    calibration_dir: Path | None,
    n_calib: int,
) -> bool:
    """
    Convierte SavedModel → TFLite con cuantización INT8 completa (Coral-ready).
    Usa cuantización de entrada y salida a uint8 para máxima compatibilidad.
    """
    import tensorflow as tf

    print(f"\n{C}2️⃣  SavedModel → TFLite INT8 (full integer)...{N}")
    print(f"   Calibrando con {n_calib} muestras...")

    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(savedmodel_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: calibration_generator(calibration_dir, n_calib)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.uint8
        converter.inference_output_type = tf.uint8

        tflite_model = converter.convert()
        output_path.write_bytes(tflite_model)

        size_mb = len(tflite_model) / 1024 / 1024
        print(f"   {G}✅ TFLite INT8 guardado: {output_path.name} ({size_mb:.2f} MB){N}")
        return True

    except Exception as e:
        print(f"   {R}❌ Conversión TFLite fallida: {e}{N}")

        # Fallback: INT8 pesos solamente (sin cuantización de I/O)
        print(f"   {Y}⚠️  Intentando fallback: INT8 pesos (float32 I/O)...{N}")
        try:
            converter2 = tf.lite.TFLiteConverter.from_saved_model(str(savedmodel_dir))
            converter2.optimizations = [tf.lite.Optimize.DEFAULT]
            converter2.representative_dataset = lambda: calibration_generator(calibration_dir, n_calib)
            tflite_model2 = converter2.convert()
            output_path.write_bytes(tflite_model2)
            size_mb2 = len(tflite_model2) / 1024 / 1024
            print(f"   {Y}✅ Fallback TFLite (float32 I/O) guardado: {size_mb2:.2f} MB{N}")
            print(f"   {Y}   Nota: El Edge TPU Compiler aún puede cuantizar las capas internas.{N}")
            return True
        except Exception as e2:
            print(f"   {R}❌ Fallback también falló: {e2}{N}")
            return False


# ─── Validación ───────────────────────────────────────────────────────────────

def validar_tflite(tflite_path: Path) -> bool:
    """Valida el modelo TFLite OCR con una placa sintética."""
    import tensorflow as tf

    print(f"\n{C}🔍 Validando {tflite_path.name}...{N}")
    try:
        interp = tf.lite.Interpreter(model_path=str(tflite_path))
        interp.allocate_tensors()

        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]

        print(f"   📥 Entrada : dtype={inp['dtype'].__name__}  shape={inp['shape'].tolist()}")
        print(f"   📤 Salida  : dtype={out['dtype'].__name__}  shape={out['shape'].tolist()}")

        # Imagen dummy (placa blanca)
        if inp["dtype"] == np.uint8:
            dummy = np.full(inp["shape"], 255, dtype=np.uint8)
        else:
            dummy = np.ones(inp["shape"], dtype=np.float32)

        interp.set_tensor(inp["index"], dummy)
        interp.invoke()
        result = interp.get_tensor(out["index"])
        print(f"   ✅ Inferencia OK  |  output shape: {result.shape}")

        coral_ready = inp["dtype"] in (np.uint8, np.int8) and out["dtype"] in (np.uint8, np.int8)
        if coral_ready:
            print(f"   {G}🪸 Coral-ready: I/O son INT8/UINT8 ✅{N}")
        else:
            print(f"   {Y}⚠️  I/O float32 — el Edge TPU Compiler cuantizará capas internas{N}")

        return True
    except Exception as e:
        print(f"   {R}❌ Validación fallida: {e}{N}")
        return False


# ─── Mostrar estado completo ──────────────────────────────────────────────────

def mostrar_estado():
    print(f"\n{B}{'='*60}{N}")
    print(f"{B}  ESTADO FINAL — models/tflite_exports/{N}")
    print(f"{B}{'='*60}{N}")

    esperados = [
        ("yolo11n_coco_vehicle_int8.tflite",  "Vehicle detector"),
        ("color_classifier_int8.tflite",       "Color classifier  🪸 Coral-ready"),
        ("marca_detector_yolo11n_int8.tflite", "Brand detector"),
        ("ocr_cct_xs_int8.tflite",             "OCR (CCT-XS)"),
    ]
    all_ok = True
    for fname, label in esperados:
        path = EXPORTS_DIR / fname
        if path.exists():
            size = path.stat().st_size / 1024 / 1024
            print(f"   {G}✅{N} {fname:50s}  ({size:.1f} MB)  ← {label}")
        else:
            print(f"   {R}❌{N} {fname:50s}  ← {label}  (falta)")
            all_ok = False

    print()
    if all_ok:
        print(f"   {G}🎉 Todos los modelos INT8 listos. Siguiente paso:{N}")
        print(f"   {C}   bash scripts/07_compilar_edgetpu.sh{N}")
    else:
        print(f"   {Y}Ejecuta este script para completar el OCR.{N}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Exportar CCT-XS OCR a TFLite INT8 para Coral")
    p.add_argument("--onnx", type=str, default=str(ONNX_MODEL),
                   help="Ruta al modelo ONNX (default: ~/.cache/fast-plate-ocr/...)")
    p.add_argument("--calibration", type=str, default=None,
                   help="Directorio con imágenes de placas recortadas para calibración")
    p.add_argument("--n-calib", type=int, default=200,
                   help="Número de muestras de calibración (default: 200)")
    p.add_argument("--validar", action="store_true",
                   help="Solo validar modelos existentes, no reexportar")
    return p.parse_args()


def main():
    print(f"\n{B}{'='*60}{N}")
    print(f"{B}  OCR CCT-XS → TFLITE INT8 (CORAL EDGE TPU){N}")
    print(f"{B}{'='*60}{N}")

    args = parse_args()
    onnx_path = Path(args.onnx)

    if args.validar:
        tflite_path = EXPORTS_DIR / "ocr_cct_xs_int8.tflite"
        if tflite_path.exists():
            validar_tflite(tflite_path)
        else:
            print(f"{Y}⚠️  No hay modelo exportado aún. Ejecuta sin --validar primero.{N}")
        mostrar_estado()
        return

    # Verificar que el ONNX existe
    if not onnx_path.exists():
        print(f"{R}❌ ONNX no encontrado: {onnx_path}{N}")
        print(f"   Ejecuta primero en tu app: from fast_plate_ocr import LicensePlateRecognizer")
        print(f"   r = LicensePlateRecognizer('cct-xs-v1-global-model'); r.run(img)")
        print(f"   Eso descarga el modelo en ~/.cache/fast-plate-ocr/")
        sys.exit(1)

    print(f"   ONNX fuente: {onnx_path}")
    print(f"   Salida:      {EXPORTS_DIR}/ocr_cct_xs_int8.tflite")

    # Directorio de calibración
    calib_dir = Path(args.calibration) if args.calibration else None
    # Intentar usar imágenes del dataset de placas si existen
    if calib_dir is None:
        candidates = [
            DATASET_DIR / "train" / "images",
            DATASET_DIR / "valid" / "images",
            PROJECT_DIR / "datasets" / "plates_crops",
        ]
        for c in candidates:
            if c.exists() and list(c.glob("*.jpg"))[:1]:
                calib_dir = c
                break

    # Convertir ONNX → SavedModel
    tmp_dir = Path(tempfile.mkdtemp())
    savedmodel_dir = tmp_dir / "ocr_savedmodel"

    try:
        ok = convert_onnx_to_savedmodel(onnx_path, savedmodel_dir)
        if not ok:
            sys.exit(1)

        # Convertir SavedModel → TFLite INT8
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = EXPORTS_DIR / "ocr_cct_xs_int8.tflite"

        ok = convert_to_tflite_int8(savedmodel_dir, output_path, calib_dir, args.n_calib)
        if not ok:
            sys.exit(1)

        # Validar
        validar_tflite(output_path)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    mostrar_estado()


if __name__ == "__main__":
    main()
