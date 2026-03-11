#!/usr/bin/env python3
"""
================================================================================
SCRIPT 07: EXPORTACIÓN DETECTOR DE MARCAS A TFLITE INT8 (CORAL EDGE TPU)
================================================================================
Exporta marca_detector_yolo11n.pt a TFLite INT8 con cuantización completa
de integers para que sea compatible con Coral Edge TPU.

También opcionalmente re-exporta placa_detector_yolo11n.pt con mejor
calibración usando el dataset de placas.

Uso:
    python scripts/07_exportar_marca_int8.py
    python scripts/07_exportar_marca_int8.py --solo-placa
    python scripts/07_exportar_marca_int8.py --validar

Salida:
    models/tflite_exports/marca_detector_yolo11n_int8.tflite
    models/tflite_exports/placa_detector_yolo11n_int8.tflite  (opcional)
================================================================================
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "models"
EXPORTS_DIR = MODELS_DIR / "tflite_exports"
DATASET_DIR = PROJECT_DIR / "dataset_combinado"

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
        ("tensorflow", f"tensorflow"),
    ]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            log(f"   ✅ {name} {ver}", GREEN)
        except ImportError:
            log(f"   ❌ {name} no instalado  →  pip install {pkg}", RED)
            ok = False
    return ok


# ─── Exportación INT8 vía ultralytics ────────────────────────────────────────

def exportar_int8(model_path: Path, data_yaml: Path, nombre_salida: str) -> Path | None:
    """
    Exporta un modelo .pt a TFLite INT8 y lo copia al directorio de exports.

    Args:
        model_path : ruta al .pt
        data_yaml  : data.yaml con imágenes de calibración
        nombre_salida : nombre del archivo .tflite de destino

    Returns:
        Path al archivo .tflite en EXPORTS_DIR o None si falló
    """
    from ultralytics import YOLO

    log(f"\n📦 Exportando {model_path.name} → INT8 TFLite...", CYAN)
    log(f"   Calibración: {data_yaml}", YELLOW)

    model = YOLO(str(model_path))

    try:
        resultado = model.export(
            format="tflite",
            imgsz=640,
            int8=True,
            half=False,
            data=str(data_yaml),
        )
        if resultado is None:
            log("   ❌ La exportación devolvió None.", RED)
            return None

        src = Path(resultado)
        if not src.exists():
            # ultralytics a veces devuelve el directorio saved_model; buscamos
            candidatos = list(src.parent.rglob("*int8*.tflite"))
            if not candidatos:
                candidatos = list(src.parent.rglob("*.tflite"))
            if candidatos:
                src = candidatos[0]
            else:
                log(f"   ❌ No se encontró .tflite generado en {src.parent}", RED)
                return None

        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        destino = EXPORTS_DIR / nombre_salida
        shutil.copy2(src, destino)

        size_mb = destino.stat().st_size / 1024 / 1024
        log(f"   ✅ Guardado: {destino.name} ({size_mb:.2f} MB)", GREEN)
        return destino

    except Exception as e:
        log(f"   ❌ Error durante exportación: {e}", RED)
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

        inp  = interp.get_input_details()[0]
        out  = interp.get_output_details()[0]

        log(f"   📥 Entrada : dtype={inp['dtype'].__name__}  shape={inp['shape'].tolist()}")
        log(f"   📤 Salida  : dtype={out['dtype'].__name__}  shape={out['shape'].tolist()}")

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

        return True

    except Exception as e:
        log(f"   ❌ Validación fallida: {e}", RED)
        return False


# ─── Resumen de archivos ──────────────────────────────────────────────────────

def mostrar_estado_exports():
    log(f"\n{'='*60}", BOLD)
    log(f"  ESTADO DE MODELOS EN {EXPORTS_DIR.name}/", BOLD)
    log(f"{'='*60}", BOLD)

    esperados = [
        "yolo11n_coco_vehicle_int8.tflite",
        "color_classifier_int8.tflite",
        "marca_detector_yolo11n_int8.tflite",
        "placa_detector_yolo11n_int8.tflite",
        "yolo11n_coco_vehicle_int8_edgetpu.tflite",
        "color_classifier_int8_edgetpu.tflite",
        "marca_detector_yolo11n_int8_edgetpu.tflite",
    ]

    for nombre in esperados:
        path = EXPORTS_DIR / nombre
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            es_edgetpu = "_edgetpu" in nombre
            icon = "🪸" if es_edgetpu else "✅"
            log(f"   {icon} {nombre:55s} ({size_mb:.1f} MB)", GREEN)
        else:
            icon = "🪸" if "_edgetpu" in nombre else "❌"
            log(f"   {icon} {nombre:55s} (falta)", YELLOW if "_edgetpu" in nombre else RED)

    log(f"\n   Siguiente paso → bash scripts/07_compilar_edgetpu.sh", CYAN)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Exportar detector de marcas a TFLite INT8")
    p.add_argument("--solo-placa",    action="store_true",
                   help="Re-exporta solo el detector de placas (no de marcas)")
    p.add_argument("--todo",          action="store_true",
                   help="Exporta tanto el detector de marcas como el de placas")
    p.add_argument("--validar",       action="store_true",
                   help="Valida todos los modelos INT8 en tflite_exports/")
    p.add_argument("--data",          type=str, default=None,
                   help="Ruta alternativa a data.yaml para calibración")
    return p.parse_args()


def main():
    log(f"\n{'='*60}", BOLD)
    log(f"  EXPORTACIÓN MARCA DETECTOR → TFLITE INT8 (CORAL)", BOLD)
    log(f"{'='*60}", BOLD)

    args = parse_args()

    if not verificar_requisitos():
        sys.exit(1)

    # Resolver data.yaml de calibración
    data_yaml = Path(args.data) if args.data else DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        log(f"\n❌ data.yaml no encontrado: {data_yaml}", RED)
        log("   Pasa --data <ruta/data.yaml>", YELLOW)
        sys.exit(1)

    exportados = {}

    # ── Marca detector (por defecto si no se pasa --solo-placa) ──────────────
    if not args.solo_placa:
        marca_pt = MODELS_DIR / "marca_detector_yolo11n.pt"
        if not marca_pt.exists():
            log(f"\n❌ No se encontró {marca_pt}", RED)
            log("   Ejecuta primero: python scripts/06_entrenar_marca.py", YELLOW)
            sys.exit(1)

        resultado = exportar_int8(
            model_path=marca_pt,
            data_yaml=data_yaml,
            nombre_salida="marca_detector_yolo11n_int8.tflite",
        )
        if resultado:
            exportados["marca"] = resultado

    # ── Placa detector (si --todo o --solo-placa) ─────────────────────────────
    if args.todo or args.solo_placa:
        placa_pt = MODELS_DIR / "placa_detector_yolo11n.pt"
        if placa_pt.exists():
            resultado = exportar_int8(
                model_path=placa_pt,
                data_yaml=data_yaml,
                nombre_salida="placa_detector_yolo11n_int8.tflite",
            )
            if resultado:
                exportados["placa"] = resultado
        else:
            log(f"\n⚠️  Placa detector no encontrado: {placa_pt}", YELLOW)

    # ── Validación ────────────────────────────────────────────────────────────
    if args.validar or exportados:
        log(f"\n{'='*60}", BOLD)
        log("  VALIDACIÓN", BOLD)
        log(f"{'='*60}", BOLD)

        # Validar los recién exportados
        for _, path in exportados.items():
            validar_tflite(path)

        # Si --validar, también los ya existentes
        if args.validar:
            for tflite in sorted(EXPORTS_DIR.glob("*int8*.tflite")):
                if tflite not in exportados.values():
                    validar_tflite(tflite)

    # ── Estado final ──────────────────────────────────────────────────────────
    mostrar_estado_exports()


if __name__ == "__main__":
    main()
