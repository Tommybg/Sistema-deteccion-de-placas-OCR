#!/usr/bin/env python3
"""
================================================================================
SCRIPT 03: EXPORTACIÓN A TENSORFLOW LITE PARA EDGE DEVICES
================================================================================
Este script convierte el modelo YOLOv11 entrenado a formato TensorFlow Lite,
optimizado para dispositivos edge como Coral TPU, Raspberry Pi, etc.

Formatos de exportación:
- TFLite FP32: Precisión completa
- TFLite FP16: Media precisión (más rápido)
- TFLite INT8: Cuantizado (más eficiente en edge)
- Edge TPU: Optimizado específicamente para Coral

Autor: Sistema ANPR Colombia
Fecha: 2025
================================================================================
"""

import os
import sys
from pathlib import Path
import argparse
import shutil
from datetime import datetime

# Configuración de rutas
PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "output"
DATASET_DIR = PROJECT_DIR / "dataset_combinado"

# Nombre del modelo entrenado
DEFAULT_MODEL = MODELS_DIR / "placa_detector_yolo11n.pt"


def verificar_requisitos():
    """Verifica que todos los requisitos estén instalados."""
    print("🔍 Verificando requisitos...")

    requisitos_faltantes = []

    try:
        from ultralytics import YOLO
        print("   ✅ ultralytics")
    except ImportError:
        requisitos_faltantes.append("ultralytics")

    try:
        import tensorflow as tf
        print(f"   ✅ TensorFlow {tf.__version__}")
    except ImportError:
        requisitos_faltantes.append("tensorflow")

    try:
        import onnx
        print(f"   ✅ ONNX {onnx.__version__}")
    except ImportError:
        requisitos_faltantes.append("onnx")

    try:
        import onnx2tf
        print("   ✅ onnx2tf")
    except ImportError:
        # onnx2tf es opcional
        print("   ⚠️  onnx2tf no instalado (opcional para conversión avanzada)")

    if requisitos_faltantes:
        print(f"\n❌ Faltan dependencias: {', '.join(requisitos_faltantes)}")
        print("   Ejecuta: pip install ultralytics tensorflow onnx onnx2tf")
        return False

    return True


def encontrar_mejor_modelo():
    """Encuentra el mejor modelo entrenado."""
    # Primero buscar en la carpeta models
    if DEFAULT_MODEL.exists():
        return DEFAULT_MODEL

    # Buscar en las carpetas de output
    modelos = list(OUTPUT_DIR.rglob("**/weights/best.pt"))
    if modelos:
        # Ordenar por fecha de modificación (más reciente primero)
        modelos.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return modelos[0]

    return None


def exportar_tflite_fp32(model, output_dir: Path) -> Path:
    """
    Exporta el modelo a TFLite con precisión FP32.

    Args:
        model: Modelo YOLO cargado
        output_dir: Directorio de salida

    Returns:
        Path al modelo exportado
    """
    print("\n📦 Exportando a TFLite FP32...")

    try:
        # Exportar usando ultralytics
        exported = model.export(
            format="tflite",
            imgsz=640,
            half=False,
            int8=False,
        )
        print(f"   ✅ Exportado: {exported}")
        return Path(exported)
    except Exception as e:
        print(f"   ❌ Error en exportación FP32: {e}")
        return None


def exportar_tflite_fp16(model, output_dir: Path) -> Path:
    """
    Exporta el modelo a TFLite con precisión FP16 (media precisión).

    Args:
        model: Modelo YOLO cargado
        output_dir: Directorio de salida

    Returns:
        Path al modelo exportado
    """
    print("\n📦 Exportando a TFLite FP16...")

    try:
        exported = model.export(
            format="tflite",
            imgsz=640,
            half=True,
            int8=False,
        )
        print(f"   ✅ Exportado: {exported}")
        return Path(exported)
    except Exception as e:
        print(f"   ❌ Error en exportación FP16: {e}")
        return None


def exportar_tflite_int8(model, output_dir: Path, data_yaml: str = None) -> Path:
    """
    Exporta el modelo a TFLite cuantizado INT8.
    Requiere datos de calibración para cuantización.

    Args:
        model: Modelo YOLO cargado
        output_dir: Directorio de salida
        data_yaml: Path al data.yaml para calibración

    Returns:
        Path al modelo exportado
    """
    print("\n📦 Exportando a TFLite INT8 (cuantizado)...")

    try:
        exported = model.export(
            format="tflite",
            imgsz=640,
            half=False,
            int8=True,
            data=data_yaml,  # Datos para calibración
        )
        print(f"   ✅ Exportado: {exported}")
        return Path(exported)
    except Exception as e:
        print(f"   ❌ Error en exportación INT8: {e}")
        return None


def exportar_edgetpu(model, output_dir: Path) -> Path:
    """
    Exporta el modelo optimizado para Coral Edge TPU.

    Nota: Requiere el compilador Edge TPU instalado.
    https://coral.ai/docs/edgetpu/compiler/

    Args:
        model: Modelo YOLO cargado
        output_dir: Directorio de salida

    Returns:
        Path al modelo exportado
    """
    print("\n📦 Exportando para Edge TPU...")

    try:
        exported = model.export(
            format="edgetpu",
            imgsz=640,
        )
        print(f"   ✅ Exportado: {exported}")
        return Path(exported)
    except Exception as e:
        print(f"   ⚠️  Error en exportación Edge TPU: {e}")
        print("   Nota: Requiere el compilador Edge TPU instalado")
        print("   Instalar: https://coral.ai/docs/edgetpu/compiler/")
        return None


def exportar_onnx(model, output_dir: Path) -> Path:
    """
    Exporta el modelo a formato ONNX (intermedio).

    Args:
        model: Modelo YOLO cargado
        output_dir: Directorio de salida

    Returns:
        Path al modelo exportado
    """
    print("\n📦 Exportando a ONNX...")

    try:
        exported = model.export(
            format="onnx",
            imgsz=640,
            simplify=True,
            dynamic=False,
        )
        print(f"   ✅ Exportado: {exported}")
        return Path(exported)
    except Exception as e:
        print(f"   ❌ Error en exportación ONNX: {e}")
        return None


def exportar_saved_model(model, output_dir: Path) -> Path:
    """
    Exporta el modelo a TensorFlow SavedModel.

    Args:
        model: Modelo YOLO cargado
        output_dir: Directorio de salida

    Returns:
        Path al modelo exportado
    """
    print("\n📦 Exportando a TensorFlow SavedModel...")

    try:
        exported = model.export(
            format="saved_model",
            imgsz=640,
        )
        print(f"   ✅ Exportado: {exported}")
        return Path(exported)
    except Exception as e:
        print(f"   ❌ Error en exportación SavedModel: {e}")
        return None


def validar_tflite(tflite_path: Path):
    """
    Valida que el modelo TFLite funcione correctamente.

    Args:
        tflite_path: Path al modelo TFLite
    """
    import tensorflow as tf
    import numpy as np

    print(f"\n🔍 Validando modelo: {tflite_path.name}")

    try:
        # Cargar modelo
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()

        # Obtener detalles de entrada/salida
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"   📥 Entrada:")
        for inp in input_details:
            print(f"      • Shape: {inp['shape']}")
            print(f"      • Dtype: {inp['dtype']}")

        print(f"   📤 Salida:")
        for out in output_details:
            print(f"      • Shape: {out['shape']}")
            print(f"      • Dtype: {out['dtype']}")

        # Test con datos aleatorios
        input_shape = input_details[0]['shape']
        input_data = np.random.rand(*input_shape).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"   ✅ Inferencia de prueba exitosa")
        print(f"      • Output shape: {output_data.shape}")

        # Tamaño del modelo
        size_mb = tflite_path.stat().st_size / (1024 * 1024)
        print(f"   📊 Tamaño: {size_mb:.2f} MB")

        return True

    except Exception as e:
        print(f"   ❌ Error en validación: {e}")
        return False


def organizar_modelos_exportados(output_dir: Path):
    """
    Organiza todos los modelos exportados en una carpeta limpia.

    Args:
        output_dir: Directorio donde están los modelos
    """
    print("\n📁 Organizando modelos exportados...")

    export_dir = MODELS_DIR / "tflite_exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Buscar todos los modelos TFLite generados
    for tflite in output_dir.parent.rglob("*.tflite"):
        destino = export_dir / tflite.name
        shutil.copy2(tflite, destino)
        print(f"   ✅ Copiado: {tflite.name}")

    print(f"\n📂 Modelos disponibles en: {export_dir}")


def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Exportación de modelo YOLO a TensorFlow Lite"
    )

    parser.add_argument(
        "--modelo", type=str, default=None,
        help="Ruta al modelo .pt entrenado"
    )
    parser.add_argument(
        "--formato", type=str, default="all",
        choices=["fp32", "fp16", "int8", "edgetpu", "onnx", "saved_model", "all"],
        help="Formato de exportación (default: all)"
    )
    parser.add_argument(
        "--validar", action="store_true", default=True,
        help="Validar modelos exportados"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Ruta a data.yaml para calibración INT8"
    )
    parser.add_argument(
        "--coco", action="store_true",
        help="También exportar el modelo COCO (yolo11n.pt) para detección de vehículos"
    )
    parser.add_argument(
        "--brand", action="store_true",
        help="También exportar el modelo de marca (marca_detector_yolo11n.pt)"
    )

    return parser.parse_args()


def main():
    """Función principal del script."""
    print("=" * 70)
    print("   EXPORTACIÓN A TENSORFLOW LITE")
    print("   Para dispositivos Edge (Coral TPU, Raspberry Pi, etc.)")
    print("=" * 70)

    args = parse_args()

    # Verificar requisitos
    if not verificar_requisitos():
        sys.exit(1)

    # Encontrar modelo
    if args.modelo:
        model_path = Path(args.modelo)
    else:
        model_path = encontrar_mejor_modelo()

    if model_path is None or not model_path.exists():
        print(f"\n❌ No se encontró el modelo entrenado")
        print(f"   Ejecuta primero: python 02_entrenar_modelo.py")
        sys.exit(1)

    print(f"\n📦 Modelo a exportar: {model_path}")

    # Cargar modelo
    from ultralytics import YOLO
    model = YOLO(str(model_path))

    # Directorio de salida
    output_dir = MODELS_DIR / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data yaml para calibración
    data_yaml = args.data
    if data_yaml is None:
        data_yaml_path = DATASET_DIR / "data.yaml"
        if data_yaml_path.exists():
            data_yaml = str(data_yaml_path)

    # Exportar según formato seleccionado
    modelos_exportados = {}

    if args.formato in ["fp32", "all"]:
        result = exportar_tflite_fp32(model, output_dir)
        if result:
            modelos_exportados["fp32"] = result

    if args.formato in ["fp16", "all"]:
        result = exportar_tflite_fp16(model, output_dir)
        if result:
            modelos_exportados["fp16"] = result

    if args.formato in ["int8", "all"]:
        result = exportar_tflite_int8(model, output_dir, data_yaml)
        if result:
            modelos_exportados["int8"] = result

    if args.formato in ["onnx", "all"]:
        result = exportar_onnx(model, output_dir)
        if result:
            modelos_exportados["onnx"] = result

    if args.formato in ["saved_model", "all"]:
        result = exportar_saved_model(model, output_dir)
        if result:
            modelos_exportados["saved_model"] = result

    if args.formato in ["edgetpu", "all"]:
        result = exportar_edgetpu(model, output_dir)
        if result:
            modelos_exportados["edgetpu"] = result

    # Validar modelos exportados
    if args.validar:
        print("\n" + "=" * 70)
        print("   VALIDACIÓN DE MODELOS")
        print("=" * 70)

        for nombre, path in modelos_exportados.items():
            if path and path.suffix == ".tflite":
                validar_tflite(path)

    # Exportar modelo COCO para detección de vehículos
    if args.coco:
        coco_model_path = Path(__file__).parent / "yolo11n.pt"
        if coco_model_path.exists():
            print("\n" + "=" * 70)
            print("   EXPORTACIÓN MODELO COCO (DETECCIÓN DE VEHÍCULOS)")
            print("=" * 70)
            coco_model = YOLO(str(coco_model_path))
            try:
                coco_exported = coco_model.export(
                    format="tflite",
                    imgsz=640,
                    int8=True,
                    data=data_yaml,
                )
                if coco_exported:
                    coco_path = Path(coco_exported)
                    dest = MODELS_DIR / "tflite_exports" / "yolo11n_coco_vehicle_int8.tflite"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(coco_path, dest)
                    modelos_exportados["coco_vehicle_int8"] = dest
                    print(f"   ✅ Modelo COCO exportado: {dest}")
            except Exception as e:
                print(f"   ❌ Error exportando modelo COCO: {e}")
        else:
            print(f"   ⚠️  Modelo COCO no encontrado: {coco_model_path}")

    # Exportar modelo de marca vehicular
    if args.brand:
        brand_model_path = MODELS_DIR / "marca_detector_yolo11n.pt"
        if brand_model_path.exists():
            print("\n" + "=" * 70)
            print("   EXPORTACIÓN MODELO MARCA (DETECCIÓN DE LOGOS)")
            print("=" * 70)
            brand_model = YOLO(str(brand_model_path))

            # Usar data.yaml de marcas para calibración INT8
            brand_data_yaml = str(PROJECT_DIR / "dataset_marcas_combinado" / "data.yaml")
            if not Path(brand_data_yaml).exists():
                brand_data_yaml = data_yaml  # Fallback

            try:
                brand_exported = brand_model.export(
                    format="tflite",
                    imgsz=640,
                    int8=True,
                    data=brand_data_yaml,
                )
                if brand_exported:
                    brand_path = Path(brand_exported)
                    dest = MODELS_DIR / "tflite_exports" / "marca_detector_yolo11n_int8.tflite"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(brand_path, dest)
                    modelos_exportados["brand_int8"] = dest
                    print(f"   ✅ Modelo de marca exportado: {dest}")
            except Exception as e:
                print(f"   ❌ Error exportando modelo de marca: {e}")
        else:
            print(f"   ⚠️  Modelo de marca no encontrado: {brand_model_path}")
            print(f"   Ejecuta primero: python scripts/06_entrenar_marca.py")

    # Organizar modelos
    organizar_modelos_exportados(output_dir)

    # Resumen
    print("\n" + "=" * 70)
    print("   RESUMEN DE EXPORTACIÓN")
    print("=" * 70)
    print(f"\n✅ Modelos exportados exitosamente:")
    for nombre, path in modelos_exportados.items():
        if path:
            size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0
            print(f"   • {nombre}: {path.name} ({size_mb:.2f} MB)")

    print(f"\n📂 Ubicación: {MODELS_DIR / 'tflite_exports'}")
    print(f"\n📍 Siguiente paso: Ejecutar 04_inferencia_tiempo_real.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
