#!/usr/bin/env python3
"""
================================================================================
SCRIPT 02: ENTRENAMIENTO DEL MODELO YOLOv11 PARA DETECCIÓN DE PLACAS
================================================================================
Este script entrena un modelo YOLOv11 nano optimizado para la detección de
placas vehiculares, diseñado para ejecutarse en dispositivos edge.

Características:
- Modelo base: YOLOv11n (nano) - 2.6M parámetros
- Optimizado para: Coral Edge TPU, Raspberry Pi, dispositivos ARM
- Resolución de entrada: 640x640 (ajustable)

Autor: Sistema ANPR Colombia
Fecha: 2025
================================================================================
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import yaml
import argparse

# Configuración de rutas
PROJECT_DIR = Path(__file__).parent.parent
DATASET_DIR = PROJECT_DIR / "dataset_combinado"
OUTPUT_DIR = PROJECT_DIR / "output"
MODELS_DIR = PROJECT_DIR / "models"

# Configuración por defecto del entrenamiento
DEFAULT_CONFIG = {
    # Modelo
    "model": "yolo11n.pt",  # YOLOv11 nano - más ligero
    "imgsz": 640,           # Tamaño de imagen

    # Entrenamiento
    "epochs": 100,          # Épocas de entrenamiento
    "batch": 16,            # Tamaño de batch (ajustar según GPU/RAM)
    "patience": 20,         # Early stopping

    # Optimización
    "optimizer": "AdamW",   # Optimizador
    "lr0": 0.01,            # Learning rate inicial
    "lrf": 0.01,            # Learning rate final (fracción)
    "momentum": 0.937,      # Momentum SGD
    "weight_decay": 0.0005, # Regularización L2

    # Augmentación de datos
    "hsv_h": 0.015,         # Augmentación de tono
    "hsv_s": 0.7,           # Augmentación de saturación
    "hsv_v": 0.4,           # Augmentación de valor
    "degrees": 0.0,         # Rotación máxima
    "translate": 0.1,       # Traslación
    "scale": 0.5,           # Escalado
    "shear": 0.0,           # Cizallamiento
    "perspective": 0.0,     # Perspectiva
    "flipud": 0.0,          # Flip vertical
    "fliplr": 0.5,          # Flip horizontal
    "mosaic": 1.0,          # Mosaic augmentation
    "mixup": 0.0,           # Mixup augmentation

    # Hardware
    "device": "auto",       # "auto", "0" (CUDA), "mps" (Mac), o "cpu"
    "workers": 8,           # Workers para dataloader

    # Guardado
    "save": True,
    "save_period": 10,      # Guardar cada N épocas
    "cache": True,          # Cachear imágenes en RAM
    "exist_ok": True,
    "pretrained": True,
    "verbose": True,
}


def verificar_requisitos():
    """Verifica que todos los requisitos estén instalados."""
    print("🔍 Verificando requisitos...")

    requisitos = []

    try:
        from ultralytics import YOLO
        print("   ✅ ultralytics instalado")
    except ImportError:
        requisitos.append("ultralytics")
        print("   ❌ ultralytics NO instalado")

    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")

        # Verificar dispositivos disponibles
        if torch.cuda.is_available():
            print(f"   ✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            print(f"   ✅ MPS disponible (Apple Silicon Metal)")
        else:
            print("   ⚠️  GPU no disponible - entrenamiento en CPU")
    except ImportError:
        requisitos.append("torch")
        print("   ❌ PyTorch NO instalado")

    if requisitos:
        print(f"\n❌ Faltan dependencias: {', '.join(requisitos)}")
        print("   Ejecuta: pip install ultralytics torch torchvision")
        return False

    return True


def detectar_dispositivo():
    """
    Detecta el mejor dispositivo disponible para entrenamiento.

    Returns:
        str: "0" para CUDA, "mps" para Apple Silicon, "cpu" para CPU
    """
    import torch

    if torch.cuda.is_available():
        return "0"  # GPU NVIDIA
    elif torch.backends.mps.is_available():
        return "mps"  # Apple Silicon (M1/M2/M3)
    else:
        return "cpu"


def verificar_dataset():
    """Verifica que el dataset esté preparado."""
    data_yaml = DATASET_DIR / "data.yaml"

    if not data_yaml.exists():
        print(f"\n❌ No se encontró el dataset combinado")
        print(f"   Esperado en: {data_yaml}")
        print(f"   Ejecuta primero: python 01_preparar_dataset.py")
        return None

    with open(data_yaml, "r") as f:
        config = yaml.safe_load(f)

    # Contar imágenes
    train_imgs = len(list((DATASET_DIR / "train" / "images").glob("*")))
    val_imgs = len(list((DATASET_DIR / "valid" / "images").glob("*")))

    print(f"\n📊 Dataset encontrado:")
    print(f"   • Imágenes entrenamiento: {train_imgs}")
    print(f"   • Imágenes validación: {val_imgs}")
    print(f"   • Clases: {config.get('names', [])}")

    return str(data_yaml)


def entrenar_modelo(config: dict, data_yaml: str, nombre_experimento: str = None):
    """
    Entrena el modelo YOLOv11 con la configuración especificada.

    Args:
        config: Diccionario de configuración
        data_yaml: Ruta al archivo data.yaml
        nombre_experimento: Nombre opcional para el experimento
    """
    from ultralytics import YOLO

    # Crear nombre de experimento único
    if nombre_experimento is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_experimento = f"anpr_yolo11n_{timestamp}"

    print("\n" + "=" * 70)
    print("   INICIANDO ENTRENAMIENTO")
    print("=" * 70)
    print(f"\n📋 Configuración:")
    print(f"   • Modelo base: {config['model']}")
    print(f"   • Épocas: {config['epochs']}")
    print(f"   • Batch size: {config['batch']}")
    print(f"   • Tamaño imagen: {config['imgsz']}x{config['imgsz']}")
    print(f"   • Dispositivo: {config['device']}")
    print(f"   • Nombre experimento: {nombre_experimento}")

    # Cargar modelo base
    print(f"\n🔄 Cargando modelo base: {config['model']}")
    model = YOLO(config["model"])

    # Crear directorio de salida
    output_path = OUTPUT_DIR / nombre_experimento
    output_path.mkdir(parents=True, exist_ok=True)

    # Guardar configuración usada
    config_file = output_path / "config_entrenamiento.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"   ✅ Configuración guardada en: {config_file}")

    # Entrenar
    print("\n🚀 Iniciando entrenamiento...")
    print("-" * 70)

    results = model.train(
        data=data_yaml,
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        patience=config["patience"],
        optimizer=config["optimizer"],
        lr0=config["lr0"],
        lrf=config["lrf"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        hsv_h=config["hsv_h"],
        hsv_s=config["hsv_s"],
        hsv_v=config["hsv_v"],
        degrees=config["degrees"],
        translate=config["translate"],
        scale=config["scale"],
        shear=config["shear"],
        perspective=config["perspective"],
        flipud=config["flipud"],
        fliplr=config["fliplr"],
        mosaic=config["mosaic"],
        mixup=config["mixup"],
        device=config["device"],
        workers=config["workers"],
        save=config["save"],
        save_period=config["save_period"],
        cache=config["cache"],
        exist_ok=config["exist_ok"],
        pretrained=config["pretrained"],
        verbose=config["verbose"],
        project=str(OUTPUT_DIR),
        name=nombre_experimento,
    )

    print("-" * 70)
    print("\n✅ Entrenamiento completado!")

    # Obtener ruta del mejor modelo
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    last_model = Path(results.save_dir) / "weights" / "last.pt"

    print(f"\n📦 Modelos guardados:")
    print(f"   • Mejor modelo: {best_model}")
    print(f"   • Último modelo: {last_model}")

    # Copiar mejor modelo al directorio de modelos
    if best_model.exists():
        import shutil
        modelo_final = MODELS_DIR / f"placa_detector_yolo11n.pt"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_model, modelo_final)
        print(f"   • Copia en models/: {modelo_final}")

    return best_model, results


def evaluar_modelo(model_path: Path, data_yaml: str):
    """Evalúa el modelo entrenado en el conjunto de prueba."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("   EVALUACIÓN DEL MODELO")
    print("=" * 70)

    model = YOLO(str(model_path))
    metrics = model.val(data=data_yaml, split="test")

    print(f"\n📊 Métricas de evaluación:")
    print(f"   • mAP50: {metrics.box.map50:.4f}")
    print(f"   • mAP50-95: {metrics.box.map:.4f}")
    print(f"   • Precisión: {metrics.box.mp:.4f}")
    print(f"   • Recall: {metrics.box.mr:.4f}")

    return metrics


def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento de YOLOv11 para detección de placas"
    )

    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_CONFIG["epochs"],
        help=f"Número de épocas (default: {DEFAULT_CONFIG['epochs']})"
    )
    parser.add_argument(
        "--batch", type=int, default=DEFAULT_CONFIG["batch"],
        help=f"Tamaño de batch (default: {DEFAULT_CONFIG['batch']})"
    )
    parser.add_argument(
        "--imgsz", type=int, default=DEFAULT_CONFIG["imgsz"],
        help=f"Tamaño de imagen (default: {DEFAULT_CONFIG['imgsz']})"
    )
    parser.add_argument(
        "--device", type=str, default=DEFAULT_CONFIG["device"],
        help="Dispositivo: 'auto' (detectar), '0' (CUDA), 'mps' (Mac Metal), 'cpu'"
    )
    parser.add_argument(
        "--nombre", type=str, default=None,
        help="Nombre del experimento"
    )
    parser.add_argument(
        "--evaluar", action="store_true",
        help="Evaluar modelo después del entrenamiento"
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_CONFIG["workers"],
        help=f"Número de workers (default: {DEFAULT_CONFIG['workers']})"
    )

    return parser.parse_args()


def main():
    """Función principal del script."""
    print("=" * 70)
    print("   ENTRENAMIENTO YOLOv11 - DETECCIÓN DE PLACAS VEHICULARES")
    print("   Optimizado para dispositivos Edge (Coral TPU)")
    print("=" * 70)

    # Parsear argumentos
    args = parse_args()

    # Verificar requisitos
    if not verificar_requisitos():
        sys.exit(1)

    # Verificar dataset
    data_yaml = verificar_dataset()
    if data_yaml is None:
        sys.exit(1)

    # Actualizar configuración con argumentos
    config = DEFAULT_CONFIG.copy()
    config["epochs"] = args.epochs
    config["batch"] = args.batch
    config["imgsz"] = args.imgsz
    config["workers"] = args.workers

    # Detectar dispositivo automáticamente si es "auto"
    if args.device == "auto":
        config["device"] = detectar_dispositivo()
        print(f"\n🔧 Dispositivo detectado automáticamente: {config['device']}")
    else:
        config["device"] = args.device

    # Nota: MPS en Macs modernos (M1 Pro+, M2, M3, M4) con 16GB+ RAM
    # pueden manejar más workers sin problemas
    if config["device"] == "mps" and config["workers"] > 8:
        print(f"   ℹ️  MPS detectado con {config['workers']} workers (OK para M4 con 24GB RAM)")

    # Entrenar
    best_model, results = entrenar_modelo(config, data_yaml, args.nombre)

    # Evaluar si se especificó
    if args.evaluar and best_model.exists():
        evaluar_modelo(best_model, data_yaml)

    print("\n" + "=" * 70)
    print("   ¡ENTRENAMIENTO COMPLETADO!")
    print("=" * 70)
    print(f"\n📍 Siguiente paso: Ejecutar 03_exportar_tflite.py")
    print(f"   para convertir el modelo a TensorFlow Lite")
    print("=" * 70)


if __name__ == "__main__":
    main()
