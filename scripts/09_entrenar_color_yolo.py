#!/usr/bin/env python3
"""
================================================================================
SCRIPT 09: ENTRENAMIENTO YOLO11n-cls PARA CLASIFICACIÓN DE COLOR VEHICULAR
================================================================================
Entrena un modelo YOLO11n-cls (clasificación) sobre el dataset vehicle_colors
con 15 clases de colores comunes en vehículos.

Características:
- Modelo base: YOLO11n-cls (nano clasificación) - ImageNet pretrained
- 15 clases: beige, black, blue, brown, gold, green, grey, orange,
             pink, purple, red, silver, tan, white, yellow
- hsv_h=0.0 — NO augmentación de tono (el color ES el objetivo)
- Resolución: 224x224

Autor: Sistema ANPR Colombia
Fecha: 2025
================================================================================
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Configuración de rutas
PROJECT_DIR = Path(__file__).parent.parent
DATASET_DIR = PROJECT_DIR / "datasets" / "vehicle_colors"
OUTPUT_DIR = PROJECT_DIR / "output"
MODELS_DIR = PROJECT_DIR / "models"

# Configuración por defecto del entrenamiento
DEFAULT_CONFIG = {
    # Modelo
    "model": "yolo11n-cls.pt",  # YOLO11 nano clasificación
    "imgsz": 224,               # Tamaño de imagen estándar para clasificación

    # Entrenamiento
    "epochs": 100,
    "batch": 32,
    "patience": 20,             # Early stopping

    # Optimización
    "optimizer": "AdamW",
    "lr0": 0.01,
    "lrf": 0.01,

    # Augmentación de datos
    "hsv_h": 0.0,     # ¡CRÍTICO! NO rotar tono — el color es lo que clasificamos
    "hsv_s": 0.3,     # Variación moderada de saturación
    "hsv_v": 0.3,     # Variación moderada de brillo
    "fliplr": 0.5,    # Flip horizontal
    "scale": 0.5,     # Escalado
    "translate": 0.1, # Traslación

    # Hardware
    "device": "auto",
    "workers": 8,

    # Guardado
    "save": True,
    "save_period": 10,
    "cache": True,
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
        return "0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def verificar_dataset():
    """Verifica que el dataset de colores exista con la estructura esperada."""
    if not DATASET_DIR.exists():
        print(f"\n❌ No se encontró el dataset de colores")
        print(f"   Esperado en: {DATASET_DIR}")
        return False

    # Verificar carpetas train/val
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"

    if not train_dir.exists():
        print(f"\n❌ No se encontró carpeta train: {train_dir}")
        return False
    if not val_dir.exists():
        print(f"\n❌ No se encontró carpeta val: {val_dir}")
        return False

    # Contar clases y imágenes
    clases_train = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    total_train = sum(len(list(d.glob("*"))) for d in train_dir.iterdir() if d.is_dir())
    total_val = sum(len(list(d.glob("*"))) for d in val_dir.iterdir() if d.is_dir())

    # Verificar si hay split de test
    test_dir = DATASET_DIR / "test"
    total_test = 0
    if test_dir.exists():
        total_test = sum(len(list(d.glob("*"))) for d in test_dir.iterdir() if d.is_dir())

    print(f"\n📊 Dataset de colores encontrado:")
    print(f"   • Clases ({len(clases_train)}): {', '.join(clases_train)}")
    print(f"   • Imágenes entrenamiento: {total_train}")
    print(f"   • Imágenes validación: {total_val}")
    if total_test > 0:
        print(f"   • Imágenes test: {total_test}")

    return True


def entrenar_modelo(config: dict, nombre_experimento: str = None):
    """
    Entrena el modelo YOLO11n-cls con la configuración especificada.

    Args:
        config: Diccionario de configuración
        nombre_experimento: Nombre opcional para el experimento
    """
    from ultralytics import YOLO

    if nombre_experimento is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_experimento = f"color_yolo11n_cls_{timestamp}"

    print("\n" + "=" * 70)
    print("   INICIANDO ENTRENAMIENTO - CLASIFICACIÓN DE COLOR")
    print("=" * 70)
    print(f"\n📋 Configuración:")
    print(f"   • Modelo base: {config['model']}")
    print(f"   • Épocas: {config['epochs']}")
    print(f"   • Batch size: {config['batch']}")
    print(f"   • Tamaño imagen: {config['imgsz']}x{config['imgsz']}")
    print(f"   • Dispositivo: {config['device']}")
    print(f"   • hsv_h: {config['hsv_h']} (sin augmentación de tono)")
    print(f"   • Nombre experimento: {nombre_experimento}")

    # Cargar modelo base
    print(f"\n🔄 Cargando modelo base: {config['model']}")
    model = YOLO(config["model"])

    # Entrenar
    print("\n🚀 Iniciando entrenamiento...")
    print("-" * 70)

    results = model.train(
        data=str(DATASET_DIR),
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        patience=config["patience"],
        optimizer=config["optimizer"],
        lr0=config["lr0"],
        lrf=config["lrf"],
        hsv_h=config["hsv_h"],
        hsv_s=config["hsv_s"],
        hsv_v=config["hsv_v"],
        fliplr=config["fliplr"],
        scale=config["scale"],
        translate=config["translate"],
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
        modelo_final = MODELS_DIR / "color_classifier_yolo11n.pt"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_model, modelo_final)
        print(f"   • Copia en models/: {modelo_final}")

    return best_model, results


def evaluar_modelo(model_path: Path):
    """Evalúa el modelo entrenado en el conjunto de test."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("   EVALUACIÓN DEL MODELO")
    print("=" * 70)

    model = YOLO(str(model_path))

    # Evaluar en test si existe, sino en val
    test_dir = DATASET_DIR / "test"
    split = "test" if test_dir.exists() else "val"
    print(f"   Evaluando en split: {split}")

    metrics = model.val(data=str(DATASET_DIR), split=split)

    print(f"\n📊 Métricas de evaluación (clasificación):")
    print(f"   • Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"   • Top-5 Accuracy: {metrics.top5:.4f}")

    return metrics


def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento YOLO11n-cls para clasificación de color vehicular"
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
    print("   ENTRENAMIENTO YOLO11n-cls - CLASIFICACIÓN DE COLOR VEHICULAR")
    print("   15 colores · 224x224 · Optimizado para Edge TPU")
    print("=" * 70)

    args = parse_args()

    # Verificar requisitos
    if not verificar_requisitos():
        sys.exit(1)

    # Verificar dataset
    if not verificar_dataset():
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

    # Entrenar
    best_model, results = entrenar_modelo(config, args.nombre)

    # Evaluar si se especificó
    if args.evaluar and best_model.exists():
        evaluar_modelo(best_model)

    print("\n" + "=" * 70)
    print("   ¡ENTRENAMIENTO COMPLETADO!")
    print("=" * 70)
    print(f"\n📍 Siguiente paso: python scripts/09_exportar_color_yolo_int8.py")
    print(f"   para exportar a TFLite INT8 (Coral Edge TPU)")
    print("=" * 70)


if __name__ == "__main__":
    main()
