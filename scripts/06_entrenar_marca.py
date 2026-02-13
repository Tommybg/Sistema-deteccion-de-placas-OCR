#!/usr/bin/env python3
"""
================================================================================
SCRIPT 06: ENTRENAMIENTO DEL MODELO YOLOv11 PARA DETECCIÓN DE MARCAS
================================================================================
Este script entrena un modelo YOLOv11 nano optimizado para la detección de
logos de marcas vehiculares (30 clases), diseñado para ejecutarse en
dispositivos edge (Coral Edge TPU).

Modelo de salida: models/marca_detector_yolo11n.pt

Prerequisito: Ejecutar 05_preparar_dataset_marcas.py primero.

Autor: Sistema ANPR Colombia
Fecha: 2026
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
DATASET_DIR = PROJECT_DIR / "dataset_marcas_combinado"
OUTPUT_DIR = PROJECT_DIR / "output"
MODELS_DIR = PROJECT_DIR / "models"

# Configuración por defecto del entrenamiento
# Ajustada para detección de logos (objetos pequeños, 30 clases)
DEFAULT_CONFIG = {
    # Modelo
    "model": "yolo11n.pt",  # YOLOv11 nano - transfer learning desde COCO
    "imgsz": 640,           # Tamaño de imagen (datasets ya en 640x640)

    # Entrenamiento
    "epochs": 150,          # Más épocas para 30 clases
    "batch": 16,            # Tamaño de batch (ajustar según GPU/RAM)
    "patience": 25,         # Early stopping con más paciencia

    # Optimización
    "optimizer": "AdamW",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,

    # Augmentación -REDUCIDA porque Dataset2 ya tiene augmentación (3x)
    # y los logos requieren cuidado especial
    "hsv_h": 0.01,         # Tono reducido (logos tienen colores distintivos)
    "hsv_s": 0.5,          # Saturación moderada
    "hsv_v": 0.3,          # Valor moderado
    "degrees": 5.0,         # Rotación mínima (DS2 ya tiene rotación)
    "translate": 0.1,
    "scale": 0.3,           # Escala reducida (logos son pequeños)
    "shear": 0.0,           # Sin shear (DS2 ya tiene shear)
    "perspective": 0.0,
    "flipud": 0.0,          # SIN flip vertical (logos no aparecen al revés)
    "fliplr": 0.0,          # SIN flip horizontal (logos NO son simétricos)
    "mosaic": 0.5,          # Mosaic reducido (logos son objetos pequeños)
    "mixup": 0.0,

    # Loss -ajustado para desbalance de clases
    "cls": 1.5,             # Class loss gain aumentado por desbalance

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
    print("[*] Verificando requisitos...")

    requisitos = []

    try:
        from ultralytics import YOLO
        print("   [OK] ultralytics instalado")
    except ImportError:
        requisitos.append("ultralytics")
        print("   [X] ultralytics NO instalado")

    try:
        import torch
        print(f"   [OK] PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            print(f"   [OK] CUDA disponible: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            print(f"   [OK] MPS disponible (Apple Silicon Metal)")
        else:
            print("   [!]  GPU no disponible - entrenamiento en CPU")
    except ImportError:
        requisitos.append("torch")
        print("   [X] PyTorch NO instalado")

    if requisitos:
        print(f"\n[X] Faltan dependencias: {', '.join(requisitos)}")
        print("   Ejecuta: pip install ultralytics torch torchvision")
        return False

    return True


def detectar_dispositivo():
    """Detecta el mejor dispositivo disponible para entrenamiento."""
    import torch

    if torch.cuda.is_available():
        return "0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def verificar_dataset(dataset_dir: Path = None):
    """Verifica que el dataset de marcas esté preparado."""
    dataset_dir = dataset_dir or DATASET_DIR
    data_yaml = dataset_dir / "data.yaml"

    if not data_yaml.exists():
        print(f"\n[X] No se encontró el dataset combinado de marcas")
        print(f"   Esperado en: {data_yaml}")
        print(f"   Ejecuta primero: python scripts/05_preparar_dataset_marcas.py")
        return None

    with open(data_yaml, "r") as f:
        config = yaml.safe_load(f)

    # Contar imágenes
    train_dir = dataset_dir / "train" / "images"
    val_dir = dataset_dir / "valid" / "images"
    train_imgs = len(list(train_dir.glob("*"))) if train_dir.exists() else 0
    val_imgs = len(list(val_dir.glob("*"))) if val_dir.exists() else 0

    nc = config.get("nc", 0)
    names = config.get("names", [])

    print(f"\n[*] Dataset de marcas encontrado:")
    print(f"   -Imágenes entrenamiento: {train_imgs}")
    print(f"   -Imágenes validación: {val_imgs}")
    print(f"   -Número de clases: {nc}")
    print(f"   -Marcas: {', '.join(names[:8])}...")

    return str(data_yaml)


def entrenar_modelo(config: dict, data_yaml: str, nombre_experimento: str = None):
    """
    Entrena el modelo YOLOv11 para detección de logos de marcas.

    Args:
        config: Diccionario de configuración
        data_yaml: Ruta al archivo data.yaml
        nombre_experimento: Nombre opcional para el experimento
    """
    from ultralytics import YOLO

    if nombre_experimento is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_experimento = f"marca_yolo11n_{timestamp}"

    print("\n" + "=" * 70)
    print("   INICIANDO ENTRENAMIENTO - DETECCIÓN DE MARCAS")
    print("=" * 70)
    print(f"\n[*] Configuración:")
    print(f"   -Modelo base: {config['model']}")
    print(f"   -Épocas: {config['epochs']}")
    print(f"   -Batch size: {config['batch']}")
    print(f"   -Tamaño imagen: {config['imgsz']}x{config['imgsz']}")
    print(f"   -Dispositivo: {config['device']}")
    print(f"   -Flip horizontal: {config['fliplr']} (desactivado para logos)")
    print(f"   -Mosaic: {config['mosaic']} (reducido para objetos pequeños)")
    print(f"   -Class loss gain: {config['cls']} (aumentado por desbalance)")
    print(f"   -Nombre experimento: {nombre_experimento}")

    # Cargar modelo base
    print(f"\n[*] Cargando modelo base: {config['model']}")
    model = YOLO(config["model"])

    # Crear directorio de salida
    output_path = OUTPUT_DIR / nombre_experimento
    output_path.mkdir(parents=True, exist_ok=True)

    # Guardar configuración usada
    config_file = output_path / "config_entrenamiento.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"   [OK] Configuración guardada en: {config_file}")

    # Entrenar
    print("\n[>>] Iniciando entrenamiento...")
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
        cls=config["cls"],
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
    print("\n[OK] Entrenamiento completado!")

    # Obtener ruta del mejor modelo
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    last_model = Path(results.save_dir) / "weights" / "last.pt"

    print(f"\n[+] Modelos guardados:")
    print(f"   -Mejor modelo: {best_model}")
    print(f"   -Último modelo: {last_model}")

    # Copiar mejor modelo al directorio de modelos
    if best_model.exists():
        import shutil
        modelo_final = MODELS_DIR / "marca_detector_yolo11n.pt"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_model, modelo_final)
        print(f"   -Copia en models/: {modelo_final}")

    return best_model, results


def evaluar_modelo(model_path: Path, data_yaml: str):
    """Evalúa el modelo entrenado en el conjunto de prueba."""
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("   EVALUACIÓN DEL MODELO DE MARCAS")
    print("=" * 70)

    model = YOLO(str(model_path))
    metrics = model.val(data=data_yaml, split="test")

    print(f"\n[*] Métricas de evaluación:")
    print(f"   -mAP50: {metrics.box.map50:.4f}")
    print(f"   -mAP50-95: {metrics.box.map:.4f}")
    print(f"   -Precisión: {metrics.box.mp:.4f}")
    print(f"   -Recall: {metrics.box.mr:.4f}")

    return metrics


def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento de YOLOv11 para detección de marcas vehiculares (30 clases)"
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
        "--data", type=str, default=None,
        help=f"Ruta al data.yaml (default: {DATASET_DIR / 'data.yaml'})"
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
    print("   ENTRENAMIENTO YOLOv11 - DETECCIÓN DE MARCAS VEHICULARES")
    print("   30 clases | Optimizado para Coral Edge TPU")
    print("=" * 70)

    args = parse_args()

    if not verificar_requisitos():
        sys.exit(1)

    # Verificar dataset
    dataset_dir = Path(args.data).parent if args.data else DATASET_DIR
    data_yaml = args.data or verificar_dataset(dataset_dir)
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
        print(f"\n[*] Dispositivo detectado automáticamente: {config['device']}")
    else:
        config["device"] = args.device

    if config["device"] == "mps" and config["workers"] > 8:
        print(f"   [i]  MPS detectado con {config['workers']} workers")

    # Entrenar
    best_model, results = entrenar_modelo(config, data_yaml, args.nombre)

    # Evaluar si se especificó
    if args.evaluar and best_model.exists():
        evaluar_modelo(best_model, data_yaml)

    print("\n" + "=" * 70)
    print("   ¡ENTRENAMIENTO DE MARCAS COMPLETADO!")
    print("=" * 70)
    print(f"\n[>] Modelo guardado en: models/marca_detector_yolo11n.pt")
    print(f"[>] Siguiente paso: El modelo se integra automáticamente al pipeline")
    print(f"   Ejecuta: python scripts/04_inferencia_tiempo_real.py --source 0")
    print("=" * 70)


if __name__ == "__main__":
    main()
