#!/usr/bin/env python3
"""
================================================================================
SCRIPT 01: PREPARACIÓN Y COMBINACIÓN DE DATASETS
================================================================================
Este script combina múltiples datasets de Roboflow en una estructura unificada
para el entrenamiento del modelo de detección de placas.

Autor: Sistema ANPR Colombia
Fecha: 2025
================================================================================
"""

import os
import shutil
import yaml
from pathlib import Path
from collections import defaultdict
import random

# Configuración de rutas
BASE_DIR = Path(__file__).parent.parent.parent  # Carpeta placas
PROJECT_DIR = Path(__file__).parent.parent  # anpr_project

# Datasets disponibles
DATASETS = [
    BASE_DIR / "Proyecto-Placas-1",
    BASE_DIR / "Proyecto Placas(200img).v1i.yolov11",
    BASE_DIR / "Proyecto Placas(600img).v2i.yolov11",
]

# Directorio de salida combinado
OUTPUT_DIR = PROJECT_DIR / "dataset_combinado"


def contar_archivos(directorio: Path, extension: str = ".jpg") -> int:
    """Cuenta archivos con una extensión específica."""
    if not directorio.exists():
        return 0
    return len(list(directorio.glob(f"*{extension}"))) + \
           len(list(directorio.glob("*.png"))) + \
           len(list(directorio.glob("*.jpeg")))


def copiar_dataset(dataset_path: Path, output_base: Path, prefix: str) -> dict:
    """
    Copia un dataset al directorio de salida con un prefijo único.

    Args:
        dataset_path: Ruta del dataset original
        output_base: Directorio base de salida
        prefix: Prefijo para evitar colisiones de nombres

    Returns:
        dict con conteos de archivos copiados
    """
    stats = defaultdict(int)

    for split in ["train", "valid", "test"]:
        img_src = dataset_path / split / "images"
        lbl_src = dataset_path / split / "labels"

        img_dst = output_base / split / "images"
        lbl_dst = output_base / split / "labels"

        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        if not img_src.exists():
            print(f"  ⚠️  No existe: {img_src}")
            continue

        # Copiar imágenes
        for img_file in img_src.iterdir():
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                new_name = f"{prefix}_{img_file.name}"
                shutil.copy2(img_file, img_dst / new_name)
                stats[f"{split}_images"] += 1

        # Copiar etiquetas
        if lbl_src.exists():
            for lbl_file in lbl_src.iterdir():
                if lbl_file.suffix == ".txt":
                    new_name = f"{prefix}_{lbl_file.name}"
                    shutil.copy2(lbl_file, lbl_dst / new_name)
                    stats[f"{split}_labels"] += 1

    return dict(stats)


def crear_data_yaml(output_dir: Path) -> Path:
    """
    Crea el archivo data.yaml para el entrenamiento.

    Returns:
        Path al archivo data.yaml creado
    """
    data_config = {
        "path": str(output_dir.absolute()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["placa"],
        # Metadatos adicionales
        "proyecto": "ANPR Colombia - Reconocimiento de Placas",
        "version": "1.0",
        "descripcion": "Dataset combinado para detección de placas vehiculares colombianas"
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

    return yaml_path


def verificar_integridad(output_dir: Path) -> bool:
    """
    Verifica que cada imagen tenga su archivo de etiquetas correspondiente.

    Returns:
        True si la integridad es correcta
    """
    problemas = []

    for split in ["train", "valid", "test"]:
        img_dir = output_dir / split / "images"
        lbl_dir = output_dir / split / "labels"

        if not img_dir.exists():
            continue

        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                if not lbl_file.exists():
                    problemas.append(f"Falta etiqueta para: {img_file.name}")

    if problemas:
        print(f"\n⚠️  Se encontraron {len(problemas)} problemas de integridad:")
        for p in problemas[:10]:
            print(f"   - {p}")
        if len(problemas) > 10:
            print(f"   ... y {len(problemas) - 10} más")
        return False

    return True


def main():
    """Función principal del script."""
    print("=" * 70)
    print("   PREPARACIÓN DE DATASET COMBINADO PARA ANPR")
    print("=" * 70)

    # Limpiar directorio de salida si existe
    if OUTPUT_DIR.exists():
        print(f"\n🗑️  Limpiando directorio existente: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True)

    # Procesar cada dataset
    total_stats = defaultdict(int)

    print("\n📂 Procesando datasets...")
    for i, dataset in enumerate(DATASETS, 1):
        if not dataset.exists():
            print(f"  ⚠️  Dataset no encontrado: {dataset}")
            continue

        prefix = f"ds{i}"
        print(f"\n  [{i}/{len(DATASETS)}] {dataset.name}")

        stats = copiar_dataset(dataset, OUTPUT_DIR, prefix)
        for key, value in stats.items():
            total_stats[key] += value

        print(f"      ✅ Copiados: {stats}")

    # Crear archivo de configuración
    print("\n📝 Creando archivo data.yaml...")
    yaml_path = crear_data_yaml(OUTPUT_DIR)
    print(f"   ✅ Creado: {yaml_path}")

    # Verificar integridad
    print("\n🔍 Verificando integridad del dataset...")
    if verificar_integridad(OUTPUT_DIR):
        print("   ✅ Integridad verificada correctamente")

    # Resumen final
    print("\n" + "=" * 70)
    print("   RESUMEN DEL DATASET COMBINADO")
    print("=" * 70)
    print(f"\n📊 Estadísticas:")
    print(f"   • Imágenes de entrenamiento: {total_stats.get('train_images', 0)}")
    print(f"   • Imágenes de validación:    {total_stats.get('valid_images', 0)}")
    print(f"   • Imágenes de prueba:        {total_stats.get('test_images', 0)}")
    print(f"   • Total de imágenes:         {sum(v for k, v in total_stats.items() if 'images' in k)}")

    print(f"\n📁 Ubicación del dataset: {OUTPUT_DIR}")
    print(f"📄 Archivo de configuración: {yaml_path}")

    print("\n✅ ¡Dataset preparado exitosamente!")
    print("   Siguiente paso: Ejecutar 02_entrenar_modelo.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
