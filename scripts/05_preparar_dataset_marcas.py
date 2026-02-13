#!/usr/bin/env python3
"""
================================================================================
SCRIPT 05: PREPARACIÓN Y COMBINACIÓN DE DATASETS DE MARCAS VEHICULARES
================================================================================
Este script combina 3 datasets de Roboflow con diferentes marcas de vehículos
en una estructura unificada para el entrenamiento del modelo de detección de
logos de marcas.

Datasets:
  - Dataset1: 9 marcas (BMW, Honda, Hyundai, Mazda, MercedesBenz, Perodua,
              Proton, Toyota, Volkswagen) — 7,627 imágenes, sin augmentación
  - Dataset2: 20 marcas (superset de DS1 + Acura, Audi, Chevrolet, Ford,
              Infiniti, KIA, Lamborghini, Lexus, Nissan, Porsche, Tesla)
              — 9,205 imágenes, con augmentación (3x)
  - Dataset3: 23 marcas (incluye Renault, Suzuki, Mitsubishi, Citroen, etc.)
              — 6,607 imágenes, formato YOLOv12 (compatible con v11)
              ATENCION: Contiene clase "Plate" (ID 16) que se EXCLUYE

Salida: dataset_marcas_combinado/ con 30 clases unificadas

Autor: Sistema ANPR Colombia
Fecha: 2026
================================================================================
"""

import os
import shutil
import yaml
from pathlib import Path
from collections import defaultdict

# ─── Configuración de rutas ──────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent.parent
DATASET_MARCAS_DIR = PROJECT_DIR / "Dataset-marcas"

DATASETS = {
    "ds1": DATASET_MARCAS_DIR / "Dataset1",
    "ds2": DATASET_MARCAS_DIR / "Dataset2",
    "ds3": DATASET_MARCAS_DIR / "Dataset3",
}

OUTPUT_DIR = PROJECT_DIR / "dataset_marcas_combinado"

# ─── Esquema unificado: 30 clases (orden alfabético) ─────────────────────────

UNIFIED_NAMES = [
    "Acura",          # 0
    "Audi",           # 1
    "BMW",            # 2
    "Chevrolet",      # 3
    "Citroen",        # 4
    "Dacia",          # 5
    "Fiat",           # 6
    "Ford",           # 7
    "Honda",          # 8
    "Hyundai",        # 9
    "Infiniti",       # 10
    "KIA",            # 11
    "Lamborghini",    # 12
    "Lexus",          # 13
    "Mazda",          # 14
    "MercedesBenz",   # 15
    "Mitsubishi",     # 16
    "Nissan",         # 17
    "Opel",           # 18
    "Perodua",        # 19
    "Peugeot",        # 20
    "Porsche",        # 21
    "Proton",         # 22
    "Renault",        # 23
    "Seat",           # 24
    "Suzuki",         # 25
    "Tesla",          # 26
    "Toyota",         # 27
    "Volkswagen",     # 28
    "Volvo",          # 29
]

NUM_CLASSES = len(UNIFIED_NAMES)  # 30

# ─── Tablas de remapeo: class_id_original -> class_id_unificado ──────────────

# Dataset1: 9 clases
# Original: BMW(0), Honda(1), Hyundai(2), Mazda(3), MercedesBenz(4),
#           Perodua(5), Proton(6), Toyota(7), Volkswagen(8)
CLASS_REMAP_DS1 = {
    0: 2,    # BMW -> 2
    1: 8,    # Honda -> 8
    2: 9,    # Hyundai -> 9
    3: 14,   # Mazda -> 14
    4: 15,   # MercedesBenz -> 15
    5: 19,   # Perodua -> 19
    6: 22,   # Proton -> 22
    7: 27,   # Toyota -> 27
    8: 28,   # Volkswagen -> 28
}

# Dataset2: 20 clases
# Original: Acura(0), Audi(1), BMW(2), Chevrolet(3), Ford(4), Honda(5),
#           Hyundai(6), Infiniti(7), KIA(8), Lamborghini(9), Lexus(10),
#           Mazda(11), MercedesBenz(12), Nissan(13), Perodua(14), Porsche(15),
#           Proton(16), Tesla(17), Toyota(18), Volkswagen(19)
CLASS_REMAP_DS2 = {
    0: 0,    # Acura -> 0
    1: 1,    # Audi -> 1
    2: 2,    # BMW -> 2
    3: 3,    # Chevrolet -> 3
    4: 7,    # Ford -> 7
    5: 8,    # Honda -> 8
    6: 9,    # Hyundai -> 9
    7: 10,   # Infiniti -> 10
    8: 11,   # KIA -> 11
    9: 12,   # Lamborghini -> 12
    10: 13,  # Lexus -> 13
    11: 14,  # Mazda -> 14
    12: 15,  # MercedesBenz -> 15
    13: 17,  # Nissan -> 17
    14: 19,  # Perodua -> 19
    15: 21,  # Porsche -> 21
    16: 22,  # Proton -> 22
    17: 26,  # Tesla -> 26
    18: 27,  # Toyota -> 27
    19: 28,  # Volkswagen -> 28
}

# Dataset3: 23 clases (EXCLUIR clase 16 = Plate)
# Original: Audi(0), Bmw(1), Citroen(2), Dacia(3), Fiat(4), Ford(5),
#           Honda(6), Hyundai(7), Kia(8), Lexus(9), Mazda(10), Mercedes(11),
#           Mitsubishi(12), Nissan(13), Opel(14), Peugeot(15), Plate(16),
#           Renault(17), Seat(18), Suzuki(19), Toyota(20), Volkswagen(21), Volvo(22)
CLASS_REMAP_DS3 = {
    0: 1,    # Audi -> 1
    1: 2,    # Bmw -> 2
    2: 4,    # Citroen -> 4
    3: 5,    # Dacia -> 5
    4: 6,    # Fiat -> 6
    5: 7,    # Ford -> 7
    6: 8,    # Honda -> 8
    7: 9,    # Hyundai -> 9
    8: 11,   # Kia -> 11
    9: 13,   # Lexus -> 13
    10: 14,  # Mazda -> 14
    11: 15,  # Mercedes -> 15
    12: 16,  # Mitsubishi -> 16
    13: 17,  # Nissan -> 17
    14: 18,  # Opel -> 18
    15: 20,  # Peugeot -> 20
    # 16: Plate -> EXCLUIR
    17: 23,  # Renault -> 23
    18: 24,  # Seat -> 24
    19: 25,  # Suzuki -> 25
    20: 27,  # Toyota -> 27
    21: 28,  # Volkswagen -> 28
    22: 29,  # Volvo -> 29
}

# Clases a excluir por dataset
EXCLUDE_CLASSES = {
    "ds1": set(),
    "ds2": set(),
    "ds3": {16},  # Plate
}

# Mapeo completo: prefix -> (remap_dict, exclude_set)
DATASET_CONFIG = {
    "ds1": (CLASS_REMAP_DS1, EXCLUDE_CLASSES["ds1"]),
    "ds2": (CLASS_REMAP_DS2, EXCLUDE_CLASSES["ds2"]),
    "ds3": (CLASS_REMAP_DS3, EXCLUDE_CLASSES["ds3"]),
}


# ─── Funciones ───────────────────────────────────────────────────────────────

def remap_label_file(label_path: Path, output_path: Path,
                     class_remap: dict, exclude_classes: set) -> int:
    """
    Lee un archivo de etiquetas YOLO, remapea class IDs y excluye clases.

    Args:
        label_path: Archivo .txt original
        output_path: Archivo .txt de salida
        class_remap: Diccionario {old_id: new_id}
        exclude_classes: Set de class IDs a excluir

    Returns:
        Número de anotaciones válidas escritas (0 si todas fueron excluidas)
    """
    valid_lines = []

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            old_cls = int(parts[0])

            # Excluir clases no deseadas
            if old_cls in exclude_classes:
                continue

            # Remapear class ID
            if old_cls not in class_remap:
                continue  # Clase desconocida, saltar

            parts[0] = str(class_remap[old_cls])
            valid_lines.append(" ".join(parts))

    if valid_lines:
        with open(output_path, "w") as f:
            f.write("\n".join(valid_lines) + "\n")

    return len(valid_lines)


def copiar_dataset_con_remap(dataset_path: Path, output_base: Path,
                             prefix: str, class_remap: dict,
                             exclude_classes: set) -> dict:
    """
    Copia un dataset al directorio de salida con remapeo de clases.

    Args:
        dataset_path: Ruta del dataset original
        output_base: Directorio base de salida
        prefix: Prefijo para evitar colisiones de nombres
        class_remap: Diccionario de remapeo de class IDs
        exclude_classes: Set de class IDs a excluir

    Returns:
        dict con conteos de archivos copiados y excluidos
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
            print(f"  [!] No existe: {img_src}")
            continue

        # Procesar cada imagen con su label
        for img_file in sorted(img_src.iterdir()):
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            lbl_file = lbl_src / f"{img_file.stem}.txt"
            new_name_base = f"{prefix}_{img_file.stem}"

            if lbl_file.exists():
                # Remapear label
                output_lbl = lbl_dst / f"{new_name_base}.txt"
                num_annotations = remap_label_file(
                    lbl_file, output_lbl, class_remap, exclude_classes
                )

                if num_annotations == 0:
                    # Todas las anotaciones fueron excluidas, no copiar imagen
                    if output_lbl.exists():
                        output_lbl.unlink()
                    stats[f"{split}_excluded"] += 1
                    continue

                # Copiar imagen
                shutil.copy2(img_file, img_dst / f"{new_name_base}{img_file.suffix}")
                stats[f"{split}_images"] += 1
                stats[f"{split}_labels"] += 1
            else:
                # Sin label, copiar imagen con label vacío
                shutil.copy2(img_file, img_dst / f"{new_name_base}{img_file.suffix}")
                (lbl_dst / f"{new_name_base}.txt").touch()
                stats[f"{split}_images"] += 1
                stats[f"{split}_labels"] += 1
                stats[f"{split}_no_label"] += 1

    return dict(stats)


def crear_data_yaml(output_dir: Path) -> Path:
    """Crea el archivo data.yaml para el entrenamiento."""
    data_config = {
        "path": str(output_dir.absolute()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": NUM_CLASSES,
        "names": UNIFIED_NAMES,
        "proyecto": "ANPR Colombia - Detección de Marcas Vehiculares",
        "version": "1.0",
        "descripcion": f"Dataset combinado de {NUM_CLASSES} marcas vehiculares (3 datasets)"
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

    return yaml_path


def verificar_integridad(output_dir: Path) -> bool:
    """Verifica que cada imagen tenga su archivo de etiquetas correspondiente."""
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
        print(f"\n[!] Se encontraron {len(problemas)} problemas de integridad:")
        for p in problemas[:10]:
            print(f"   - {p}")
        if len(problemas) > 10:
            print(f"   ... y {len(problemas) - 10} mas")
        return False

    return True


def generar_reporte_clases(output_dir: Path):
    """
    Cuenta anotaciones por clase por split y genera reporte.
    Guarda el reporte en output_dir/reporte_clases.txt
    """
    print("\n[*] Distribucion de clases en el dataset combinado:")
    print("-" * 70)

    reporte_lines = []
    reporte_lines.append("REPORTE DE DISTRIBUCIÓN DE CLASES")
    reporte_lines.append("=" * 60)

    totals = defaultdict(int)

    for split in ["train", "valid", "test"]:
        lbl_dir = output_dir / split / "labels"
        if not lbl_dir.exists():
            continue

        counts = defaultdict(int)
        for lbl_file in lbl_dir.iterdir():
            if lbl_file.suffix != ".txt":
                continue
            with open(lbl_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if parts:
                        cls_id = int(parts[0])
                        counts[cls_id] += 1
                        totals[cls_id] += 1

        total_anns = sum(counts.values())
        reporte_lines.append(f"\n{split.upper()} ({total_anns} anotaciones):")
        reporte_lines.append(f"{'ID':>4} {'Marca':<16} {'Count':>6} {'%':>6}")
        reporte_lines.append("-" * 36)

        for cls_id in range(NUM_CLASSES):
            count = counts.get(cls_id, 0)
            pct = (count / total_anns * 100) if total_anns > 0 else 0
            name = UNIFIED_NAMES[cls_id]
            marker = " [!]" if count < 50 and split == "train" else ""
            reporte_lines.append(f"{cls_id:>4} {name:<16} {count:>6} {pct:>5.1f}%{marker}")

    # Totales
    grand_total = sum(totals.values())
    reporte_lines.append(f"\nTOTAL GENERAL ({grand_total} anotaciones):")
    reporte_lines.append(f"{'ID':>4} {'Marca':<16} {'Count':>6} {'%':>6}")
    reporte_lines.append("-" * 36)

    for cls_id in range(NUM_CLASSES):
        count = totals.get(cls_id, 0)
        pct = (count / grand_total * 100) if grand_total > 0 else 0
        name = UNIFIED_NAMES[cls_id]
        marker = " [!] BAJO" if count < 100 else ""
        reporte_lines.append(f"{cls_id:>4} {name:<16} {count:>6} {pct:>5.1f}%{marker}")

    # Imprimir resumen en consola (solo totales)
    print(f"\n{'ID':>4} {'Marca':<16} {'Total':>6}")
    print("-" * 30)
    for cls_id in range(NUM_CLASSES):
        count = totals.get(cls_id, 0)
        name = UNIFIED_NAMES[cls_id]
        bar = "#" * min(int(count / 50), 40)
        print(f"{cls_id:>4} {name:<16} {count:>6} {bar}")

    # Guardar reporte
    reporte_path = output_dir / "reporte_clases.txt"
    with open(reporte_path, "w", encoding="utf-8") as f:
        f.write("\n".join(reporte_lines))

    print(f"\n[+] Reporte guardado en: {reporte_path}")


def main():
    """Función principal del script."""
    print("=" * 70)
    print("   PREPARACIÓN DE DATASET COMBINADO DE MARCAS VEHICULARES")
    print("   30 clases | 3 datasets | YOLOv11")
    print("=" * 70)

    # Verificar que los datasets existan
    print("\n[*] Verificando datasets...")
    for prefix, path in DATASETS.items():
        if path.exists():
            data_yaml = path / "data.yaml"
            if data_yaml.exists():
                with open(data_yaml, "r") as f:
                    config = yaml.safe_load(f)
                nc = config.get("nc", "?")
                names = config.get("names", [])
                print(f"   [OK] {prefix}: {path.name} ({nc} clases: {', '.join(str(n) for n in names[:5])}...)")
            else:
                print(f"   [OK] {prefix}: {path.name} (sin data.yaml)")
        else:
            print(f"   [X] {prefix}: No encontrado en {path}")

    # Limpiar directorio de salida si existe
    if OUTPUT_DIR.exists():
        print(f"\n[*] Limpiando directorio existente: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True)

    # Procesar cada dataset
    total_stats = defaultdict(int)
    process_order = ["ds2", "ds1", "ds3"]  # DS2 primero (más grande)

    print("\n[*] Procesando datasets...")
    for i, prefix in enumerate(process_order, 1):
        dataset_path = DATASETS[prefix]
        if not dataset_path.exists():
            print(f"  [!] Dataset no encontrado: {dataset_path}")
            continue

        class_remap, exclude_classes = DATASET_CONFIG[prefix]
        exclude_info = f" (excluyendo clase Plate)" if exclude_classes else ""

        print(f"\n  [{i}/{len(process_order)}] {dataset_path.name} (prefijo: {prefix}){exclude_info}")

        stats = copiar_dataset_con_remap(
            dataset_path, OUTPUT_DIR, prefix, class_remap, exclude_classes
        )

        for key, value in stats.items():
            total_stats[key] += value

        # Mostrar stats del dataset
        imgs = sum(v for k, v in stats.items() if "images" in k)
        excluded = sum(v for k, v in stats.items() if "excluded" in k)
        print(f"      [OK] Copiados: {imgs} imagenes")
        if excluded:
            print(f"      [X] Excluidos (solo Plate): {excluded} imagenes")

    # Crear archivo de configuración
    print("\n[*] Creando archivo data.yaml...")
    yaml_path = crear_data_yaml(OUTPUT_DIR)
    print(f"   [OK] Creado: {yaml_path}")
    print(f"   - Clases: {NUM_CLASSES}")
    print(f"   - Nombres: {UNIFIED_NAMES[:5]}...{UNIFIED_NAMES[-3:]}")

    # Verificar integridad
    print("\n[*] Verificando integridad del dataset...")
    if verificar_integridad(OUTPUT_DIR):
        print("   [OK] Integridad verificada correctamente")

    # Generar reporte de clases
    generar_reporte_clases(OUTPUT_DIR)

    # Resumen final
    print("\n" + "=" * 70)
    print("   RESUMEN DEL DATASET COMBINADO DE MARCAS")
    print("=" * 70)
    print(f"\nEstadisticas:")
    print(f"   - Imagenes de entrenamiento: {total_stats.get('train_images', 0)}")
    print(f"   - Imagenes de validacion:    {total_stats.get('valid_images', 0)}")
    print(f"   - Imagenes de prueba:        {total_stats.get('test_images', 0)}")

    total_imgs = sum(v for k, v in total_stats.items() if "images" in k)
    total_excluded = sum(v for k, v in total_stats.items() if "excluded" in k)
    print(f"   - Total de imagenes:         {total_imgs}")
    if total_excluded:
        print(f"   - Imagenes excluidas (Plate): {total_excluded}")

    print(f"\nUbicacion del dataset: {OUTPUT_DIR}")
    print(f"Archivo de configuracion: {yaml_path}")
    print(f"Clases: {NUM_CLASSES} marcas vehiculares")

    print("\n[OK] Dataset de marcas preparado exitosamente!")
    print("   Siguiente paso: Ejecutar 06_entrenar_marca.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
