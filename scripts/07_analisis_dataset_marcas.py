"""
07_analisis_dataset_marcas.py
=============================
Script de analisis y visualizacion del dataset combinado de marcas vehiculares.
Genera graficas detalladas para entender la distribucion, balance y
caracteristicas de las anotaciones del dataset.

Uso:
    python scripts/07_analisis_dataset_marcas.py
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import random

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset_marcas_combinado"
OUTPUT_DIR = BASE_DIR / "graficas_dataset"
OUTPUT_DIR.mkdir(exist_ok=True)

SPLITS = ["train", "valid", "test"]
SPLIT_LABELS = {"train": "Train", "valid": "Valid", "test": "Test"}

# Colores consistentes
COLOR_TRAIN = "#2196F3"
COLOR_VALID = "#FF9800"
COLOR_TEST = "#4CAF50"
SPLIT_COLORS = {"train": COLOR_TRAIN, "valid": COLOR_VALID, "test": COLOR_TEST}

# Paleta para barras de clases (gradient azul-rojo)
def get_class_colors(n):
    cmap = plt.colormaps.get_cmap("tab20").resampled(n)
    return [cmap(i) for i in range(n)]


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
def load_config():
    """Carga data.yaml del dataset."""
    yaml_path = DATASET_DIR / "data.yaml"
    if not yaml_path.exists():
        print(f"[X] No se encontro {yaml_path}")
        sys.exit(1)
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def parse_labels(split):
    """Lee todas las anotaciones de un split. Retorna lista de dicts."""
    labels_dir = DATASET_DIR / split / "labels"
    annotations = []
    files_info = []

    if not labels_dir.exists():
        print(f"[!] No existe directorio: {labels_dir}")
        return annotations, files_info

    label_files = sorted(labels_dir.glob("*.txt"))
    for lf in label_files:
        file_annots = []
        with open(lf, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    # Detectar dataset fuente por prefijo del nombre
                    fname = lf.stem
                    if fname.startswith("ds1_"):
                        source = "DS1"
                    elif fname.startswith("ds2_"):
                        source = "DS2"
                    elif fname.startswith("ds3_"):
                        source = "DS3"
                    else:
                        source = "Desconocido"

                    annotations.append({
                        "class_id": cls_id,
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                        "split": split,
                        "source": source,
                        "file": fname,
                    })
                    file_annots.append(cls_id)
        files_info.append({
            "file": lf.stem,
            "num_annotations": len(file_annots),
            "classes": file_annots,
        })

    return annotations, files_info


def load_all_data(class_names):
    """Carga datos de todos los splits."""
    all_annotations = []
    all_files = {}

    for split in SPLITS:
        print(f"  [>] Leyendo {split}...")
        annots, files = parse_labels(split)
        all_annotations.extend(annots)
        all_files[split] = files
        print(f"      {len(annots)} anotaciones en {len(files)} archivos")

    print(f"  [OK] Total: {len(all_annotations)} anotaciones")
    return all_annotations, all_files


# ---------------------------------------------------------------------------
# Graficas
# ---------------------------------------------------------------------------

def plot_01_class_distribution_total(annotations, class_names):
    """Grafica 1: Distribucion total de clases (barras horizontales)."""
    fig, ax = plt.subplots(figsize=(12, 10))

    counts = Counter(a["class_id"] for a in annotations)
    classes = list(range(len(class_names)))
    values = [counts.get(c, 0) for c in classes]

    # Ordenar por cantidad
    sorted_pairs = sorted(zip(classes, values), key=lambda x: x[1])
    sorted_classes, sorted_values = zip(*sorted_pairs)
    sorted_names = [class_names[c] for c in sorted_classes]

    # Colores: gradient segun cantidad
    norm = plt.Normalize(min(sorted_values), max(sorted_values))
    colors = plt.cm.RdYlGn_r(norm(sorted_values))

    bars = ax.barh(range(len(sorted_names)), sorted_values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel("Numero de anotaciones", fontsize=11)
    ax.set_title("Distribucion de clases - Total del dataset", fontsize=14, fontweight="bold")

    # Etiquetas en las barras
    for bar, val in zip(bars, sorted_values):
        pct = val / sum(sorted_values) * 100
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                f"{val} ({pct:.1f}%)", va="center", fontsize=8)

    ax.set_xlim(0, max(sorted_values) * 1.25)

    # Linea de distribucion uniforme ideal
    ideal = len(annotations) / len(class_names)
    ax.axvline(ideal, color="red", linestyle="--", alpha=0.7, label=f"Ideal uniforme ({ideal:.0f})")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_distribucion_clases_total.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 01_distribucion_clases_total.png")


def plot_02_class_distribution_by_split(annotations, class_names):
    """Grafica 2: Distribucion de clases por split (barras agrupadas)."""
    fig, ax = plt.subplots(figsize=(16, 8))

    split_counts = {}
    for split in SPLITS:
        counts = Counter(a["class_id"] for a in annotations if a["split"] == split)
        split_counts[split] = [counts.get(c, 0) for c in range(len(class_names))]

    x = np.arange(len(class_names))
    width = 0.25

    for i, split in enumerate(SPLITS):
        offset = (i - 1) * width
        ax.bar(x + offset, split_counts[split], width,
               label=SPLIT_LABELS[split], color=SPLIT_COLORS[split], alpha=0.85,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Numero de anotaciones", fontsize=11)
    ax.set_title("Distribucion de clases por split", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_distribucion_clases_por_split.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 02_distribucion_clases_por_split.png")


def plot_03_split_distribution(all_files):
    """Grafica 3: Distribucion de imagenes y anotaciones por split (pie + bar)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart - imagenes
    img_counts = [len(all_files[s]) for s in SPLITS]
    labels = [f"{SPLIT_LABELS[s]}\n{c} imgs" for s, c in zip(SPLITS, img_counts)]
    colors = [SPLIT_COLORS[s] for s in SPLITS]

    axes[0].pie(img_counts, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 10})
    axes[0].set_title("Imagenes por split", fontsize=13, fontweight="bold")

    # Bar chart - anotaciones
    annot_counts = [sum(f["num_annotations"] for f in all_files[s]) for s in SPLITS]
    bars = axes[1].bar(
        [SPLIT_LABELS[s] for s in SPLITS], annot_counts,
        color=colors, edgecolor="white", linewidth=1.5
    )
    for bar, val in zip(bars, annot_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Numero de anotaciones", fontsize=11)
    axes[1].set_title("Anotaciones por split", fontsize=13, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_distribucion_splits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 03_distribucion_splits.png")


def plot_04_annotations_per_image(all_files):
    """Grafica 4: Histograma de anotaciones por imagen."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for i, split in enumerate(SPLITS):
        counts = [f["num_annotations"] for f in all_files[split]]

        max_count = max(counts) if counts else 1
        bins = range(0, max_count + 2)

        axes[i].hist(counts, bins=bins, color=SPLIT_COLORS[split],
                     edgecolor="white", alpha=0.85, align="left")
        axes[i].set_title(f"{SPLIT_LABELS[split]} ({len(counts)} imgs)", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Anotaciones por imagen", fontsize=10)
        if i == 0:
            axes[i].set_ylabel("Numero de imagenes", fontsize=10)

        # Stats
        mean_v = np.mean(counts)
        median_v = np.median(counts)
        axes[i].axvline(mean_v, color="red", linestyle="--", alpha=0.7, label=f"Media: {mean_v:.2f}")
        axes[i].axvline(median_v, color="green", linestyle=":", alpha=0.7, label=f"Mediana: {median_v:.0f}")
        axes[i].legend(fontsize=8)
        axes[i].grid(axis="y", alpha=0.3)

    fig.suptitle("Distribucion de anotaciones por imagen", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_anotaciones_por_imagen.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 04_anotaciones_por_imagen.png")


def plot_05_bbox_size_distribution(annotations, class_names):
    """Grafica 5: Distribucion de tamanos de bounding boxes."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    widths = [a["width"] for a in annotations]
    heights = [a["height"] for a in annotations]
    areas = [a["width"] * a["height"] for a in annotations]

    # Scatter: ancho vs alto
    ax = axes[0]
    ax.scatter(widths, heights, alpha=0.05, s=3, c="#2196F3")
    ax.set_xlabel("Ancho (normalizado)", fontsize=10)
    ax.set_ylabel("Alto (normalizado)", fontsize=10)
    ax.set_title("Ancho vs Alto de BBoxes", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [0, 1], "r--", alpha=0.4, label="Cuadrado")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Histograma de areas
    ax = axes[1]
    ax.hist(areas, bins=50, color="#9C27B0", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Area (normalizada)", fontsize=10)
    ax.set_ylabel("Frecuencia", fontsize=10)
    ax.set_title("Distribucion de areas de BBoxes", fontsize=12, fontweight="bold")
    ax.axvline(np.mean(areas), color="red", linestyle="--", alpha=0.7, label=f"Media: {np.mean(areas):.4f}")
    ax.axvline(np.median(areas), color="green", linestyle=":", alpha=0.7, label=f"Mediana: {np.median(areas):.4f}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Histograma de aspect ratios
    ax = axes[2]
    aspect_ratios = [a["width"] / max(a["height"], 1e-6) for a in annotations]
    # Clip extremos para mejor visualizacion
    ar_clipped = [min(ar, 5) for ar in aspect_ratios]
    ax.hist(ar_clipped, bins=60, color="#FF5722", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Aspect Ratio (ancho/alto)", fontsize=10)
    ax.set_ylabel("Frecuencia", fontsize=10)
    ax.set_title("Distribucion de Aspect Ratio", fontsize=12, fontweight="bold")
    ax.axvline(1.0, color="blue", linestyle="--", alpha=0.5, label="Cuadrado (1:1)")
    ax.axvline(np.median(aspect_ratios), color="green", linestyle=":", alpha=0.7,
               label=f"Mediana: {np.median(aspect_ratios):.2f}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Analisis de tamano de Bounding Boxes", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_tamano_bboxes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 05_tamano_bboxes.png")


def plot_06_bbox_center_heatmap(annotations):
    """Grafica 6: Heatmap de centros de bounding boxes."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    titles = ["Todos", "Train", "Valid", "Test"]
    filters = [
        lambda a: True,
        lambda a: a["split"] == "train",
        lambda a: a["split"] == "valid",
        lambda a: a["split"] == "test",
    ]

    for ax, title, filt in zip(axes, titles, filters):
        subset = [a for a in annotations if filt(a)]
        xs = [a["x_center"] for a in subset]
        ys = [a["y_center"] for a in subset]

        heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=50, range=[[0, 1], [0, 1]])

        im = ax.imshow(heatmap.T, origin="lower", cmap="hot",
                       extent=[0, 1, 0, 1], aspect="equal", interpolation="gaussian")
        ax.set_title(f"{title} ({len(subset)})", fontsize=11, fontweight="bold")
        ax.set_xlabel("X center", fontsize=9)
        ax.set_ylabel("Y center", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Heatmap de centros de BBoxes (posicion en imagen)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_heatmap_centros_bbox.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 06_heatmap_centros_bbox.png")


def plot_07_source_dataset_contribution(annotations, class_names):
    """Grafica 7: Contribucion de cada dataset fuente por clase (stacked bar)."""
    fig, ax = plt.subplots(figsize=(16, 8))

    sources = ["DS1", "DS2", "DS3"]
    source_colors = {"DS1": "#E91E63", "DS2": "#3F51B5", "DS3": "#009688"}

    class_source_counts = defaultdict(lambda: defaultdict(int))
    for a in annotations:
        class_source_counts[a["class_id"]][a["source"]] += 1

    x = np.arange(len(class_names))
    bottom = np.zeros(len(class_names))

    for src in sources:
        values = [class_source_counts[c][src] for c in range(len(class_names))]
        ax.bar(x, values, bottom=bottom, label=src, color=source_colors[src],
               edgecolor="white", linewidth=0.5, alpha=0.85)
        bottom += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Numero de anotaciones", fontsize=11)
    ax.set_title("Contribucion de cada dataset fuente por clase", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, title="Dataset fuente")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_contribucion_datasets_fuente.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 07_contribucion_datasets_fuente.png")


def plot_08_class_imbalance(annotations, class_names):
    """Grafica 8: Ratio de desbalance de clases y metricas."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    counts = Counter(a["class_id"] for a in annotations)
    values = [counts.get(c, 0) for c in range(len(class_names))]

    # Ratio max/min para cada clase vs la media
    mean_count = np.mean(values)
    ratios = [v / mean_count for v in values]

    # Grafica de ratios
    ax = axes[0]
    sorted_pairs = sorted(zip(class_names, ratios), key=lambda x: x[1])
    names_sorted, ratios_sorted = zip(*sorted_pairs)

    colors = ["#F44336" if r < 0.5 else "#FF9800" if r < 0.8 else "#4CAF50" if r < 1.5 else "#2196F3"
              for r in ratios_sorted]

    bars = ax.barh(range(len(names_sorted)), ratios_sorted, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.axvline(1.0, color="black", linestyle="-", alpha=0.5, label="Promedio (1.0)")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="50% del promedio")
    ax.axvline(2.0, color="blue", linestyle="--", alpha=0.5, label="200% del promedio")
    ax.set_xlabel("Ratio vs promedio", fontsize=10)
    ax.set_title("Ratio de cada clase vs promedio", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # Metricas de balance
    ax = axes[1]
    ax.axis("off")

    max_class = class_names[np.argmax(values)]
    min_class = class_names[np.argmin(values)]
    max_val = max(values)
    min_val = min(values)
    imbalance_ratio = max_val / max(min_val, 1)

    # Calcular entropia normalizada (1 = perfectamente balanceado)
    probs = np.array(values) / sum(values)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(class_names))
    norm_entropy = entropy / max_entropy

    # Gini coefficient
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumulative = np.cumsum(sorted_vals)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_vals))) / (n * np.sum(sorted_vals)) - (n + 1) / n

    metrics_text = (
        f"METRICAS DE BALANCE DEL DATASET\n"
        f"{'='*45}\n\n"
        f"Total de anotaciones:  {sum(values):,}\n"
        f"Numero de clases:      {len(class_names)}\n"
        f"Promedio por clase:    {mean_count:,.0f}\n"
        f"Desviacion estandar:   {np.std(values):,.0f}\n\n"
        f"Clase mayoritaria:     {max_class} ({max_val:,})\n"
        f"Clase minoritaria:     {min_class} ({min_val:,})\n"
        f"Ratio max/min:         {imbalance_ratio:.1f}x\n\n"
        f"Entropia normalizada:  {norm_entropy:.4f}\n"
        f"  (1.0 = perfecto balance)\n\n"
        f"Coeficiente de Gini:   {gini:.4f}\n"
        f"  (0.0 = perfecto balance)\n\n"
        f"{'='*45}\n"
        f"Clases sub-representadas (<50% promedio):\n"
    )

    under = [(class_names[i], v) for i, v in enumerate(values) if v < mean_count * 0.5]
    under.sort(key=lambda x: x[1])
    for name, count in under:
        metrics_text += f"  - {name}: {count} ({count/mean_count*100:.0f}% del promedio)\n"

    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#CCCCCC"))

    fig.suptitle("Analisis de desbalance de clases", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_analisis_desbalance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 08_analisis_desbalance.png")


def plot_09_bbox_size_per_class(annotations, class_names):
    """Grafica 9: Box plot de area de bbox por clase."""
    fig, ax = plt.subplots(figsize=(16, 8))

    class_areas = defaultdict(list)
    for a in annotations:
        area = a["width"] * a["height"]
        class_areas[a["class_id"]].append(area)

    data = [class_areas[c] for c in range(len(class_names))]

    bp = ax.boxplot(data, tick_labels=class_names, patch_artist=True,
                    showfliers=False, medianprops=dict(color="red", linewidth=1.5))

    colors = get_class_colors(len(class_names))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Area de BBox (normalizada)", fontsize=11)
    ax.set_title("Distribucion de area de BBox por clase", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "09_area_bbox_por_clase.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 09_area_bbox_por_clase.png")


def plot_10_split_consistency(annotations, class_names):
    """Grafica 10: Consistencia de proporcion de clases entre splits."""
    fig, ax = plt.subplots(figsize=(14, 8))

    split_proportions = {}
    for split in SPLITS:
        subset = [a for a in annotations if a["split"] == split]
        total = len(subset)
        counts = Counter(a["class_id"] for a in subset)
        split_proportions[split] = [counts.get(c, 0) / total * 100 for c in range(len(class_names))]

    x = np.arange(len(class_names))
    width = 0.25

    for i, split in enumerate(SPLITS):
        offset = (i - 1) * width
        ax.bar(x + offset, split_proportions[split], width,
               label=SPLIT_LABELS[split], color=SPLIT_COLORS[split], alpha=0.85,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Proporcion (%)", fontsize=11)
    ax.set_title("Consistencia de distribuciones entre splits", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "10_consistencia_splits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 10_consistencia_splits.png")


def plot_11_source_images_per_split(all_files):
    """Grafica 11: Distribucion de dataset fuente por split."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    source_colors = {"DS1": "#E91E63", "DS2": "#3F51B5", "DS3": "#009688", "Desconocido": "#9E9E9E"}

    for i, split in enumerate(SPLITS):
        source_counts = Counter()
        for f in all_files[split]:
            fname = f["file"]
            if fname.startswith("ds1_"):
                source_counts["DS1"] += 1
            elif fname.startswith("ds2_"):
                source_counts["DS2"] += 1
            elif fname.startswith("ds3_"):
                source_counts["DS3"] += 1
            else:
                source_counts["Desconocido"] += 1

        sources = sorted(source_counts.keys())
        vals = [source_counts[s] for s in sources]
        colors = [source_colors[s] for s in sources]

        axes[i].pie(vals, labels=[f"{s}\n{v}" for s, v in zip(sources, vals)],
                    colors=colors, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
        axes[i].set_title(f"{SPLIT_LABELS[split]}", fontsize=12, fontweight="bold")

    fig.suptitle("Imagenes por dataset fuente en cada split", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "11_fuentes_por_split.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 11_fuentes_por_split.png")


def plot_12_cooccurrence_matrix(all_files, class_names):
    """Grafica 12: Matriz de co-ocurrencia de clases en la misma imagen."""
    n_classes = len(class_names)
    cooccurrence = np.zeros((n_classes, n_classes), dtype=int)

    for split in SPLITS:
        for f in all_files[split]:
            unique_classes = list(set(f["classes"]))
            for i in range(len(unique_classes)):
                for j in range(len(unique_classes)):
                    cooccurrence[unique_classes[i]][unique_classes[j]] += 1

    # Solo mostrar si hay co-ocurrencias interesantes (off-diagonal > 0)
    off_diag_sum = np.sum(cooccurrence) - np.trace(cooccurrence)

    fig, ax = plt.subplots(figsize=(14, 12))

    if off_diag_sum > 0:
        # Normalizar por diagonal para ver proporcion
        diag = np.diag(cooccurrence).astype(float)
        diag[diag == 0] = 1  # evitar division por cero
        cooc_norm = cooccurrence / diag[:, np.newaxis]
        np.fill_diagonal(cooc_norm, 0)  # quitar diagonal para visualizar

        im = ax.imshow(cooc_norm, cmap="YlOrRd", aspect="equal")
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Proporcion de co-ocurrencia")
        ax.set_title("Matriz de co-ocurrencia de clases\n(proporcion de veces que aparecen juntas)",
                     fontsize=13, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No hay co-ocurrencia significativa\n(la mayoria de imagenes tiene solo 1 clase)",
                transform=ax.transAxes, ha="center", va="center", fontsize=14)
        ax.set_title("Matriz de co-ocurrencia de clases", fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "12_coocurrencia_clases.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 12_coocurrencia_clases.png")


def plot_13_bbox_position_per_class(annotations, class_names):
    """Grafica 13: Posicion promedio del centro de bbox por clase."""
    fig, ax = plt.subplots(figsize=(12, 10))

    class_positions = defaultdict(lambda: {"x": [], "y": []})
    for a in annotations:
        class_positions[a["class_id"]]["x"].append(a["x_center"])
        class_positions[a["class_id"]]["y"].append(a["y_center"])

    colors = get_class_colors(len(class_names))

    for c in range(len(class_names)):
        if class_positions[c]["x"]:
            mean_x = np.mean(class_positions[c]["x"])
            mean_y = np.mean(class_positions[c]["y"])
            std_x = np.std(class_positions[c]["x"])
            std_y = np.std(class_positions[c]["y"])

            ax.scatter(mean_x, mean_y, s=100, color=colors[c], zorder=5, edgecolors="black", linewidths=0.5)
            ax.errorbar(mean_x, mean_y, xerr=std_x, yerr=std_y, fmt="none",
                       color=colors[c], alpha=0.3, capsize=2)
            ax.annotate(class_names[c], (mean_x, mean_y), fontsize=7,
                       textcoords="offset points", xytext=(5, 5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("X center (normalizado)", fontsize=11)
    ax.set_ylabel("Y center (normalizado)", fontsize=11)
    ax.set_title("Posicion promedio del centro de BBox por clase\n(con barras de desviacion estandar)",
                fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "13_posicion_centro_por_clase.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 13_posicion_centro_por_clase.png")


def plot_14_sample_images(class_names):
    """Grafica 14: Mosaico de imagenes de ejemplo por clase."""
    fig, axes = plt.subplots(5, 6, figsize=(18, 15))
    axes = axes.flatten()

    images_dir = DATASET_DIR / "train" / "images"
    labels_dir = DATASET_DIR / "train" / "labels"

    # Para cada clase, buscar una imagen de ejemplo
    class_examples = {c: None for c in range(len(class_names))}

    label_files = list(labels_dir.glob("*.txt"))
    random.seed(42)
    random.shuffle(label_files)

    for lf in label_files:
        if all(v is not None for v in class_examples.values()):
            break
        with open(lf, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if class_examples[cls_id] is None:
                        # Buscar la imagen correspondiente
                        stem = lf.stem
                        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                            img_path = images_dir / (stem + ext)
                            if img_path.exists():
                                class_examples[cls_id] = (img_path, float(parts[1]),
                                                          float(parts[2]), float(parts[3]), float(parts[4]))
                                break

    for c in range(len(class_names)):
        ax = axes[c]
        if class_examples[c] is not None:
            img_path, xc, yc, w, h = class_examples[c]
            try:
                img = Image.open(img_path)
                img_w, img_h = img.size
                ax.imshow(img)

                # Dibujar bbox
                x1 = (xc - w/2) * img_w
                y1 = (yc - h/2) * img_h
                rect_w = w * img_w
                rect_h = h * img_h
                rect = plt.Rectangle((x1, y1), rect_w, rect_h, linewidth=2,
                                    edgecolor="lime", facecolor="none")
                ax.add_patch(rect)
            except Exception:
                ax.text(0.5, 0.5, "Error al cargar", ha="center", va="center",
                       transform=ax.transAxes, fontsize=8)
        else:
            ax.text(0.5, 0.5, "Sin ejemplo", ha="center", va="center",
                   transform=ax.transAxes, fontsize=8)

        ax.set_title(class_names[c], fontsize=9, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Ejemplo de imagen por clase (con BBox)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "14_ejemplos_por_clase.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 14_ejemplos_por_clase.png")


def plot_15_summary_dashboard(annotations, all_files, class_names):
    """Grafica 15: Dashboard resumen con las metricas clave."""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # --- Panel 1: KPIs ---
    ax_kpi = fig.add_subplot(gs[0, 0])
    ax_kpi.axis("off")

    total_imgs = sum(len(all_files[s]) for s in SPLITS)
    total_annots = len(annotations)
    counts = Counter(a["class_id"] for a in annotations)
    values = list(counts.values())

    kpi_text = (
        f"RESUMEN DEL DATASET\n"
        f"{'='*35}\n\n"
        f"Imagenes totales:     {total_imgs:,}\n"
        f"Anotaciones totales:  {total_annots:,}\n"
        f"Clases:               {len(class_names)}\n"
        f"Annot/imagen (prom):  {total_annots/total_imgs:.2f}\n\n"
        f"Train: {len(all_files['train']):,} imgs\n"
        f"Valid: {len(all_files['valid']):,} imgs\n"
        f"Test:  {len(all_files['test']):,} imgs\n\n"
        f"Clase mayor: {class_names[max(counts, key=counts.get)]}\n"
        f"  ({max(values):,} annot)\n"
        f"Clase menor: {class_names[min(counts, key=counts.get)]}\n"
        f"  ({min(values):,} annot)\n"
        f"Ratio desbalance: {max(values)/min(values):.1f}x"
    )
    ax_kpi.text(0.05, 0.95, kpi_text, transform=ax_kpi.transAxes, fontsize=10,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", edgecolor="#90CAF9"))

    # --- Panel 2: Top 10 clases ---
    ax_top = fig.add_subplot(gs[0, 1])
    top10 = counts.most_common(10)
    names_t = [class_names[c] for c, _ in top10]
    vals_t = [v for _, v in top10]
    ax_top.barh(range(len(names_t)), vals_t, color="#2196F3", edgecolor="white")
    ax_top.set_yticks(range(len(names_t)))
    ax_top.set_yticklabels(names_t, fontsize=9)
    ax_top.invert_yaxis()
    ax_top.set_title("Top 10 clases", fontsize=12, fontweight="bold")
    ax_top.set_xlabel("Anotaciones")
    for j, v in enumerate(vals_t):
        ax_top.text(v + 20, j, str(v), va="center", fontsize=8)

    # --- Panel 3: Split pie ---
    ax_pie = fig.add_subplot(gs[0, 2])
    img_counts = [len(all_files[s]) for s in SPLITS]
    ax_pie.pie(img_counts, labels=[SPLIT_LABELS[s] for s in SPLITS],
               colors=[SPLIT_COLORS[s] for s in SPLITS], autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 10})
    ax_pie.set_title("Distribucion de splits", fontsize=12, fontweight="bold")

    # --- Panel 4: Histograma areas ---
    ax_area = fig.add_subplot(gs[1, 0])
    areas = [a["width"] * a["height"] for a in annotations]
    ax_area.hist(areas, bins=50, color="#9C27B0", edgecolor="white", alpha=0.85)
    ax_area.set_title("Distribucion de areas BBox", fontsize=12, fontweight="bold")
    ax_area.set_xlabel("Area normalizada")
    ax_area.set_ylabel("Frecuencia")
    ax_area.grid(axis="y", alpha=0.3)

    # --- Panel 5: Heatmap centros ---
    ax_heat = fig.add_subplot(gs[1, 1])
    xs = [a["x_center"] for a in annotations]
    ys = [a["y_center"] for a in annotations]
    heatmap, _, _ = np.histogram2d(xs, ys, bins=40, range=[[0, 1], [0, 1]])
    ax_heat.imshow(heatmap.T, origin="lower", cmap="hot", extent=[0, 1, 0, 1],
                   aspect="equal", interpolation="gaussian")
    ax_heat.set_title("Heatmap centros BBox", fontsize=12, fontweight="bold")
    ax_heat.set_xlabel("X")
    ax_heat.set_ylabel("Y")

    # --- Panel 6: Contribucion fuentes ---
    ax_src = fig.add_subplot(gs[1, 2])
    source_counts = Counter(a["source"] for a in annotations)
    sources = sorted(source_counts.keys())
    src_vals = [source_counts[s] for s in sources]
    src_colors = {"DS1": "#E91E63", "DS2": "#3F51B5", "DS3": "#009688", "Desconocido": "#9E9E9E"}
    ax_src.pie(src_vals, labels=[f"{s}\n{v:,}" for s, v in zip(sources, src_vals)],
               colors=[src_colors.get(s, "#9E9E9E") for s in sources],
               autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
    ax_src.set_title("Anotaciones por fuente", fontsize=12, fontweight="bold")

    fig.suptitle("Dashboard - Dataset Marcas Vehiculares Combinado",
                fontsize=16, fontweight="bold", y=1.01)
    fig.savefig(OUTPUT_DIR / "15_dashboard_resumen.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 15_dashboard_resumen.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  ANALISIS DEL DATASET COMBINADO DE MARCAS")
    print("=" * 60)
    print()

    # Cargar configuracion
    print("[1/3] Cargando configuracion...")
    config = load_config()
    class_names = config["names"]
    print(f"  [OK] {len(class_names)} clases definidas")
    print()

    # Cargar datos
    print("[2/3] Cargando anotaciones de todos los splits...")
    annotations, all_files = load_all_data(class_names)
    print()

    # Generar graficas
    print("[3/3] Generando graficas...")
    print(f"  Directorio de salida: {OUTPUT_DIR}")
    print()

    plot_01_class_distribution_total(annotations, class_names)
    plot_02_class_distribution_by_split(annotations, class_names)
    plot_03_split_distribution(all_files)
    plot_04_annotations_per_image(all_files)
    plot_05_bbox_size_distribution(annotations, class_names)
    plot_06_bbox_center_heatmap(annotations)
    plot_07_source_dataset_contribution(annotations, class_names)
    plot_08_class_imbalance(annotations, class_names)
    plot_09_bbox_size_per_class(annotations, class_names)
    plot_10_split_consistency(annotations, class_names)
    plot_11_source_images_per_split(all_files)
    plot_12_cooccurrence_matrix(all_files, class_names)
    plot_13_bbox_position_per_class(annotations, class_names)
    plot_14_sample_images(class_names)
    plot_15_summary_dashboard(annotations, all_files, class_names)

    print()
    print("=" * 60)
    print(f"  [OK] 15 graficas generadas en: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
