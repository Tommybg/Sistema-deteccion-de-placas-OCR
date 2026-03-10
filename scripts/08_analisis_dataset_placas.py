"""
08_analisis_dataset_placas.py
=============================
Script de analisis y visualizacion del dataset combinado de deteccion de placas
vehiculares colombianas.
Al ser un dataset de 1 sola clase, las graficas se enfocan en:
- Distribucion espacial y de tamano de las placas
- Cantidad de placas por imagen
- Comparacion entre splits y datasets fuente
- Calidad y diversidad de las anotaciones

Uso:
    python scripts/08_analisis_dataset_placas.py
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import random

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset_combinado"
OUTPUT_DIR = BASE_DIR / "graficas_dataset_placas"
OUTPUT_DIR.mkdir(exist_ok=True)

SPLITS = ["train", "valid", "test"]
SPLIT_LABELS = {"train": "Train", "valid": "Valid", "test": "Test"}

COLOR_TRAIN = "#2196F3"
COLOR_VALID = "#FF9800"
COLOR_TEST = "#4CAF50"
SPLIT_COLORS = {"train": COLOR_TRAIN, "valid": COLOR_VALID, "test": COLOR_TEST}


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
    """Lee todas las anotaciones de un split."""
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
        })

    return annotations, files_info


def load_all_data():
    """Carga datos de todos los splits."""
    all_annotations = []
    all_files = {}

    for split in SPLITS:
        print(f"  [>] Leyendo {split}...")
        annots, files = parse_labels(split)
        all_annotations.extend(annots)
        all_files[split] = files
        print(f"      {len(annots)} anotaciones en {len(files)} imagenes")

    print(f"  [OK] Total: {len(all_annotations)} anotaciones en "
          f"{sum(len(all_files[s]) for s in SPLITS)} imagenes")
    return all_annotations, all_files


def get_image_sizes(split, sample_n=None):
    """Lee dimensiones reales de imagenes de un split."""
    images_dir = DATASET_DIR / split / "images"
    if not images_dir.exists():
        return []

    img_files = list(images_dir.glob("*.*"))
    if sample_n and len(img_files) > sample_n:
        random.seed(42)
        img_files = random.sample(img_files, sample_n)

    sizes = []
    for img_path in img_files:
        try:
            with Image.open(img_path) as img:
                sizes.append({"file": img_path.stem, "width": img.width, "height": img.height})
        except Exception:
            pass
    return sizes


# ---------------------------------------------------------------------------
# Graficas
# ---------------------------------------------------------------------------

def plot_01_split_distribution(all_files, annotations):
    """Grafica 1: Distribucion de imagenes y anotaciones por split."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Pie - imagenes
    img_counts = [len(all_files[s]) for s in SPLITS]
    labels = [f"{SPLIT_LABELS[s]}\n{c} imgs" for s, c in zip(SPLITS, img_counts)]
    colors = [SPLIT_COLORS[s] for s in SPLITS]

    axes[0].pie(img_counts, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 11})
    axes[0].set_title("Imagenes por split", fontsize=13, fontweight="bold")

    # Pie - anotaciones
    annot_counts = [len([a for a in annotations if a["split"] == s]) for s in SPLITS]
    labels_a = [f"{SPLIT_LABELS[s]}\n{c} annot" for s, c in zip(SPLITS, annot_counts)]

    axes[1].pie(annot_counts, labels=labels_a, colors=colors, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 11})
    axes[1].set_title("Anotaciones por split", fontsize=13, fontweight="bold")

    # Bar comparativo
    x = np.arange(len(SPLITS))
    width = 0.35
    bars1 = axes[2].bar(x - width/2, img_counts, width, label="Imagenes", color="#42A5F5", edgecolor="white")
    bars2 = axes[2].bar(x + width/2, annot_counts, width, label="Anotaciones", color="#EF5350", edgecolor="white")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([SPLIT_LABELS[s] for s in SPLITS], fontsize=11)
    axes[2].set_ylabel("Cantidad", fontsize=11)
    axes[2].set_title("Imagenes vs Anotaciones", fontsize=13, fontweight="bold")
    axes[2].legend(fontsize=10)
    axes[2].grid(axis="y", alpha=0.3)
    for bar in bars1:
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     str(int(bar.get_height())), ha="center", fontsize=9, fontweight="bold")
    for bar in bars2:
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     str(int(bar.get_height())), ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Distribucion del dataset de placas por split", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_distribucion_splits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 01_distribucion_splits.png")


def plot_02_annotations_per_image(all_files):
    """Grafica 2: Histograma de placas por imagen."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Por split individual
    for i, split in enumerate(SPLITS):
        ax = axes[i // 2][i % 2]
        counts = [f["num_annotations"] for f in all_files[split]]
        max_c = max(counts) if counts else 1
        bins = range(0, max_c + 2)

        ax.hist(counts, bins=bins, color=SPLIT_COLORS[split], edgecolor="white",
                alpha=0.85, align="left")
        ax.set_title(f"{SPLIT_LABELS[split]} ({len(counts)} imgs)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Placas por imagen", fontsize=10)
        ax.set_ylabel("Num. imagenes", fontsize=10)

        mean_v = np.mean(counts)
        ax.axvline(mean_v, color="red", linestyle="--", alpha=0.7, label=f"Media: {mean_v:.2f}")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Stats box
        stats_text = (f"Min: {min(counts)}\nMax: {max(counts)}\n"
                      f"Media: {mean_v:.2f}\nMediana: {np.median(counts):.0f}")
        ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                va="top", ha="right", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Total
    ax = axes[1][1]
    all_counts = []
    for split in SPLITS:
        all_counts.extend([f["num_annotations"] for f in all_files[split]])
    max_c = max(all_counts) if all_counts else 1
    bins = range(0, max_c + 2)
    ax.hist(all_counts, bins=bins, color="#9C27B0", edgecolor="white", alpha=0.85, align="left")
    ax.set_title(f"Total ({len(all_counts)} imgs)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Placas por imagen", fontsize=10)
    ax.set_ylabel("Num. imagenes", fontsize=10)
    mean_v = np.mean(all_counts)
    ax.axvline(mean_v, color="red", linestyle="--", alpha=0.7, label=f"Media: {mean_v:.2f}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Porcentaje con 0, 1, 2+ placas
    c0 = sum(1 for c in all_counts if c == 0)
    c1 = sum(1 for c in all_counts if c == 1)
    c2p = sum(1 for c in all_counts if c >= 2)
    stats_text = (f"0 placas: {c0} ({c0/len(all_counts)*100:.1f}%)\n"
                  f"1 placa:  {c1} ({c1/len(all_counts)*100:.1f}%)\n"
                  f"2+ placas: {c2p} ({c2p/len(all_counts)*100:.1f}%)")
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
            va="top", ha="right", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.suptitle("Distribucion de placas por imagen", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_placas_por_imagen.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 02_placas_por_imagen.png")


def plot_03_bbox_size_analysis(annotations):
    """Grafica 3: Analisis de tamano de bounding boxes de placas."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    widths = np.array([a["width"] for a in annotations])
    heights = np.array([a["height"] for a in annotations])
    areas = widths * heights
    aspect_ratios = widths / np.maximum(heights, 1e-6)

    # --- Row 1 ---

    # Scatter: ancho vs alto
    ax = axes[0][0]
    ax.scatter(widths, heights, alpha=0.15, s=8, c="#2196F3")
    ax.set_xlabel("Ancho (normalizado)", fontsize=10)
    ax.set_ylabel("Alto (normalizado)", fontsize=10)
    ax.set_title("Ancho vs Alto de placas", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(widths.max() * 1.1, 0.5))
    ax.set_ylim(0, max(heights.max() * 1.1, 0.5))
    ax.plot([0, 1], [0, 1], "r--", alpha=0.3, label="Cuadrado (1:1)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Histograma de anchos
    ax = axes[0][1]
    ax.hist(widths, bins=50, color="#42A5F5", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(widths), color="red", linestyle="--", label=f"Media: {np.mean(widths):.3f}")
    ax.axvline(np.median(widths), color="green", linestyle=":", label=f"Mediana: {np.median(widths):.3f}")
    ax.set_xlabel("Ancho (normalizado)", fontsize=10)
    ax.set_ylabel("Frecuencia", fontsize=10)
    ax.set_title("Distribucion de anchos", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Histograma de altos
    ax = axes[0][2]
    ax.hist(heights, bins=50, color="#66BB6A", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(heights), color="red", linestyle="--", label=f"Media: {np.mean(heights):.3f}")
    ax.axvline(np.median(heights), color="green", linestyle=":", label=f"Mediana: {np.median(heights):.3f}")
    ax.set_xlabel("Alto (normalizado)", fontsize=10)
    ax.set_ylabel("Frecuencia", fontsize=10)
    ax.set_title("Distribucion de altos", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- Row 2 ---

    # Histograma de areas
    ax = axes[1][0]
    ax.hist(areas, bins=50, color="#AB47BC", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(areas), color="red", linestyle="--", label=f"Media: {np.mean(areas):.4f}")
    ax.axvline(np.median(areas), color="green", linestyle=":", label=f"Mediana: {np.median(areas):.4f}")
    ax.set_xlabel("Area (normalizada)", fontsize=10)
    ax.set_ylabel("Frecuencia", fontsize=10)
    ax.set_title("Distribucion de areas", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Histograma de aspect ratios
    ax = axes[1][1]
    ar_clipped = np.clip(aspect_ratios, 0, 8)
    ax.hist(ar_clipped, bins=60, color="#FF7043", edgecolor="white", alpha=0.85)
    ax.axvline(1.0, color="blue", linestyle="--", alpha=0.5, label="Cuadrado (1:1)")
    ax.axvline(np.median(aspect_ratios), color="green", linestyle=":",
               label=f"Mediana: {np.median(aspect_ratios):.2f}")
    # Placas colombianas tipicas: ~2:1 a 3:1
    ax.axvspan(2.0, 3.5, alpha=0.1, color="green", label="Rango tipico placas (2:1 - 3.5:1)")
    ax.set_xlabel("Aspect Ratio (ancho/alto)", fontsize=10)
    ax.set_ylabel("Frecuencia", fontsize=10)
    ax.set_title("Distribucion de Aspect Ratio", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # Box plots comparativos por split
    ax = axes[1][2]
    data_by_split = {
        "areas": [],
        "labels": [],
    }
    for split in SPLITS:
        split_areas = [a["width"] * a["height"] for a in annotations if a["split"] == split]
        data_by_split["areas"].append(split_areas)
        data_by_split["labels"].append(SPLIT_LABELS[split])

    bp = ax.boxplot(data_by_split["areas"], tick_labels=data_by_split["labels"],
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=2, alpha=0.3),
                    medianprops=dict(color="red", linewidth=2))
    for patch, split in zip(bp["boxes"], SPLITS):
        patch.set_facecolor(SPLIT_COLORS[split])
        patch.set_alpha(0.7)

    ax.set_ylabel("Area (normalizada)", fontsize=10)
    ax.set_title("Area de placa por split", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Analisis de tamano de BBoxes de placas", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_tamano_bboxes_placas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 03_tamano_bboxes_placas.png")


def plot_04_bbox_center_heatmap(annotations):
    """Grafica 4: Heatmap de posicion de centros de placas."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

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

        heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=30, range=[[0, 1], [0, 1]])
        im = ax.imshow(heatmap.T, origin="lower", cmap="hot",
                       extent=[0, 1, 0, 1], aspect="equal", interpolation="gaussian")
        ax.set_title(f"{title} (n={len(subset)})", fontsize=11, fontweight="bold")
        ax.set_xlabel("X center", fontsize=9)
        ax.set_ylabel("Y center", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Heatmap de posicion de centros de placas", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_heatmap_centros_placas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 04_heatmap_centros_placas.png")


def plot_05_bbox_position_distribution(annotations):
    """Grafica 5: Distribucion marginal de posiciones X e Y de centros."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    xs = [a["x_center"] for a in annotations]
    ys = [a["y_center"] for a in annotations]

    # Distribucion X
    axes[0][0].hist(xs, bins=50, color="#42A5F5", edgecolor="white", alpha=0.85)
    axes[0][0].set_xlabel("X center (normalizado)", fontsize=10)
    axes[0][0].set_ylabel("Frecuencia", fontsize=10)
    axes[0][0].set_title("Distribucion horizontal del centro", fontsize=12, fontweight="bold")
    axes[0][0].axvline(0.5, color="red", linestyle="--", alpha=0.5, label="Centro imagen")
    axes[0][0].axvline(np.mean(xs), color="green", linestyle=":", label=f"Media: {np.mean(xs):.3f}")
    axes[0][0].legend(fontsize=8)
    axes[0][0].grid(axis="y", alpha=0.3)

    # Distribucion Y
    axes[0][1].hist(ys, bins=50, color="#66BB6A", edgecolor="white", alpha=0.85)
    axes[0][1].set_xlabel("Y center (normalizado)", fontsize=10)
    axes[0][1].set_ylabel("Frecuencia", fontsize=10)
    axes[0][1].set_title("Distribucion vertical del centro", fontsize=12, fontweight="bold")
    axes[0][1].axvline(0.5, color="red", linestyle="--", alpha=0.5, label="Centro imagen")
    axes[0][1].axvline(np.mean(ys), color="green", linestyle=":", label=f"Media: {np.mean(ys):.3f}")
    axes[0][1].legend(fontsize=8)
    axes[0][1].grid(axis="y", alpha=0.3)

    # X por split
    for split in SPLITS:
        xs_s = [a["x_center"] for a in annotations if a["split"] == split]
        axes[1][0].hist(xs_s, bins=40, alpha=0.5, label=SPLIT_LABELS[split],
                        color=SPLIT_COLORS[split], edgecolor="white")
    axes[1][0].set_xlabel("X center (normalizado)", fontsize=10)
    axes[1][0].set_ylabel("Frecuencia", fontsize=10)
    axes[1][0].set_title("Distribucion X por split", fontsize=12, fontweight="bold")
    axes[1][0].legend(fontsize=9)
    axes[1][0].grid(axis="y", alpha=0.3)

    # Y por split
    for split in SPLITS:
        ys_s = [a["y_center"] for a in annotations if a["split"] == split]
        axes[1][1].hist(ys_s, bins=40, alpha=0.5, label=SPLIT_LABELS[split],
                        color=SPLIT_COLORS[split], edgecolor="white")
    axes[1][1].set_xlabel("Y center (normalizado)", fontsize=10)
    axes[1][1].set_ylabel("Frecuencia", fontsize=10)
    axes[1][1].set_title("Distribucion Y por split", fontsize=12, fontweight="bold")
    axes[1][1].legend(fontsize=9)
    axes[1][1].grid(axis="y", alpha=0.3)

    fig.suptitle("Distribucion espacial de centros de placas", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_distribucion_posicion_centros.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 05_distribucion_posicion_centros.png")


def plot_06_source_dataset_analysis(annotations, all_files):
    """Grafica 6: Analisis de contribucion de cada dataset fuente."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    source_colors = {"DS1": "#E91E63", "DS2": "#3F51B5", "DS3": "#009688", "Desconocido": "#9E9E9E"}

    # Pie - anotaciones por fuente
    ax = axes[0][0]
    source_annot = Counter(a["source"] for a in annotations)
    sources = sorted(source_annot.keys())
    vals = [source_annot[s] for s in sources]
    colors = [source_colors.get(s, "#9E9E9E") for s in sources]
    ax.pie(vals, labels=[f"{s}\n{v} annot" for s, v in zip(sources, vals)],
           colors=colors, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
    ax.set_title("Anotaciones por fuente", fontsize=12, fontweight="bold")

    # Pie - imagenes por fuente
    ax = axes[0][1]
    source_imgs = Counter()
    for split in SPLITS:
        for f in all_files[split]:
            fname = f["file"]
            if fname.startswith("ds1_"):
                source_imgs["DS1"] += 1
            elif fname.startswith("ds2_"):
                source_imgs["DS2"] += 1
            elif fname.startswith("ds3_"):
                source_imgs["DS3"] += 1
            else:
                source_imgs["Desconocido"] += 1

    sources_i = sorted(source_imgs.keys())
    vals_i = [source_imgs[s] for s in sources_i]
    colors_i = [source_colors.get(s, "#9E9E9E") for s in sources_i]
    ax.pie(vals_i, labels=[f"{s}\n{v} imgs" for s, v in zip(sources_i, vals_i)],
           colors=colors_i, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
    ax.set_title("Imagenes por fuente", fontsize=12, fontweight="bold")

    # Stacked bar: fuentes por split (imagenes)
    ax = axes[1][0]
    sources_all = sorted(set(source_imgs.keys()))
    x = np.arange(len(SPLITS))
    bottom = np.zeros(len(SPLITS))

    for src in sources_all:
        vals_s = []
        for split in SPLITS:
            count = sum(1 for f in all_files[split]
                       if f["file"].startswith(f"ds{src[-1]}_") if src != "Desconocido")
            if src == "Desconocido":
                count = sum(1 for f in all_files[split]
                           if not any(f["file"].startswith(p) for p in ["ds1_", "ds2_", "ds3_"]))
            vals_s.append(count)
        ax.bar(x, vals_s, bottom=bottom, label=src,
               color=source_colors.get(src, "#9E9E9E"), edgecolor="white")
        bottom += np.array(vals_s)

    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLITS], fontsize=11)
    ax.set_ylabel("Imagenes", fontsize=10)
    ax.set_title("Fuentes por split", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Box plot de area por fuente
    ax = axes[1][1]
    source_areas = defaultdict(list)
    for a in annotations:
        source_areas[a["source"]].append(a["width"] * a["height"])

    sources_sorted = sorted(source_areas.keys())
    data = [source_areas[s] for s in sources_sorted]
    bp = ax.boxplot(data, tick_labels=sources_sorted, patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker=".", markersize=2, alpha=0.3),
                    medianprops=dict(color="red", linewidth=2))
    for patch, s in zip(bp["boxes"], sources_sorted):
        patch.set_facecolor(source_colors.get(s, "#9E9E9E"))
        patch.set_alpha(0.7)

    ax.set_ylabel("Area BBox (normalizada)", fontsize=10)
    ax.set_title("Area de placa por fuente", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Analisis de datasets fuente", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_analisis_fuentes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 06_analisis_fuentes.png")


def plot_07_bbox_coverage(annotations):
    """Grafica 7: Cobertura del bbox en la imagen (que % de la imagen ocupa la placa)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    areas = [a["width"] * a["height"] for a in annotations]
    areas_pct = [a * 100 for a in areas]

    # Histograma de cobertura
    ax = axes[0]
    ax.hist(areas_pct, bins=50, color="#FF5722", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Cobertura de imagen (%)", fontsize=10)
    ax.set_ylabel("Frecuencia", fontsize=10)
    ax.set_title("Que % de la imagen ocupa la placa", fontsize=12, fontweight="bold")
    ax.axvline(np.mean(areas_pct), color="red", linestyle="--", label=f"Media: {np.mean(areas_pct):.2f}%")
    ax.axvline(np.median(areas_pct), color="green", linestyle=":", label=f"Mediana: {np.median(areas_pct):.2f}%")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Categorias de tamano
    ax = axes[1]
    small = sum(1 for a in areas_pct if a < 1)
    medium = sum(1 for a in areas_pct if 1 <= a < 5)
    large = sum(1 for a in areas_pct if 5 <= a < 15)
    xlarge = sum(1 for a in areas_pct if a >= 15)
    cats = ["Pequena\n(<1%)", "Mediana\n(1-5%)", "Grande\n(5-15%)", "Muy grande\n(>15%)"]
    cat_vals = [small, medium, large, xlarge]
    cat_colors = ["#F44336", "#FF9800", "#4CAF50", "#2196F3"]

    bars = ax.bar(cats, cat_vals, color=cat_colors, edgecolor="white")
    for bar, val in zip(bars, cat_vals):
        pct = val / len(areas_pct) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{val}\n({pct:.1f}%)", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Num. anotaciones", fontsize=10)
    ax.set_title("Categorias de tamano de placa", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # CDF (distribucion acumulada)
    ax = axes[2]
    sorted_areas = np.sort(areas_pct)
    cdf = np.arange(1, len(sorted_areas) + 1) / len(sorted_areas)
    ax.plot(sorted_areas, cdf, color="#9C27B0", linewidth=2)
    ax.set_xlabel("Cobertura de imagen (%)", fontsize=10)
    ax.set_ylabel("Proporcion acumulada", fontsize=10)
    ax.set_title("Distribucion acumulada (CDF)", fontsize=12, fontweight="bold")
    # Marcar percentiles
    for pct in [25, 50, 75, 90]:
        val = np.percentile(areas_pct, pct)
        ax.axhline(pct/100, color="gray", linestyle=":", alpha=0.3)
        ax.axvline(val, color="gray", linestyle=":", alpha=0.3)
        ax.annotate(f"P{pct}: {val:.2f}%", (val, pct/100),
                   textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Cobertura de la placa en la imagen", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_cobertura_placa.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 07_cobertura_placa.png")


def plot_08_image_dimensions(annotations):
    """Grafica 8: Scatter de posicion vs tamano de placa (patron de perspectiva)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ys = np.array([a["y_center"] for a in annotations])
    areas = np.array([a["width"] * a["height"] for a in annotations])
    widths = np.array([a["width"] for a in annotations])
    heights = np.array([a["height"] for a in annotations])

    # Y position vs area (placas mas abajo suelen ser mas grandes por perspectiva)
    ax = axes[0]
    ax.scatter(ys, areas * 100, alpha=0.15, s=8, c="#E91E63")
    ax.set_xlabel("Posicion Y del centro (normalizada)", fontsize=10)
    ax.set_ylabel("Area de la placa (%)", fontsize=10)
    ax.set_title("Posicion vertical vs Tamano\n(efecto de perspectiva)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    # Binned means
    bins_y = np.linspace(0, 1, 11)
    bin_centers = (bins_y[:-1] + bins_y[1:]) / 2
    bin_means = []
    for i in range(len(bins_y) - 1):
        mask = (ys >= bins_y[i]) & (ys < bins_y[i+1])
        if mask.sum() > 0:
            bin_means.append(np.mean(areas[mask]) * 100)
        else:
            bin_means.append(0)
    ax.plot(bin_centers, bin_means, "b-o", linewidth=2, markersize=6,
            label="Media por bin", zorder=5)
    ax.legend(fontsize=9)

    # X position vs width
    ax = axes[1]
    xs = np.array([a["x_center"] for a in annotations])
    ax.scatter(xs, widths, alpha=0.15, s=8, c="#3F51B5")
    ax.set_xlabel("Posicion X del centro (normalizada)", fontsize=10)
    ax.set_ylabel("Ancho de la placa (normalizado)", fontsize=10)
    ax.set_title("Posicion horizontal vs Ancho", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    fig.suptitle("Relacion posicion-tamano de placas", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_posicion_vs_tamano.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 08_posicion_vs_tamano.png")


def plot_09_bbox_edge_proximity(annotations):
    """Grafica 9: Proximidad de bboxes a los bordes de la imagen."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Distancia minima de cada bbox a cada borde
    dist_left = []
    dist_right = []
    dist_top = []
    dist_bottom = []
    dist_min = []

    for a in annotations:
        x1 = a["x_center"] - a["width"] / 2
        y1 = a["y_center"] - a["height"] / 2
        x2 = a["x_center"] + a["width"] / 2
        y2 = a["y_center"] + a["height"] / 2

        dl = max(x1, 0)
        dr = max(1 - x2, 0)
        dt = max(y1, 0)
        db = max(1 - y2, 0)

        dist_left.append(dl)
        dist_right.append(dr)
        dist_top.append(dt)
        dist_bottom.append(db)
        dist_min.append(min(dl, dr, dt, db))

    # Histograma de distancia minima al borde
    ax = axes[0]
    ax.hist(dist_min, bins=50, color="#FF5722", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Distancia minima al borde (normalizada)", fontsize=10)
    ax.set_ylabel("Frecuencia", fontsize=10)
    ax.set_title("Proximidad al borde mas cercano", fontsize=12, fontweight="bold")

    # Cuantas estan cortadas (distancia ~0)
    cut_threshold = 0.01
    n_cut = sum(1 for d in dist_min if d < cut_threshold)
    ax.axvline(cut_threshold, color="red", linestyle="--", alpha=0.7,
               label=f"Umbral corte ({cut_threshold})")
    ax.text(0.95, 0.95,
            f"Posiblemente cortadas:\n{n_cut} ({n_cut/len(dist_min)*100:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#FFEBEE", edgecolor="#EF9A9A"))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Distancia a cada borde
    ax = axes[1]
    borde_data = [dist_left, dist_right, dist_top, dist_bottom]
    borde_labels = ["Izquierdo", "Derecho", "Superior", "Inferior"]
    borde_colors = ["#E91E63", "#3F51B5", "#4CAF50", "#FF9800"]
    bp = ax.boxplot(borde_data, tick_labels=borde_labels, patch_artist=True,
                    showfliers=False, medianprops=dict(color="red", linewidth=2))
    for patch, color in zip(bp["boxes"], borde_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Distancia al borde (normalizada)", fontsize=10)
    ax.set_title("Distancia a cada borde de imagen", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Analisis de proximidad a bordes", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "09_proximidad_bordes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 09_proximidad_bordes.png")


def plot_10_bbox_visualization(annotations):
    """Grafica 10: Visualizacion de todas las bboxes superpuestas (silueta)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Mapa de densidad de bboxes (donde caen las cajas)
    ax = axes[0]
    resolution = 200
    density_map = np.zeros((resolution, resolution))

    for a in annotations:
        x1 = int(max(0, (a["x_center"] - a["width"]/2)) * resolution)
        y1 = int(max(0, (a["y_center"] - a["height"]/2)) * resolution)
        x2 = int(min(1, (a["x_center"] + a["width"]/2)) * resolution)
        y2 = int(min(1, (a["y_center"] + a["height"]/2)) * resolution)
        x2 = min(x2, resolution)
        y2 = min(y2, resolution)
        if x2 > x1 and y2 > y1:
            density_map[y1:y2, x1:x2] += 1

    im = ax.imshow(density_map, cmap="inferno", aspect="equal",
                   extent=[0, 1, 1, 0], interpolation="bilinear")
    ax.set_xlabel("X (normalizado)", fontsize=10)
    ax.set_ylabel("Y (normalizado)", fontsize=10)
    ax.set_title("Mapa de densidad de BBoxes\n(donde caen las placas)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Num. BBoxes superpuestas")

    # Dibujar subset de bboxes aleatorias
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    random.seed(42)
    sample = random.sample(annotations, min(200, len(annotations)))
    for a in sample:
        x1 = a["x_center"] - a["width"]/2
        y1 = a["y_center"] - a["height"]/2
        rect = plt.Rectangle((x1, y1), a["width"], a["height"],
                             linewidth=0.5, edgecolor="#2196F3", facecolor="#2196F3", alpha=0.05)
        ax.add_patch(rect)

    ax.set_xlabel("X (normalizado)", fontsize=10)
    ax.set_ylabel("Y (normalizado)", fontsize=10)
    ax.set_title(f"Muestra de 200 BBoxes superpuestas", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)

    fig.suptitle("Visualizacion espacial de BBoxes", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "10_visualizacion_bboxes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 10_visualizacion_bboxes.png")


def plot_11_sample_images():
    """Grafica 11: Mosaico de imagenes de ejemplo con BBoxes dibujadas."""
    fig, axes = plt.subplots(3, 4, figsize=(18, 14))
    axes = axes.flatten()

    images_dir = DATASET_DIR / "train" / "images"
    labels_dir = DATASET_DIR / "train" / "labels"

    label_files = list(labels_dir.glob("*.txt"))
    random.seed(42)
    random.shuffle(label_files)

    # Buscar 12 imagenes con al menos 1 anotacion
    examples = []
    for lf in label_files:
        if len(examples) >= 12:
            break
        bboxes = []
        with open(lf, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    bboxes.append([float(p) for p in parts[1:5]])
        if bboxes:
            stem = lf.stem
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                img_path = images_dir / (stem + ext)
                if img_path.exists():
                    examples.append((img_path, bboxes))
                    break

    for i, ax in enumerate(axes):
        if i < len(examples):
            img_path, bboxes = examples[i]
            try:
                img = Image.open(img_path)
                img_w, img_h = img.size
                ax.imshow(img)

                for bbox in bboxes:
                    xc, yc, w, h = bbox
                    x1 = (xc - w/2) * img_w
                    y1 = (yc - h/2) * img_h
                    rect = plt.Rectangle((x1, y1), w * img_w, h * img_h,
                                        linewidth=2, edgecolor="lime", facecolor="none")
                    ax.add_patch(rect)

                ax.set_title(f"{len(bboxes)} placa(s)", fontsize=9)
            except Exception:
                ax.text(0.5, 0.5, "Error", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "-", ha="center", va="center", transform=ax.transAxes)

        ax.axis("off")

    fig.suptitle("Ejemplos del dataset de placas (con BBoxes)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "11_ejemplos_placas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 11_ejemplos_placas.png")


def plot_12_image_resolutions():
    """Grafica 12: Analisis de resoluciones de imagenes."""
    print("  [>] Leyendo resoluciones de imagenes (puede tardar)...")

    all_sizes = []
    for split in SPLITS:
        sizes = get_image_sizes(split)
        for s in sizes:
            s["split"] = split
        all_sizes.extend(sizes)

    if not all_sizes:
        print("  [!] No se pudieron leer resoluciones de imagenes")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ws = [s["width"] for s in all_sizes]
    hs = [s["height"] for s in all_sizes]
    resolutions = [f"{s['width']}x{s['height']}" for s in all_sizes]
    res_counts = Counter(resolutions)

    # Scatter de resoluciones
    ax = axes[0][0]
    ax.scatter(ws, hs, alpha=0.2, s=10, c="#2196F3")
    ax.set_xlabel("Ancho (px)", fontsize=10)
    ax.set_ylabel("Alto (px)", fontsize=10)
    ax.set_title("Resoluciones de imagenes", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    # Top resoluciones
    ax = axes[0][1]
    top_res = res_counts.most_common(10)
    res_names = [r for r, _ in top_res]
    res_vals = [v for _, v in top_res]
    bars = ax.barh(range(len(res_names)), res_vals, color="#FF9800", edgecolor="white")
    ax.set_yticks(range(len(res_names)))
    ax.set_yticklabels(res_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Num. imagenes", fontsize=10)
    ax.set_title("Top 10 resoluciones", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, res_vals):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f"{val} ({val/len(all_sizes)*100:.1f}%)", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    # Histograma de ancho
    ax = axes[1][0]
    ax.hist(ws, bins=30, color="#4CAF50", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Ancho (px)", fontsize=10)
    ax.set_ylabel("Frecuencia", fontsize=10)
    ax.set_title("Distribucion de anchos", fontsize=12, fontweight="bold")
    ax.axvline(np.mean(ws), color="red", linestyle="--", label=f"Media: {np.mean(ws):.0f}px")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Histograma de alto
    ax = axes[1][1]
    ax.hist(hs, bins=30, color="#9C27B0", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Alto (px)", fontsize=10)
    ax.set_ylabel("Frecuencia", fontsize=10)
    ax.set_title("Distribucion de altos", fontsize=12, fontweight="bold")
    ax.axvline(np.mean(hs), color="red", linestyle="--", label=f"Media: {np.mean(hs):.0f}px")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Analisis de resoluciones de imagenes", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "12_resoluciones_imagenes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 12_resoluciones_imagenes.png")


def plot_13_summary_dashboard(annotations, all_files):
    """Grafica 13: Dashboard resumen del dataset de placas."""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    total_imgs = sum(len(all_files[s]) for s in SPLITS)
    total_annots = len(annotations)
    areas = [a["width"] * a["height"] for a in annotations]
    aspect_ratios = [a["width"] / max(a["height"], 1e-6) for a in annotations]
    all_counts = []
    for split in SPLITS:
        all_counts.extend([f["num_annotations"] for f in all_files[split]])

    # --- Panel 1: KPIs ---
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")

    kpi_text = (
        f"DATASET DE PLACAS COLOMBIANAS\n"
        f"{'='*40}\n\n"
        f"Imagenes totales:    {total_imgs:,}\n"
        f"Anotaciones totales: {total_annots:,}\n"
        f"Clases:              1 (placa)\n"
        f"Placas/imagen (prom): {total_annots/total_imgs:.2f}\n\n"
        f"Train: {len(all_files['train']):>5} imgs "
        f"({len(all_files['train'])/total_imgs*100:.1f}%)\n"
        f"Valid: {len(all_files['valid']):>5} imgs "
        f"({len(all_files['valid'])/total_imgs*100:.1f}%)\n"
        f"Test:  {len(all_files['test']):>5} imgs "
        f"({len(all_files['test'])/total_imgs*100:.1f}%)\n\n"
        f"Area BBox (media):   {np.mean(areas)*100:.2f}%\n"
        f"Aspect ratio (med):  {np.median(aspect_ratios):.2f}\n"
        f"Max placas/imagen:   {max(all_counts)}\n"
        f"Imgs con 0 placas:   {sum(1 for c in all_counts if c == 0)}"
    )
    ax.text(0.05, 0.95, kpi_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9", edgecolor="#A5D6A7"))

    # --- Panel 2: Split pie ---
    ax = fig.add_subplot(gs[0, 1])
    img_counts = [len(all_files[s]) for s in SPLITS]
    ax.pie(img_counts, labels=[f"{SPLIT_LABELS[s]}\n{c}" for s, c in zip(SPLITS, img_counts)],
           colors=[SPLIT_COLORS[s] for s in SPLITS], autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 10})
    ax.set_title("Distribucion de splits", fontsize=12, fontweight="bold")

    # --- Panel 3: Source pie ---
    ax = fig.add_subplot(gs[0, 2])
    source_counts = Counter(a["source"] for a in annotations)
    sources = sorted(source_counts.keys())
    src_colors = {"DS1": "#E91E63", "DS2": "#3F51B5", "DS3": "#009688", "Desconocido": "#9E9E9E"}
    ax.pie([source_counts[s] for s in sources],
           labels=[f"{s}\n{source_counts[s]}" for s in sources],
           colors=[src_colors.get(s, "#9E9E9E") for s in sources],
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
    ax.set_title("Fuentes de datos", fontsize=12, fontweight="bold")

    # --- Panel 4: Heatmap centros ---
    ax = fig.add_subplot(gs[1, 0])
    xs = [a["x_center"] for a in annotations]
    ys = [a["y_center"] for a in annotations]
    heatmap, _, _ = np.histogram2d(xs, ys, bins=30, range=[[0, 1], [0, 1]])
    ax.imshow(heatmap.T, origin="lower", cmap="hot", extent=[0, 1, 0, 1],
              aspect="equal", interpolation="gaussian")
    ax.set_title("Heatmap centros", fontsize=12, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # --- Panel 5: Histograma areas ---
    ax = fig.add_subplot(gs[1, 1])
    ax.hist([a * 100 for a in areas], bins=40, color="#AB47BC", edgecolor="white", alpha=0.85)
    ax.set_title("Cobertura de placa (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("% de imagen")
    ax.set_ylabel("Frecuencia")
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 6: Placas por imagen ---
    ax = fig.add_subplot(gs[1, 2])
    max_c = max(all_counts) if all_counts else 1
    ax.hist(all_counts, bins=range(0, max_c + 2), color="#26A69A", edgecolor="white",
            alpha=0.85, align="left")
    ax.set_title("Placas por imagen", fontsize=12, fontweight="bold")
    ax.set_xlabel("Num. placas")
    ax.set_ylabel("Num. imagenes")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Dashboard - Dataset Placas Vehiculares Colombianas",
                fontsize=16, fontweight="bold", y=1.01)
    fig.savefig(OUTPUT_DIR / "13_dashboard_resumen.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] 13_dashboard_resumen.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  ANALISIS DEL DATASET DE PLACAS VEHICULARES")
    print("=" * 60)
    print()

    print("[1/3] Cargando configuracion...")
    config = load_config()
    print(f"  [OK] Dataset: {config.get('descripcion', 'N/A')}")
    print(f"  [OK] {config['nc']} clase(s): {config['names']}")
    print()

    print("[2/3] Cargando anotaciones...")
    annotations, all_files = load_all_data()
    print()

    print("[3/3] Generando graficas...")
    print(f"  Directorio de salida: {OUTPUT_DIR}")
    print()

    plot_01_split_distribution(all_files, annotations)
    plot_02_annotations_per_image(all_files)
    plot_03_bbox_size_analysis(annotations)
    plot_04_bbox_center_heatmap(annotations)
    plot_05_bbox_position_distribution(annotations)
    plot_06_source_dataset_analysis(annotations, all_files)
    plot_07_bbox_coverage(annotations)
    plot_08_image_dimensions(annotations)
    plot_09_bbox_edge_proximity(annotations)
    plot_10_bbox_visualization(annotations)
    plot_11_sample_images()
    plot_12_image_resolutions()
    plot_13_summary_dashboard(annotations, all_files)

    print()
    print("=" * 60)
    print(f"  [OK] 13 graficas generadas en: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
