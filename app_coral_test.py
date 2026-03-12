#!/usr/bin/env python3
"""
app_coral_test.py — Streamlit app para testear modelos cuantizados INT8 / Edge TPU.

Uso:
    streamlit run app_coral_test.py

Tabs:
  1. Pipeline completo  — sube una imagen y corre los 5 modelos en secuencia
  2. Modelos individuales — prueba cada modelo por separado con info de debugging
  3. Benchmark — mide latencias reales (N runs) para cada modelo y proyecta Coral TPU

Pipeline híbrido: 4 modelos INT8 TFLite vía CoralInterpreter + OCR en CPU:
  - Vehicle detector (YOLO11n COCO) — Coral INT8
  - Color classifier (EfficientNet-lite) — Coral INT8
  - Brand detector (YOLO11n custom) — Coral INT8
  - Plate detector (YOLO11n custom) — Coral INT8
  - OCR (CCT-XS) — CPU vía ONNX Runtime (transformer no compatible con Edge TPU)
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ─── Rutas ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
EXPORTS_DIR = PROJECT_DIR / "models" / "tflite_exports"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from coral_simulator import CoralInterpreter

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ANPR Coral — Model Tester",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .result-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 6px 0;
  }
  .result-card .rc-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
  .result-card .rc-val   { font-size: 1.4rem; font-weight: 700; color: #e6edf3; margin-top: 2px; }
  .result-card .rc-conf  { font-size: 0.75rem; color: #3fb950; margin-top: 2px; }

  .metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 16px 20px; text-align: center; margin: 4px 0;
  }
  .metric-card .val { font-size: 2rem; font-weight: 700; color: #58a6ff; }
  .metric-card .lbl { font-size: 0.8rem; color: #8b949e; margin-top: 2px; }
  .metric-card .sub { font-size: 0.75rem; color: #3fb950; margin-top: 2px; }

  .model-badge-ok   { background:#1a4129; color:#3fb950; padding:3px 8px; border-radius:6px; font-size:0.75rem; font-weight:600; }
  .model-badge-warn { background:#2d1d06; color:#d29922; padding:3px 8px; border-radius:6px; font-size:0.75rem; font-weight:600; }
  .model-badge-miss { background:#2d1117; color:#f85149; padding:3px 8px; border-radius:6px; font-size:0.75rem; font-weight:600; }

  .stButton>button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white; border: none; border-radius: 8px;
    padding: 0.5rem 1.5rem; font-weight: 600;
    transition: all 0.2s ease;
  }
  .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(31,111,235,0.5); }

  .pipeline-row {
    display: flex; align-items: center; gap: 8px;
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 12px 16px; margin: 4px 0;
  }
  .pipeline-row .stage { font-weight: 600; color: #58a6ff; min-width: 110px; }
  .pipeline-row .bar-wrap { flex: 1; height: 8px; background: #21262d; border-radius: 4px; overflow: hidden; }
  .pipeline-row .bar { height: 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ─── Constantes ───────────────────────────────────────────────────────────────

# Color classifier — 15 colores, ordenados alfabéticamente (igual que en el entrenamiento)
COLOR_CLASSES = sorted([
    "beige", "black", "blue", "brown", "gold", "green", "grey",
    "orange", "pink", "purple", "red", "silver", "tan", "white", "yellow",
])
COLOR_ES = {
    "beige": "Beige", "black": "Negro", "blue": "Azul", "brown": "Café",
    "gold": "Dorado", "green": "Verde", "grey": "Gris", "orange": "Naranja",
    "pink": "Rosa", "purple": "Morado", "red": "Rojo", "silver": "Plata",
    "tan": "Canela", "white": "Blanco", "yellow": "Amarillo",
}
COLOR_EMOJI = {
    "black": "⚫", "white": "⚪", "grey": "🩶", "silver": "🩶",
    "red": "🔴", "blue": "🔵", "green": "🟢", "yellow": "🟡",
    "orange": "🟠", "brown": "🟤", "gold": "🥇",
}

# Brand detector — 30 marcas (mismo orden que en BRAND_CLASSES de brand_detector.py)
BRAND_CLASSES = {
    0: "Acura", 1: "Audi", 2: "BMW", 3: "Chevrolet", 4: "Citroën",
    5: "Dacia", 6: "Fiat", 7: "Ford", 8: "Honda", 9: "Hyundai",
    10: "Infiniti", 11: "KIA", 12: "Lamborghini", 13: "Lexus", 14: "Mazda",
    15: "Mercedes-Benz", 16: "Mitsubishi", 17: "Nissan", 18: "Opel",
    19: "Perodua", 20: "Peugeot", 21: "Porsche", 22: "Proton",
    23: "Renault", 24: "Seat", 25: "Suzuki", 26: "Tesla", 27: "Toyota",
    28: "Volkswagen", 29: "Volvo",
}

# Vehicle detector — clases COCO de vehículos (IDs que devuelve yolo11n entrenado con COCO)
VEHICLE_CLASSES = {
    2: "Auto 🚗", 3: "Moto 🏍️", 5: "Bus 🚌",
    7: "Camión 🚛", 1: "Bicicleta 🚲",
}

MODELS_CONFIG = {
    "vehicle": {
        "file": "yolo11n_coco_vehicle_int8.tflite",
        "edgetpu": "yolo11n_coco_vehicle_int8_edgetpu.tflite",
        "label": "🚗 Tipo de vehículo",
        "hint": "yolo", "scale": 11,
        "desc": "YOLO11n entrenado en COCO — detecta auto, moto, bus, camión, bicicleta. Devuelve bounding boxes con clase y confianza.",
    },
    "color": {
        "file": "color_classifier_int8.tflite",
        "edgetpu": "color_classifier_int8_edgetpu.tflite",
        "label": "🎨 Color del vehículo",
        "hint": "efficient", "scale": 6,
        "desc": "EfficientNet-lite clasificador de 15 colores (negro, blanco, rojo, azul...). Recibe imagen 224×224 y devuelve probabilidad por color.",
    },
    "brand": {
        "file": "marca_detector_yolo11n_int8.tflite",
        "edgetpu": "marca_detector_yolo11n_int8_edgetpu.tflite",
        "label": "🏭 Marca del vehículo",
        "hint": "yolo", "scale": 11,
        "desc": "YOLO11n personalizado entrenado para detectar 30 logos de marcas: Toyota, Volkswagen, BMW, Chevrolet, Honda, etc.",
    },
    "plate": {
        "file": "placa_detector_yolo11n_int8.tflite",
        "edgetpu": "placa_detector_yolo11n_int8_edgetpu.tflite",
        "label": "📍 Detector de placa",
        "hint": "yolo", "scale": 11,
        "desc": "YOLO11n INT8 para detectar placas. Devuelve bbox para recortar y enviar al OCR.",
    },
    "ocr": {
        "file": None,
        "edgetpu": None,
        "label": "🔤 OCR de placa",
        "hint": "cpu_only", "scale": 1,
        "desc": "CCT-XS (Compact Convolutional Transformer) de fast-plate-ocr. Lee caracteres alfanuméricos de la placa recortada. Corre en CPU vía ONNX Runtime (~3ms).",
    },
}


# ─── Cache de modelos ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando modelos INT8...")
def load_interpreters():
    loaded = {}
    for key, cfg in MODELS_CONFIG.items():
        if cfg["file"] is None:
            loaded[key] = None
            continue
        for candidate in [cfg.get("edgetpu"), cfg["file"]]:
            if not candidate:
                continue
            path = EXPORTS_DIR / candidate
            if path.exists():
                try:
                    interp = CoralInterpreter(str(path), device="auto", verbose=False)
                    loaded[key] = {"interp": interp, "file": candidate, "path": path}
                    break
                except Exception:
                    continue
        else:
            loaded[key] = None
    return loaded


@st.cache_resource(show_spinner="Cargando OCR...")
def load_ocr():
    try:
        from fast_plate_ocr import LicensePlateRecognizer
        return LicensePlateRecognizer("cct-xs-v1-global-model")
    except Exception:
        return None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def run_model_timed(interp_cfg, image_rgb, target_size=None, hint="default"):
    interp = interp_cfg["interp"]
    interp.set_input_from_image(image_rgb, target_size=target_size)
    r = interp.invoke(model_hint=hint)
    return r.raw_output, r.latency_ms, r.estimated_tpu_ms, r.backend


def decode_yolo_top(outputs, conf_thresh=0.25):
    """Extrae la detección YOLO de mayor confianza. Retorna (class_id, conf, bbox_norm)."""
    raw = outputs[0]
    if raw.ndim == 3:
        raw = raw[0]
    # YOLO output: [num_det, 4+num_classes] o transpuesto
    if raw.shape[0] < raw.shape[-1]:
        raw = raw.T
    if raw.ndim < 2 or raw.shape[1] < 5:
        return None, 0.0
    scores = raw[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confs = scores[np.arange(len(scores)), class_ids]
    best = int(np.argmax(confs))
    if float(confs[best]) < conf_thresh:
        return None, 0.0
    return int(class_ids[best]), float(confs[best])


def decode_yolo_best_bbox(outputs, conf_thresh=0.25, img_shape=(640, 640)):
    """Extract best YOLO bbox in pixel coords. Returns ((x1,y1,x2,y2), conf)."""
    raw = outputs[0]
    if raw.ndim == 3:
        raw = raw[0]
    if raw.shape[0] < raw.shape[-1]:
        raw = raw.T
    if raw.ndim < 2 or raw.shape[1] < 5:
        return None, 0.0
    scores = raw[:, 4:]
    max_scores = np.max(scores, axis=1)
    best = int(np.argmax(max_scores))
    conf = float(max_scores[best])
    if conf < conf_thresh:
        return None, 0.0
    cx, cy, w, h = raw[best, :4]
    img_h, img_w = img_shape
    # YOLO outputs are in 640×640 space, scale to original image
    sx, sy = img_w / 640.0, img_h / 640.0
    x1 = int((cx - w / 2) * sx)
    y1 = int((cy - h / 2) * sy)
    x2 = int((cx + w / 2) * sx)
    y2 = int((cy + h / 2) * sy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    return (x1, y1, x2, y2), conf


def result_card(label, value, sub=None, emoji=""):
    sub_html = f"<div class='rc-conf'>{sub}</div>" if sub else ""
    st.markdown(
        f"<div class='result-card'>"
        f"  <div class='rc-label'>{label}</div>"
        f"  <div class='rc-val'>{emoji} {value}</div>"
        f"  {sub_html}"
        f"</div>",
        unsafe_allow_html=True
    )


# ─── Sidebar ──────────────────────────────────────────────────────────────────

interpreters = load_interpreters()

with st.sidebar:
    st.markdown("## 🪸 ANPR Coral Tester")
    st.markdown("Test de modelos **INT8 cuantizados** para Google Coral Edge TPU")
    st.divider()

    st.markdown("### 📦 Estado de modelos")
    for key, cfg in MODELS_CONFIG.items():
        interp_data = interpreters.get(key)
        label = cfg["label"]
        if cfg["file"] is None:
            # OCR — runs on CPU via ONNX
            ocr_obj = load_ocr()
            badge = "model-badge-ok" if ocr_obj else "model-badge-miss"
            status = "ONNX CPU ✓" if ocr_obj else "NO INSTALADO"
        elif interp_data:
            fname = interp_data["file"]
            is_edgetpu = "_edgetpu" in fname
            badge = "model-badge-ok" if is_edgetpu else "model-badge-warn"
            status = "Edge TPU ✓" if is_edgetpu else "INT8 ✓"
        else:
            badge = "model-badge-miss"
            status = "NO ENCONTRADO"
        st.markdown(
            f"<div style='margin:6px 0'>{label}<br>"
            f"<span class='{badge}'>{status}</span></div>",
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown("**Runs para benchmark:**")
    n_runs = st.slider("", 5, 50, 15, key="bench_runs")

    st.divider()
    st.markdown("""
    **Modos de ejecución:**
    - 🟡 **INT8 sim** = TFLite en CPU, mismos resultados que Coral
    - 🟢 **Edge TPU** = compilado `_edgetpu.tflite`, requiere hardware
    - 🟠 **OCR** = ONNX Runtime CPU (transformer no compatible con Edge TPU)
    """)


# ─── Título ───────────────────────────────────────────────────────────────────

st.markdown("# 🪸 ANPR Coral — Quantized Model Tester")
st.markdown(
    "Pipeline híbrido: **4 modelos INT8** en Coral + **OCR** en CPU. "
    "Los modelos INT8 dan resultados idénticos a ejecución real en hardware Coral Edge TPU."
)
st.divider()

tab_pipeline, tab_individual, tab_benchmark = st.tabs([
    "🔁 Pipeline completo", "🔬 Modelos individuales", "📊 Benchmark"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PIPELINE COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

with tab_pipeline:
    st.markdown("""
    **¿Qué hace este tab?**
    Corre los 5 modelos en secuencia sobre la imagen que subas, igual que el pipeline ANPR real:
    1. 🚗 **Tipo de vehículo** — detecta si es auto, moto, bus o camión (YOLO11n COCO INT8)
    2. 🎨 **Color** — clasifica el color dominante entre 15 opciones (EfficientNet INT8)
    3. 🏭 **Marca** — detecta el logo de la marca entre 30 marcas (YOLO11n personalizado INT8)
    4. 📍 **Detector de placa** — detecta bbox de la placa en la imagen (YOLO11n INT8)
    5. 🔤 **OCR de placa** — lee caracteres de la placa recortada (CCT-XS vía ONNX Runtime CPU)

    Pipeline híbrido: 4 modelos en **Coral INT8** + OCR en **CPU** (ONNX).
    """)

    st.divider()

    # ── Input source selector ────────────────────────────────────────────
    TEST_DIRS = {
        "dataset_combinado/test": PROJECT_DIR / "dataset_combinado" / "test" / "images",
        "samples": PROJECT_DIR / "samples",
    }
    input_mode = st.radio(
        "Fuente de imagen:", ["Upload", "Camera", "Test dataset"],
        horizontal=True, key="pipe_input_mode",
    )

    frame_rgb = None
    if input_mode == "Upload":
        uploaded = st.file_uploader("📁 Sube una imagen de vehículo con placa", type=["jpg", "jpeg", "png"])
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    elif input_mode == "Camera":
        cam_img = st.camera_input("📷 Toma una foto")
        if cam_img:
            file_bytes = np.asarray(bytearray(cam_img.read()), dtype=np.uint8)
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    else:  # Test dataset
        ds_choice = st.selectbox("Carpeta:", list(TEST_DIRS.keys()), key="pipe_ds")
        ds_path = TEST_DIRS[ds_choice]
        if ds_path.exists():
            imgs = sorted([f.name for f in ds_path.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
            if imgs:
                sel_img = st.selectbox(f"Imagen ({len(imgs)} disponibles):", imgs, key="pipe_img_sel")
                img_path = ds_path / sel_img
                frame_bgr = cv2.imread(str(img_path))
                if frame_bgr is not None:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            else:
                st.warning("No hay imágenes en esta carpeta.")
        else:
            st.warning(f"Carpeta `{ds_path}` no encontrada.")

    if frame_rgb is not None:
        col_img, col_results = st.columns([1, 1])
        with col_img:
            st.image(frame_rgb, caption="Imagen de entrada", use_container_width=True)

        if st.button("▶ Ejecutar Pipeline ANPR Coral", use_container_width=True):
            latencias = {}
            detections = {}

            with st.spinner("Corriendo los 5 modelos..."):

                # ── 1. Tipo de vehículo (YOLO COCO) ──────────────────────────
                veh_data = interpreters.get("vehicle")
                if veh_data:
                    outputs, lat, est, _ = run_model_timed(veh_data, frame_rgb, hint="yolo")
                    latencias["🚗 Vehículo"] = (lat, est)
                    cls_id, conf = decode_yolo_top(outputs)
                    detections["vehicle"] = {
                        "class_id": cls_id, "conf": conf,
                        "name": VEHICLE_CLASSES.get(cls_id, f"clase {cls_id}") if cls_id is not None else None
                    }

                # ── 2. Color ─────────────────────────────────────────────────
                col_data = interpreters.get("color")
                if col_data:
                    outputs, lat, est, _ = run_model_timed(
                        col_data, frame_rgb, target_size=(224, 224), hint="efficient"
                    )
                    latencias["🎨 Color"] = (lat, est)
                    out_flat = outputs[0].flatten()
                    top_idx = int(np.argmax(out_flat))
                    top_score = float(out_flat[top_idx])
                    eng_color = COLOR_CLASSES[top_idx] if top_idx < len(COLOR_CLASSES) else "?"
                    detections["color"] = {
                        "eng": eng_color,
                        "es": COLOR_ES.get(eng_color, eng_color),
                        "score": top_score,
                        "top3": [
                            (COLOR_ES.get(COLOR_CLASSES[i], COLOR_CLASSES[i]), float(out_flat[i]))
                            for i in np.argsort(out_flat)[-3:][::-1]
                            if i < len(COLOR_CLASSES)
                        ]
                    }

                # ── 3. Marca ─────────────────────────────────────────────────
                brand_data = interpreters.get("brand")
                if brand_data:
                    outputs, lat, est, _ = run_model_timed(brand_data, frame_rgb, hint="yolo")
                    latencias["🏭 Marca"] = (lat, est)
                    cls_id, conf = decode_yolo_top(outputs, conf_thresh=0.2)
                    detections["brand"] = {
                        "class_id": cls_id,
                        "name": BRAND_CLASSES.get(cls_id) if cls_id is not None else None,
                        "conf": conf,
                    }

                # ── 4. Plate detection ───────────────────────────────────────
                plate_data = interpreters.get("plate")
                plate_bbox = None
                if plate_data:
                    outputs, lat, est, _ = run_model_timed(plate_data, frame_rgb, hint="yolo")
                    latencias["📍 Placa"] = (lat, est)
                    plate_bbox, plate_conf = decode_yolo_best_bbox(
                        outputs, conf_thresh=0.3, img_shape=frame_rgb.shape[:2]
                    )
                    detections["plate"] = {"bbox": plate_bbox, "conf": plate_conf}

                # ── 5. OCR (ONNX CPU) ────────────────────────────────────────
                ocr_obj = load_ocr()
                if ocr_obj:
                    h, w = frame_rgb.shape[:2]
                    fallback = False
                    if plate_bbox:
                        x1, y1, x2, y2 = plate_bbox
                        plate_crop = frame_bgr[y1:y2, x1:x2]
                        if plate_crop.size == 0:
                            plate_crop = frame_bgr[int(h * 0.55):, int(w * 0.15):int(w * 0.85)]
                            fallback = True
                    else:
                        plate_crop = frame_bgr[int(h * 0.55):, int(w * 0.15):int(w * 0.85)]
                        fallback = True
                    # Pass raw BGR crop to OCR — same as app_demo.py
                    t0 = time.perf_counter()
                    try:
                        res = ocr_obj.run(plate_crop)
                        ocr_text = res[0].strip() if isinstance(res, (list, tuple)) else str(res).strip()
                    except Exception:
                        ocr_text = "—"
                    ocr_lat = (time.perf_counter() - t0) * 1000
                    latencias["🔤 OCR"] = (ocr_lat, ocr_lat)  # CPU-only, no TPU speedup
                    plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                    detections["ocr"] = {"text": ocr_text, "fallback": fallback, "crop": plate_crop_rgb}

            # ── Mostrar resultados ────────────────────────────────────────────
            with col_results:
                st.markdown("### 🎯 Resultados del pipeline")

                # Tipo de vehículo
                veh = detections.get("vehicle", {})
                if veh.get("name"):
                    result_card("Tipo de vehículo", veh["name"], f"Confianza: {veh['conf']:.0%}")
                else:
                    result_card("Tipo de vehículo", "No detectado", "Confianza < 25%", "🚫")

                # Color
                col_det = detections.get("color", {})
                if col_det:
                    emoji_c = COLOR_EMOJI.get(col_det["eng"], "🎨")
                    top3_str = "  |  ".join([f"{n} ({int(s)}/255)" for n, s in col_det["top3"]])
                    result_card("Color del vehículo", col_det["es"], f"Top 3: {top3_str}", emoji_c)

                # Marca
                brand = detections.get("brand", {})
                if brand.get("name"):
                    result_card("Marca detectada", brand["name"], f"Confianza: {brand['conf']:.0%}", "🏭")
                else:
                    result_card("Marca detectada", "No detectada", "Logo no visible o confianza < 20%", "🚫")

                # Placa (bbox)
                plate_det = detections.get("plate", {})
                if plate_det.get("bbox"):
                    x1, y1, x2, y2 = plate_det["bbox"]
                    result_card(
                        "Placa detectada",
                        f"({x1},{y1}) → ({x2},{y2})",
                        f"Confianza: {plate_det['conf']:.0%}",
                        "📍",
                    )
                else:
                    result_card("Placa detectada", "No detectada", "Usando recorte fallback", "🚫")

                # OCR
                ocr_det = detections.get("ocr", {})
                if ocr_det:
                    txt = ocr_det.get("text", "—")
                    fb_note = " (recorte fallback)" if ocr_det.get("fallback") else " (bbox detectado)"
                    result_card("Texto de placa (OCR)", txt if txt else "—", fb_note, "🔤")
                    if ocr_det.get("crop") is not None:
                        with st.expander("Ver crop enviado al OCR"):
                            st.image(ocr_det["crop"], caption="Crop de placa", width=256)

                # ── Latencias ─────────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### ⏱ Latencias")

                max_lat = max((v[0] for v in latencias.values()), default=1)
                total_cpu = sum(v[0] for v in latencias.values())
                total_tpu = sum(v[1] for v in latencias.values())

                for stage, (lat_cpu, lat_tpu) in latencias.items():
                    pct = max(int(lat_cpu / max_lat * 100), 3)
                    is_cpu_only = lat_cpu == lat_tpu
                    color = "#d29922" if is_cpu_only else "#388bfd"
                    tpu_label = "CPU only" if is_cpu_only else f"≈{lat_tpu:.1f}ms TPU"
                    st.markdown(
                        f"<div class='pipeline-row'>"
                        f"  <div class='stage'>{stage}</div>"
                        f"  <div class='bar-wrap'><div class='bar' style='width:{pct}%;background:{color}'></div></div>"
                        f"  <div style='min-width:70px;text-align:right;color:#e6edf3;font-weight:600'>{lat_cpu:.1f} ms</div>"
                        f"  <div style='min-width:110px;text-align:right;color:#8b949e;font-size:0.72rem'>{tpu_label}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                st.divider()
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        f"<div class='metric-card'><div class='val'>{total_cpu:.0f}"
                        f"<span style='font-size:1rem'>ms</span></div>"
                        f"<div class='lbl'>Total CPU (simulación)</div></div>",
                        unsafe_allow_html=True
                    )
                with c2:
                    if total_tpu > 0:
                        speedup = total_cpu / max(total_tpu, 1)
                        st.markdown(
                            f"<div class='metric-card'><div class='val' style='color:#3fb950'>{total_tpu:.0f}"
                            f"<span style='font-size:1rem'>ms</span></div>"
                            f"<div class='lbl'>Estimado Coral TPU</div>"
                            f"<div class='sub'>~{speedup:.1f}× más rápido</div></div>",
                            unsafe_allow_html=True
                        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODELOS INDIVIDUALES
# ══════════════════════════════════════════════════════════════════════════════

with tab_individual:
    st.markdown("""
    **¿Qué hace este tab?**
    Prueba **un solo modelo a la vez**. Útil para:
    - Ver exactamente qué devuelve cada modelo (output crudo)
    - Debuggear imágenes problemáticas
    - Ver la latencia individual sin el resto del pipeline
    - Entender el formato de input/output de cada modelo
    """)
    st.divider()

    model_sel = st.selectbox(
        "Selecciona el modelo:",
        options=["vehicle", "color", "brand", "plate", "ocr"],
        format_func=lambda k: MODELS_CONFIG[k]["label"],
    )

    cfg_sel = MODELS_CONFIG[model_sel]
    st.info(f"ℹ️ **{cfg_sel['label']}** — {cfg_sel['desc']}")

    # ── Input source selector ────────────────────────────────────────────
    ind_input_mode = st.radio(
        "Fuente de imagen:", ["Upload", "Test dataset"],
        horizontal=True, key="ind_input_mode",
    )

    frame2_rgb = None
    if ind_input_mode == "Upload":
        uploaded2 = st.file_uploader("📁 Imagen de prueba", type=["jpg", "jpeg", "png"], key="ind_upload")
        if uploaded2:
            file_bytes2 = np.asarray(bytearray(uploaded2.read()), dtype=np.uint8)
            frame2_bgr = cv2.imdecode(file_bytes2, cv2.IMREAD_COLOR)
            frame2_rgb = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2RGB)
    else:  # Test dataset
        ds_choice2 = st.selectbox("Carpeta:", list(TEST_DIRS.keys()), key="ind_ds")
        ds_path2 = TEST_DIRS[ds_choice2]
        if ds_path2.exists():
            imgs2 = sorted([f.name for f in ds_path2.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
            if imgs2:
                sel_img2 = st.selectbox(f"Imagen ({len(imgs2)} disponibles):", imgs2, key="ind_img_sel")
                img_path2 = ds_path2 / sel_img2
                frame2_bgr = cv2.imread(str(img_path2))
                if frame2_bgr is not None:
                    frame2_rgb = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2RGB)
            else:
                st.warning("No hay imágenes en esta carpeta.")
        else:
            st.warning(f"Carpeta `{ds_path2}` no encontrada.")

    if frame2_rgb is not None:
        st.image(frame2_rgb, caption="Imagen seleccionada", use_container_width=True)

        if st.button(f"▶ Ejecutar {cfg_sel['label']}", key="run_ind"):
            interp_data = interpreters.get(model_sel)

            if not interp_data:
                st.error(f"Modelo `{model_sel}` no encontrado en `models/tflite_exports/`.")

            # ── Plate detector ───────────────────────────────────────────────
            elif model_sel == "plate":
                outputs, lat, est, backend = run_model_timed(interp_data, frame2_rgb, hint="yolo")
                inp = interp_data["interp"].input_details[0]

                c1, c2, c3 = st.columns(3)
                c1.metric("Latencia CPU", f"{lat:.1f} ms")
                c2.metric("Est. Coral TPU", f"{est:.1f} ms", delta=f"-{lat - est:.1f}ms")
                c3.metric("Speedup estimado", f"~{cfg_sel['scale']}×")

                st.divider()
                bbox, conf = decode_yolo_best_bbox(outputs, conf_thresh=0.2, img_shape=frame2_rgb.shape[:2])
                if bbox:
                    x1, y1, x2, y2 = bbox
                    st.success(f"**Placa detectada** — bbox: ({x1},{y1})→({x2},{y2}) — confianza: {conf:.1%}")
                    # Draw bbox on image
                    img_draw = frame2_rgb.copy()
                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    st.image(img_draw, caption="Placa detectada", use_container_width=True)
                    # Show crop
                    plate_crop = frame2_rgb[y1:y2, x1:x2]
                    st.image(plate_crop, caption="Recorte de placa", width=256)
                else:
                    st.warning("Sin detecciones de placa con confianza > 20%.")

                st.caption(
                    f"Archivo: `{interp_data['file']}` | Backend: `{backend}` | "
                    f"Input: `{inp['dtype'].__name__} {inp['shape'].tolist()}`"
                )

            # ── OCR (ONNX CPU) ───────────────────────────────────────────────
            elif model_sel == "ocr":
                ocr2 = load_ocr()
                if not ocr2:
                    st.error("OCR no disponible. Instala: `pip install 'fast-plate-ocr[onnx]'`")
                else:
                    # First detect plate bbox to get a good crop
                    plate_data = interpreters.get("plate")
                    plate_crop_bgr = None
                    if plate_data:
                        p_out, _, _, _ = run_model_timed(plate_data, frame2_rgb, hint="yolo")
                        bbox, pconf = decode_yolo_best_bbox(p_out, conf_thresh=0.3, img_shape=frame2_rgb.shape[:2])
                        if bbox:
                            x1, y1, x2, y2 = bbox
                            crop = frame2_bgr[y1:y2, x1:x2]
                            if crop.size > 0:
                                plate_crop_bgr = crop
                                st.info(f"📍 Placa detectada ({pconf:.0%}) — usando bbox para crop")

                    if plate_crop_bgr is None:
                        h, w = frame2_rgb.shape[:2]
                        plate_crop_bgr = frame2_bgr[int(h * 0.55):, int(w * 0.15):int(w * 0.85)]
                        st.warning("⚠️ Detector de placa no disponible — usando recorte fallback")

                    # Pass raw BGR crop — the OCR library handles resize/preprocess
                    t0 = time.perf_counter()
                    try:
                        res = ocr2.run(plate_crop_bgr)
                        text = res[0].strip() if isinstance(res, (list, tuple)) else str(res).strip()
                    except Exception as e:
                        text = f"Error: {e}"
                    lat = (time.perf_counter() - t0) * 1000

                    st.success(f"**Texto de placa:** `{text or '(vacío)'}`")
                    c1, c2 = st.columns(2)
                    c1.metric("Latencia CPU (ONNX)", f"{lat:.1f} ms")
                    c2.metric("Mode", "CPU only", help="CCT Transformer no soportado por Edge TPU")
                    plate_crop_rgb = cv2.cvtColor(plate_crop_bgr, cv2.COLOR_BGR2RGB)
                    st.image(plate_crop_rgb, caption="Crop enviado al OCR", width=256)

            # ── TFLite models (vehicle, color, brand) ────────────────────────
            else:
                target = (224, 224) if model_sel == "color" else None
                outputs, lat, est, backend = run_model_timed(
                    interp_data, frame2_rgb, target_size=target, hint=cfg_sel["hint"]
                )
                inp = interp_data["interp"].input_details[0]

                c1, c2, c3 = st.columns(3)
                c1.metric("Latencia CPU", f"{lat:.1f} ms")
                c2.metric("Est. Coral TPU", f"{est:.1f} ms", delta=f"-{lat - est:.1f}ms")
                c3.metric("Speedup estimado", f"~{cfg_sel['scale']}×")

                st.divider()

                # ── Color: mostrar top-5 ─────────────────────────────────────
                if model_sel == "color":
                    st.markdown("**Top 5 colores detectados:**")
                    out_flat = outputs[0].flatten()
                    for i in np.argsort(out_flat)[-5:][::-1]:
                        if i >= len(COLOR_CLASSES):
                            continue
                        name = COLOR_ES.get(COLOR_CLASSES[i], COLOR_CLASSES[i])
                        emoji_c = COLOR_EMOJI.get(COLOR_CLASSES[i], "")
                        pct = float(out_flat[i]) / 255.0
                        st.progress(pct, text=f"{emoji_c} {name}  ({pct:.1%})")

                # ── Brand / Vehicle: top detección ───────────────────────────
                else:
                    st.markdown("**Detección de mayor confianza:**")
                    cls_id, conf = decode_yolo_top(outputs, conf_thresh=0.1)
                    if cls_id is not None and conf > 0.1:
                        if model_sel == "brand":
                            name = BRAND_CLASSES.get(cls_id, f"clase {cls_id}")
                            st.success(f"**{name}** — confianza: {conf:.1%}")
                        else:
                            name = VEHICLE_CLASSES.get(cls_id, f"clase COCO {cls_id}")
                            st.success(f"**{name}** — confianza: {conf:.1%}")
                    else:
                        st.warning("Sin detecciones con confianza > 10% en esta imagen.")

                    st.caption(
                        f"Output shape: `{outputs[0].shape}` | dtype: `{outputs[0].dtype}` | "
                        f"input: `{inp['dtype'].__name__} {inp['shape'].tolist()}`"
                    )
                    with st.expander("Ver output raw (primeras 5 detecciones)"):
                        raw = outputs[0]
                        if raw.ndim == 3:
                            raw = raw[0]
                        if raw.shape[0] < raw.shape[-1]:
                            raw = raw.T
                        st.dataframe(raw[:5])

                st.caption(
                    f"Archivo: `{interp_data['file']}` | "
                    f"Backend: `{backend}` | "
                    f"Input: `{inp['dtype'].__name__} {inp['shape'].tolist()}`"
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

with tab_benchmark:
    st.markdown(f"""
    **¿Qué hace este tab?**
    Corre **{n_runs} inferencias por modelo** con imágenes dummy de ceros para medir
    latencia estable (sin carga de imagen, solo inferencia pura). Luego proyecta cuánto
    tardaría en **Coral Edge TPU real** basándose en benchmarks públicos de hardware similar.

    > Los modelos _edgetpu.tflite compilados tardarían ≈10× menos en hardware real.
    """)
    st.divider()

    if st.button("▶ Ejecutar benchmark completo", use_container_width=True, key="run_bench"):
        results = {}
        progress = st.progress(0, text="Iniciando...")
        available = [(k, v) for k, v in interpreters.items() if v is not None]
        total_steps = len(available)

        for i, (key, interp_data) in enumerate(available):
            cfg = MODELS_CONFIG[key]
            progress.progress((i) / max(total_steps, 1), text=f"Benchmarking {cfg['label']}...")
            interp = interp_data["interp"]
            inp_det = interp.input_details[0]
            dummy = np.zeros(inp_det["shape"], dtype=inp_det["dtype"])

            lats = []
            for _ in range(n_runs):
                interp.set_input_tensor(dummy.copy())
                r = interp.invoke(model_hint=cfg["hint"])
                lats.append(r.latency_ms)

            arr = np.array(lats)
            results[key] = {
                "mean": float(arr.mean()), "min": float(arr.min()),
                "p95": float(np.percentile(arr, 95)),
                "est_tpu": float(arr.mean() / cfg["scale"]),
                "scale": cfg["scale"], "label": cfg["label"],
                "file": interp_data["file"],
            }

        # OCR benchmark (ONNX CPU)
        progress.progress(len(available) / max(total_steps + 1, 1), text="Benchmarking OCR (ONNX)...")
        ocr3 = load_ocr()
        ocr_mean = None
        if ocr3:
            dummy_plate = np.zeros((64, 128, 3), dtype=np.uint8)
            lats_ocr = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                try:
                    ocr3.run(dummy_plate)
                except Exception:
                    pass
                lats_ocr.append((time.perf_counter() - t0) * 1000)
            ocr_mean = float(np.mean(lats_ocr))

        progress.progress(1.0, text="¡Listo!")

        # ── Tabla de resultados ───────────────────────────────────────────────
        st.markdown("### Resultados")
        cols = st.columns([2.5, 1, 1, 1, 1.2, 0.8])
        for col, h in zip(cols, ["Modelo", "Media", "Mín", "P95", "Est. TPU", "Speedup"]):
            col.markdown(f"**{h}**")

        total_cpu = 0.0
        total_tpu = 0.0
        for key, res in results.items():
            c1, c2, c3, c4, c5, c6 = st.columns([2.5, 1, 1, 1, 1.2, 0.8])
            c1.markdown(res["label"])
            c2.markdown(f"`{res['mean']:.1f} ms`")
            c3.markdown(f"`{res['min']:.1f} ms`")
            c4.markdown(f"`{res['p95']:.1f} ms`")
            c5.markdown(f"**`{res['est_tpu']:.1f} ms`**")
            c6.markdown(f"~{res['scale']}×")
            total_cpu += res["mean"]
            total_tpu += res["est_tpu"]

        if ocr_mean is not None:
            c1, c2, c3, c4, c5, c6 = st.columns([2.5, 1, 1, 1, 1.2, 0.8])
            c1.markdown("🔤 OCR (ONNX)")
            c2.markdown(f"`{ocr_mean:.1f} ms`")
            c3.markdown("—")
            c4.markdown("—")
            c5.markdown("`CPU only`")
            c6.markdown("1×")
            total_cpu += ocr_mean
            total_tpu += ocr_mean  # OCR stays on CPU

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"<div class='metric-card'><div class='val'>{total_cpu:.0f}"
                f"<span style='font-size:1rem'>ms</span></div>"
                f"<div class='lbl'>Total pipeline CPU</div></div>",
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"<div class='metric-card'><div class='val' style='color:#3fb950'>{total_tpu:.0f}"
                f"<span style='font-size:1rem'>ms</span></div>"
                f"<div class='lbl'>Estimado Coral TPU</div></div>",
                unsafe_allow_html=True
            )
        with c3:
            speedup = total_cpu / max(total_tpu, 1)
            st.markdown(
                f"<div class='metric-card'><div class='val' style='color:#d29922'>{speedup:.1f}×</div>"
                f"<div class='lbl'>Speedup total estimado</div></div>",
                unsafe_allow_html=True
            )

        st.info("""
        **Notas sobre los estimados Coral:**
        - **YOLO nano INT8 (640px):** ~17ms en Coral USB → factor ~11× sobre CPU M4
        - **EfficientNet-lite INT8:** ~4ms en Coral → factor ~6×
        - **OCR CCT-XS:** queda en CPU — los transformers (self-attention) no son compatibles con Edge TPU
        - Los `_edgetpu.tflite` requieren compilar con `bash scripts/07_compilar_edgetpu.sh` en Linux/Docker amd64
        """)

    # ── Estado de archivos ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📁 Estado de archivos en `models/tflite_exports/`")
    esperados = [
        ("yolo11n_coco_vehicle_int8.tflite",           "Vehicle INT8 ✅"),
        ("yolo11n_coco_vehicle_int8_edgetpu.tflite",   "Vehicle Edge TPU"),
        ("color_classifier_int8.tflite",               "Color INT8 ✅"),
        ("color_classifier_int8_edgetpu.tflite",        "Color Edge TPU"),
        ("marca_detector_yolo11n_int8.tflite",         "Brand INT8 ✅"),
        ("marca_detector_yolo11n_int8_edgetpu.tflite", "Brand Edge TPU"),
        ("placa_detector_yolo11n_int8.tflite",         "Plate INT8 ✅"),
        ("placa_detector_yolo11n_int8_edgetpu.tflite", "Plate Edge TPU"),
    ]
    for fname, label in esperados:
        path = EXPORTS_DIR / fname
        if path.exists():
            size = path.stat().st_size / 1024 / 1024
            st.markdown(f"✅ `{fname}` — {size:.1f} MB — *{label}*")
        else:
            is_edgetpu = "_edgetpu" in fname
            icon = "⏳" if is_edgetpu else "❌"
            hint = "→ `bash scripts/07_compilar_edgetpu.sh` (requiere Docker + Linux/amd64)" if is_edgetpu else "→ run export script"
            st.markdown(f"{icon} `{fname}` — *{label}* {hint}")
