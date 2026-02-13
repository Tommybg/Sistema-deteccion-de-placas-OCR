#!/usr/bin/env python3
"""
================================================================================
APP CLOUD - STREAMLIT PARA SISTEMA ANPR (VERSIÓN CLOUD)
================================================================================
Versión adaptada para deploy en Railway.
- Modo "Subir imagen" + "Imágenes de prueba" (samples/)
- Sin webcam ni dataset local

Ejecutar: streamlit run app_cloud.py
================================================================================
"""

import streamlit as st
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io
from scripts.vehicle_detector import VehicleDetector
from scripts.brand_detector import BrandDetector

# Rutas del proyecto
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"
SAMPLES_DIR = PROJECT_DIR / "samples"

# Configuración de página
st.set_page_config(
    page_title="ANPR - Detección de Placas",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .plate-text {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        font-family: monospace;
    }
    .confidence {
        font-size: 1rem;
        color: #666;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    """Carga el modelo YOLO para detección de placas."""
    from ultralytics import YOLO
    
    model_path = MODELS_DIR / "placa_detector_yolo11n.pt"
    if not model_path.exists():
        # Buscar en output si no está en models
        model_paths = list(PROJECT_DIR.rglob("**/weights/best.pt"))
        if model_paths:
            model_path = model_paths[0]
        else:
            return None
    
    return YOLO(str(model_path))


@st.cache_resource
def load_vehicle_detector():
    """Carga el modelo COCO para detección de tipo de vehículo."""
    return VehicleDetector()


@st.cache_resource
def load_brand_detector():
    """Carga el modelo para detección de marca de vehículo."""
    model_path = MODELS_DIR / "marca_detector_yolo11n.pt"
    if not model_path.exists():
        return None
    return BrandDetector(model_path=str(model_path))


@st.cache_resource
def load_ocr():
    """Carga el modelo OCR para lectura de placas."""
    try:
        from fast_plate_ocr import LicensePlateRecognizer
        return LicensePlateRecognizer('cct-xs-v1-global-model')
    except Exception as e:
        st.warning(f"OCR no disponible: {e}")
        return None


def detect_plates(model, image_np, vehicle_detector=None, brand_detector=None):
    """
    Detecta placas, vehículos y marcas en la imagen.

    Returns:
        annotated: Imagen con anotaciones
        plates: Lista de recortes de placas (con vehicle_type y brand si disponible)
        vehicles: Lista de vehículos detectados
    """
    # Detectar vehículos primero
    vehicles = []
    if vehicle_detector is not None:
        vehicles = vehicle_detector.detect(image_np)

    # Detectar marcas
    brand_detections = []
    vehicle_brands = {}
    if brand_detector is not None:
        brand_detections = brand_detector.detect(image_np)
        for bd in brand_detections:
            matched = BrandDetector.associate_brand_to_vehicle(bd["bbox"], vehicles)
            if matched is not None:
                v_id = id(matched)
                if v_id not in vehicle_brands or bd["confidence"] > vehicle_brands[v_id][1]:
                    vehicle_brands[v_id] = (bd["brand"], bd["confidence"])

    # Detectar placas
    results = model(image_np, verbose=False, device="cpu")

    # Construir imagen anotada manualmente
    annotated = image_np.copy()

    # Dibujar vehículos en azul
    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f'{v["type"]} {v["confidence"]:.0%}'
        v_id = id(v)
        if v_id in vehicle_brands:
            label += f' | {vehicle_brands[v_id][0]}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), (255, 0, 0), -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Dibujar logos de marca en naranja
    for bd in brand_detections:
        bx1, by1, bx2, by2 = bd["bbox"]
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 165, 255), 2)

    # Extraer recortes de placas y dibujar en verde
    plates = []
    for r in results[0].boxes:
        box = r.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        conf = float(r.conf[0])

        plate_crop = image_np[y1:y2, x1:x2]
        plate_info = {
            'image': plate_crop,
            'box': (x1, y1, x2, y2),
            'confidence': conf,
            'vehicle_type': None,
            'brand': None,
        }

        # Asociar placa a vehículo
        if vehicle_detector is not None and vehicles:
            match = VehicleDetector.associate_plate_to_vehicle((x1, y1, x2, y2), vehicles)
            if match:
                plate_info['vehicle_type'] = match["type"]
                v_id = id(match)
                if v_id in vehicle_brands:
                    plate_info['brand'] = vehicle_brands[v_id][0]

        plates.append(plate_info)

        # Dibujar placa en verde
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        plabel = f'Placa {conf:.0%}'
        (tw, th), _ = cv2.getTextSize(plabel, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(annotated, plabel, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return annotated, plates, vehicles


def read_plate_text(ocr, plate_image):
    """Lee el texto de una placa usando OCR."""
    if ocr is None:
        return "OCR no disponible"
    
    try:
        if isinstance(plate_image, np.ndarray):
            temp_path = "/tmp/plate_temp.jpg"
            cv2.imwrite(temp_path, plate_image)
            result = ocr.run(temp_path)
            return result[0] if result else "No detectado"
    except Exception as e:
        return f"Error: {str(e)[:30]}"


def get_sample_images():
    """Lista las imágenes de prueba disponibles en samples/."""
    if not SAMPLES_DIR.exists():
        return []
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in extensions:
        images.extend(SAMPLES_DIR.glob(ext))
    
    return sorted(images, key=lambda x: x.name)


def main():
    # Header
    st.markdown('<p class="main-header">🚗 Sistema ANPR</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Reconocimiento Automático de Placas Vehiculares</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Fuente de imagen
        source = st.radio(
            "Fuente de imagen:",
            ["📂 Imágenes de prueba", "📁 Subir imagen"],
            index=0
        )
        
        # Umbral de confianza
        conf_threshold = st.slider(
            "Umbral de confianza:",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        st.divider()
        
        # Info del modelo
        st.header("📊 Info del Modelo")
        st.info("""
        **Detector Placas:** YOLOv11n
        **Detector Vehículos:** YOLOv11n (COCO)
        **Detector Marcas:** YOLOv11n (30 marcas)
        **OCR:** fast-plate-ocr
        **Modelo OCR:** cct-xs-v1-global
        """)
        
        st.divider()
        st.caption("🚀 Desplegado en Railway")
    
    # Cargar modelos
    with st.spinner("Cargando modelos..."):
        detector = load_detector()
        ocr = load_ocr()
        vehicle_detector = load_vehicle_detector()
        brand_detector = load_brand_detector()

    if detector is None:
        st.error("❌ No se encontró el modelo de detección.")
        return
    
    # Contenedor principal
    image = None
    image_np = None
    
    if source == "📂 Imágenes de prueba":
        sample_images = get_sample_images()
        
        if sample_images:
            st.info(f"📂 {len(sample_images)} imágenes de prueba disponibles. Selecciona una para detectar placas.")
            
            selected = st.selectbox(
                "Selecciona una imagen de prueba:",
                sample_images,
                format_func=lambda x: x.name
            )
            
            if selected:
                image = Image.open(selected)
                image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            st.warning("No se encontraron imágenes de prueba en samples/")
    
    elif source == "📁 Subir imagen":
        uploaded_file = st.file_uploader(
            "Selecciona una imagen",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Sube una imagen con vehículos para detectar placas"
        )
        
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Procesar imagen
    if image is not None and image_np is not None:
        st.divider()
        
        # Detectar placas y vehículos
        with st.spinner("⏳ Detectando vehículos y placas..."):
            annotated, plates, vehicles = detect_plates(detector, image_np, vehicle_detector, brand_detector)
        
        # Mostrar resultados
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Imagen Original")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Detecciones")
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)
        
        # Mostrar placas detectadas
        st.divider()
        st.subheader(f"🚘 Placas Detectadas: {len(plates)}")
        
        if plates:
            plate_cols = st.columns(min(len(plates), 4))
            
            for i, plate in enumerate(plates):
                with plate_cols[i % 4]:
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    
                    plate_rgb = cv2.cvtColor(plate['image'], cv2.COLOR_BGR2RGB)
                    st.image(plate_rgb, caption=f"Placa {i+1}", use_container_width=True)
                    
                    with st.spinner("Leyendo..."):
                        plate_text = read_plate_text(ocr, plate['image'])
                    
                    st.markdown(f'<p class="plate-text">{plate_text}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="confidence">Confianza: {plate["confidence"]:.1%}</p>', unsafe_allow_html=True)
                    if plate.get('vehicle_type'):
                        st.markdown(f'<p class="confidence">🚗 Tipo: {plate["vehicle_type"]}</p>', unsafe_allow_html=True)
                    if plate.get('brand'):
                        st.markdown(f'<p class="confidence">🏭 Marca: {plate["brand"]}</p>', unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No se detectaron placas en esta imagen. Intenta con otra imagen o ajusta el umbral de confianza.")
        
        # Estadísticas
        st.divider()
        col_stats = st.columns(5)

        with col_stats[0]:
            st.metric("Vehículos Detectados", len(vehicles))

        with col_stats[1]:
            st.metric("Placas Detectadas", len(plates))

        with col_stats[2]:
            if plates:
                avg_conf = sum(p['confidence'] for p in plates) / len(plates)
                st.metric("Confianza Promedio", f"{avg_conf:.1%}")

        with col_stats[3]:
            st.metric("Modelo", "YOLOv11n")

        with col_stats[4]:
            st.metric("OCR", "cct-xs-v1")


if __name__ == "__main__":
    main()
