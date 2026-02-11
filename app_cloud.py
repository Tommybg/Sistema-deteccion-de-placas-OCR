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
def load_ocr():
    """Carga el modelo OCR para lectura de placas."""
    try:
        from fast_plate_ocr import LicensePlateRecognizer
        return LicensePlateRecognizer('cct-xs-v1-global-model')
    except Exception as e:
        st.warning(f"OCR no disponible: {e}")
        return None


def detect_plates(model, image_np):
    """
    Detecta placas en la imagen.
    
    Returns:
        results: Resultados de YOLO
        annotated: Imagen con anotaciones
        plates: Lista de recortes de placas
    """
    results = model(image_np, verbose=False)
    annotated = results[0].plot()
    
    plates = []
    for r in results[0].boxes:
        box = r.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        conf = float(r.conf[0])
        
        plate_crop = image_np[y1:y2, x1:x2]
        plates.append({
            'image': plate_crop,
            'box': (x1, y1, x2, y2),
            'confidence': conf
        })
    
    return results, annotated, plates


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
        **Detector:** YOLOv11n  
        **OCR:** fast-plate-ocr  
        **Modelo:** cct-xs-v1-global
        """)
        
        st.divider()
        st.caption("🚀 Desplegado en Railway")
    
    # Cargar modelos
    with st.spinner("Cargando modelos..."):
        detector = load_detector()
        ocr = load_ocr()
    
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
        
        # Detectar placas
        with st.spinner("⏳ Detectando placas..."):
            results, annotated, plates = detect_plates(detector, image_np)
        
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
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No se detectaron placas en esta imagen. Intenta con otra imagen o ajusta el umbral de confianza.")
        
        # Estadísticas
        st.divider()
        col_stats = st.columns(4)
        
        with col_stats[0]:
            st.metric("Placas Detectadas", len(plates))
        
        with col_stats[1]:
            if plates:
                avg_conf = sum(p['confidence'] for p in plates) / len(plates)
                st.metric("Confianza Promedio", f"{avg_conf:.1%}")
        
        with col_stats[2]:
            st.metric("Modelo", "YOLOv11n")
        
        with col_stats[3]:
            st.metric("OCR", "cct-xs-v1")


if __name__ == "__main__":
    main()
