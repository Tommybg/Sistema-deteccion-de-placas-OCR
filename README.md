# 🚗 Sistema deteccion de placas + OCR 

Guía detallada del sistema de Reconocimiento Automático de Placas Vehiculares (ANPR) para Colombia.

---

## 📋 Tabla de Contenidos

1. [Estructura del Proyecto](#-estructura-del-proyecto)
2. [Qué Hace Cada Script](#-qué-hace-cada-script)
3. [Cómo Funciona el Entrenamiento](#-cómo-funciona-el-entrenamiento)
4. [Los Modelos Generados](#-los-modelos-generados)
5. [Deployment para Coral Edge TPU](#-deployment-para-coral-edge-tpu)
6. [Comandos Rápidos](#-comandos-rápidos)
7. [Instalación Detallada](#-instalación-detallada)
8. [Troubleshooting](#-troubleshooting)

---

## 📂 Estructura del Proyecto

```
anpr_project/
├── setup.sh                 # Instalación del entorno
├── app_demo.py              # Demo web Streamlit
├── requirements.txt         # Dependencias
├── scripts/                 # Pipeline de ML
│   ├── 01_preparar_dataset.py
│   ├── 02_entrenar_modelo.py
│   ├── 03_exportar_tflite.py
│   └── 04_inferencia_tiempo_real.py
├── models/                  # Modelos entrenados
│   ├── placa_detector_yolo11n.pt       # PyTorch (5.2 MB)
│   ├── placa_detector_yolo11n.onnx     # ONNX (10 MB)
│   └── placa_detector_yolo11n_saved_model/
│       ├── placa_detector_yolo11n_float32.tflite    # 10 MB
│       ├── placa_detector_yolo11n_float16.tflite    # 5.1 MB
│       └── placa_detector_yolo11n_dynamic_range_quant.tflite  # 2.8 MB ⭐
├── dataset_combinado/       # Dataset unificado
└── output/                  # Resultados de entrenamiento
```

---

## 🔧 Qué Hace Cada Script

### [setup.sh](setup.sh)

Configura el entorno de desarrollo completo:

1. **Busca Python 3.11** (requerido por PyTorch/TensorFlow)
2. **Crea entorno virtual** `anpr_env/` aislado
3. **Instala dependencias**: PyTorch, Ultralytics, TensorFlow, fast-plate-ocr
4. **Crea directorios** necesarios

```bash
./setup.sh  # Ejecutar una sola vez
```
## Dataset: https://drive.google.com/drive/folders/1sVXDxPxJC0eKLjrj66TlhHmid9qZFTnq?usp=sharing 

### [01_preparar_dataset.py](scripts/01_preparar_dataset.py)

Combina múltiples datasets de Roboflow en uno solo, fusionando carpetas y re-etiquetando para evitar colisiones.

**Resultado:** `dataset_combinado/` con 1,212 imágenes organizadas para YOLO.

### [02_entrenar_modelo.py](scripts/02_entrenar_modelo.py)

Entrena el modelo YOLOv11 nano para detección de placas usando Transfer Learning.

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `--epochs` | 200 | Iteraciones de entrenamiento |
| `--batch` | 16 | Imágenes por lote |
| `--device` | auto | Detecta MPS/CUDA/CPU |
| `--patience` | 20 | Early stopping |

**Métricas obtenidas:** mAP50: **99.5%** | Precisión: **99.9%** | Recall: **100%**

### [03_exportar_tflite.py](scripts/03_exportar_tflite.py)

Convierte el modelo PyTorch a formatos para dispositivos edge (TFLite FP32, FP16, INT8) y prepara la compilación para Edge TPU.

### [04_inferencia_tiempo_real.py](scripts/04_inferencia_tiempo_real.py)

Ejecuta detección + OCR en video/webcam.

```bash
python 04_inferencia_tiempo_real.py --source 0  # Webcam
```

---

## 🧠 Cómo Funciona el Entrenamiento

### Modelo Base: YOLOv11n (nano)
- **Parámetros**: 2.6 millones (muy ligero)
- **Tamaño**: ~5 MB
- **Arquitectura**: YOLO v11 optimizada

### Proceso Técnico
1. **Carga modelo preentrenado** (`yolo11n.pt`)
2. **Fine-tuning** con dataset de placas colombianas
3. **Transfer Learning** (ajuste de pesos para clase "placa")
4. **Data Augmentation**: Mosaic, HSV, rotación

---

## 📦 Los Modelos Generados

| Formato | Tamaño | Precisión | Velocidad | Uso Recomendado |
|---------|--------|-----------|-----------|-----------------|
| `.pt` (PyTorch) | 5.2 MB | 100% | Lento | Demo / Desarrollo |
| `.onnx` | 10 MB | 100% | Medio | Multiplataforma |
| `float16.tflite` | 5.1 MB | ~99% | Rápido | Móviles / RPi 4 |
| **`dynamic_range_quant.tflite`** | **2.8 MB** | ~98% | **Muy rápido** | **Coral TPU ⭐** |

### 🎯 ¿Cuál usar?

- **Para Demo (PC/Mac):** `models/placa_detector_yolo11n.pt`
- **Para Cliente (Coral TPU):** `models/.../placa_detector_yolo11n_dynamic_range_quant.tflite` (Requiere compilación, ver abajo)

---

## 🐚 Deployment para Coral Edge TPU

### Paso 1: Ya Completado
Tenemos el modelo exportado a INT8: `placa_detector_yolo11n_dynamic_range_quant.tflite`

### Paso 2: Compilar para Edge TPU (en Linux)
El compilador de Edge TPU **solo funciona en Linux x86_64**.

**En el dispositivo del cliente (Linux):**
```bash
# Instalar compilador
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install edgetpu-compiler

# Compilar
edgetpu_compiler placa_detector_yolo11n_dynamic_range_quant.tflite
```
Genera → `placa_detector_yolo11n_dynamic_range_quant_edgetpu.tflite`

---

## 🚀 Comandos Rápidos

```bash
# Activar entorno
source anpr_env/bin/activate

# Preparar dataset
python scripts/01_preparar_dataset.py

# Entrenar
python scripts/02_entrenar_modelo.py --epochs 200

# Exportar para Edge
python scripts/03_exportar_tflite.py --formato int8

# Demo Web (Streamlit)
streamlit run app_demo.py

# Inferencia Webcam
python scripts/04_inferencia_tiempo_real.py --source 0
```

---

## 🛠 Instalación Detallada

### Requisitos Previos
- Python 3.8 - 3.11
- Hardware: CPU (básico), GPU NVIDIA o Apple Silicon (recomendado para entrenamiento)

### Paso 1: Configurar Entorno
```bash
./setup.sh
```
O manualmente:
```bash
python -m venv anpr_env
source anpr_env/bin/activate
pip install -r requirements.txt
```

### Paso 2: Verificar Dependencias
```bash
python -c "from ultralytics import YOLO; print('YOLO OK')"
python -c "from fast_plate_ocr import LicensePlateRecognizer; print('OCR OK')"
```

---

## ❓ Troubleshooting

### Error: CUDA out of memory
Reduce el tamaño del batch:
```bash
python 02_entrenar_modelo.py --batch 8
```

### Error: OCR no detecta nada
El OCR descargará el modelo automáticamente la primera vez. Asegúrate de tener internet.
El modelo usado es `cct-xs-v1-global-model`.

### Soporte Mac (M1/M2/M3/M4/M5)
El sistema detecta automáticamente MPS (Metal Performance Shaders). Usa `--device auto` o `--device mps`.


