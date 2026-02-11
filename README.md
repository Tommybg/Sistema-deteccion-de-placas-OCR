# 🚗 Sistema ANPR — Detección de Placas + Tipo de Vehículo + OCR

Sistema de Reconocimiento Automático de Placas Vehiculares (ANPR) con clasificación de tipo de vehículo, diseñado para despliegue en dispositivos edge (Coral Edge TPU).

---

## 📋 Tabla de Contenidos

1. [Arquitectura del Pipeline](#-arquitectura-del-pipeline)
2. [Estructura del Proyecto](#-estructura-del-proyecto)
3. [Qué Hace Cada Script](#-qué-hace-cada-script)
4. [Cómo Funciona el Entrenamiento](#-cómo-funciona-el-entrenamiento)
5. [Los Modelos Generados](#-los-modelos-generados)
6. [Deployment para Coral Edge TPU](#-deployment-para-coral-edge-tpu)
7. [Comandos Rápidos](#-comandos-rápidos)
8. [Instalación Detallada](#-instalación-detallada)
9. [Troubleshooting](#-troubleshooting)

---

## 🏗 Arquitectura del Pipeline

El sistema utiliza un pipeline multi-modelo que procesa cada frame de video en secuencia:

```
              Frame de Cámara
                    │
         ┌──────────▼──────────┐
         │   Modelo 1: YOLOv11n │   Detecta vehículos y clasifica
         │   Tipo de Vehículo   │   por tipo (Automóvil, Bus,
         │   (~2.8 MB INT8)     │   Camión, Motocicleta)
         └──────────┬──────────┘
                    │ recortes de vehículos
         ┌──────────┼──────────────┐
         ▼          ▼              ▼
   ┌───────────┐ ┌───────────┐ ┌──────────────┐
   │ Modelo 2  │ │ Modelo 3  │ │  Modelo 4    │
   │ Color     │ │ Marca     │ │  Detección   │
   │ (Fase 2)  │ │ (Fase 3)  │ │  de Placas   │
   │ ~3 MB     │ │ ~4 MB     │ │  ~2.8 MB     │
   │ Próximo   │ │ Próximo   │ │  ✅ Listo     │
   └───────────┘ └───────────┘ └──────┬───────┘
                                      │
                                      ▼
                                ┌───────────┐
                                │    OCR    │   Lee texto de la placa
                                │  ✅ Listo  │
                                └───────────┘
```

### Estado de los modelos

| Modelo | Función | Estado | Tamaño (INT8) |
|--------|---------|--------|---------------|
| **Tipo de Vehículo** | Automóvil, Motocicleta, Bus, Camión | ✅ Listo | ~2.8 MB |
| **Detección de Placas** | Localiza placas en el frame | ✅ Listo | ~2.8 MB |
| **OCR** | Lee caracteres alfanuméricos de la placa | ✅ Listo | — |
| **Color** | Blanco, negro, rojo, azul, etc. | 🔜 Fase 2 | ~3 MB |
| **Marca** | Chevrolet, Renault, Mazda, etc. | 🔜 Fase 3 | ~4 MB |

**Huella total en Coral Edge TPU:** ~12-13 MB (los modelos se ejecutan secuencialmente, ~30-40ms por frame → **25+ FPS en tiempo real**).

---

## 📂 Estructura del Proyecto

```
anpr_project/
├── setup.sh                 # Instalación del entorno
├── app_demo.py              # Demo web Streamlit (local)
├── app_cloud.py             # Demo web Streamlit (Railway)
├── requirements.txt         # Dependencias
├── scripts/                 # Pipeline de ML
│   ├── 01_preparar_dataset.py
│   ├── 02_entrenar_modelo.py
│   ├── 03_exportar_tflite.py
│   ├── 04_inferencia_tiempo_real.py
│   ├── vehicle_detector.py          # Módulo de detección de tipo de vehículo
│   └── yolo11n.pt                   # Modelo de detección de vehículos (~5 MB)
├── models/                  # Modelos entrenados
│   ├── placa_detector_yolo11n.pt       # Detección de placas - PyTorch (5.2 MB)
│   ├── placa_detector_yolo11n.onnx     # Detección de placas - ONNX (10 MB)
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

### [vehicle_detector.py](scripts/vehicle_detector.py)

Módulo compartido de detección de tipo de vehículo. Clasifica cada vehículo detectado en una de las siguientes categorías:

| Tipo | Descripción |
|------|-------------|
| 🚗 Automóvil | Sedán, SUV, hatchback, etc. |
| 🏍 Motocicleta | Motos de cualquier tipo |
| 🚌 Bus | Buses, busetas |
| 🚛 Camión | Camiones, furgones |

El módulo también se encarga de **asociar cada placa detectada con su vehículo correspondiente**, usando la posición espacial de los bounding boxes (la placa debe estar contenida dentro del vehículo).

### [04_inferencia_tiempo_real.py](scripts/04_inferencia_tiempo_real.py)

Ejecuta el pipeline completo (detección de vehículos + detección de placas + OCR) en video/webcam.

```bash
# Con detección de tipo de vehículo (activado por defecto)
python scripts/04_inferencia_tiempo_real.py --source 0

# Sin detección de vehículos
python scripts/04_inferencia_tiempo_real.py --source 0 --no-vehicle-detection
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

## 🚗 Detección de Tipo de Vehículo

El sistema detecta automáticamente el tipo de cada vehículo en el frame y lo asocia con su placa correspondiente.

### ¿Cómo funciona?

1. **Detección de vehículos** — El modelo `yolo11n.pt` analiza el frame completo y localiza cada vehículo, clasificándolo por tipo (Automóvil, Motocicleta, Bus, Camión)
2. **Detección de placas** — El modelo `placa_detector_yolo11n.pt` localiza las placas vehiculares
3. **Asociación placa → vehículo** — El sistema vincula cada placa con el vehículo que la contiene usando la posición espacial de los bounding boxes
4. **OCR** — Lee los caracteres de cada placa detectada

### Resultado por vehículo

Para cada vehículo detectado, el sistema entrega:

```
┌─────────────────────────────┐
│  [recorte de la placa]      │
│  📋 Placa: ABC-123          │
│  📊 Confianza: 98.5%        │
│  🚗 Tipo: Automóvil         │
│  🎨 Color: Próximamente     │
│  🏭 Marca: Próximamente     │
└─────────────────────────────┘
```

### Modelos en Coral Edge TPU

| Modelo | Archivo | Tamaño |
|--------|---------|--------|
| Detección de placas | `placa_detector_yolo11n_dynamic_range_quant.tflite` | ~2.8 MB |
| Detección de vehículos | `yolo11n_coco_vehicle_int8.tflite` | ~2.89 MB |

Ambos modelos se ejecutan secuencialmente en el Edge TPU con latencia mínima.

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


