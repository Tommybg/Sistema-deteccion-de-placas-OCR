# 🚀 Guía Rápida - Sistema ANPR

## Instalación en 5 Minutos

### Linux/Mac
```bash
cd anpr_project
./setup.sh
source anpr_env/bin/activate
```

### Windows
```cmd
cd anpr_project
setup.bat
anpr_env\Scripts\activate
```

---

## Flujo de Trabajo Completo

### 1️⃣ Preparar Dataset
```bash
python scripts/01_preparar_dataset.py
```
**Resultado**: Combina todos los datasets en `dataset_combinado/`

### 2️⃣ Entrenar Modelo
```bash
# Con GPU
python scripts/02_entrenar_modelo.py --epochs 100 --batch 16 --device 0

# Sin GPU (CPU)
python scripts/02_entrenar_modelo.py --epochs 50 --batch 8 --device cpu
```
**Resultado**: Modelo guardado en `models/placa_detector_yolo11n.pt`

### 3️⃣ Exportar a TFLite
```bash
python scripts/03_exportar_tflite.py
```
**Resultado**: Modelos en `models/tflite_exports/`

### 4️⃣ Probar en Tiempo Real
```bash
# Webcam
python scripts/04_inferencia_tiempo_real.py --source 0

# Video
python scripts/04_inferencia_tiempo_real.py --source video.mp4

# Cámara IP
python scripts/04_inferencia_tiempo_real.py --source "rtsp://ip:554/stream"
```

---

## Atajos Útiles

| Comando | Descripción |
|---------|-------------|
| `python scripts/02_entrenar_modelo.py --evaluar` | Entrena y evalúa |
| `python scripts/03_exportar_tflite.py --formato int8` | Solo INT8 |
| `python scripts/04_inferencia_tiempo_real.py --log` | Con logging |
| `python scripts/04_inferencia_tiempo_real.py --tflite` | Usar TFLite |

---

## Métricas Esperadas

| Métrica | Valor Objetivo |
|---------|----------------|
| mAP50 | > 0.90 |
| FPS (GPU) | 30+ |
| FPS (CPU) | 5-10 |
| FPS (Coral) | 15-25 |

---

## ¿Problemas?

Ver [README.md](../README.md) sección **Troubleshooting**
