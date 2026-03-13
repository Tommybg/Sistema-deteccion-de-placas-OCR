# Sistema ANPR — Guia de Implementacion en Google Coral Edge TPU

## 1. Resumen Ejecutivo

El sistema ANPR (Automatic Number Plate Recognition) identifica vehiculos en tiempo real extrayendo 5 datos clave de cada vehiculo capturado por camara:

1. **Tipo de vehiculo** (auto, moto, bus, camion, bicicleta)
2. **Color del vehiculo** (15 colores)
3. **Marca del vehiculo** (30 marcas)
4. **Ubicacion de la placa** (coordenadas en la imagen)
5. **Texto de la placa** (caracteres alfanumericos)

El pipeline esta disenado para correr en hardware **Google Coral Edge TPU**, un acelerador de inferencia que ejecuta modelos INT8 con latencia minima y bajo consumo energetico. El sistema es 100% local — no requiere conexion a internet ni servicios en la nube.

---

## 2. Arquitectura del Pipeline

El pipeline procesa cada frame de camara en 5 pasos secuenciales:

```
Imagen de camara (frame RGB)
        |
        v
  [1. DETECTOR DE VEHICULOS]  ──>  Tipo + bounding box del vehiculo
        |
        | (crop del vehiculo)
        v
  [2. CLASIFICADOR DE COLOR]  ──>  Color del vehiculo (sobre el recorte)
        |
        v
  [3. DETECTOR DE MARCA]      ──>  Logo/marca del vehiculo
        |
        v
  [4. DETECTOR DE PLACA]      ──>  Bounding box de la placa
        |
        | (crop de la placa)
        v
  [5. OCR DE PLACA]           ──>  Texto alfanumerico de la placa
```

### Flujo inteligente de recortes

El sistema no envia la imagen completa a todos los modelos. Utiliza un flujo inteligente:

- **Paso 1** detecta el vehiculo y extrae su bounding box
- **Paso 2** recorta la region del vehiculo y clasifica el color **solo sobre ese recorte**, evitando que el fondo (cielo, carretera, edificios) contamine la prediccion
- **Paso 4** detecta la placa en la imagen completa
- **Paso 5** recorta la region de la placa y lee los caracteres **solo sobre ese recorte**

---

## 3. Modelos del Sistema

### 3.1 Detector de Vehiculos

| Propiedad | Valor |
|-----------|-------|
| **Archivo** | `yolo11n_coco_vehicle_int8.tflite` |
| **Peso** | 2.9 MB |
| **Arquitectura** | YOLO11 nano (cuantizado INT8) |
| **Entrada** | 640 x 640 x 3 (INT8) |
| **Salida** | Bounding boxes + clase + confianza |
| **Hardware** | Coral Edge TPU |
| **Latencia estimada Coral** | ~3 ms |
| **Clases** | Auto, Moto, Bus, Camion, Bicicleta (5 clases) |

**Funcion:** Detecta vehiculos en la imagen completa. Devuelve el tipo de vehiculo (auto, moto, etc.) y las coordenadas del bounding box. Este bounding box se usa para recortar el vehiculo y enviarlo al clasificador de color.

Base: modelo YOLO11n preentrenado en el dataset COCO, filtrado a las 5 clases vehiculares relevantes.

---

### 3.2 Clasificador de Color

| Propiedad | Valor |
|-----------|-------|
| **Archivo** | `color_classifier_int8.tflite` |
| **Peso** | 1.6 MB |
| **Arquitectura** | YOLO11n-cls (clasificacion, cuantizado INT8) |
| **Entrada** | 224 x 224 x 3 (INT8) |
| **Salida** | Probabilidades para 15 clases de color |
| **Hardware** | Coral Edge TPU |
| **Latencia estimada Coral** | ~2-4 ms |
| **Clases (15)** | Beige, Negro, Azul, Cafe, Dorado, Verde, Gris, Naranja, Rosa, Morado, Rojo, Plata, Canela, Blanco, Amarillo |

**Funcion:** Recibe el recorte del vehiculo (no la imagen completa) y clasifica su color dominante entre 15 opciones. El modelo fue entrenado especificamente con imagenes recortadas de vehiculos, por lo que requiere recibir un crop limpio del vehiculo para funcionar correctamente.

Es el modelo mas liviano del sistema (1.6 MB) gracias a la arquitectura de clasificacion YOLO11n-cls optimizada para esta tarea.

---

### 3.3 Detector de Marca

| Propiedad | Valor |
|-----------|-------|
| **Archivo** | `marca_detector_yolo11n_int8.tflite` |
| **Peso** | 2.9 MB |
| **Arquitectura** | YOLO11 nano (deteccion, cuantizado INT8) |
| **Entrada** | 640 x 640 x 3 (INT8) |
| **Salida** | Bounding boxes de logos + clase + confianza |
| **Hardware** | Coral Edge TPU |
| **Latencia estimada Coral** | ~3 ms |
| **Clases (30)** | Acura, Audi, BMW, Chevrolet, Citroen, Dacia, Fiat, Ford, Honda, Hyundai, Infiniti, KIA, Lamborghini, Lexus, Mazda, Mercedes-Benz, Mitsubishi, Nissan, Opel, Perodua, Peugeot, Porsche, Proton, Renault, Seat, Suzuki, Tesla, Toyota, Volkswagen, Volvo |

**Funcion:** Detecta el logo de la marca del vehiculo en la imagen. Es un modelo de deteccion de objetos (no clasificacion), ya que busca el logo como un objeto dentro de la imagen. Funciona sobre la imagen completa porque el logo puede estar en distintas ubicaciones del vehiculo.

---

### 3.4 Detector de Placa

| Propiedad | Valor |
|-----------|-------|
| **Archivo** | `placa_detector_yolo11n_int8.tflite` |
| **Peso** | 2.8 MB |
| **Arquitectura** | YOLO11 nano (deteccion, cuantizado INT8) |
| **Entrada** | 640 x 640 x 3 (INT8) |
| **Salida** | Bounding box de la placa + confianza |
| **Hardware** | Coral Edge TPU |
| **Latencia estimada Coral** | ~3 ms |
| **Clases** | 1 clase (placa) |

**Funcion:** Localiza la placa vehicular en la imagen. Devuelve las coordenadas exactas (bounding box) de la placa, que se usan para recortar esa region y enviarla al paso de OCR. Es un modelo de deteccion entrenado especificamente para placas vehiculares.

---

### 3.5 OCR de Placa

| Propiedad | Valor |
|-----------|-------|
| **Modelo** | CCT-XS (Compact Convolutional Transformer) |
| **Libreria** | `fast-plate-ocr` (paquete pip) |
| **Peso** | ~2.0 MB (ONNX) |
| **Entrada** | 128 x 64 x 3 (imagen de placa recortada) |
| **Salida** | Texto alfanumerico (hasta 9 caracteres) |
| **Hardware** | CPU (via ONNX Runtime) |
| **Latencia** | ~3-5 ms en CPU |
| **Caracteres** | 37 clases (A-Z + 0-9 + vacio) |

**Funcion:** Lee los caracteres de la placa recortada. Recibe el crop del paso anterior y devuelve el texto alfanumerico (ej: "ABC123").

**Importante:** Este modelo corre en **CPU, no en Coral Edge TPU**. La arquitectura Transformer que usa para reconocer secuencias de caracteres no es compatible con el Edge TPU. Sin embargo, al ser un modelo extremadamente pequeno (~2 MB), su latencia en CPU es de solo 3-5 ms, por lo que no representa un cuello de botella.

La libreria `fast-plate-ocr` se instala via pip y descarga el modelo automaticamente a cache local en la primera ejecucion. Despues de eso funciona 100% offline.

---

## 4. Resumen de Pesos y Hardware

| # | Modelo | Archivo | Peso | Hardware | Latencia Est. |
|---|--------|---------|------|----------|---------------|
| 1 | Detector vehiculos | `yolo11n_coco_vehicle_int8.tflite` | 2.9 MB | Coral TPU | ~3 ms |
| 2 | Clasificador color | `color_classifier_int8.tflite` | 1.6 MB | Coral TPU | ~2-4 ms |
| 3 | Detector marca | `marca_detector_yolo11n_int8.tflite` | 2.9 MB | Coral TPU | ~3 ms |
| 4 | Detector placa | `placa_detector_yolo11n_int8.tflite` | 2.8 MB | Coral TPU | ~3 ms |
| 5 | OCR placa | CCT-XS (ONNX, via pip) | ~2.0 MB | CPU | ~3-5 ms |
| | **TOTAL** | | **~12.2 MB** | | **~14-18 ms** |

**Peso total en disco:** ~12.2 MB para los 5 modelos combinados.

**Latencia total estimada por frame:** 14-18 ms en Coral Edge TPU + CPU, lo que permite procesar aproximadamente **55-70 frames por segundo (FPS)** en teoria, limitado por la velocidad de captura de la camara.

---

## 5. Cuantizacion INT8 — Por que es necesaria

Todos los modelos que corren en Coral Edge TPU estan cuantizados en formato **INT8** (enteros de 8 bits). Esto significa que:

- Los pesos y activaciones del modelo se representan con numeros enteros de -128 a 127, en lugar de numeros flotantes de 32 bits
- Esto reduce el tamano del modelo en ~4x y acelera la inferencia significativamente
- El Edge TPU de Google **solo acepta modelos INT8** completamente cuantizados
- La cuantizacion se realiza con un dataset de calibracion que asegura que la precision se mantenga

La perdida de precision por cuantizacion es minima (generalmente <1-2% en accuracy) y es el estandar de la industria para inferencia en dispositivos edge.

---

## 6. Hardware Recomendado

### Opcion 1: Coral USB Accelerator + Mini PC

- **Coral USB Accelerator** (~$60 USD) — conecta via USB 3.0
- **Mini PC ARM** (Raspberry Pi 5, NVIDIA Jetson, o similar)
- Los 4 modelos TFLite corren en el Coral, el OCR corre en la CPU del mini PC

### Opcion 2: Coral Dev Board

- **Coral Dev Board** (~$150 USD) — sistema completo con Edge TPU integrado
- Todo-en-uno: CPU ARM + Edge TPU en una sola placa

### Opcion 3: Coral M.2 Accelerator

- **Coral M.2 Module** — se instala directamente en un slot M.2 de un PC industrial
- Ideal para integracion en equipos existentes tipo rack o gabinete

### Requerimientos de software

| Componente | Version |
|------------|---------|
| Python | 3.9+ |
| pycoral | 2.0+ (driver Coral) |
| libedgetpu | 2.0+ (runtime Edge TPU) |
| ONNX Runtime | 1.15+ (para OCR) |
| fast-plate-ocr | 1.0+ (modelo OCR) |
| OpenCV | 4.5+ (procesamiento imagen) |

---

## 7. Estructura de Archivos para Despliegue

Para desplegar el sistema en un dispositivo Coral, se necesitan los siguientes archivos:

```
anpr_coral/
├── models/
│   └── tflite_exports/
│       ├── yolo11n_coco_vehicle_int8.tflite       (2.9 MB)
│       ├── color_classifier_int8.tflite            (1.6 MB)
│       ├── marca_detector_yolo11n_int8.tflite      (2.9 MB)
│       └── placa_detector_yolo11n_int8.tflite      (2.8 MB)
├── scripts/
│   └── coral_simulator.py        (capa de abstraccion Coral/CPU)
├── app_coral_test.py             (aplicacion de prueba Streamlit)
└── requirements.txt
```

El modelo OCR (CCT-XS) se descarga automaticamente via `pip install fast-plate-ocr[onnx]` y se almacena en `~/.cache/fast-plate-ocr/`. No requiere copia manual.

---

## 8. Compilacion para Edge TPU

Los archivos `*_int8.tflite` que tenemos actualmente son modelos INT8 que corren en **CPU como simulacion**. Para que corran directamente en el **hardware Coral Edge TPU**, se deben compilar con el `edgetpu_compiler`:

```bash
edgetpu_compiler -s modelo_int8.tflite
```

Esto genera un archivo `*_int8_edgetpu.tflite` que es el que se carga en el dispositivo Coral. La compilacion se debe hacer una sola vez y requiere un entorno Linux x86_64 (puede ser Docker).

El sistema esta disenado para detectar automaticamente si hay un Coral conectado:
- **Si hay Coral:** usa los archivos `_edgetpu.tflite` con aceleracion hardware
- **Si no hay Coral:** usa los archivos `_int8.tflite` en CPU como fallback (mas lento, pero funcional)

---

## 9. Latencias Medidas y Estimadas

### En CPU (simulacion — Mac M4 / Intel i7)

| Modelo | Latencia CPU | Latencia estimada Coral | Factor de aceleracion |
|--------|-------------|------------------------|----------------------|
| Detector vehiculos | ~30-40 ms | ~3 ms | ~11x |
| Clasificador color | ~12-20 ms | ~2-4 ms | ~6x |
| Detector marca | ~30-40 ms | ~3 ms | ~11x |
| Detector placa | ~30-40 ms | ~3 ms | ~11x |
| OCR (CPU only) | ~3-5 ms | ~3-5 ms | 1x (sin cambio) |
| **Total pipeline** | **~105-145 ms** | **~14-18 ms** | **~7-8x** |

Las latencias de CPU se midieron con el simulador incluido en el sistema. Las latencias de Coral se estiman basandose en benchmarks publicos de hardware similar:
- YOLO11 nano INT8 a 640px: ~3 ms en Coral USB 3.0
- Clasificadores INT8 a 224px: ~2-4 ms en Coral USB 3.0

### Proyeccion de FPS

| Escenario | Latencia por frame | FPS estimados |
|-----------|-------------------|---------------|
| Solo CPU (sin Coral) | ~105-145 ms | ~7-10 FPS |
| Con Coral Edge TPU | ~14-18 ms | ~55-70 FPS |
| Camara tipica 30 FPS | 33 ms budget | Sobra capacidad con Coral |

---

## 10. Limitaciones Conocidas

1. **OCR en CPU:** El modelo de lectura de placa (CCT-XS) no es compatible con Edge TPU por usar arquitectura Transformer. Corre en CPU, pero a solo ~3-5ms no es cuello de botella.

2. **Marca depende de visibilidad del logo:** Si el logo esta tapado, sucio o el angulo no lo muestra, el detector de marca no puede identificarla. Es la deteccion con menor tasa de exito por naturaleza.

3. **Color depende del recorte:** El clasificador de color funciona bien cuando recibe un crop limpio del vehiculo. Si el detector de vehiculos falla, el color se clasifica sobre la imagen completa con menor precision.

4. **Condiciones de iluminacion:** Como todo sistema de vision por computadora, funciona mejor con buena iluminacion. Condiciones nocturnas o contraluz pueden reducir la precision, especialmente del color y la marca.

5. **30 marcas:** El detector de marca cubre 30 marcas comerciales. Vehiculos de marcas no incluidas (ej: marcas chinas emergentes) no seran identificados.

6. **Compilacion Edge TPU:** La compilacion final de los modelos para Coral (`edgetpu_compiler`) requiere un entorno Linux x86_64. Se puede hacer en Docker o en una maquina virtual, pero no directamente en Mac o Windows.
