#!/usr/bin/env bash
# ============================================================================
# Compilar modelos YOLO11 para Google Coral Edge TPU (macOS via Docker)
# ============================================================================
#
# Soluciona dos problemas conocidos:
#   1. "dynamic-sized tensors" → fijamos el batch a 1 con concrete_functions
#   2. "CONV_2D version 6"    → usamos TF 2.13 dentro de Docker
#
# Modelos compilados (4 de 5 — OCR queda en CPU):
#   - yolo11n_coco_vehicle_int8_edgetpu.tflite   (640px, 5 clases COCO)
#   - color_classifier_int8_edgetpu.tflite        (224px, 15 colores)
#   - marca_detector_yolo11n_int8_edgetpu.tflite  (640px, 30 marcas)
#   - placa_detector_yolo11n_int8_edgetpu.tflite  (640px, 1 clase)
#
# Uso:
#     bash scripts/compile_edgetpu.sh
#
# Salida:
#     models/compiled_edgetpu/
# ============================================================================
set -e

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_DIR="${PROJECT_DIR}/models/compiled_edgetpu"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  CORAL EDGE TPU — Conversión + Compilación (Docker amd64)  ${NC}"
echo -e "${BOLD}============================================================${NC}"

if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker no está corriendo.${NC}"; exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ─── Dockerfile: TF 2.13 + edgetpu-compiler ─────────────────────────────────
DOCKERFILE="/tmp/Dockerfile.edgetpu_compile"
cat << 'DOCKEREOF' > "$DOCKERFILE"
FROM --platform=linux/amd64 python:3.10-slim-bullseye
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y curl gnupg ca-certificates && \
    curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
      | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    apt-get update && \
    apt-get install -y edgetpu-compiler && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "tensorflow==2.13.1" "numpy<2" "Pillow"
DOCKEREOF

echo -e "\n${YELLOW}📦 Construyendo imagen Docker (TF 2.13 + edgetpu-compiler)...${NC}"
docker build --platform linux/amd64 \
    -t anpr-edgetpu-compiler \
    -f "$DOCKERFILE" /tmp/ 2>&1 | tail -5

# ─── Script Python de conversión (se inyecta al container) ───────────────────
PYSCRIPT="/tmp/edgetpu_convert.py"
cat << 'PYEOF' > "$PYSCRIPT"
#!/usr/bin/env python3
"""
Convierte SavedModel → TFLite INT8 full-integer (TF 2.13) → edgetpu_compiler.

Resuelve:
  - Dynamic tensors: carga el saved_model, crea concrete_function con shape fijo
  - INT8 puro: representative_dataset + inference_input/output_type = uint8
  - CONV_2D v5: TF 2.13 genera ops compatibles con edgetpu-compiler v16
"""
import os, sys, glob, subprocess
import numpy as np
import tensorflow as tf

print(f"TensorFlow {tf.__version__}")

# ── Configuración de modelos ─────────────────────────────────────────────────
MODELS = [
    {
        "saved_model": "/project/scripts/yolo11n_saved_model",
        "name": "yolo11n_coco_vehicle_int8",
        "imgsz": 640,
        "calib_dir": None,  # sin dataset propio, usar sintético
    },
    {
        "saved_model": "/project/models/color_classifier_yolo11n_saved_model",
        "name": "color_classifier_int8",
        "imgsz": 224,
        "calib_dir": "/project/datasets/vehicle_colors",
    },
    {
        "saved_model": "/project/models/marca_detector_yolo11n_saved_model",
        "name": "marca_detector_yolo11n_int8",
        "imgsz": 640,
        "calib_dir": None,
    },
    {
        "saved_model": "/project/models/placa_detector_yolo11n_saved_model",
        "name": "placa_detector_yolo11n_int8",
        "imgsz": 640,
        "calib_dir": "/project/datasets/dataset_combinado/train/images",
    },
]

OUTPUT = "/output"
os.makedirs(OUTPUT, exist_ok=True)


def make_representative_dataset(calib_dir, imgsz, n=200):
    """
    Genera datos de calibración para cuantización INT8.
    Usa imágenes reales si hay, sino genera sintéticas (ruido uniforme).
    """
    from PIL import Image

    images = []
    if calib_dir and os.path.isdir(calib_dir):
        for ext in ("jpg", "jpeg", "png"):
            images.extend(glob.glob(os.path.join(calib_dir, "**", f"*.{ext}"), recursive=True))
    images = images[:n]

    if images:
        print(f"    Calibración: {len(images)} imágenes reales de {calib_dir}")
    else:
        print(f"    Calibración: {n} muestras sintéticas (sin dataset disponible)")

    def gen():
        if images:
            for p in images:
                try:
                    img = Image.open(p).convert("RGB").resize((imgsz, imgsz))
                    arr = np.array(img, dtype=np.float32) / 255.0
                    yield [arr[np.newaxis, ...]]
                except Exception:
                    continue
        else:
            for _ in range(n):
                yield [np.random.uniform(0, 1, (1, imgsz, imgsz, 3)).astype(np.float32)]
    return gen


def convert_and_compile(cfg):
    """
    1. Carga SavedModel
    2. Crea concrete function con input shape fijo [1, H, W, 3]
    3. Convierte a TFLite INT8 full-integer (uint8 I/O)
    4. Compila para Edge TPU
    """
    sm_dir = cfg["saved_model"]
    name   = cfg["name"]
    imgsz  = cfg["imgsz"]
    calib  = cfg["calib_dir"]

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  SavedModel: {sm_dir}")
    print(f"  Input: [1, {imgsz}, {imgsz}, 3]")
    print(f"{'='*60}")

    if not os.path.isdir(sm_dir):
        print(f"  ❌ SavedModel no encontrado: {sm_dir}")
        return False

    tflite_path = os.path.join(OUTPUT, f"{name}.tflite")

    # ── Paso 1: Cargar SavedModel y fijar dimensiones ────────────────────────
    print(f"\n  [1/3] Cargando SavedModel y fijando shape a [1,{imgsz},{imgsz},3]...")

    loaded = tf.saved_model.load(sm_dir)
    infer = loaded.signatures["serving_default"]

    # Identificar el nombre del input tensor
    input_keys = list(infer.structured_input_signature[1].keys())
    input_name = input_keys[0]
    print(f"    Input key: '{input_name}'")
    print(f"    Original spec: {infer.structured_input_signature[1][input_name]}")

    # Crear concrete function con shape estático
    fixed_shape = [1, imgsz, imgsz, 3]
    fixed_spec = tf.TensorSpec(shape=fixed_shape, dtype=tf.float32, name=input_name)

    @tf.function(input_signature=[fixed_spec])
    def fixed_model(x):
        return infer(**{input_name: x})

    concrete_func = fixed_model.get_concrete_function()
    print(f"    ✅ Concrete function creada con shape fijo {fixed_shape}")

    # ── Paso 2: Convertir a TFLite INT8 full-integer ────────────────────────
    print(f"\n  [2/3] Convirtiendo a TFLite INT8 (full integer, uint8 I/O)...")

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_func], loaded
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(calib, imgsz)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"  ❌ Conversión fallida: {e}")
        return False

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"    ✅ {tflite_path} ({size_mb:.2f} MB)")

    # Verificar dtypes
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    inp_dtype = inp_det["dtype"].__name__
    out_dtype = out_det["dtype"].__name__
    inp_shape = inp_det["shape"].tolist()
    out_shape = out_det["shape"].tolist()
    print(f"    📥 Input:  {inp_dtype} {inp_shape}")
    print(f"    📤 Output: {out_dtype} {out_shape}")

    is_full_int8 = "uint8" in inp_dtype and "uint8" in out_dtype
    if is_full_int8:
        print(f"    ✅ Full INT8 (uint8 I/O) — listo para Edge TPU")
    else:
        print(f"    ⚠️  I/O no es uint8 — puede tener ops en CPU")

    # ── Paso 3: Compilar para Edge TPU ───────────────────────────────────────
    print(f"\n  [3/3] Compilando con edgetpu_compiler...")

    result = subprocess.run(
        ["edgetpu_compiler", "-s", "-o", OUTPUT, tflite_path],
        capture_output=True, text=True, timeout=300,
    )

    stdout = result.stdout + result.stderr
    print(stdout)

    edgetpu_file = tflite_path.replace(".tflite", "_edgetpu.tflite")
    if os.path.exists(edgetpu_file):
        edgetpu_size = os.path.getsize(edgetpu_file) / (1024 * 1024)
        print(f"    ✅ {os.path.basename(edgetpu_file)} ({edgetpu_size:.2f} MB)")

        # Parsear % mapeado al TPU del log del compilador
        for line in stdout.splitlines():
            if "mapped" in line.lower() or "on-chip" in line.lower() or "operations" in line.lower():
                print(f"    📊 {line.strip()}")
        return True
    else:
        print(f"    ❌ No se generó {edgetpu_file}")
        log_path = os.path.join(OUTPUT, f"{name}_compile.log")
        with open(log_path, "w") as f:
            f.write(stdout)
        print(f"    Log guardado en {log_path}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────
results = {}
for cfg in MODELS:
    ok = convert_and_compile(cfg)
    results[cfg["name"]] = ok

print(f"\n{'='*60}")
print(f"  RESUMEN FINAL")
print(f"{'='*60}")
all_ok = True
for name, ok in results.items():
    status = "✅ EDGE TPU" if ok else "❌ FALLÓ"
    print(f"  {status}  {name}")
    if not ok:
        all_ok = False

# Listar archivos generados
print(f"\n  Archivos en /output/:")
for f in sorted(os.listdir(OUTPUT)):
    fpath = os.path.join(OUTPUT, f)
    size = os.path.getsize(fpath) / (1024 * 1024)
    marker = "🪸" if "_edgetpu" in f else "  "
    print(f"  {marker} {f:50s} {size:.2f} MB")

if all_ok:
    print(f"\n  ✅ 4/4 modelos compilados para Edge TPU")
else:
    failed = sum(1 for v in results.values() if not v)
    print(f"\n  ⚠️  {failed} modelo(s) fallaron")
    sys.exit(1)
PYEOF

# ─── Ejecutar ────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}🔧 Modelos a compilar:${NC}"
echo -e "   ${GREEN}▶${NC} yolo11n_coco_vehicle_int8    (640px) ← scripts/yolo11n_saved_model"
echo -e "   ${GREEN}▶${NC} color_classifier_int8         (224px) ← models/color_classifier_yolo11n_saved_model"
echo -e "   ${GREEN}▶${NC} marca_detector_yolo11n_int8   (640px) ← models/marca_detector_yolo11n_saved_model"
echo -e "   ${GREEN}▶${NC} placa_detector_yolo11n_int8   (640px) ← models/placa_detector_yolo11n_saved_model"
echo -e "   ${CYAN}⏭${NC}  OCR (CCT-XS)                         → CPU only, no se compila\n"

echo -e "${YELLOW}🚀 Ejecutando conversión + compilación en Docker...${NC}"
echo -e "${CYAN}   (TF 2.13 + edgetpu-compiler v16, amd64 via Rosetta)${NC}\n"

docker run --rm --platform linux/amd64 \
    -v "${PROJECT_DIR}:/project:ro" \
    -v "${OUTPUT_DIR}:/output" \
    -v "${PYSCRIPT}:/opt/convert.py:ro" \
    anpr-edgetpu-compiler \
    python3 /opt/convert.py

# ─── Resultado ────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}============================================================${NC}"
echo -e "${BOLD}  ARCHIVOS EN models/compiled_edgetpu/${NC}"
echo -e "${BOLD}============================================================${NC}"
ls -lh "$OUTPUT_DIR"

EDGETPU_COUNT=$(ls "$OUTPUT_DIR"/*_edgetpu.tflite 2>/dev/null | wc -l | tr -d ' ')
echo -e "\n${GREEN}✅ ${EDGETPU_COUNT}/4 modelo(s) _edgetpu.tflite listos para Coral${NC}"
