#!/usr/bin/env bash
# ==============================================================================
# SCRIPT 07: COMPILACIÓN EDGE TPU PARA CORAL
# ==============================================================================
# Compila todos los modelos TFLite INT8 a formato Edge TPU usando Docker.
# No requiere instalación del compilador Edge TPU — solo Docker.
#
# Uso:
#   bash scripts/07_compilar_edgetpu.sh
#   bash scripts/07_compilar_edgetpu.sh --dry-run   # solo muestra comandos
#
# Pre-requisito: Docker corriendo
#   docker info >/dev/null 2>&1 || open -a Docker
#
# Salida:
#   models/tflite_exports/*_edgetpu.tflite
#   models/tflite_exports/*_edgetpu.log    (reporte del compilador)
#
# Referencia:
#   https://coral.ai/docs/edgetpu/compiler/
#   Imagen Docker: gcr.io/edgetpu-compiler/compiler:release
# ==============================================================================

set -euo pipefail

# ── Colores ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Rutas ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXPORTS_DIR="$PROJECT_DIR/models/tflite_exports"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ── Cabecera ──────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  COMPILACIÓN EDGE TPU — Google Coral${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
echo -e "  Directorio de modelos: ${CYAN}$EXPORTS_DIR${NC}"
echo -e "  Docker image: ${CYAN}gcr.io/edgetpu-compiler/compiler:release${NC}"
echo ""

# ── Verificar Docker ──────────────────────────────────────────────────────────
echo -e "${CYAN}🔍 Verificando Docker...${NC}"
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker no está corriendo.${NC}"
    echo -e "   Inicia Docker Desktop y vuelve a intentar:"
    echo -e "   ${YELLOW}open -a 'Docker Desktop'${NC}   (macOS)"
    exit 1
fi
echo -e "${GREEN}   ✅ Docker OK${NC}"

# ── Verificar que existan modelos INT8 ───────────────────────────────────────
echo -e "\n${CYAN}📂 Buscando modelos INT8 en tflite_exports/...${NC}"
INT8_MODELS=()
while IFS= read -r f; do
    # Excluir ya compilados (_edgetpu.tflite)
    [[ "$f" == *_edgetpu.tflite ]] && continue
    INT8_MODELS+=("$f")
done < <(find "$EXPORTS_DIR" -name "*int8*.tflite" 2>/dev/null | sort)

if [[ ${#INT8_MODELS[@]} -eq 0 ]]; then
    echo -e "${RED}❌ No se encontraron modelos INT8.${NC}"
    echo -e "   Ejecuta primero:"
    echo -e "   ${YELLOW}python scripts/07_exportar_marca_int8.py${NC}"
    exit 1
fi

for m in "${INT8_MODELS[@]}"; do
    echo -e "   • $(basename "$m")"
done

$DRY_RUN && echo -e "\n${YELLOW}⚠️  DRY RUN — no se ejecutará nada realmente${NC}"

# ── Pull de la imagen (solo si no existe) ─────────────────────────────────────
# gcr.io/edgetpu-compiler is deprecated (auth-gated). Use Debian + coral apt instead.
IMAGE="debian:bookworm-slim"
COMPILER_SETUP="apt-get update -qq && \
  apt-get install -y -q curl gnupg && \
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu-archive-keyring.gpg && \
  echo 'deb [signed-by=/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main' > /etc/apt/sources.list.d/coral-edgetpu.list && \
  apt-get update -qq && \
  apt-get install -y -q edgetpu-compiler"

echo -e "\n${CYAN}🐳 Preparando compilador Edge TPU (debian:bookworm amd64 + coral apt)...${NC}"
echo -e "   ${YELLOW}Nota: edgetpu-compiler es x86_64 — usando emulación QEMU en Apple Silicon${NC}"
if $DRY_RUN; then
    echo -e "   [dry-run] docker run --platform linux/amd64 --rm -v ... debian:bookworm-slim ..."
else
    echo -e "   Verificando imagen base $IMAGE (amd64)..."
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        echo -e "   Descargando imagen base amd64..."
        docker pull --platform linux/amd64 "$IMAGE"
    fi
    echo -e "${GREEN}   ✅ Imagen base disponible${NC}"
fi

# ── Compilar cada modelo ──────────────────────────────────────────────────────
echo -e "\n${BOLD}============================================================${NC}"
echo -e "${BOLD}  COMPILANDO MODELOS${NC}"
echo -e "${BOLD}============================================================${NC}"

COMPILED=()
FAILED=()

for MODEL_PATH in "${INT8_MODELS[@]}"; do
    FILENAME=$(basename "$MODEL_PATH")
    STEM="${FILENAME%.tflite}"
    LOG_FILE="$EXPORTS_DIR/${STEM}_edgetpu_compile.log"

    echo ""
    echo -e "${CYAN}► $FILENAME${NC}"

    CMD=(
        docker run --rm
        --platform linux/amd64
        -v "$EXPORTS_DIR:/models"
        "$IMAGE"
        bash -c "$COMPILER_SETUP && edgetpu_compiler --num_segments 1 --out_dir /models /models/$FILENAME"
    )

    if $DRY_RUN; then
        echo -e "   ${YELLOW}[dry-run]${NC} ${CMD[*]}"
        continue
    fi

    # Ejecutar compilador
    if "${CMD[@]}" 2>&1 | tee "$LOG_FILE"; then
        EXPECTED_OUTPUT="$EXPORTS_DIR/${STEM}_edgetpu.tflite"
        if [[ -f "$EXPECTED_OUTPUT" ]]; then
            SIZE=$(du -sh "$EXPECTED_OUTPUT" | cut -f1)
            echo -e "${GREEN}   ✅ Compilado: ${STEM}_edgetpu.tflite ($SIZE)${NC}"
            COMPILED+=("$EXPECTED_OUTPUT")

            # Leer resumen del log
            if grep -q "Operator" "$LOG_FILE" 2>/dev/null; then
                OPS_TPU=$(grep -oP '\d+(?= Operations mapped)' "$LOG_FILE" 2>/dev/null || echo "?")
                OPS_CPU=$(grep -oP '\d+(?= Operations not mapped)' "$LOG_FILE" 2>/dev/null || echo "?")
                echo -e "   📊 Operaciones en TPU: ${BOLD}${OPS_TPU}${NC}  |  en CPU: ${OPS_CPU}"
            fi
        else
            echo -e "${RED}   ❌ Compilación completó pero no se generó _edgetpu.tflite${NC}"
            echo -e "   Revisa el log: $LOG_FILE"
            FAILED+=("$FILENAME")
        fi
    else
        echo -e "${RED}   ❌ Error en compilación. Revisa: $LOG_FILE${NC}"
        FAILED+=("$FILENAME")
    fi
done

# ── Resumen final ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}  RESUMEN${NC}"
echo -e "${BOLD}============================================================${NC}"

if [[ ${#COMPILED[@]} -gt 0 ]]; then
    echo -e "\n${GREEN}✅ Compilados exitosamente:${NC}"
    for f in "${COMPILED[@]}"; do
        SIZE=$(du -sh "$f" | cut -f1)
        echo -e "   • $(basename "$f")  ($SIZE)"
    done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo -e "\n${RED}❌ Fallidos:${NC}"
    for f in "${FAILED[@]}"; do
        echo -e "   • $f"
    done
    echo -e "\n${YELLOW}Tip: Si el modelo tiene operaciones no soportadas, el compilador${NC}"
    echo -e "${YELLOW}     moverá esas capas a CPU pero igual generará el archivo.${NC}"
fi

echo ""
echo -e "${CYAN}📍 Siguiente paso:${NC}"
echo -e "   python scripts/08_inferencia_coral.py --imagen samples/<foto.jpg> --simulate"
echo ""

# Nota sobre MPACT-CoralNPU simulator
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}ℹ️  NOTA SOBRE SIMULADORES DE INSTRUCCIONES CORAL:${NC}"
echo -e "   Para simulación a nivel de instrucciones RISC-V del Coral NPU:"
echo -e "   • MPACT-CoralNPU (behavioral): https://github.com/google-coral/coralnpu-mpact"
echo -e "   • Verilator cycle-accurate:    https://github.com/google-coral/coralnpu"
echo -e "   Estos simuladores aceptan .elf/.bin (NO TFLite directamente)."
echo -e "   Para desarrollo Python/ML, usa coral_simulator.py (TFLite CPU fallback)."
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
