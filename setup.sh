#!/bin/bash
# ========================================
# SCRIPT DE INSTALACIÓN - SISTEMA ANPR
# ========================================
# Este script configura el entorno completo
# para el sistema de reconocimiento de placas.
#
# Uso: ./setup.sh
# ========================================

set -e  # Salir si hay error

echo "========================================"
echo "  INSTALACIÓN DEL SISTEMA ANPR"
echo "  Reconocimiento de Placas Vehiculares"
echo "========================================"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para mostrar progreso
show_progress() {
    echo -e "${GREEN}[✓]${NC} $1"
}

show_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

show_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Buscar Python 3.11 (requerido para compatibilidad con PyTorch/ML libraries)
echo ""
echo "📦 Buscando Python 3.11..."

PYTHON311=""

# Buscar Python 3.11 en ubicaciones comunes
if command -v python3.11 &> /dev/null; then
    PYTHON311="python3.11"
elif [ -f "/usr/local/opt/python@3.11/bin/python3.11" ]; then
    PYTHON311="/usr/local/opt/python@3.11/bin/python3.11"
elif [ -f "/opt/homebrew/opt/python@3.11/bin/python3.11" ]; then
    PYTHON311="/opt/homebrew/opt/python@3.11/bin/python3.11"
elif [ -f "$HOME/.pyenv/versions/3.11.*/bin/python3.11" ]; then
    PYTHON311=$(ls -1 $HOME/.pyenv/versions/3.11.*/bin/python3.11 2>/dev/null | head -1)
fi

if [ -z "$PYTHON311" ]; then
    show_error "Python 3.11 no encontrado."
    echo ""
    echo "Por favor instala Python 3.11:"
    echo "  brew install python@3.11"
    echo ""
    echo "O con pyenv:"
    echo "  pyenv install 3.11"
    exit 1
fi

PYTHON_VERSION=$($PYTHON311 --version)
show_progress "Python 3.11 encontrado: $PYTHON_VERSION"
show_progress "Ubicación: $PYTHON311"

# Crear entorno virtual si no existe
echo ""
echo "📦 Configurando entorno virtual con Python 3.11..."
if [ ! -d "anpr_env" ]; then
    $PYTHON311 -m venv anpr_env
    show_progress "Entorno virtual creado: anpr_env (Python 3.11)"
else
    show_warning "Entorno virtual ya existe"
fi

# Activar entorno virtual
echo ""
echo "📦 Activando entorno virtual..."
source anpr_env/bin/activate
show_progress "Entorno activado"

# Actualizar pip
echo ""
echo "📦 Actualizando pip..."
pip install --upgrade pip > /dev/null 2>&1
show_progress "pip actualizado"

# Instalar dependencias
echo ""
echo "📦 Instalando dependencias (esto puede tardar varios minutos)..."

# PyTorch (detectar CUDA)
echo "   Instalando PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1
    show_progress "PyTorch instalado (con CUDA)"
else
    pip install torch torchvision > /dev/null 2>&1
    show_progress "PyTorch instalado (CPU)"
fi

# Ultralytics (YOLO)
echo "   Instalando Ultralytics (YOLO)..."
pip install ultralytics > /dev/null 2>&1
show_progress "Ultralytics instalado"

# TensorFlow
echo "   Instalando TensorFlow..."
pip install tensorflow > /dev/null 2>&1
show_progress "TensorFlow instalado"

# OCR con ONNX runtime (cross-platform, funciona en Mac)
echo "   Instalando fast-plate-ocr[onnx]..."
pip install "fast-plate-ocr[onnx]" > /dev/null 2>&1
show_progress "fast-plate-ocr[onnx] instalado"

# OpenCV
echo "   Instalando OpenCV..."
pip install opencv-python > /dev/null 2>&1
show_progress "OpenCV instalado"

# Otras dependencias
echo "   Instalando utilidades..."
pip install pyyaml numpy pillow tqdm onnx onnxruntime > /dev/null 2>&1
show_progress "Utilidades instaladas"

# Crear directorios necesarios
echo ""
echo "📁 Creando estructura de directorios..."
mkdir -p models output logs configs docs
show_progress "Directorios creados"

# Verificar instalación
echo ""
echo "🔍 Verificando instalación..."

python3 -c "from ultralytics import YOLO" 2>/dev/null && show_progress "YOLO: OK" || show_error "YOLO: Error"
python3 -c "import tensorflow" 2>/dev/null && show_progress "TensorFlow: OK" || show_error "TensorFlow: Error"
python3 -c "import cv2" 2>/dev/null && show_progress "OpenCV: OK" || show_error "OpenCV: Error"
python3 -c "from fast_plate_ocr import ONNXPlateRecognizer" 2>/dev/null && show_progress "fast-plate-ocr: OK" || show_warning "fast-plate-ocr: Requiere modelo"

# Resumen
echo ""
echo "========================================"
echo "  ✅ INSTALACIÓN COMPLETADA"
echo "========================================"
echo ""
echo "Próximos pasos:"
echo ""
echo "1. Activar el entorno virtual:"
echo "   source anpr_env/bin/activate"
echo ""
echo "2. Preparar el dataset:"
echo "   python scripts/01_preparar_dataset.py"
echo ""
echo "3. Entrenar el modelo:"
echo "   python scripts/02_entrenar_modelo.py"
echo ""
echo "4. Exportar a TFLite:"
echo "   python scripts/03_exportar_tflite.py"
echo ""
echo "5. Ejecutar inferencia en tiempo real:"
echo "   python scripts/04_inferencia_tiempo_real.py --source 0"
echo ""
echo "========================================"
echo "  📚 Ver README.md para más detalles"
echo "========================================"
