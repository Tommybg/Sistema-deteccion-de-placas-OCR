@echo off
REM ========================================
REM SCRIPT DE INSTALACIÓN - SISTEMA ANPR
REM Para Windows
REM ========================================

echo ========================================
echo   INSTALACIÓN DEL SISTEMA ANPR
echo   Reconocimiento de Placas Vehiculares
echo ========================================
echo.

REM Verificar Python
echo [*] Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python no encontrado. Por favor instala Python 3.8+
    pause
    exit /b 1
)
echo [OK] Python encontrado

REM Crear entorno virtual
echo.
echo [*] Creando entorno virtual...
if not exist "anpr_env" (
    python -m venv anpr_env
    echo [OK] Entorno virtual creado
) else (
    echo [!] Entorno virtual ya existe
)

REM Activar entorno virtual
echo.
echo [*] Activando entorno virtual...
call anpr_env\Scripts\activate.bat

REM Actualizar pip
echo.
echo [*] Actualizando pip...
pip install --upgrade pip >nul 2>&1
echo [OK] pip actualizado

REM Instalar dependencias
echo.
echo [*] Instalando dependencias (esto puede tardar varios minutos)...

echo    Instalando PyTorch...
pip install torch torchvision >nul 2>&1
echo [OK] PyTorch instalado

echo    Instalando Ultralytics (YOLO)...
pip install ultralytics >nul 2>&1
echo [OK] Ultralytics instalado

echo    Instalando TensorFlow...
pip install tensorflow >nul 2>&1
echo [OK] TensorFlow instalado

echo    Instalando fast-plate-ocr...
pip install fast-plate-ocr >nul 2>&1
echo [OK] fast-plate-ocr instalado

echo    Instalando OpenCV...
pip install opencv-python >nul 2>&1
echo [OK] OpenCV instalado

echo    Instalando utilidades...
pip install pyyaml numpy pillow tqdm onnx onnxruntime >nul 2>&1
echo [OK] Utilidades instaladas

REM Crear directorios
echo.
echo [*] Creando estructura de directorios...
if not exist "models" mkdir models
if not exist "output" mkdir output
if not exist "logs" mkdir logs
if not exist "configs" mkdir configs
if not exist "docs" mkdir docs
echo [OK] Directorios creados

REM Resumen
echo.
echo ========================================
echo   INSTALACIÓN COMPLETADA
echo ========================================
echo.
echo Proximos pasos:
echo.
echo 1. Activar el entorno virtual:
echo    anpr_env\Scripts\activate
echo.
echo 2. Preparar el dataset:
echo    python scripts\01_preparar_dataset.py
echo.
echo 3. Entrenar el modelo:
echo    python scripts\02_entrenar_modelo.py
echo.
echo 4. Exportar a TFLite:
echo    python scripts\03_exportar_tflite.py
echo.
echo 5. Ejecutar inferencia en tiempo real:
echo    python scripts\04_inferencia_tiempo_real.py --source 0
echo.
echo ========================================
echo   Ver README.md para mas detalles
echo ========================================
echo.
pause
