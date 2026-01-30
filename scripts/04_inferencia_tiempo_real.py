#!/usr/bin/env python3
"""
================================================================================
SCRIPT 04: INFERENCIA EN TIEMPO REAL CON DETECCIÓN Y OCR
================================================================================
Este script realiza la detección de placas vehiculares en tiempo real usando:
- YOLOv11 (o TFLite) para detección de placas
- fast-plate-ocr para reconocimiento de caracteres

Características:
- Soporte para cámaras USB, IP (RTSP), y archivos de video
- Visualización en tiempo real con FPS
- Logging de placas detectadas
- Soporte para modelos TFLite (edge devices)

Autor: Sistema ANPR Colombia
Fecha: 2025
================================================================================
"""

import os
import sys
from pathlib import Path
import argparse
import time
from datetime import datetime
from collections import deque
import json
import csv

import cv2
import numpy as np

# Configuración de rutas
PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "output"
LOGS_DIR = PROJECT_DIR / "logs"

# Colores para visualización (BGR)
COLOR_BBOX = (0, 255, 0)      # Verde para bounding box
COLOR_TEXT_BG = (0, 0, 0)      # Negro para fondo de texto
COLOR_TEXT = (255, 255, 255)   # Blanco para texto
COLOR_OCR = (0, 255, 255)      # Amarillo para texto OCR


class PlateDetector:
    """
    Detector de placas usando YOLOv11 o TFLite.
    """

    def __init__(self, model_path: str, use_tflite: bool = False,
                 confidence: float = 0.5, device: str = "cpu"):
        """
        Inicializa el detector.

        Args:
            model_path: Ruta al modelo (.pt o .tflite)
            use_tflite: Usar TFLite en lugar de YOLO
            confidence: Umbral de confianza
            device: Dispositivo (cpu, cuda, etc.)
        """
        self.confidence = confidence
        self.use_tflite = use_tflite
        self.device = device

        if use_tflite:
            self._init_tflite(model_path)
        else:
            self._init_yolo(model_path)

        print(f"✅ Detector inicializado: {model_path}")

    def _init_yolo(self, model_path: str):
        """Inicializa el detector YOLO."""
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def _init_tflite(self, model_path: str):
        """Inicializa el detector TFLite."""
        import tensorflow as tf

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

    def detect(self, frame: np.ndarray) -> list:
        """
        Detecta placas en un frame.

        Args:
            frame: Imagen BGR de OpenCV

        Returns:
            Lista de detecciones: [(x1, y1, x2, y2, confidence), ...]
        """
        if self.use_tflite:
            return self._detect_tflite(frame)
        else:
            return self._detect_yolo(frame)

    def _detect_yolo(self, frame: np.ndarray) -> list:
        """Detección usando YOLO."""
        results = self.model(frame, conf=self.confidence, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))

        return detections

    def _detect_tflite(self, frame: np.ndarray) -> list:
        """Detección usando TFLite."""
        # Preprocesar imagen
        input_size = (self.input_shape[1], self.input_shape[2])
        img_resized = cv2.resize(frame, input_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        # Inferencia
        self.interpreter.set_tensor(self.input_details[0]['index'], img_batch)
        self.interpreter.invoke()

        # Obtener salida
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Procesar detecciones (formato YOLO)
        detections = []
        h, w = frame.shape[:2]

        # El formato de salida depende de la versión de YOLO
        # Ajustar según sea necesario
        for detection in output[0]:
            if len(detection) >= 5:
                x_center, y_center, width, height = detection[:4]
                conf = detection[4] if len(detection) > 4 else 0.5

                if conf > self.confidence:
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)
                    detections.append((x1, y1, x2, y2, float(conf)))

        return detections


class PlateOCR:
    """
    Reconocimiento de caracteres usando fast-plate-ocr.
    """

    def __init__(self, model_name: str = "argentinian-plates-cnn-model"):
        """
        Inicializa el OCR.

        Args:
            model_name: Nombre del modelo pre-entrenado
        """
        try:
            from fast_plate_ocr import ONNXPlateRecognizer

            self.recognizer = ONNXPlateRecognizer(model_name)
            print(f"✅ OCR inicializado: {model_name}")
        except ImportError:
            print("⚠️  fast-plate-ocr no instalado. Instalando...")
            os.system("pip install fast-plate-ocr")
            from fast_plate_ocr import ONNXPlateRecognizer
            self.recognizer = ONNXPlateRecognizer(model_name)
        except Exception as e:
            print(f"❌ Error inicializando OCR: {e}")
            self.recognizer = None

    def recognize(self, plate_img: np.ndarray) -> tuple:
        """
        Reconoce el texto de una placa.

        Args:
            plate_img: Imagen de la placa (BGR)

        Returns:
            Tupla (texto, confianza)
        """
        if self.recognizer is None:
            return "", 0.0

        try:
            # Convertir a RGB si es necesario
            if len(plate_img.shape) == 3:
                plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
            else:
                plate_rgb = plate_img

            # Reconocer
            result = self.recognizer.run(plate_rgb)

            if result:
                # El resultado puede ser string o lista según la versión
                if isinstance(result, str):
                    return result, 0.9
                elif isinstance(result, (list, tuple)):
                    return result[0], 0.9
            return "", 0.0

        except Exception as e:
            print(f"⚠️  Error en OCR: {e}")
            return "", 0.0


class ANPRSystem:
    """
    Sistema completo de ANPR (Automatic Number Plate Recognition).
    """

    def __init__(self, detector: PlateDetector, ocr: PlateOCR,
                 log_file: Path = None, min_plate_size: int = 50):
        """
        Inicializa el sistema ANPR.

        Args:
            detector: Instancia de PlateDetector
            ocr: Instancia de PlateOCR
            log_file: Archivo para logging de placas
            min_plate_size: Tamaño mínimo de placa para OCR
        """
        self.detector = detector
        self.ocr = ocr
        self.log_file = log_file
        self.min_plate_size = min_plate_size

        # Historial de placas (para evitar duplicados)
        self.plate_history = deque(maxlen=100)

        # Estadísticas
        self.stats = {
            "frames_processed": 0,
            "plates_detected": 0,
            "plates_recognized": 0,
            "start_time": time.time()
        }

        # Inicializar archivo de log
        if log_file:
            self._init_log_file()

    def _init_log_file(self):
        """Inicializa el archivo de log."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "plate_text", "confidence", "x1", "y1", "x2", "y2"])

    def _log_plate(self, plate_text: str, conf: float, bbox: tuple):
        """Registra una placa detectada."""
        if self.log_file is None:
            return

        timestamp = datetime.now().isoformat()
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, plate_text, f"{conf:.2f}", *bbox])

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Procesa un frame y retorna las placas detectadas.

        Args:
            frame: Imagen BGR de OpenCV

        Returns:
            Tupla (frame_anotado, lista_de_resultados)
        """
        self.stats["frames_processed"] += 1

        # Detectar placas
        detections = self.detector.detect(frame)

        results = []
        frame_annotated = frame.copy()

        for (x1, y1, x2, y2, det_conf) in detections:
            self.stats["plates_detected"] += 1

            # Extraer región de la placa
            plate_width = x2 - x1
            plate_height = y2 - y1

            # Verificar tamaño mínimo
            if plate_width < self.min_plate_size or plate_height < self.min_plate_size // 3:
                continue

            # Extraer imagen de la placa con un pequeño margen
            margin = 5
            y1_crop = max(0, y1 - margin)
            y2_crop = min(frame.shape[0], y2 + margin)
            x1_crop = max(0, x1 - margin)
            x2_crop = min(frame.shape[1], x2 + margin)

            plate_img = frame[y1_crop:y2_crop, x1_crop:x2_crop]

            # Reconocer texto
            plate_text, ocr_conf = self.ocr.recognize(plate_img)

            if plate_text:
                self.stats["plates_recognized"] += 1

                # Evitar duplicados recientes
                if plate_text not in self.plate_history:
                    self.plate_history.append(plate_text)
                    self._log_plate(plate_text, det_conf, (x1, y1, x2, y2))
                    print(f"🚗 Placa detectada: {plate_text} (conf: {det_conf:.2f})")

            results.append({
                "bbox": (x1, y1, x2, y2),
                "det_confidence": det_conf,
                "text": plate_text,
                "ocr_confidence": ocr_conf
            })

            # Anotar frame
            frame_annotated = self._annotate_frame(
                frame_annotated, x1, y1, x2, y2, det_conf, plate_text
            )

        return frame_annotated, results

    def _annotate_frame(self, frame: np.ndarray, x1: int, y1: int,
                        x2: int, y2: int, conf: float, text: str) -> np.ndarray:
        """Anota el frame con la detección."""
        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX, 2)

        # Etiqueta de confianza
        label = f"Placa: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), COLOR_TEXT_BG, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

        # Texto OCR
        if text:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y2), (x1 + tw + 10, y2 + th + 10), COLOR_TEXT_BG, -1)
            cv2.putText(frame, text, (x1 + 5, y2 + th + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_OCR, 2)

        return frame

    def get_stats(self) -> dict:
        """Retorna estadísticas de procesamiento."""
        elapsed = time.time() - self.stats["start_time"]
        fps = self.stats["frames_processed"] / elapsed if elapsed > 0 else 0

        return {
            **self.stats,
            "elapsed_time": elapsed,
            "fps": fps
        }


def abrir_fuente_video(source):
    """
    Abre una fuente de video.

    Args:
        source: Puede ser:
            - int: índice de cámara (0, 1, ...)
            - str: ruta a archivo de video
            - str: URL RTSP para cámara IP

    Returns:
        cv2.VideoCapture
    """
    if isinstance(source, int) or source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir la fuente de video: {source}")

    return cap


def run_realtime(source, detector: PlateDetector, ocr: PlateOCR,
                 show_video: bool = True, save_video: str = None,
                 log_file: Path = None):
    """
    Ejecuta el sistema ANPR en tiempo real.

    Args:
        source: Fuente de video
        detector: Detector de placas
        ocr: OCR de placas
        show_video: Mostrar ventana de video
        save_video: Ruta para guardar video (opcional)
        log_file: Archivo para logging
    """
    print("\n" + "=" * 70)
    print("   INICIANDO INFERENCIA EN TIEMPO REAL")
    print("=" * 70)

    # Abrir fuente de video
    cap = abrir_fuente_video(source)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    print(f"\n📹 Fuente: {source}")
    print(f"   • Resolución: {width}x{height}")
    print(f"   • FPS: {fps}")

    # Inicializar sistema ANPR
    anpr = ANPRSystem(detector, ocr, log_file)

    # Inicializar grabador de video (opcional)
    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(save_video, fourcc, fps, (width, height))
        print(f"   • Guardando video en: {save_video}")

    print("\n🚀 Procesando... (Presiona 'q' para salir)")
    print("-" * 70)

    # Variables para FPS
    fps_counter = deque(maxlen=30)
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n⚠️  Fin del video o error de lectura")
                break

            # Procesar frame
            frame_annotated, results = anpr.process_frame(frame)

            # Calcular FPS
            current_time = time.time()
            fps_counter.append(1.0 / (current_time - prev_time + 1e-6))
            prev_time = current_time
            current_fps = np.mean(fps_counter)

            # Mostrar FPS en el frame
            fps_text = f"FPS: {current_fps:.1f}"
            cv2.putText(frame_annotated, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Guardar video
            if video_writer:
                video_writer.write(frame_annotated)

            # Mostrar video
            if show_video:
                cv2.imshow("ANPR - Detección de Placas", frame_annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n⏹️  Detenido por el usuario")
                    break
                elif key == ord("s"):
                    # Guardar captura
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = OUTPUT_DIR / f"captura_{timestamp}.jpg"
                    cv2.imwrite(str(screenshot_path), frame_annotated)
                    print(f"📷 Captura guardada: {screenshot_path}")

    finally:
        # Limpiar recursos
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

    # Mostrar estadísticas finales
    stats = anpr.get_stats()
    print("\n" + "=" * 70)
    print("   ESTADÍSTICAS FINALES")
    print("=" * 70)
    print(f"\n📊 Resumen:")
    print(f"   • Frames procesados: {stats['frames_processed']}")
    print(f"   • Placas detectadas: {stats['plates_detected']}")
    print(f"   • Placas reconocidas: {stats['plates_recognized']}")
    print(f"   • Tiempo total: {stats['elapsed_time']:.2f}s")
    print(f"   • FPS promedio: {stats['fps']:.1f}")

    if log_file:
        print(f"\n📁 Log guardado en: {log_file}")


def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Inferencia en tiempo real para detección de placas"
    )

    parser.add_argument(
        "--source", type=str, default="0",
        help="Fuente de video: 0 (webcam), ruta a video, o URL RTSP"
    )
    parser.add_argument(
        "--modelo", type=str, default=None,
        help="Ruta al modelo de detección (.pt o .tflite)"
    )
    parser.add_argument(
        "--tflite", action="store_true",
        help="Usar modelo TFLite"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Umbral de confianza para detección (default: 0.5)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Dispositivo: cpu o cuda (default: cpu)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="No mostrar ventana de video"
    )
    parser.add_argument(
        "--save-video", type=str, default=None,
        help="Ruta para guardar video procesado"
    )
    parser.add_argument(
        "--log", action="store_true",
        help="Guardar log de placas detectadas"
    )
    parser.add_argument(
        "--ocr-model", type=str, default="argentinian-plates-cnn-model",
        help="Modelo OCR a usar (default: argentinian-plates-cnn-model)"
    )

    return parser.parse_args()


def encontrar_modelo(use_tflite: bool = False) -> Path:
    """Encuentra el mejor modelo disponible."""
    if use_tflite:
        # Buscar modelo TFLite
        tflite_dir = MODELS_DIR / "tflite_exports"
        if tflite_dir.exists():
            modelos = list(tflite_dir.glob("*.tflite"))
            if modelos:
                return modelos[0]

        modelos = list(MODELS_DIR.rglob("*.tflite"))
        if modelos:
            return modelos[0]
    else:
        # Buscar modelo PyTorch
        modelo_default = MODELS_DIR / "placa_detector_yolo11n.pt"
        if modelo_default.exists():
            return modelo_default

        modelos = list(OUTPUT_DIR.rglob("**/weights/best.pt"))
        if modelos:
            modelos.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return modelos[0]

    return None


def main():
    """Función principal del script."""
    print("=" * 70)
    print("   SISTEMA ANPR - RECONOCIMIENTO DE PLACAS EN TIEMPO REAL")
    print("   Detección + OCR para cámaras de vigilancia")
    print("=" * 70)

    args = parse_args()

    # Encontrar modelo
    if args.modelo:
        model_path = Path(args.modelo)
    else:
        model_path = encontrar_modelo(args.tflite)

    if model_path is None or not model_path.exists():
        print(f"\n❌ No se encontró el modelo")
        print(f"   Ejecuta primero: python 02_entrenar_modelo.py")
        sys.exit(1)

    print(f"\n📦 Modelo: {model_path}")

    # Inicializar detector
    detector = PlateDetector(
        model_path=str(model_path),
        use_tflite=args.tflite or model_path.suffix == ".tflite",
        confidence=args.confidence,
        device=args.device
    )

    # Inicializar OCR
    ocr = PlateOCR(model_name=args.ocr_model)

    # Configurar log
    log_file = None
    if args.log:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"placas_{timestamp}.csv"

    # Ejecutar
    run_realtime(
        source=args.source,
        detector=detector,
        ocr=ocr,
        show_video=not args.no_display,
        save_video=args.save_video,
        log_file=log_file
    )


if __name__ == "__main__":
    main()
