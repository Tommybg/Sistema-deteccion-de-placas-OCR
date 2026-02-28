"""
Vehicle color classification module.
Supports both Keras (.h5) and TFLite backends.
"""

from pathlib import Path

import cv2
import numpy as np

# 15 classes — alphabetical order, must match training
CLASS_NAMES = sorted([
    "beige", "black", "blue", "brown", "gold", "green", "grey",
    "orange", "pink", "purple", "red", "silver", "tan", "white", "yellow"
])

EN_TO_ES = {
    "beige": "Beige", "black": "Negro", "blue": "Azul", "brown": "Café",
    "gold": "Dorado", "green": "Verde", "grey": "Gris", "orange": "Naranja",
    "pink": "Rosa", "purple": "Morado", "red": "Rojo", "silver": "Plata",
    "tan": "Canela", "white": "Blanco", "yellow": "Amarillo",
}

DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "color_classifier_efficientnet.h5"
DEFAULT_TFLITE_PATH = Path(__file__).parent.parent / "models" / "tflite_exports" / "color_classifier_int8.tflite"


class ColorClassifier:
    def __init__(self, model_path=None, input_size=224, use_tflite=False):
        self.input_size = input_size
        self.use_tflite = use_tflite

        if use_tflite:
            # Try ai-edge-litert (Google's successor), then tflite-runtime, then full TF
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                try:
                    from tflite_runtime.interpreter import Interpreter
                except ImportError:
                    from tensorflow.lite import Interpreter
            model_path = str(model_path or DEFAULT_TFLITE_PATH)
            self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            import tensorflow as tf
            model_path = str(model_path or DEFAULT_MODEL_PATH)
            self.model = tf.keras.models.load_model(model_path)

    def classify(self, vehicle_crop):
        """Classify the color of a vehicle crop (BGR numpy array).

        Returns: {"color": "Blanco", "confidence": 0.92}
        """
        # Resize to model input size
        img = cv2.resize(vehicle_crop, (self.input_size, self.input_size))
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.use_tflite:
            img_input = np.expand_dims(img.astype(np.uint8), axis=0)
            self.interpreter.set_tensor(self.input_details[0]["index"], img_input)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]["index"])
            probs = output[0].astype(np.float32)
            # If uint8 output, scale to 0-1
            if probs.max() > 1.0:
                probs = probs / 255.0
        else:
            img_input = np.expand_dims(img.astype(np.float32), axis=0)
            probs = self.model.predict(img_input, verbose=0)[0]

        idx = int(np.argmax(probs))
        en_name = CLASS_NAMES[idx]
        return {
            "color": EN_TO_ES[en_name],
            "color_en": en_name,
            "confidence": float(probs[idx]),
        }
