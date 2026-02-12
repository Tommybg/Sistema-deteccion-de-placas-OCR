#!/usr/bin/env python3
"""
Export trained color classifier to INT8 TFLite for Coral Edge TPU.

Strategy: Keras → SavedModel → TFLite INT8 (full integer quantization).
The intermediate SavedModel step avoids Keras 3 direct-converter issues.

Usage:
    python scripts/06_exportar_color_tflite.py
    python scripts/06_exportar_color_tflite.py --model models/color_classifier_efficientnet.h5
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

IMG_SIZE = 224
NUM_CLASSES = 15
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


def representative_dataset_gen(calibration_dir, num_samples=300):
    """Generator yielding calibration images for INT8 quantization."""
    ds = tf.keras.utils.image_dataset_from_directory(
        calibration_dir, image_size=(IMG_SIZE, IMG_SIZE), batch_size=1,
        label_mode=None, shuffle=True, seed=42,
    )
    count = 0
    for batch in ds:
        if count >= num_samples:
            break
        yield [batch.numpy().astype(np.float32)]
        count += 1


def main():
    parser = argparse.ArgumentParser(description="Export color classifier to TFLite INT8")
    parser.add_argument("--model", type=str, default="models/color_classifier_efficientnet.h5")
    parser.add_argument("--calibration-data", type=str, default="datasets/vehicle_colors/train")
    parser.add_argument("--output-dir", type=str, default="models/tflite_exports")
    parser.add_argument("--num-calibration", type=int, default=300)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Keras model
    print(f"Loading model from {args.model}...")
    model = tf.keras.models.load_model(args.model)
    model.summary()

    # Step 1: Export to SavedModel via model.export() (Keras 3 compatible)
    tmp_dir = tempfile.mkdtemp()
    saved_model_dir = Path(tmp_dir) / "saved_model"
    print(f"\nExporting to SavedModel at {saved_model_dir}...")
    model.export(str(saved_model_dir))

    # Step 2: Convert SavedModel → TFLite INT8 (full integer, Coral-compatible)
    print("\nConverting SavedModel to TFLite INT8 (full integer quantization)...")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(
        args.calibration_data, args.num_calibration
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    tflite_path = output_dir / "color_classifier_int8.tflite"
    tflite_path.write_bytes(tflite_model)
    size_mb = len(tflite_model) / 1024 / 1024
    print(f"TFLite INT8 model saved to: {tflite_path} ({size_mb:.1f} MB)")

    # Cleanup temp SavedModel
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Validate: Compare Keras vs TFLite accuracy on test set ──
    print("\nValidating TFLite model against Keras model...")

    test_dir = Path(args.calibration_data).parent / "test"
    if not test_dir.exists():
        print(f"Test directory {test_dir} not found, skipping validation.")
        return

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=(IMG_SIZE, IMG_SIZE), batch_size=32,
        label_mode="categorical", class_names=CLASS_NAMES, shuffle=False,
    )

    # Keras predictions
    y_true, y_keras, y_tflite = [], [], []
    for images, labels in test_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_keras.extend(np.argmax(model.predict(images, verbose=0), axis=1))

    # TFLite INT8 predictions
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"TFLite input: {input_details[0]['dtype']}, shape: {input_details[0]['shape']}")
    print(f"TFLite output: {output_details[0]['dtype']}, shape: {output_details[0]['shape']}")

    for images, _ in test_ds:
        for img in images:
            img_uint8 = tf.cast(img, tf.uint8).numpy()
            img_uint8 = np.expand_dims(img_uint8, axis=0)
            interpreter.set_tensor(input_details[0]["index"], img_uint8)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]["index"])
            y_tflite.append(np.argmax(output[0]))

    keras_acc = np.mean(np.array(y_true) == np.array(y_keras))
    tflite_acc = np.mean(np.array(y_true) == np.array(y_tflite))

    print(f"\nKeras accuracy:  {keras_acc:.4f}")
    print(f"TFLite accuracy: {tflite_acc:.4f}")
    print(f"Accuracy drop:   {keras_acc - tflite_acc:.4f}")

    es_names = [EN_TO_ES[c] for c in CLASS_NAMES]
    print("\nTFLite Classification Report:")
    print(classification_report(y_true, y_tflite, target_names=es_names))


if __name__ == "__main__":
    main()
