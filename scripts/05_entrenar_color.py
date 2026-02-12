#!/usr/bin/env python3
"""
Train EfficientNetB0 color classifier for vehicle color recognition.
Two-phase training: head-only then full fine-tuning.

Usage:
    python scripts/05_entrenar_color.py
    python scripts/05_entrenar_color.py --data datasets/vehicle_colors --epochs-head 30 --epochs-finetune 20
"""

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# 15 classes — alphabetical order matching image_dataset_from_directory
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

IMG_SIZE = 224
NUM_CLASSES = 15


def build_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])


def build_model():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    return model


def compute_weights(train_dir):
    """Compute class weights from training directory to handle imbalance."""
    labels = []
    for i, cls_name in enumerate(CLASS_NAMES):
        cls_dir = Path(train_dir) / cls_name
        if cls_dir.exists():
            count = len(list(cls_dir.iterdir()))
            labels.extend([i] * count)
    weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=np.array(labels))
    return dict(enumerate(weights))


def main():
    parser = argparse.ArgumentParser(description="Train vehicle color classifier")
    parser.add_argument("--data", type=str, default="datasets/vehicle_colors")
    parser.add_argument("--epochs-head", type=int, default=30)
    parser.add_argument("--epochs-finetune", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data_dir = Path(args.data)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    # Device setup
    if args.device == "auto":
        if tf.config.list_physical_devices("GPU"):
            print("Using GPU")
        elif hasattr(tf, "config") and hasattr(tf.config, "list_physical_devices"):
            # Check for Apple Silicon MPS (TF-metal)
            print(f"Available devices: {[d.device_type for d in tf.config.list_physical_devices()]}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Load datasets
    print(f"\nLoading datasets from {data_dir}...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(IMG_SIZE, IMG_SIZE), batch_size=args.batch,
        label_mode="categorical", class_names=CLASS_NAMES, shuffle=True, seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=(IMG_SIZE, IMG_SIZE), batch_size=args.batch,
        label_mode="categorical", class_names=CLASS_NAMES, shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=(IMG_SIZE, IMG_SIZE), batch_size=args.batch,
        label_mode="categorical", class_names=CLASS_NAMES, shuffle=False,
    )

    # Augmentation + prefetch
    augment = build_augmentation()
    train_ds_aug = train_ds.map(lambda x, y: (augment(x, training=True), y),
                                num_parallel_calls=tf.data.AUTOTUNE)
    train_ds_aug = train_ds_aug.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    # Class weights
    class_weights = compute_weights(train_dir)
    print(f"\nClass weights: { {CLASS_NAMES[k]: f'{v:.2f}' for k, v in class_weights.items()} }")

    # Build model
    model = build_model()
    model.summary()

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
    )

    # ── Phase 1: Train head only ──
    print("\n" + "=" * 60)
    print("PHASE 1: Training classification head (base frozen)")
    print("=" * 60)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    model.fit(
        train_ds_aug, validation_data=val_ds, epochs=args.epochs_head,
        callbacks=[early_stop, reduce_lr], class_weight=class_weights,
    )

    # ── Phase 2: Fine-tune all layers ──
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning all layers")
    print("=" * 60)

    model.layers[0].trainable = True  # Unfreeze base

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    early_stop_ft = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    reduce_lr_ft = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
    )

    model.fit(
        train_ds_aug, validation_data=val_ds, epochs=args.epochs_finetune,
        callbacks=[early_stop_ft, reduce_lr_ft], class_weight=class_weights,
    )

    # ── Save model ──
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    h5_path = models_dir / "color_classifier_efficientnet.h5"
    keras_path = models_dir / "color_classifier_efficientnet.keras"

    model.save(str(h5_path))
    model.save(str(keras_path))
    print(f"\nModel saved to: {h5_path}")
    print(f"Keras model saved to: {keras_path}")

    # ── Evaluate on test set ──
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Classification report
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    es_names = [EN_TO_ES[c] for c in CLASS_NAMES]
    print("\n" + classification_report(y_true, y_pred, target_names=es_names))


if __name__ == "__main__":
    main()
