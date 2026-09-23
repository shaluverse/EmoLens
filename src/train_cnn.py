"""
EmoLens - CNN Training

Trains a CNN on the processed FERPlus dataset
for five-class facial emotion recognition.
"""

import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from sklearn.utils.class_weight import compute_class_weight


# -----------------------------
# Configuration
# -----------------------------

IMG_SIZE = 48
NUM_CLASSES = 5
BATCH_SIZE = 64
EPOCHS = 30

CLASS_NAMES = [
    "angry",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

SEED = 42

TRAIN_DIR = "data/processed/ferplus_5class/train"
VAL_DIR = "data/processed/ferplus_5class/validation"
TEST_DIR = "data/processed/ferplus_5class/test"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "emolens_cnn.keras")


# -----------------------------
# Reproducibility
# -----------------------------

np.random.seed(SEED)
tf.random.set_seed(SEED)


# -----------------------------
# CNN Architecture
# -----------------------------

def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.30),

        # Classification head
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.50),

        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# -----------------------------
# Main
# -----------------------------

def main():
    print("EmoLens CNN Training")
    print("=" * 40)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # -------------------------
    # Data augmentation
    # -------------------------

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        horizontal_flip=True,
    )

    val_test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    # -------------------------
    # Dataset generators
    # -------------------------

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        classes=CLASS_NAMES,
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )

    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        classes=CLASS_NAMES,
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        classes=CLASS_NAMES,
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("\nClass indices:")
    print(train_generator.class_indices)

    print("\nDataset sizes:")
    print("Training   :", train_generator.samples)
    print("Validation :", val_generator.samples)
    print("Test       :", test_generator.samples)

    # -------------------------
    # Class weights
    # -------------------------

    train_labels = train_generator.classes

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels,
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(
            np.unique(train_labels),
            class_weights_array
        )
    }

    print("\nClass weights:")
    for class_id, weight in class_weights.items():
        print(f"{CLASS_NAMES[class_id]:<10}: {weight:.4f}")

    # -------------------------
    # Build model
    # -------------------------

    model = build_model()

    print("\nModel summary:")
    model.summary()

    # -------------------------
    # Callbacks
    # -------------------------

    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),

        EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # -------------------------
    # Training
    # -------------------------

    print("\nStarting training...")
    print("=" * 40)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # -------------------------
    # Load best model
    # -------------------------

    print("\nLoading best saved model...")
    best_model = tf.keras.models.load_model(MODEL_PATH)

    # -------------------------
    # Test evaluation
    # -------------------------

    print("\nEvaluating on test set...")
    print("=" * 40)

    test_loss, test_accuracy = best_model.evaluate(
        test_generator,
        verbose=1,
    )

    print(f"\nTest Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

    print("\nBest model saved to:")
    print(MODEL_PATH)

    print("\nCNN training completed successfully.")


if __name__ == "__main__":
    main()