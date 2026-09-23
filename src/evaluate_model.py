
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMG_SIZE = 48
BATCH_SIZE = 64

CLASS_NAMES = [
    "angry",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

TEST_DIR = Path("data/processed/ferplus_5class/test")
MODEL_PATH = Path("models/emolens_cnn.keras")


# --------------------------------------------------
# Check required paths
# --------------------------------------------------

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test dataset not found: {TEST_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )


# --------------------------------------------------
# Load test dataset
# --------------------------------------------------

print("=" * 50)
print("Loading test dataset...")
print("=" * 50)

test_data = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0
)

test_generator = test_data.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    class_mode="categorical",
    classes=CLASS_NAMES,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print(f"\nTest images: {test_generator.samples}")
print(f"Classes: {test_generator.class_indices}")


# --------------------------------------------------
# Load trained CNN
# --------------------------------------------------

print("\nLoading saved CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print(f"Model: {MODEL_PATH}")


# --------------------------------------------------
# Generate predictions
# --------------------------------------------------

print("\nGenerating predictions...")

test_generator.reset()

probabilities = model.predict(
    test_generator,
    verbose=1
)

y_pred = np.argmax(probabilities, axis=1)
y_true = test_generator.classes


# --------------------------------------------------
# Accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(y_true, y_pred)

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("Rows = Actual")
print("Columns = Predicted\n")

print(f"{'':>12}", end="")

for name in CLASS_NAMES:
    print(f"{name:>12}", end="")

print()

for i, name in enumerate(CLASS_NAMES):
    print(f"{name:>12}", end="")

    for value in cm[i]:
        print(f"{value:>12}", end="")

    print()


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("EMOLENS CNN EVALUATION SUMMARY")
print("=" * 60)

print(f"Test Samples : {len(y_true)}")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Accuracy     : {accuracy * 100:.2f}%")

print("\nEvaluation completed successfully.")


