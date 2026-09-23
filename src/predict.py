"""
EmoLens - Prediction Backend

Handles:
1. Face detection
2. Face cropping
3. Image preprocessing
4. CNN prediction
"""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "emolens_cnn.keras"

IMAGE_SIZE = (48, 48)

EMOTION_CLASSES = [
    "angry",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

EMOTION_EMOJIS = {
    "angry": "😠",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😮",
}


# --------------------------------------------------
# Load model once
# --------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# Face detector
# --------------------------------------------------

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(CASCADE_PATH)


# --------------------------------------------------
# Face detection
# --------------------------------------------------

def detect_face(image: Image.Image):
    """
    Detect the largest face in an uploaded image.

    Returns:
        face_image: cropped PIL image
        face_box: (x, y, width, height)
    """

    # Convert PIL image to RGB
    rgb_image = image.convert("RGB")

    # Convert RGB → OpenCV BGR
    image_array = np.array(rgb_image)
    bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        return None, None, "no_face"

    if len(faces) > 1:
        return None, None, "multiple_faces"

    x, y, w, h = faces[0]

    # Crop face from original RGB image
    face = image.crop((x, y, x + w, y + h))

    return face, (x, y, w, h), "success"


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def preprocess_face(face: Image.Image):
    """
    Convert detected face into the format expected
    by the trained CNN.
    """

    # Convert to grayscale
    face = face.convert("L")

    # Resize to 48 × 48
    face = face.resize(IMAGE_SIZE)

    # Convert to NumPy array
    image_array = np.array(face, dtype=np.float32)

    # Normalize exactly like validation/test data
    image_array = image_array / 255.0

    # Add CNN dimensions:
    # (48, 48)
    #     ↓
    # (1, 48, 48, 1)
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_emotion(image: Image.Image):
    """
    Detect a face and predict its emotion.

    Returns a dictionary containing:
        emotion
        confidence
        emoji
        face
        face_box
    """

    face, face_box, detection_status = detect_face(image)

    if detection_status == "no_face":
        return {
            "success": False,
            "status": "no_face",
            "message": "No face detected in the image.",
            "emotion": None,
            "confidence": None,
            "emoji": None,
            "face": None,
            "face_box": None,
        }

    if detection_status == "multiple_faces":
        return {
            "success": False,
            "status": "multiple_faces",
            "message": "Multiple faces detected. Please upload an image containing one face.",
            "emotion": None,
            "confidence": None,
            "emoji": None,
            "face": None,
            "face_box": None,
        }

    # Prepare face for CNN
    processed_face = preprocess_face(face)

    # Predict
    probabilities = model.predict(
        processed_face,
        verbose=0,
    )[0]

    # Get highest probability
    predicted_index = int(np.argmax(probabilities))

    emotion = EMOTION_CLASSES[predicted_index]
    confidence = float(probabilities[predicted_index])

    return {
        "success": True,
        "message": "Prediction successful.",
        "emotion": emotion,
        "confidence": confidence,
        "emoji": EMOTION_EMOJIS[emotion],
        "face": face,
        "face_box": face_box,
        "status": "success",
        "probabilities": {
            emotion_name: float(probability)
            for emotion_name, probability
            in zip(EMOTION_CLASSES, probabilities)
        },
    }