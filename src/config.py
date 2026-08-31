from pathlib import Path

# =========================
# Project Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"


# =========================
# Image Configuration
# =========================

IMAGE_SIZE = (48, 48)
IMAGE_CHANNELS = 1


# =========================
# Emotion Classes
# =========================

EMOTION_CLASSES = [
    "angry",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

NUM_CLASSES = len(EMOTION_CLASSES)


# =========================
# Training Configuration
# =========================

BATCH_SIZE = 64
RANDOM_SEED = 42


# =========================
# Upload Configuration
# =========================

MAX_UPLOAD_SIZE_MB = 5

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}