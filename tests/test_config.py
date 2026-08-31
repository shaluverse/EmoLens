from src.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    BATCH_SIZE,
    EMOTION_CLASSES,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    MAX_UPLOAD_SIZE_MB,
    NUM_CLASSES,
    RANDOM_SEED,
)


def test_emotion_classes():
    assert len(EMOTION_CLASSES) == 5
    assert NUM_CLASSES == 5


def test_image_configuration():
    assert IMAGE_SIZE == (48, 48)
    assert IMAGE_CHANNELS == 1


def test_training_configuration():
    assert BATCH_SIZE > 0
    assert RANDOM_SEED == 42


def test_upload_configuration():
    assert MAX_UPLOAD_SIZE_MB > 0
    assert ".jpg" in ALLOWED_IMAGE_EXTENSIONS
    assert ".jpeg" in ALLOWED_IMAGE_EXTENSIONS
    assert ".png" in ALLOWED_IMAGE_EXTENSIONS