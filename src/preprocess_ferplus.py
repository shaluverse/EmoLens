from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FER2013_PATH = PROJECT_ROOT / "data" / "raw" / "fer2013.csv"
FERPLUS_PATH = PROJECT_ROOT / "data" / "raw" / "fer2013new.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ferplus_5class"


# -----------------------------
# Configuration
# -----------------------------
TARGET_COLUMNS = {
    "anger": "angry",
    "happiness": "happy",
    "sadness": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}

SPLIT_NAMES = {
    "Training": "train",
    "PublicTest": "validation",
    "PrivateTest": "test",
}


def load_data():
    """Load and verify the FER2013 and FERPlus CSV files."""
    print("Loading FER2013...")
    fer2013 = pd.read_csv(FER2013_PATH)

    print("Loading FERPlus annotations...")
    ferplus = pd.read_csv(FERPLUS_PATH)

    if len(fer2013) != len(ferplus):
        raise ValueError(
            f"Row count mismatch: FER2013={len(fer2013)}, "
            f"FERPlus={len(ferplus)}"
        )

    if not fer2013["Usage"].equals(ferplus["Usage"]):
        raise ValueError("FER2013 and FERPlus Usage columns do not match.")

    print(f"Verified {len(fer2013):,} aligned rows.")

    return fer2013, ferplus


def assign_labels(ferplus):
    """
    Assign one of the five target emotions only when there is
    a unique highest vote.

    Tied highest votes are discarded.
    """

    vote_columns = list(TARGET_COLUMNS.keys())

    votes = ferplus[vote_columns]

    max_votes = votes.max(axis=1)

    # Count how many target emotions share the maximum vote.
    number_of_max_votes = votes.eq(max_votes, axis=0).sum(axis=1)

    # Keep only rows with one unique highest-voted emotion.
    valid = (max_votes > 0) & (number_of_max_votes == 1)

    labels = votes.idxmax(axis=1).map(TARGET_COLUMNS)

    return labels, valid


def pixels_to_image(pixel_string):
    """Convert FER2013 pixel string into a 48x48 grayscale image."""
    pixels = np.fromstring(pixel_string, dtype=np.uint8, sep=" ")

    if pixels.size != 48 * 48:
        raise ValueError(
            f"Expected 2304 pixels, received {pixels.size}."
        )

    return pixels.reshape(48, 48)


def save_dataset(fer2013, labels, valid):
    """Convert valid rows into PNG images organized by split and class."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_counts = {}

    valid_indices = np.where(valid.to_numpy())[0]

    for index in valid_indices:
        split = SPLIT_NAMES[fer2013.iloc[index]["Usage"]]
        label = labels.iloc[index]

        output_class_dir = OUTPUT_DIR / split / label
        output_class_dir.mkdir(parents=True, exist_ok=True)

        image = pixels_to_image(fer2013.iloc[index]["pixels"])

        image_path = output_class_dir / f"{index:05d}.png"

        Image.fromarray(image, mode="L").save(image_path)

        key = (split, label)
        saved_counts[key] = saved_counts.get(key, 0) + 1

    return saved_counts


def main():
    print("=" * 60)
    print("EmoLens - FERPlus 5-Class Preprocessing")
    print("=" * 60)

    fer2013, ferplus = load_data()

    labels, valid = assign_labels(ferplus)

    print("\nLabel filtering:")
    print(f"Total rows:       {len(ferplus):,}")
    print(f"Valid rows:       {valid.sum():,}")
    print(f"Discarded rows:   {(~valid).sum():,}")

    print("\nSelected class distribution:")
    print(labels[valid].value_counts().to_string())

    print("\nSaving processed images...")

    counts = save_dataset(fer2013, labels, valid)

    print("\nProcessed dataset:")
    for (split, label), count in sorted(counts.items()):
        print(f"{split:10s} | {label:8s} | {count:,}")

    print("\nDone.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()