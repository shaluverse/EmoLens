from pathlib import Path

from PIL import Image


ROOT = Path("data/processed/ferplus_5class")

SPLITS = ["train", "validation", "test"]
CLASSES = ["angry", "happy", "sad", "surprise", "neutral"]


def main():
    print("Dataset integrity check")
    print("=" * 50)

    total = 0
    bad = 0

    for split in SPLITS:
        for emotion in CLASSES:
            folder = ROOT / split / emotion
            files = list(folder.glob("*.png"))

            print(f"{split:10s} | {emotion:8s} | {len(files):,}")

            total += len(files)

            for file in files:
                try:
                    with Image.open(file) as image:
                        image.verify()
                except Exception:
                    print(f"BAD IMAGE: {file}")
                    bad += 1

    print("\n" + "=" * 50)
    print(f"Total PNG files: {total:,}")
    print(f"Corrupted images: {bad:,}")

    if bad == 0:
        print("Status: PASS")
    else:
        print("Status: FAIL")


if __name__ == "__main__":
    main()