from pathlib import Path

import numpy as np

from models.ml.data import load_split
from models.ml.features import extract_features_from_paths

OUTPUT_DIR = Path(__file__).parent / "outputs" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_split_features(dataset_dir: Path, split: str):
    print(f"\nLoading {split} dataset...")

    image_paths, labels = load_split(dataset_dir, split)
    print(f"{split}: {len(image_paths)} images")

    features = extract_features_from_paths(image_paths)

    output_file = OUTPUT_DIR / f"{split}_features.npz"

    np.savez_compressed(
        output_file,
        features=features,
        labels=np.asarray(labels),
    )

    print(f"Saved to: {output_file}")
    print(f"Feature shape: {features.shape}")


def main():
    print("Extracting features...")

    dataset_dir = Path("dataset") / "processed_dataset"

    for split in ("train", "val", "test"):
        save_split_features(dataset_dir, split)

    print("\nFeature extraction completed.")


if __name__ == "__main__":
    main()