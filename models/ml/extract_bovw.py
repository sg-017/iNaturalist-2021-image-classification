from pathlib import Path

import joblib
import numpy as np

from .data import load_split
from .bovw import build_vocabulary, create_bovw_features


BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR.parent.parent / "dataset" / "processed_dataset"

OUTPUT_DIR = BASE_DIR / "outputs" / "features"
MODEL_DIR = BASE_DIR / "outputs" / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def save_features(name, features, labels):
    np.savez_compressed(
        OUTPUT_DIR / f"{name}_bovw_features.npz",
        features=features,
        labels=labels
    )


def main():

    # Load dataset
    train_images, train_labels = load_split(DATASET_DIR,"train")
    val_images, val_labels = load_split(DATASET_DIR,"val")
    test_images, test_labels = load_split(DATASET_DIR,"test")

    # Build visual vocabulary
    kmeans = build_vocabulary(train_images)

    joblib.dump(
        kmeans,
        BASE_DIR / "outputs" / "models" / "bovw_kmeans.joblib"
    )

    # Create BoVW features
    train_features = create_bovw_features(train_images, kmeans)
    val_features = create_bovw_features(val_images, kmeans)
    test_features = create_bovw_features(test_images, kmeans)

    # Save features
    save_features("train", train_features, train_labels)
    save_features("val", val_features, val_labels)
    save_features("test", test_features, test_labels)

    print("BoVW feature extraction finished")


if __name__ == "__main__":
    main()