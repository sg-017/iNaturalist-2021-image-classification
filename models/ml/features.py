from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

IMAGE_SIZE = (128, 128)

LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)


def load_grayscale_image(
    image_path: str | Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    return cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)


def extract_lbp_feature(
    image: np.ndarray,
    radius: int = LBP_RADIUS,
    points: int = LBP_POINTS,
    method: str = LBP_METHOD,
) -> np.ndarray:
    lbp = local_binary_pattern(image, P=points, R=radius, method=method)

    bins = points + 2 if method == "uniform" else 2 ** points

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(bins + 1),
        range=(0, bins),
    )

    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-8

    return hist


def extract_hog_feature(
    image: np.ndarray,
    orientations: int = HOG_ORIENTATIONS,
    pixels_per_cell: tuple[int, int] = HOG_PIXELS_PER_CELL,
    cells_per_block: tuple[int, int] = HOG_CELLS_PER_BLOCK,
) -> np.ndarray:
    feature = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )

    return feature.astype(np.float32)


def extract_lbp_hog_feature(
    image_path: str | Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    image = load_grayscale_image(image_path, image_size)

    lbp = extract_lbp_feature(image)
    hog_feature = extract_hog_feature(image)

    return np.concatenate((lbp, hog_feature)).astype(np.float32)


def extract_features_from_paths(
    image_paths: list[Path],
    show_progress: bool = True,
) -> np.ndarray:
    features = []
    total = len(image_paths)

    for i, path in enumerate(image_paths, start=1):
        features.append(extract_lbp_hog_feature(path))

        if show_progress and (i == 1 or i % 500 == 0 or i == total):
            print(f"Processed {i}/{total} images")

    return np.asarray(features, dtype=np.float32)


if __name__ == "__main__":
    from models.ml.data import load_split

    dataset_dir = Path("dataset") / "processed_dataset"

    image_paths, labels = load_split(dataset_dir, "train")

    sample_paths = image_paths[:10]
    features = extract_features_from_paths(sample_paths)

    print(f"Sample images: {len(sample_paths)}")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Labels: {labels[:10]}")