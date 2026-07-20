import json
from pathlib import Path
from time import perf_counter

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .data import load_split

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR.parent.parent / "dataset" / "processed_dataset"

MODEL_DIR = BASE_DIR / "outputs" / "models"
RESULT_DIR = BASE_DIR / "outputs" / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
NUM_WORDS = 200
MAX_KEYPOINTS = 200

rng = np.random.default_rng(RANDOM_SEED)
sift = cv2.SIFT_create(nfeatures=MAX_KEYPOINTS)


def add_gaussian_noise(image, severity):
    sigma = [10, 20, 30, 40][severity - 1]
    noise = rng.normal(0, sigma, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_gaussian_blur(image, severity):
    kernel = [3, 5, 7, 9][severity - 1]
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def add_motion_blur(image, severity):
    kernel_size = [3, 5, 7, 9][severity - 1]

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1
    kernel /= kernel_size

    return cv2.filter2D(image, -1, kernel)


def reduce_brightness(image, severity):
    factor = [0.8, 0.6, 0.4, 0.2][severity - 1]
    dark = image.astype(np.float32) * factor
    return np.clip(dark, 0, 255).astype(np.uint8)


def add_jpeg_compression(image, severity):
    quality = [80, 60, 40, 20][severity - 1]

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )

    if not success:
        return image

    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def apply_degradation(image, degradation, severity):
    if degradation == "gaussian_noise":
        return add_gaussian_noise(image, severity)
    elif degradation == "gaussian_blur":
        return add_gaussian_blur(image, severity)
    elif degradation == "motion_blur":
        return add_motion_blur(image, severity)
    elif degradation == "brightness_reduction":
        return reduce_brightness(image, severity)
    elif degradation == "jpeg_compression":
        return add_jpeg_compression(image, severity)

    return image


def create_histogram(image, kmeans):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, descriptors = sift.detectAndCompute(gray, None)

    histogram = np.zeros(NUM_WORDS, dtype=np.float32)

    if descriptors is None:
        return histogram

    words = kmeans.predict(descriptors)

    for word in words:
        histogram[word] += 1

    if histogram.sum() > 0:
        histogram /= histogram.sum()

    return histogram


def create_robustness_features(image_paths, kmeans, degradation, severity):
    features = []

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))

        if image is None:
            histogram = np.zeros(NUM_WORDS, dtype=np.float32)
        else:
            degraded = apply_degradation(image, degradation, severity)
            histogram = create_histogram(degraded, kmeans)

        features.append(histogram)

        if (i + 1) % 1000 == 0:
            print(f"{degradation} (severity {severity}): {i + 1}/{len(image_paths)}")

    return np.asarray(features, dtype=np.float32)


def calculate_metrics(labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def main():
    test_images, test_labels = load_split(DATASET_DIR, "test")

    kmeans = joblib.load(MODEL_DIR / "bovw_kmeans.joblib")
    model = joblib.load(MODEL_DIR / "bovw_svm.joblib")

    degradations = [
        "gaussian_noise",
        "gaussian_blur",
        "motion_blur",
        "brightness_reduction",
        "jpeg_compression",
    ]

    all_results = {}

    for degradation in degradations:
        all_results[degradation] = {}

        for severity in range(1, 5):
            print(f"\nTesting {degradation} (severity {severity})...")

            start = perf_counter()

            features = create_robustness_features(
                test_images,
                kmeans,
                degradation,
                severity,
            )

            predictions = model.predict(features)
            elapse_time = perf_counter() - start

            metrics = calculate_metrics(test_labels, predictions)
            metrics["test_time_seconds"] = elapse_time

            all_results[degradation][f"severity_{severity}"] = metrics

            print(f"Accuracy : {metrics['accuracy']:.4f}")
            print(f"Macro F1 : {metrics['macro_f1']:.4f}")
            print(f"Time      : {elapse_time:.2f} seconds")

    output_path = RESULT_DIR / "bovw_svm_robustness_metrics.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nRobustness testing finished.")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()