import json
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from tqdm import tqdm


# ============================================================
# SETTINGS
# Change the variables in this section before running the file.
# ============================================================

# The repository root.
# This file should be stored at: utils/robustness.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The processed dataset folder.
# It contains train/, val/, test/ and their JSON files.
DATA_ROOT = PROJECT_ROOT / "dataset" / "processed_dataset"

# Metadata file for the selected test set.
TEST_JSON = DATA_ROOT / "test.json"

# Folder used to save generated degraded test images.
OUTPUT_DIR = DATA_ROOT / "robustness_data"

# Degradations to generate.
# Available options:
# "gaussian_noise", "gaussian_blur", "motion_blur",
# "brightness", "jpeg"
# The default list runs every available degradation.
DEGRADATIONS_TO_RUN = [
    "gaussian_noise",
    "gaussian_blur",
    "motion_blur",
    "brightness",
    "jpeg"
]

# Severity levels to generate.
# Available options: 1, 2, 3, 4
# Higher values produce stronger degradation.
SEVERITIES_TO_RUN = [1, 2, 3, 4]

# Random seed used for Gaussian noise.
SEED = 42

# Generated images are saved as PNG.
OUTPUT_FORMAT = "PNG"

# Limit the number of test images to process.
# Set to None to process all test images.
MAX_IMAGES = None

# Skip images that have already been generated.
# Set to True to skip existing output images.
SKIP_EXISTING = False


# ============================================================
# DEGRADATION STRENGTHS
# ============================================================

NOISE_SIGMA = [5, 10, 20, 35]
GAUSSIAN_BLUR_RADIUS = [1, 2, 3, 5]
MOTION_BLUR_KERNEL = [3, 5, 9, 15]
BRIGHTNESS_FACTOR = [0.8, 0.6, 0.4, 0.25]
JPEG_QUALITY = [80, 60, 40, 20]

AVAILABLE_DEGRADATIONS = [
    "gaussian_noise",
    "gaussian_blur",
    "motion_blur",
    "brightness",
    "jpeg"
]


def check_settings():
    """Check paths and selected settings."""
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Processed dataset folder not found: {DATA_ROOT}"
        )

    if not TEST_JSON.exists():
        raise FileNotFoundError(
            f"Test JSON not found: {TEST_JSON}"
        )

    for degradation in DEGRADATIONS_TO_RUN:
        if degradation not in AVAILABLE_DEGRADATIONS:
            raise ValueError(
                f"Unknown degradation: {degradation}"
            )

    for severity in SEVERITIES_TO_RUN:
        if severity not in [1, 2, 3, 4]:
            raise ValueError(
                "Severity must be 1, 2, 3, or 4."
            )


def load_test_data():
    """Load test.json and check its required fields."""
    with open(TEST_JSON, "r", encoding="utf-8") as file:
        test_list = json.load(file)

    test_data = pd.DataFrame(test_list)

    required_columns = ["file_name", "label"]

    for column in required_columns:
        if column not in test_data.columns:
            raise ValueError(
                f"Missing field in test.json: {column}"
            )

    if len(test_data) == 0:
        raise ValueError("test.json is empty.")

    return test_data


def add_gaussian_noise(image, severity, seed):
    """Add Gaussian noise to an image."""
    sigma = NOISE_SIGMA[severity - 1]

    image_array = np.array(image).astype(np.float32)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, image_array.shape)

    output = image_array + noise
    output = np.clip(output, 0, 255)
    output = output.astype(np.uint8)

    return Image.fromarray(output)


def add_gaussian_blur(image, severity):
    """Apply Gaussian blur to an image."""
    radius = GAUSSIAN_BLUR_RADIUS[severity - 1]

    return image.filter(
        ImageFilter.GaussianBlur(radius=radius)
    )


def add_motion_blur(image, severity):
    """Apply horizontal motion blur to an image."""
    kernel_size = MOTION_BLUR_KERNEL[severity - 1]

    kernel = np.zeros(
        (kernel_size, kernel_size),
        dtype=np.float32
    )

    middle = kernel_size // 2
    kernel[middle, :] = 1.0 / kernel_size

    image_array = np.array(image)

    output = cv2.filter2D(
        image_array,
        -1,
        kernel
    )

    return Image.fromarray(output)


def reduce_brightness(image, severity):
    """Reduce image brightness."""
    factor = BRIGHTNESS_FACTOR[severity - 1]

    enhancer = ImageEnhance.Brightness(image)

    return enhancer.enhance(factor)


def add_jpeg_compression(image, severity):
    """Apply JPEG compression in memory."""
    quality = JPEG_QUALITY[severity - 1]

    memory_file = BytesIO()

    image.save(
        memory_file,
        format="JPEG",
        quality=quality
    )

    memory_file.seek(0)

    compressed_image = Image.open(
        memory_file
    ).convert("RGB")

    output = compressed_image.copy()

    compressed_image.close()
    memory_file.close()

    return output


def apply_degradation(
    image,
    degradation,
    severity,
    seed
):
    """Apply one selected degradation."""
    if degradation == "gaussian_noise":
        return add_gaussian_noise(
            image,
            severity,
            seed
        )

    if degradation == "gaussian_blur":
        return add_gaussian_blur(
            image,
            severity
        )

    if degradation == "motion_blur":
        return add_motion_blur(
            image,
            severity
        )

    if degradation == "brightness":
        return reduce_brightness(
            image,
            severity
        )

    if degradation == "jpeg":
        return add_jpeg_compression(
            image,
            severity
        )

    raise ValueError(
        f"Unknown degradation: {degradation}"
    )


def get_class_folder(file_name, label):
    """Get the class folder such as 000, 001, ..., 499."""
    parent_folder = Path(file_name).parent.name

    if parent_folder:
        return parent_folder

    return f"{int(label):03d}"


def generate_one_dataset(
    test_data,
    degradation,
    severity
):
    """Generate one degraded version of the test set."""
    output_folder = (
        OUTPUT_DIR
        / degradation
        / f"severity_{severity}"
    )

    manifest_rows = []

    description = (
        f"{degradation}, severity {severity}"
    )

    # Apply MAX_IMAGES limit
    if MAX_IMAGES is not None:
        test_data = test_data.head(MAX_IMAGES)

    for row_number, row in tqdm(
        test_data.iterrows(),
        total=len(test_data),
        desc=description
    ):
        input_path = DATA_ROOT / row["file_name"]

        if not input_path.exists():
            raise FileNotFoundError(
                f"Image not found: {input_path}"
            )

        # Check if output already exists
        if SKIP_EXISTING:
            original_name = Path(row["file_name"]).stem
            parent_folder = Path(row["file_name"]).parent.name
            if parent_folder:
                class_folder = parent_folder
            else:
                class_folder = f"{int(row['label']):03d}"
            
            existing_path = (
                output_folder
                / class_folder
                / f"{original_name}.png"
            )
            
            if existing_path.exists():
                # Still add to manifest but skip generation
                manifest_rows.append({
                    "original_file_name": row["file_name"],
                    "degraded_file_name": str(
                        existing_path.relative_to(DATA_ROOT)
                    ),
                    "label": int(row["label"]),
                    "original_category_id": int(
                        row.get("original_category_id", -1)
                    ),
                    "degradation": degradation,
                    "severity": severity
                })
                continue

        with Image.open(input_path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")

            output_image = apply_degradation(
                image=image,
                degradation=degradation,
                severity=severity,
                seed=SEED + row_number
            )

        class_folder = get_class_folder(
            row["file_name"],
            row["label"]
        )

        output_class_folder = (
            output_folder
            / class_folder
        )

        output_class_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        original_name = Path(
            row["file_name"]
        ).stem

        output_path = (
            output_class_folder
            / f"{original_name}.png"
        )

        output_image.save(
            output_path,
            format=OUTPUT_FORMAT
        )

        manifest_rows.append({
            "original_file_name": row["file_name"],
            "degraded_file_name": str(
                output_path.relative_to(DATA_ROOT)
            ),
            "label": int(row["label"]),
            "original_category_id": int(
                row.get("original_category_id", -1)
            ),
            "degradation": degradation,
            "severity": severity
        })

    manifest_folder = OUTPUT_DIR / "manifests"

    manifest_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest_path = (
        manifest_folder
        / f"{degradation}_severity_{severity}.csv"
    )

    manifest_data = pd.DataFrame(
        manifest_rows
    )

    manifest_data.to_csv(
        manifest_path,
        index=False
    )

    print()
    print("Finished:", description)
    print("Generated images:", len(manifest_data))
    print("Image folder:", output_folder)
    print("Manifest:", manifest_path)
    print()


def main():
    """Generate every requested degradation and severity."""
    check_settings()

    test_data = load_test_data()

    print("Processed dataset:", DATA_ROOT)
    print("Test JSON:", TEST_JSON)
    print("Output folder:", OUTPUT_DIR)
    print("Test images:", len(test_data))
    if MAX_IMAGES is not None:
        print("Max images to process:", MAX_IMAGES)
        test_data = test_data.head(MAX_IMAGES)
    else:
        print("Max images to process: All")
    print("Skip existing:", SKIP_EXISTING)
    print("Degradations:", DEGRADATIONS_TO_RUN)
    print("Severities:", SEVERITIES_TO_RUN)
    print()

    total_runs = (
        len(DEGRADATIONS_TO_RUN)
        * len(SEVERITIES_TO_RUN)
    )

    current_run = 0

    for degradation in DEGRADATIONS_TO_RUN:
        for severity in SEVERITIES_TO_RUN:
            current_run += 1

            print(
                f"Run {current_run}/{total_runs}"
            )

            generate_one_dataset(
                test_data=test_data,
                degradation=degradation,
                severity=severity
            )

    print("All robustness datasets completed.")


if __name__ == "__main__":
    main()
