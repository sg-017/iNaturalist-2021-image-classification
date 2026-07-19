# Robustness Dataset Generator

```text
iNaturalist-2021-image-classification/
└── utils/
    └── robustness.py
```

Run it from the repository root:

```bash
python3 utils/robustness.py
```

No command-line arguments are needed.

## Expected Project Structure

```text
iNaturalist-2021-image-classification/
├── dataset/
│   └── processed_dataset/
│       ├── train/
│       ├── train.json
│       ├── val/
│       ├── val.json
│       ├── test/
│       ├── test.json
│       ├── selected_classes.json
│       └── robustness_data/          ← output folder
├── models/
├── explainability/
├── utils/
│   ├── robustness.py
│   └── ROBUSTNESS_GUIDE.md
└── README.md
```
It saves generated degraded images to:

```text
dataset/processed_dataset/robustness_data/
```

## Settings to Change

All main settings are at the top of `robustness.py`.

### Dataset paths

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "dataset" / "processed_dataset"
TEST_JSON = DATA_ROOT / "test.json"
OUTPUT_DIR = DATA_ROOT / "robustness_data"
```

Change these only when your folders are stored somewhere else.

### Image limit

```python
MAX_IMAGES = None
```

- Set to a number (e.g. `20`) to process only that many test images (useful for quick testing).
- Set to `None` to process all test images.

### Skip existing

```python
SKIP_EXISTING = False
```

- Set to `True` to skip images that have already been generated (useful for resuming interrupted runs).
- Set to `False` to regenerate all images.

### Degradations to run

```python
DEGRADATIONS_TO_RUN = [
    "gaussian_noise",
    "gaussian_blur",
    "motion_blur",
    "brightness",
    "jpeg"
]
```

The default list runs all degradation types.

To run only Gaussian blur and motion blur:

```python
DEGRADATIONS_TO_RUN = [
    "gaussian_blur",
    "motion_blur"
]
```

### Severity levels

```python
SEVERITIES_TO_RUN = [1, 2, 3, 4]
```

Available levels:

```text
1 = light
2 = moderate
3 = strong
4 = very strong
```

To run only two levels:

```python
SEVERITIES_TO_RUN = [1, 2]
```

### Random seed

```python
SEED = 42
```

This makes Gaussian noise reproducible.

## Input

The main input is:

```text
dataset/processed_dataset/test.json
```

The JSON file is a list of records. Each record must contain:

```text
file_name
label
```

It may also contain:

```text
image_id
original_category_id
```

Example record:

```json
{
    "image_id": 2760336,
    "file_name": "test/000/2760336_083b38d0-2c65-4e82-9111-4a7ef467ba84.jpg",
    "label": 0,
    "original_category_id": 6
}
```

Each `file_name` is joined with `DATA_ROOT` to form the full path.

Example:

```text
DATA_ROOT:
dataset/processed_dataset/

file_name:
test/000/2760336_083b38d0-2c65-4e82-9111-4a7ef467ba84.jpg

full image path:
dataset/processed_dataset/test/000/2760336_083b38d0-2c65-4e82-9111-4a7ef467ba84.jpg
```

## Output

Generated images are saved as PNG files.

```text
dataset/processed_dataset/robustness_data/
├── gaussian_noise/
│   ├── severity_1/
│   ├── severity_2/
│   ├── severity_3/
│   └── severity_4/
├── gaussian_blur/
│   ├── severity_1/
│   ├── severity_2/
│   ├── severity_3/
│   └── severity_4/
├── motion_blur/
│   ├── severity_1/
│   ├── severity_2/
│   ├── severity_3/
│   └── severity_4/
├── brightness/
│   ├── severity_1/
│   ├── severity_2/
│   ├── severity_3/
│   └── severity_4/
├── jpeg/
│   ├── severity_1/
│   ├── severity_2/
│   ├── severity_3/
│   └── severity_4/
└── manifests/
    ├── gaussian_noise_severity_1.csv
    ├── gaussian_noise_severity_2.csv
    ├── gaussian_noise_severity_3.csv
    ├── gaussian_noise_severity_4.csv
    ├── gaussian_blur_severity_1.csv
    ├── gaussian_blur_severity_2.csv
    ├── gaussian_blur_severity_3.csv
    ├── gaussian_blur_severity_4.csv
    ├── motion_blur_severity_1.csv
    ├── motion_blur_severity_2.csv
    ├── motion_blur_severity_3.csv
    ├── motion_blur_severity_4.csv
    ├── brightness_severity_1.csv
    ├── brightness_severity_2.csv
    ├── brightness_severity_3.csv
    ├── brightness_severity_4.csv
    ├── jpeg_severity_1.csv
    ├── jpeg_severity_2.csv
    ├── jpeg_severity_3.csv
    └── jpeg_severity_4.csv
```

Each degradation and severity gets its own copy of the test set.

Manifest files are saved in:

```text
dataset/processed_dataset/robustness_data/manifests/
```

Each manifest contains:

```text
original_file_name
degraded_file_name
label
original_category_id
degradation
severity
```

## What Each Degradation Does

### Gaussian noise

Adds random pixel noise.

```text
Severity 1: sigma 5
Severity 2: sigma 10
Severity 3: sigma 20
Severity 4: sigma 35
```

### Gaussian blur

Makes the image appear out of focus.

```text
Severity 1: radius 1
Severity 2: radius 2
Severity 3: radius 3
Severity 4: radius 5
```

### Motion blur

Creates horizontal motion blur.

```text
Severity 1: kernel 3
Severity 2: kernel 5
Severity 3: kernel 9
Severity 4: kernel 15
```

### Brightness

Makes the image darker.

```text
Severity 1: factor 0.80
Severity 2: factor 0.60
Severity 3: factor 0.40
Severity 4: factor 0.25
```

### JPEG compression

Reduces JPEG quality.

```text
Severity 1: quality 80
Severity 2: quality 60
Severity 3: quality 40
Severity 4: quality 20
```

## Processing Flow

For each test image:

```text
read original image
→ correct EXIF orientation
→ convert to RGB
→ apply one degradation
→ save degraded image as PNG
→ record output path in a manifest CSV
```

The original test images are never modified.

## Function Summary

### `check_settings()`

Checks that:

- the dataset folder exists;
- the test JSON exists;
- degradation names are valid;
- severity values are valid.

### `load_test_data()`

Reads `test.json` and checks that it contains:

```text
file_name
label
```

### `add_gaussian_noise(image, severity, seed)`

Adds Gaussian noise with sigma based on severity.

### `add_gaussian_blur(image, severity)`

Applies Gaussian blur with radius based on severity.

### `add_motion_blur(image, severity)`

Applies horizontal motion blur with kernel size based on severity.

### `reduce_brightness(image, severity)`

Reduces brightness with factor based on severity.

### `add_jpeg_compression(image, severity)`

Applies JPEG compression with quality based on severity.

### `apply_degradation(image, degradation, severity, seed)`

Input:

```text
image
degradation name
severity
seed
```

Output:

```text
new degraded PIL image
```

### `get_class_folder(file_name, label)`

Returns the class folder name from the file path or label.

### `generate_one_dataset(test_data, degradation, severity)`

Processes the test set for one degradation and one severity.

Respects `MAX_IMAGES` and `SKIP_EXISTING` settings.

Output:

- one generated image folder;
- one manifest CSV.

### `main()`

Reads:

```python
DEGRADATIONS_TO_RUN
SEVERITIES_TO_RUN
MAX_IMAGES
SKIP_EXISTING
```

and runs every combination.

With the default settings:

```text
5 degradation types × 4 severity levels = 20 runs
```

For a 5,000-image test set, this creates about:

```text
100,000 generated images
```

## Runtime Estimates

Tested on MacBook Air M3:

| Setting | Images | Time | Storage |
|---------|--------|------|---------|
| Smoke test (MAX_IMAGES=20, 1 severity) | 100 | ~3 seconds | ~24 MB |
| Full run (all 5,000 images, 4 severities) | 100,000 | ~43 minutes | ~24 GB |

Processing speed: ~39 images/second.

### Quick test (20 images, 1 severity)

Edit the settings at the top of `robustness.py`:

```python
MAX_IMAGES = 20
SEVERITIES_TO_RUN = [1]
```

Then run:

```bash
python3 utils/robustness.py
```

### Full run (all images, all severities)

Edit the settings at the top of `robustness.py`:

```python
MAX_IMAGES = None
SEVERITIES_TO_RUN = [1, 2, 3, 4]
```

Then run:

```bash
python3 utils/robustness.py
```

### Resume interrupted run

If the run was interrupted, set:

```python
SKIP_EXISTING = True
```

This will skip already generated images and continue from where it left off.

The script prints:

- dataset path;
- test JSON path;
- output path;
- number of test images;
- max images to process;
- skip existing setting;
- selected degradations;
- selected severity levels;
- progress for every run.

## Using the Generated Data

Model owners should load generated images through the manifest CSV.

Then apply the same test preprocessing used by the original model:

```text
resize
centre crop
tensor conversion
normalisation
```

Recommended order:

```text
degraded image
→ model test transform
→ trained model
→ prediction
```

For the main robustness experiment:

- do not retrain the model;
- do not degrade training images;
- compare original test performance against each degradation and severity;
- report top-1 accuracy and macro F1;
- use the same degraded test sets for every model.
