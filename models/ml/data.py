import json
from pathlib import Path

NUM_CLASSES = 500

EXPECTED_CLASS_NAMES = [
    f"{label:03d}"
    for label in range(NUM_CLASSES)
]

EXPECTED_IMAGES_PER_CLASS = {
    "train": 40,
    "val": 10,
    "test": 10,
}

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def get_class_directories(split_dir: Path) -> list[Path]:
    class_directories = sorted(
        path
        for path in split_dir.iterdir()
        if path.is_dir()
    )

    class_names = [
        path.name
        for path in class_directories
    ]

    if class_names != EXPECTED_CLASS_NAMES:
        raise ValueError(
            f"{split_dir} must contain class folders "
            "000 to 499."
        )

    return class_directories


def get_image_paths(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
    )


def load_split(
    dataset_dir: str | Path,
    split: str,
) -> tuple[list[Path], list[int]]:
    if split not in EXPECTED_IMAGES_PER_CLASS:
        raise ValueError(
            "split must be 'train', 'val', or 'test'"
        )

    dataset_dir = Path(dataset_dir)
    split_dir = dataset_dir / split

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset split does not exist: {split_dir}"
        )

    class_directories = get_class_directories(split_dir)

    expected_count = EXPECTED_IMAGES_PER_CLASS[split]

    image_paths: list[Path] = []
    labels: list[int] = []

    for class_dir in class_directories:
        label = int(class_dir.name)
        class_images = get_image_paths(class_dir)

        if len(class_images) != expected_count:
            raise ValueError(
                f"{split}/{class_dir.name} contains "
                f"{len(class_images)} images, "
                f"but expected {expected_count}."
            )

        image_paths.extend(class_images)
        labels.extend([label] * len(class_images))

    expected_total = NUM_CLASSES * expected_count

    if len(image_paths) != expected_total:
        raise ValueError(
            f"{split} contains {len(image_paths)} images, "
            f"but expected {expected_total}."
        )

    return image_paths, labels


def load_original_category_mapping(
    dataset_dir: str | Path,
) -> dict[int, int]:
    dataset_dir = Path(dataset_dir)
    mapping_path = dataset_dir / "selected_classes.json"

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Mapping file does not exist: {mapping_path}"
        )

    with mapping_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    classes = data.get("classes", [])

    mapping = {
        int(item["label"]): int(item["original_category_id"])
        for item in classes
    }

    expected_labels = set(range(NUM_CLASSES))

    if set(mapping.keys()) != expected_labels:
        raise ValueError(
            "selected_classes.json must contain all labels "
            "from 0 to 499."
        )

    return mapping


def load_all_splits(
    dataset_dir: str | Path,
) -> dict[str, tuple[list[Path], list[int]]]:
    return {
        split: load_split(dataset_dir, split)
        for split in ("train", "val", "test")
    }


if __name__ == "__main__":
    dataset_path = Path("dataset") / "processed_dataset"

    splits = load_all_splits(dataset_path)

    for split_name, (paths, labels) in splits.items():
        print(
            f"{split_name}: "
            f"{len(paths)} images, "
            f"{len(set(labels))} classes"
        )