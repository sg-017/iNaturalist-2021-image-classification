import json
import random
import shutil
from pathlib import Path
from collections import defaultdict


SOURCE_DATA = Path("dataset") / "source_data"

TRAIN_IMAGES = SOURCE_DATA / "train_mini"
TEST_IMAGES = SOURCE_DATA / "val"

TRAIN_JSON = SOURCE_DATA / "train_mini.json"
TEST_JSON = SOURCE_DATA / "val.json"

OUTPUT = Path("dataset") / "processed_dataset"


# Dataset settings
NUM_CLASSES = 500
TRAIN_PER_CLASS = 40
VAL_PER_CLASS = 10
TEST_PER_CLASS = 10
RANDOM_SEED = 42



def load_json(file_path):
    print(f"Loading {file_path}...")

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


# Group images by category
def group_images_by_category(data):
    category_images = defaultdict(list)

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        category_images[category_id].append(image_id)
    return category_images



# image mapping for access
def create_image_mapping(data):
    image_mapping = {}
    for image in data["images"]:
        image_id = image["id"]
        image_mapping[image_id] = image
    return image_mapping



# actual image file
def find_image(image_root, file_name):
    file_path = Path(file_name)
    possible_paths = [
        SOURCE_DATA / file_path,
        image_root / file_path,
        image_root / file_path.name
    ]

    if len(file_path.parts) > 1:
        possible_paths.append(
            image_root / Path(*file_path.parts[1:])
        )
    for path in possible_paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Cannot find image: {file_name}"
    )


def copy_image(source_path, split, label, image_id):
    class_folder = OUTPUT / split / f"{label:03d}"
    class_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    
    new_file_name = f"{image_id}_{source_path.name}" # Add image ID to prevent duplicate file names

    destination = class_folder / new_file_name

    shutil.copy2(source_path, destination)

    relative_path = destination.relative_to(OUTPUT)   # Save relative path in JSON

    return str(relative_path)



def save_json(data, file_path):
    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


def prepare_dataset():  # main function 
    train_data = load_json(TRAIN_JSON)
    test_data = load_json(TEST_JSON)
    print("JSON files loaded successfully.")

    train_images_by_category = group_images_by_category(
        train_data
    )
    test_images_by_category = group_images_by_category(
        test_data
    )
    train_image_mapping = create_image_mapping(
        train_data
    )
    test_image_mapping = create_image_mapping(
        test_data
    )

    valid_categories = []

    for category_id in train_images_by_category:
        train_count = len(
            train_images_by_category[category_id]
        )
        test_count = len(
            test_images_by_category.get(category_id, [])
        )

        if train_count >= 50 and test_count >= 10:
            valid_categories.append(category_id)


    print(
        f"Valid categories found: "
        f"{len(valid_categories)}"
    )


    if len(valid_categories) < NUM_CLASSES:
        raise ValueError(
            "Not enough valid categories."
        )



    random.seed(RANDOM_SEED)
    valid_categories.sort()

    selected_categories = random.sample(
        valid_categories,
        NUM_CLASSES
    )

    
    selected_categories.sort()

    if OUTPUT.exists():
        print("Removing old processed dataset...")
        shutil.rmtree(OUTPUT)

    (OUTPUT / "train").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / "val").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / "test").mkdir(
        parents=True,
        exist_ok=True
    )


    train_records = []
    val_records = []
    test_records = []

    selected_classes = []

    print("\nStarting data preprocessing...\n")


    for label, category_id in enumerate(
        selected_categories
    ):
        train_ids = list(
            train_images_by_category[category_id]
        )
        test_ids = list(
            test_images_by_category[category_id]
        )
        random.shuffle(train_ids)
        random.shuffle(test_ids)

        selected_train = train_ids[:TRAIN_PER_CLASS]
        selected_val = train_ids[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS]
        selected_test = test_ids[:TEST_PER_CLASS]

        selected_classes.append({
            "label": label,
            "original_category_id": category_id
        })

        for image_id in selected_train:
            image_info = train_image_mapping[image_id]
            source_path = find_image(
                TRAIN_IMAGES,
                image_info["file_name"]
            )
            new_path = copy_image(
                source_path,
                "train",
                label,
                image_id
            )
            train_records.append({
                "image_id": image_id,
                "file_name": new_path,
                "label": label,
                "original_category_id": category_id
            })


# valid images

        for image_id in selected_val:
            image_info = train_image_mapping[image_id]
            source_path = find_image(
                TRAIN_IMAGES,
                image_info["file_name"]
            )
            new_path = copy_image(
                source_path,
                "val",
                label,
                image_id
            )
            val_records.append({
                "image_id": image_id,
                "file_name": new_path,
                "label": label,
                "original_category_id": category_id
            })


       
# Test images
        for image_id in selected_test:
            image_info = test_image_mapping[image_id]
            source_path = find_image(
                TEST_IMAGES,
                image_info["file_name"]
            )
            new_path = copy_image(
                source_path,
                "test",
                label,
                image_id
            )
            test_records.append({
                "image_id": image_id,
                "file_name": new_path,
                "label": label,
                "original_category_id": category_id
            })

# progress update
        if (label + 1) % 25 == 0:
            print(
                f"Completed "
                f"{label + 1}/{NUM_CLASSES} classes"
            )
    save_json(
        train_records,
        OUTPUT / "train.json"
    )
    save_json(
        val_records,
        OUTPUT / "val.json"
    )
    save_json(
        test_records,
        OUTPUT / "test.json"
    )
    class_info = {
        "random_seed": RANDOM_SEED,
        "num_classes": NUM_CLASSES,
        "train_per_class": TRAIN_PER_CLASS,
        "val_per_class": VAL_PER_CLASS,
        "test_per_class": TEST_PER_CLASS,
        "classes": selected_classes
    }
    save_json(
        class_info,
        OUTPUT / "selected_classes.json"
    )
 
    print("\nFinal dataset check")


    print(f"Classes: {len(selected_classes)}")
    print(f"Training images: {len(train_records)}")
    print(f"Validation images: {len(val_records)}")
    print(f"Test images: {len(test_records)}")


    assert len(selected_classes) == 500
    assert len(train_records) == 20000
    assert len(val_records) == 5000
    assert len(test_records) == 5000


    print("\nAll checks passed!")
    print(f"Dataset saved to: {OUTPUT}")


if __name__ == "__main__":
    prepare_dataset()