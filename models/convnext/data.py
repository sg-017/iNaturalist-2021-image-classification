import json
import random
import numpy as np
import timm
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

NUM_CLASSES = 500
EXPECTED_CLASS_TO_IDX = {f"{label:03d}": label for label in range(NUM_CLASSES)}

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image, target, self.samples[index][0]

def validate_imagefolder(dataset, split):
    if dataset.class_to_idx != EXPECTED_CLASS_TO_IDX:
        raise ValueError(f"{split} folders must map 000-499 to labels 0-499")
    expected = {"train": 40, "val": 10, "test": 10}[split]
    counts = [0] * NUM_CLASSES
    for _, label in dataset.samples:
        counts[label] += 1
    if any(count != expected for count in counts):
        raise ValueError(f"{split} must contain {expected} images per class")

def load_original_category_mapping(dataset_dir):
    with (dataset_dir / "selected_classes.json").open("r", encoding="utf-8") as file:
        classes = json.load(file).get("classes", [])
    mapping = {
        int(item["label"]): int(item["original_category_id"])
        for item in classes
    }
    if set(mapping) != set(range(NUM_CLASSES)):
        raise ValueError("selected_classes.json must contain all 500 labels")
    return mapping

def create_transforms(data_config, min_crop_scale):
    val_transform = timm.data.create_transform(**data_config, is_training=False)
    train_transform = timm.data.create_transform(
        **data_config,
        is_training=True,
        scale=(min_crop_scale, 1.0),
        color_jitter=0.1,
        auto_augment=None,
        re_prob=0.0,
        hflip=0.5,
        vflip=0.0,
    )
    return train_transform, val_transform

def build_datasets(dataset_dir, data_config, min_crop_scale):
    train_transform, val_transform = create_transforms(data_config, min_crop_scale)
    train = ImageFolder(dataset_dir / "train", transform=train_transform)
    val = ImageFolder(dataset_dir / "val", transform=val_transform)
    validate_imagefolder(train, "train")
    validate_imagefolder(val, "val")
    return train, val

def build_test_dataset(dataset_dir, data_config):
    transform = timm.data.create_transform(**data_config, is_training=False)
    test = ImageFolderWithPaths(dataset_dir / "test", transform=transform)
    validate_imagefolder(test, "test")
    return test

def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_loader(dataset, batch_size, num_workers, shuffle, pin_memory, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )
