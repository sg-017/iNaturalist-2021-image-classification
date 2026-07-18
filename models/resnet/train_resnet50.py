import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


METHODS = ("scratch", "linear_probe", "full_finetune")


def parse_args():
    parser = argparse.ArgumentParser(description="ResNet50 training")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augmentation",
        action="store_true",
        help="Use random crop and horizontal flip on training images.",
    )
    return parser.parse_args()

# Set random seeds for reproducible experiments
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Create training, validation and test data loaders
def make_loaders(data_root, batch_size, num_workers, augmentation):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if augmentation:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
        )
    eval_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    )

    train_set = datasets.ImageFolder(data_root / "train", train_transform)
    val_set = datasets.ImageFolder(data_root / "val", eval_transform)
    test_set = datasets.ImageFolder(data_root / "test", eval_transform)

    # Check whether all splits use the same class label
    if train_set.class_to_idx != val_set.class_to_idx:
        raise ValueError("Train and validation class folders do not match")
    if train_set.class_to_idx != test_set.class_to_idx:
        raise ValueError("Train and test class folders do not match")

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    loaders = {
        "train": DataLoader(train_set, shuffle=True, **common),
        "val": DataLoader(val_set, shuffle=False, **common),
        "test": DataLoader(test_set, shuffle=False, **common),
    }
    return loaders, train_set.classes

def make_model(method, num_classes):
    """
    Create a ResNet50 model for the selected training method

    Args:
        method: Training method
        num_classes: Number of output classes

    Returns:
        A configured ResNet50 model
    """

    # scratch: using random initialisation
    if method == "scratch":
        weights = None
    # linear-probe and full-finetune: start with ImageNet pretrained weights
    else:
        weights = models.ResNet50_Weights.DEFAULT    
    model = models.resnet50(weights=weights)

    if method == "linear_probe":
        # only updates the final classification layer
        for parameter in model.parameters():
            parameter.requires_grad = False
    # Replace the ImageNet classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

# Run one epoch and return the loss and accuracy
def run_epoch(model, loader, criterion, device, optimizer=None, linear_probe=False):
    training = optimizer is not None
    if training and linear_probe:
        # Keep frozen backbone fixed
        model.eval()
        model.fc.train()
    else:
        model.train(training)

    loss_sum = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total

# Save epoch results and plot the loss and accuracy curves
def plot_history(history, output_dir):
    table = pd.DataFrame(history)
    table.to_csv(output_dir / "history.csv", index=False)
    for name in ("loss", "accuracy"):
        plt.figure(figsize=(8, 5))
        plt.plot(table["epoch"], table[f"train_{name}"], label="Train")
        plt.plot(table["epoch"], table[f"val_{name}"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(name.capitalize())
        plt.title(f"ResNet50 {name.capitalize()}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_curve.png", dpi=200)
        plt.close()

def evaluate(model, loader, device, num_classes, output_dir):
    """
    Evaluate the trained model on the test set.

    Args:
        model: Trained ResNet50 model.
        loader: Test data loader.
        device: Device used for inference.
        num_classes: Number of species classes.
        output_dir: Directory used to save the confusion matrix.

    Returns:
        A dictionary containing the test metrics.
    """
    
    model.eval()
    labels_all, predictions_all = [], []
    top1_correct = top5_correct = total = 0
    started = time.perf_counter()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(1)
            k = min(5, num_classes)
            top5 = outputs.topk(k=k, dim=1).indices
            top1_correct += (predictions == labels).sum().item()
            top5_correct += (top5 == labels.unsqueeze(1)).any(1).sum().item()
            total += labels.size(0)
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())
    test_seconds = time.perf_counter() - started
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_all,
        predictions_all,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(
        labels_all, predictions_all, labels=list(range(num_classes))
    )
    np.save(output_dir / "confusion_matrix.npy", matrix)
    return {
        "test_top1_accuracy": top1_correct / total,
        "test_top5_accuracy": top5_correct / total,
        "overall_accuracy": top1_correct / total,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "test_seconds": test_seconds,
        "test_images": total,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, classes = make_loaders(
        args.data_root, args.batch_size, args.num_workers, args.augmentation
    )
    with (args.output_dir / "classes.json").open("w", encoding="utf-8") as file:
        json.dump(classes, file, indent=2, ensure_ascii=False)

    model = make_model(args.method, len(classes)).to(device)
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 0.0001 if args.method == "full_finetune" else 0.001
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    criterion = nn.CrossEntropyLoss()
    history = {key: [] for key in (
        "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"
    )}
    best_val_accuracy = -1.0
    best_epoch = 0
    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, loaders["train"], criterion, device, optimizer,
            linear_probe=args.method == "linear_probe",
        )
        val_loss, val_accuracy = run_epoch(
            model, loaders["val"], criterion, device
        )
        values = (epoch, train_loss, train_accuracy, val_loss, val_accuracy)
        for key, value in zip(history, values):
            history[key].append(value)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), args.output_dir / "best_model.pth")
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} | "
            f"train acc {train_accuracy:.4f} | val loss {val_loss:.4f} | "
            f"val acc {val_accuracy:.4f}"
        )

    training_seconds = time.perf_counter() - training_started
    plot_history(history, args.output_dir)
    model.load_state_dict(
        torch.load(args.output_dir / "best_model.pth", map_location=device)
    )
    metrics = evaluate(model, loaders["test"], device, len(classes), args.output_dir)
    metrics.update(
        {
            "model": "ResNet50",
            "method": args.method,
            "num_classes": len(classes),
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_val_accuracy,
            "training_seconds": training_seconds,
            "learning_rate": learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "augmentation": args.augmentation,
            "seed": args.seed,
            "device": str(device),
        }
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
