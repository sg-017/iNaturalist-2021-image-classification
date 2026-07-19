#!/usr/bin/env python3
"""Full fine-tuning of ResNet-50 with PMG on an iNat2021 subset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, ResNetModel


MODEL_NAME = "microsoft/resnet-50"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full fine-tuning of ImageNet-pretrained ResNet-50 with PMG"
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--num-classes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument(
        "--resize-size", type=int, default=550,
        help="Square resize before the 448x448 crop (550 matches the PMG reference)",
    )
    parser.add_argument("--feature-size", type=int, default=512)
    parser.add_argument(
        "--backbone-lr", type=float, default=2e-4,
        help="Learning rate for every pretrained ResNet-50 parameter",
    )
    parser.add_argument(
        "--head-lr", type=float, default=2e-3,
        help="Learning rate for the newly initialized PMG heads",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--concat-loss-weight", type=float, default=2.0)
    parser.add_argument(
        "--grad-clip", type=float, default=0.0,
        help="Maximum gradient norm; 0 disables clipping",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Checkpoint used with --test-only",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class INatJsonDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, transform: Any, num_classes: int):
        self.data_dir = data_dir.expanduser().resolve()
        self.transform = transform
        json_path = self.data_dir / f"{split}.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {json_path}")
        with json_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        if not isinstance(raw, list):
            raise ValueError(f"{json_path} must contain a JSON list")
        self.records = raw
        for index, item in enumerate(self.records):
            if not isinstance(item, dict) or "file_name" not in item or "label" not in item:
                raise ValueError(
                    f"Record {index} in {json_path} must contain file_name and label"
                )
            label = int(item["label"])
            if not 0 <= label < num_classes:
                raise ValueError(
                    f"Label {label} in record {index} is outside [0, {num_classes - 1}]"
                )

    def __len__(self) -> int:
        return len(self.records)

    def _image_path(self, file_name: str) -> Path:
        path = Path(file_name)
        if path.is_absolute():
            return path
        candidate = self.data_dir / path
        if candidate.is_file():
            return candidate
        if path.parts and path.parts[0] == self.data_dir.name:
            return self.data_dir.parent / path
        return candidate

    def __getitem__(self, index: int):
        item = self.records[index]
        image_path = self._image_path(str(item["file_name"]))
        try:
            with Image.open(image_path) as image:
                pixel_values = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {image_path}") from exc
        return pixel_values, int(item["label"]), index


class BasicConv(nn.Module):
    """Conv-BN-ReLU block used by the original PMG implementation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=1, padding=padding, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(features)))


class PMGClassifier(nn.Module):
    """PMG heads on the final three feature stages of a Hugging Face ResNet-50."""

    def __init__(self, model_name: str, num_classes: int, feature_size: int):
        super().__init__()
        self.backbone = ResNetModel.from_pretrained(model_name)
        hidden_sizes = [int(value) for value in self.backbone.config.hidden_sizes]
        if hidden_sizes[-3:] != [512, 1024, 2048]:
            raise ValueError(
                "PMG ResNet-50 expects the final three hidden sizes to be "
                f"[512, 1024, 2048], but received {hidden_sizes[-3:]}"
            )

        branch_channels = 1024
        self.conv_block1 = nn.Sequential(
            BasicConv(512, feature_size, kernel_size=1, padding=0),
            BasicConv(feature_size, branch_channels, kernel_size=3, padding=1),
        )
        self.conv_block2 = nn.Sequential(
            BasicConv(1024, feature_size, kernel_size=1, padding=0),
            BasicConv(feature_size, branch_channels, kernel_size=3, padding=1),
        )
        self.conv_block3 = nn.Sequential(
            BasicConv(2048, feature_size, kernel_size=1, padding=0),
            BasicConv(feature_size, branch_channels, kernel_size=3, padding=1),
        )
        self.classifier1 = self._make_branch_classifier(
            branch_channels, feature_size, num_classes
        )
        self.classifier2 = self._make_branch_classifier(
            branch_channels, feature_size, num_classes
        )
        self.classifier3 = self._make_branch_classifier(
            branch_channels, feature_size, num_classes
        )
        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(branch_channels * 3),
            nn.Linear(branch_channels * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, num_classes),
        )
        self._initialize_pmg_heads()

    @staticmethod
    def _make_branch_classifier(
        in_features: int, feature_size: int, num_classes: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, num_classes),
        )

    def _initialize_pmg_heads(self) -> None:
        head_modules = [
            self.conv_block1, self.conv_block2, self.conv_block3,
            self.classifier1, self.classifier2, self.classifier3,
            self.classifier_concat,
        ]
        for head in head_modules:
            for module in head.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
        if outputs.hidden_states is None or len(outputs.hidden_states) < 4:
            raise RuntimeError("The ResNet backbone did not return its stage hidden states")
        stage2, stage3, stage4 = outputs.hidden_states[-3:]
        expected_channels = (512, 1024, 2048)
        actual_channels = (stage2.shape[1], stage3.shape[1], stage4.shape[1])
        if actual_channels != expected_channels:
            raise RuntimeError(
                f"Unexpected ResNet feature channels: {actual_channels}; "
                f"expected {expected_channels}"
            )

        feature1 = F.adaptive_max_pool2d(self.conv_block1(stage2), 1).flatten(1)
        feature2 = F.adaptive_max_pool2d(self.conv_block2(stage3), 1).flatten(1)
        feature3 = F.adaptive_max_pool2d(self.conv_block3(stage4), 1).flatten(1)
        logits1 = self.classifier1(feature1)
        logits2 = self.classifier2(feature2)
        logits3 = self.classifier3(feature3)
        concat_logits = self.classifier_concat(
            torch.cat((feature1, feature2, feature3), dim=1)
        )
        return logits1, logits2, logits3, concat_logits

    def optimizer_parameter_groups(self, args: argparse.Namespace) -> list[dict[str, Any]]:
        heads: list[nn.Module] = [
            self.conv_block1, self.conv_block2, self.conv_block3,
            self.classifier1, self.classifier2, self.classifier3,
            self.classifier_concat,
        ]
        head_parameters = [parameter for head in heads for parameter in head.parameters()]
        return [
            {
                "params": self.backbone.parameters(),
                "lr": args.backbone_lr,
                "initial_lr": args.backbone_lr,
                "group_name": "backbone",
            },
            {
                "params": head_parameters,
                "lr": args.head_lr,
                "initial_lr": args.head_lr,
                "group_name": "pmg_heads",
            },
        ]


def make_transforms(model_name: str, image_size: int, resize_size: int):
    processor = AutoImageProcessor.from_pretrained(model_name)
    mean, std = processor.image_mean, processor.image_std
    train_transform = transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(image_size, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def make_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
    drop_last: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def jigsaw_generator(images: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Shuffle non-overlapping patches using one permutation for the batch."""
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    height, width = images.shape[-2:]
    if height % grid_size or width % grid_size:
        raise ValueError(
            f"Image shape {(height, width)} must be divisible by grid size {grid_size}"
        )
    batch, channels = images.shape[:2]
    patch_h, patch_w = height // grid_size, width // grid_size
    patches = (
        images.reshape(batch, channels, grid_size, patch_h, grid_size, patch_w)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, grid_size * grid_size, channels, patch_h, patch_w)
    )
    permutation = torch.randperm(grid_size * grid_size, device=images.device)
    patches = patches[:, permutation]
    return (
        patches.reshape(batch, grid_size, grid_size, channels, patch_h, patch_w)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels, height, width)
        .contiguous()
    )


def set_cosine_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    epochs: int,
    warmup_epochs: float,
) -> None:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        factor = float(epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1.0, epochs - warmup_epochs)
        factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor


def optimizer_step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    grad_clip: float,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    model.train()
    set_cosine_lr(optimizer, epoch, args.epochs, args.warmup_epochs)
    amp_enabled = device.type == "cuda" and not args.no_amp
    loss_sums = np.zeros(4, dtype=np.float64)
    correct, count = 0, 0
    progress = tqdm(loader, desc=f"Train {epoch + 1}/{args.epochs}")

    for images, targets, _ in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        step_losses: list[float] = []

        for branch_index, grid_size in enumerate((8, 4, 2)):
            jigsaw_images = jigsaw_generator(images, grid_size)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                branch_logits = model(jigsaw_images)[branch_index]
                loss = F.cross_entropy(
                    branch_logits, targets, label_smoothing=args.label_smoothing
                )
            optimizer_step(loss, model, optimizer, scaler, args.grad_clip)
            step_losses.append(float(loss.detach()))

        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            concat_logits = model(images)[3]
            concat_loss = F.cross_entropy(
                concat_logits, targets, label_smoothing=args.label_smoothing
            ) * args.concat_loss_weight
        optimizer_step(concat_loss, model, optimizer, scaler, args.grad_clip)
        step_losses.append(float(concat_loss.detach()))

        batch_size = targets.size(0)
        count += batch_size
        loss_sums += np.asarray(step_losses) * batch_size
        correct += concat_logits.detach().argmax(dim=1).eq(targets).sum().item()
        progress.set_postfix(
            loss=f"{loss_sums.sum() / count:.4f}",
            acc=f"{correct / count:.4f}",
        )

    if count == 0:
        raise ValueError("Training loader produced no batches; reduce --batch-size")
    return {
        "loss": float(loss_sums.sum() / count),
        "branch1_loss": float(loss_sums[0] / count),
        "branch2_loss": float(loss_sums[1] / count),
        "branch3_loss": float(loss_sums[2] / count),
        "concat_weighted_loss": float(loss_sums[3] / count),
        "concat_top1_accuracy": correct / count,
    }


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, description: str):
    model.eval()
    all_targets, all_probabilities, all_indices = [], [], []
    loss_sum, count = 0.0, 0
    for images, targets, indices in tqdm(loader, desc=description):
        images = images.to(device, non_blocking=True)
        device_targets = targets.to(device, non_blocking=True)
        logits = model(images)
        combined_logits = logits[0] + logits[1] + logits[2] + logits[3]
        loss_sum += F.cross_entropy(
            combined_logits, device_targets, reduction="sum"
        ).item()
        count += targets.size(0)
        all_targets.append(targets)
        all_probabilities.append(combined_logits.float().softmax(dim=1).cpu())
        all_indices.append(indices)
    if count == 0:
        raise ValueError(f"{description} dataset is empty")
    return (
        loss_sum / count,
        torch.cat(all_targets).numpy(),
        torch.cat(all_probabilities).numpy(),
        torch.cat(all_indices).numpy(),
    )


def compute_metrics(
    targets: np.ndarray, probabilities: np.ndarray, num_classes: int
) -> dict[str, float]:
    max_k = min(5, num_classes)
    top_order = np.argsort(-probabilities, axis=1)[:, :max_k]
    metrics = {
        f"top{k}_accuracy": float(
            np.mean(np.any(top_order[:, :k] == targets[:, None], axis=1))
        )
        for k in range(1, max_k + 1)
    }
    predictions = top_order[:, 0]
    precision, recall, f1 = [], [], []
    for class_id in range(num_classes):
        true_positive = int(np.sum((targets == class_id) & (predictions == class_id)))
        false_positive = int(np.sum((targets != class_id) & (predictions == class_id)))
        false_negative = int(np.sum((targets == class_id) & (predictions != class_id)))
        p = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        r = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2.0 * p * r / (p + r) if p + r else 0.0)
    metrics.update(
        macro_precision=float(np.mean(precision)),
        macro_recall=float(np.mean(recall)),
        macro_f1=float(np.mean(f1)),
    )
    return metrics


def save_test_outputs(
    output_dir: Path,
    dataset: INatJsonDataset,
    targets: np.ndarray,
    probabilities: np.ndarray,
    indices: np.ndarray,
    test_loss: float,
) -> None:
    metrics = compute_metrics(targets, probabilities, probabilities.shape[1])
    lines = [
        "Test classification report (combined PMG logits)",
        f"test_loss: {test_loss:.6f}",
        f"num_samples: {len(targets)}",
    ]
    lines.extend(
        f"{name}: {value:.6f} ({100.0 * value:.2f}%)"
        for name, value in metrics.items()
    )
    report = "\n".join(lines) + "\n"
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    with (output_dir / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "image_id", "file_name", "ground_truth", "predicted_label",
                "predicted_probability", "top5_labels", "top5_probabilities",
            ]
        )
        for target, probability, index in zip(targets, probabilities, indices):
            record = dataset.records[int(index)]
            top = np.argsort(-probability)[: min(5, len(probability))]
            writer.writerow(
                [
                    record.get("image_id", ""),
                    record["file_name"],
                    f"{int(target):03d}",
                    f"{int(top[0]):03d}",
                    f"{float(probability[top[0]]):.8f}",
                    json.dumps([f"{int(label):03d}" for label in top]),
                    json.dumps([round(float(probability[label]), 8) for label in top]),
                ]
            )
    print(report, end="")


def serializable_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_top1": best_val,
            "args": serializable_arguments(args),
        },
        path,
    )


def load_model_weights(model: nn.Module, path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    return checkpoint


def validate_args(args: argparse.Namespace) -> None:
    if args.num_classes < 2:
        raise ValueError("--num-classes must be at least 2")
    if args.epochs < 1 or args.batch_size < 2 or args.eval_batch_size < 1:
        raise ValueError("--epochs and batch sizes must be positive; train batch size must be >= 2")
    if args.image_size < 8 or args.image_size % 8:
        raise ValueError("--image-size must be at least 8 and divisible by 8")
    if args.resize_size < args.image_size:
        raise ValueError("--resize-size must be greater than or equal to --image-size")
    if args.feature_size < 1 or args.num_workers < 0:
        raise ValueError("--feature-size must be positive and --num-workers cannot be negative")
    if args.backbone_lr <= 0 or args.head_lr <= 0:
        raise ValueError("Learning rates must be positive")
    if args.weight_decay < 0 or args.momentum < 0:
        raise ValueError("--weight-decay and --momentum cannot be negative")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.concat_loss_weight <= 0 or args.warmup_epochs < 0 or args.grad_clip < 0:
        raise ValueError("Loss weight must be positive; warmup and grad clip cannot be negative")
    if args.test_only and args.checkpoint is None:
        raise ValueError("--test-only requires --checkpoint")
    if args.resume is not None and args.test_only:
        raise ValueError("--resume and --test-only cannot be used together")


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(serializable_arguments(args), file, indent=2)

    train_transform, eval_transform = make_transforms(
        args.model_name, args.image_size, args.resize_size
    )
    test_set = INatJsonDataset(args.data_dir, "test", eval_transform, args.num_classes)
    test_loader = make_loader(
        test_set, args.eval_batch_size, args.num_workers, False, device, args.seed
    )
    model = PMGClassifier(
        args.model_name, args.num_classes, args.feature_size
    ).to(device)

    if args.test_only:
        load_model_weights(model, args.checkpoint, device)
    else:
        train_set = INatJsonDataset(
            args.data_dir, "train", train_transform, args.num_classes
        )
        val_set = INatJsonDataset(
            args.data_dir, "val", eval_transform, args.num_classes
        )
        train_loader = make_loader(
            train_set, args.batch_size, args.num_workers, True, device,
            args.seed, drop_last=True,
        )
        val_loader = make_loader(
            val_set, args.eval_batch_size, args.num_workers, False, device, args.seed
        )
        optimizer = torch.optim.SGD(
            model.optimizer_parameter_groups(args),
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        amp_enabled = device.type == "cuda" and not args.no_amp
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        start_epoch, best_val = 0, -1.0
        if args.resume is not None:
            checkpoint = load_model_weights(model, args.resume, device)
            if not isinstance(checkpoint, dict):
                raise ValueError("--resume requires a complete training checkpoint")
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val = float(checkpoint.get("best_val_top1", -1.0))

        history_path = args.output_dir / "training_history.jsonl"
        for epoch in range(start_epoch, args.epochs):
            started = time.time()
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scaler, device, args, epoch
            )
            val_loss, val_targets, val_probabilities, _ = predict(
                model, val_loader, device, "Validation"
            )
            val_metrics = compute_metrics(
                val_targets, val_probabilities, args.num_classes
            )
            row = {
                "epoch": epoch + 1,
                "seconds": time.time() - started,
                "learning_rates": {
                    group.get("group_name", str(index)): group["lr"]
                    for index, group in enumerate(optimizer.param_groups)
                },
                **{f"train_{key}": value for key, value in train_metrics.items()},
                "val_loss": val_loss,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            with history_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row) + "\n")
            print(json.dumps(row, indent=2))

            current_top1 = val_metrics["top1_accuracy"]
            if current_top1 > best_val:
                best_val = current_top1
                save_checkpoint(
                    args.output_dir / "best.pt",
                    model, optimizer, scaler, epoch, best_val, args,
                )
            save_checkpoint(
                args.output_dir / "last.pt",
                model, optimizer, scaler, epoch, best_val, args,
            )

        best_path = args.output_dir / "best.pt"
        if not best_path.is_file():
            raise RuntimeError(
                "No best checkpoint was produced; check --epochs and --resume"
            )
        load_model_weights(model, best_path, device)

    test_loss, targets, probabilities, indices = predict(
        model, test_loader, device, "Test"
    )
    save_test_outputs(
        args.output_dir, test_set, targets, probabilities, indices, test_loss
    )


if __name__ == "__main__":
    main()
