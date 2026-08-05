#!/usr/bin/env python3
"""Full fine-tuning of DINOv2 ViT-S/14 on an iNaturalist subset."""

from __future__ import annotations

import argparse
import csv
import getpass
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
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from transformers import AutoImageProcessor, Dinov2Model


MODEL_NAME = "facebook/dinov2-small"  # ViT-S/14, 21M-parameter backbone.
PATCH_SIZE = 14

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full fine-tuning of DINOv2 ViT-S/14 on an iNaturalist subset"
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--num-classes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Accumulate this many mini-batches before each optimizer update",
    )
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--backbone-lr", type=float, default=5e-5)
    parser.add_argument("--classifier-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.01,
        help="Final LR as a fraction of each parameter group's initial LR",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square crop size; it must be divisible by DINOv2's patch size 14",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16", "none"),
        default="float16",
        help="CUDA automatic mixed precision type",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Reduce backbone activation memory at the cost of extra computation",
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="Resume a complete training checkpoint"
    )
    parser.add_argument(
        "--test-only", action="store_true", help="Skip training and evaluate --checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.data_dir.parent / "outputs" / "dinov2-small"
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.num_classes < 2:
        raise ValueError("--num-classes must be at least 2")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.accumulation_steps < 1:
        raise ValueError("--accumulation-steps must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.image_size < PATCH_SIZE or args.image_size % PATCH_SIZE != 0:
        raise ValueError("--image-size must be a positive multiple of 14")
    if args.backbone_lr <= 0 or args.classifier_lr <= 0:
        raise ValueError("Learning rates must be positive")
    if args.weight_decay < 0 or args.warmup_epochs < 0:
        raise ValueError("Weight decay and warmup epochs cannot be negative")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.grad_clip < 0 or not 0.0 <= args.dropout < 1.0:
        raise ValueError("Invalid --grad-clip or --dropout")
    if args.test_only and args.checkpoint is None:
        raise ValueError("--test-only requires --checkpoint")
    if args.test_only and args.resume is not None:
        raise ValueError("--test-only cannot be combined with --resume")


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
    """Image dataset backed by one of train.json, val.json, or test.json."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        transform: Any,
        num_classes: int,
    ) -> None:
        self.data_dir = data_dir.expanduser().resolve()
        self.split = split
        self.transform = transform
        annotation_path = self.data_dir / f"{split}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

        with annotation_path.open("r", encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list):
            raise ValueError(f"{annotation_path} must contain a JSON list")
        if not records:
            raise ValueError(f"{annotation_path} contains no samples")

        self.records: list[dict[str, Any]] = records
        self.labels: list[int] = []
        for index, record in enumerate(self.records):
            if not isinstance(record, dict):
                raise ValueError(f"Record {index} in {annotation_path} is not an object")
            if "file_name" not in record or "label" not in record:
                raise ValueError(
                    f"Record {index} in {annotation_path} lacks file_name or label"
                )
            label = int(record["label"])
            if not 0 <= label < num_classes:
                raise ValueError(
                    f"Label {label} in record {index} is outside "
                    f"[0, {num_classes - 1}]"
                )
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.records)

    def image_path(self, file_name: str) -> Path:
        path = Path(file_name)
        if path.is_absolute():
            return path

        direct_path = self.data_dir / path
        if direct_path.is_file():
            return direct_path

        if path.parts and path.parts[0] == self.data_dir.name:
            return self.data_dir.parent / path
        return direct_path

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        record = self.records[index]
        path = self.image_path(str(record["file_name"]))
        try:
            with Image.open(path) as image:
                pixel_values = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {path}") from exc
        return pixel_values, self.labels[index], index


class DINOv2Classifier(nn.Module):
    """DINOv2 backbone plus a newly initialized 500-way linear head.

    The classifier consumes the final CLS token, matching the standard
    DINOv2 image-classification head.  Every backbone parameter remains
    trainable for full fine-tuning.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        dropout: float,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained(model_name)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        hidden_size = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=pixel_values)
        tokens = output.last_hidden_state
        cls_token = tokens[:, 0]
        return self.classifier(self.dropout(cls_token))

    def optimizer_parameter_groups(
        self,
        backbone_lr: float,
        classifier_lr: float,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        groups: dict[tuple[str, bool], list[nn.Parameter]] = {
            ("backbone", True): [],
            ("backbone", False): [],
            ("classifier", True): [],
            ("classifier", False): [],
        }
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            section = "classifier" if name.startswith("classifier.") else "backbone"
            # Do not decay bias or normalization scale parameters.
            apply_decay = parameter.ndim > 1 and not name.endswith(".bias")
            groups[(section, apply_decay)].append(parameter)

        parameter_groups: list[dict[str, Any]] = []
        for (section, apply_decay), parameters in groups.items():
            if not parameters:
                continue
            lr = classifier_lr if section == "classifier" else backbone_lr
            parameter_groups.append(
                {
                    "params": parameters,
                    "lr": lr,
                    "initial_lr": lr,
                    "weight_decay": weight_decay if apply_decay else 0.0,
                    "group_name": f"{section}_{'decay' if apply_decay else 'no_decay'}",
                }
            )
        return parameter_groups


def make_transforms(model_name: str, image_size: int):
    processor = AutoImageProcessor.from_pretrained(model_name)
    mean = processor.image_mean
    std = processor.image_std
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.6, 1.0),
                ratio=(0.75, 4.0 / 3.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                int(round(image_size / 0.875)),
                interpolation=InterpolationMode.BICUBIC,
            ),
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
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "drop_last": False,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def amp_settings(device: torch.device, amp_dtype: str) -> tuple[bool, torch.dtype]:
    enabled = device.type == "cuda" and amp_dtype != "none"
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    if enabled and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bfloat16; use --amp-dtype float16")
    return enabled, dtype


def make_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def set_cosine_lr(
    optimizer: torch.optim.Optimizer,
    update: int,
    total_updates: int,
    warmup_updates: int,
    min_lr_ratio: float,
) -> None:
    if warmup_updates > 0 and update < warmup_updates:
        factor = float(update + 1) / float(warmup_updates)
    else:
        progress = (update - warmup_updates) / max(1, total_updates - warmup_updates)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        factor = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    total_updates: int,
    warmup_updates: int,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    amp_enabled, autocast_dtype = amp_settings(device, args.amp_dtype)
    updates_per_epoch = math.ceil(len(loader) / args.accumulation_steps)
    update_in_epoch = 0
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    accumulation_divisor = args.accumulation_steps
    progress = tqdm(loader, desc=f"Train {epoch + 1}/{args.epochs}")

    for batch_index, (images, targets, _) in enumerate(progress):
        if batch_index % args.accumulation_steps == 0:
            global_update = epoch * updates_per_epoch + update_in_epoch
            accumulation_divisor = min(
                args.accumulation_steps, len(loader) - batch_index
            )
            set_cosine_lr(
                optimizer,
                global_update,
                total_updates,
                warmup_updates,
                args.min_lr_ratio,
            )

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=amp_enabled,
        ):
            logits = model(images)
            unscaled_loss = F.cross_entropy(
                logits,
                targets,
                label_smoothing=args.label_smoothing,
            )
            loss = unscaled_loss / accumulation_divisor
        scaler.scale(loss).backward()

        should_update = (
            (batch_index + 1) % args.accumulation_steps == 0
            or batch_index + 1 == len(loader)
        )
        if should_update:
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update_in_epoch += 1

        batch_size = targets.size(0)
        sample_count += batch_size
        loss_sum += unscaled_loss.detach().item() * batch_size
        correct += logits.detach().argmax(dim=1).eq(targets).sum().item()
        progress.set_postfix(
            loss=f"{loss_sum / sample_count:.4f}",
            top1=f"{correct / sample_count:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    if sample_count == 0:
        raise ValueError("Training dataset is empty")
    return {
        "loss": loss_sum / sample_count,
        "top1_accuracy": correct / sample_count,
    }


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: str,
    description: str,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    amp_enabled, autocast_dtype = amp_settings(device, amp_dtype)
    all_targets: list[torch.Tensor] = []
    all_probabilities: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    loss_sum = 0.0
    sample_count = 0

    for images, targets, indices in tqdm(loader, desc=description):
        images = images.to(device, non_blocking=True)
        device_targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=amp_enabled,
        ):
            logits = model(images)
        float_logits = logits.float()
        loss_sum += F.cross_entropy(
            float_logits, device_targets, reduction="sum"
        ).item()
        sample_count += targets.size(0)
        all_targets.append(targets.cpu())
        all_probabilities.append(float_logits.softmax(dim=1).cpu())
        all_indices.append(indices.cpu())

    if sample_count == 0:
        raise ValueError(f"{description} dataset is empty")
    return (
        loss_sum / sample_count,
        torch.cat(all_targets).numpy(),
        torch.cat(all_probabilities).numpy(),
        torch.cat(all_indices).numpy(),
    )


def compute_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    if targets.ndim != 1 or probabilities.ndim != 2:
        raise ValueError("Invalid target or probability array shape")
    if len(targets) != len(probabilities):
        raise ValueError("Targets and probabilities have different lengths")
    if probabilities.shape[1] != num_classes:
        raise ValueError("Probability width does not equal --num-classes")

    max_k = min(5, num_classes)
    top_order = np.argsort(-probabilities, axis=1)[:, :max_k]
    metrics = {
        f"top{k}_accuracy": float(
            np.mean(np.any(top_order[:, :k] == targets[:, None], axis=1))
        )
        for k in range(1, max_k + 1)
    }

    predictions = top_order[:, 0]
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    for class_id in range(num_classes):
        true_positive = int(
            np.sum((targets == class_id) & (predictions == class_id))
        )
        false_positive = int(
            np.sum((targets != class_id) & (predictions == class_id))
        )
        false_negative = int(
            np.sum((targets == class_id) & (predictions != class_id))
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    metrics.update(
        macro_precision=float(np.mean(precisions)),
        macro_recall=float(np.mean(recalls)),
        macro_f1=float(np.mean(f1_scores)),
    )
    return metrics


def save_test_outputs(
    output_dir: Path,
    dataset: INatJsonDataset,
    targets: np.ndarray,
    probabilities: np.ndarray,
    indices: np.ndarray,
    test_loss: float,
    num_classes: int,
) -> None:
    metrics = compute_metrics(targets, probabilities, num_classes)
    report_lines = [
        "DINOv2 ViT-S/14 test classification report",
        f"num_samples: {len(targets)}",
        f"num_classes: {num_classes}",
        f"test_loss: {test_loss:.6f}",
    ]
    report_lines.extend(
        f"{name}: {value:.6f} ({100.0 * value:.2f}%)"
        for name, value in metrics.items()
    )
    report = "\n".join(report_lines) + "\n"
    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    prediction_path = output_dir / "test_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "image_id",
                "file_name",
                "ground_truth",
                "ground_truth_padded",
                "predicted_label",
                "predicted_label_padded",
                "predicted_probability",
                "correct",
                "top5_labels",
                "top5_probabilities",
            ]
        )
        for target, probability, index in zip(targets, probabilities, indices):
            record = dataset.records[int(index)]
            top_indices = np.argsort(-probability)[: min(5, num_classes)]
            predicted = int(top_indices[0])
            writer.writerow(
                [
                    record.get("image_id", ""),
                    record["file_name"],
                    int(target),
                    f"{int(target):03d}",
                    predicted,
                    f"{predicted:03d}",
                    f"{float(probability[predicted]):.8f}",
                    int(predicted == int(target)),
                    json.dumps([int(label) for label in top_indices]),
                    json.dumps(
                        [float(probability[label]) for label in top_indices]
                    ),
                ]
            )

    print(report, end="")
    print(f"Saved report to {report_path}")
    print(f"Saved per-image predictions to {prediction_path}")


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
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
    best_val_top1: float,
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "best_val_top1": best_val_top1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": serializable_args(args),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    model: nn.Module,
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
        checkpoint = {"model": state_dict}
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def print_dataset_summary(dataset: INatJsonDataset, num_classes: int) -> None:
    counts = np.bincount(dataset.labels, minlength=num_classes)
    present = int(np.count_nonzero(counts))
    print(
        f"{dataset.split}: {len(dataset):,} images, {present}/{num_classes} classes, "
        f"per-class min={int(counts.min())}, max={int(counts.max())}"
    )

def build_model(args: argparse.Namespace, device: torch.device) -> DINOv2Classifier:
    model = DINOv2Classifier(
        model_name=args.model_name,
        num_classes=args.num_classes,
        dropout=args.dropout,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    return model.to(device)

def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    if device.type != "cuda" and args.amp_dtype != "none":
        print("AMP is CUDA-only in this script; running without AMP")

    args.output_dir = args.output_dir.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(serializable_args(args), file, indent=2)

    train_transform, eval_transform = make_transforms(
        args.model_name, args.image_size
    )
    test_set = INatJsonDataset(
        args.data_dir, "test", eval_transform, args.num_classes
    )
    print_dataset_summary(test_set, args.num_classes)
    test_loader = make_loader(
        test_set,
        args.eval_batch_size,
        args.num_workers,
        False,
        device,
        args.seed + 2,
    )

    model = build_model(args, device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters != total_parameters:
        raise RuntimeError("Some model parameters are frozen; expected full fine-tuning")
    print(
        f"Trainable parameters: {trainable_parameters:,}/{total_parameters:,} "
        "(full fine-tuning)"
    )

    if args.test_only:
        load_checkpoint(model, args.checkpoint, device)
    else:
        train_set = INatJsonDataset(
            args.data_dir, "train", train_transform, args.num_classes
        )
        val_set = INatJsonDataset(
            args.data_dir, "val", eval_transform, args.num_classes
        )
        print_dataset_summary(train_set, args.num_classes)
        print_dataset_summary(val_set, args.num_classes)
        train_loader = make_loader(
            train_set,
            args.batch_size,
            args.num_workers,
            True,
            device,
            args.seed,
        )
        val_loader = make_loader(
            val_set,
            args.eval_batch_size,
            args.num_workers,
            False,
            device,
            args.seed + 1,
        )

        optimizer = torch.optim.AdamW(
            model.optimizer_parameter_groups(
                args.backbone_lr,
                args.classifier_lr,
                args.weight_decay,
            ),
            betas=(0.9, 0.999),
        )
        amp_enabled, amp_dtype = amp_settings(device, args.amp_dtype)
        scaler = make_grad_scaler(amp_enabled and amp_dtype == torch.float16)
        start_epoch = 0
        best_val_top1 = -1.0

        if args.resume is not None:
            checkpoint = load_checkpoint(model, args.resume, device)
            if "optimizer" not in checkpoint or "epoch" not in checkpoint:
                raise ValueError("--resume requires a complete training checkpoint")
            optimizer.load_state_dict(checkpoint["optimizer"])
            if "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val_top1 = float(checkpoint.get("best_val_top1", -1.0))
            if start_epoch >= args.epochs:
                raise ValueError(
                    f"Checkpoint has completed {start_epoch} epochs, so --epochs "
                    f"must be greater than {start_epoch}"
                )

        updates_per_epoch = math.ceil(
            len(train_loader) / args.accumulation_steps
        )
        total_updates = args.epochs * updates_per_epoch
        warmup_updates = int(args.warmup_epochs * updates_per_epoch)
        history_path = args.output_dir / "training_history.jsonl"
        if args.resume is None:
            history_path.write_text("", encoding="utf-8")

        for epoch in range(start_epoch, args.epochs):
            started = time.time()
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                args,
                epoch,
                total_updates,
                warmup_updates,
            )
            val_loss, val_targets, val_probabilities, _ = predict(
                model,
                val_loader,
                device,
                args.amp_dtype,
                "Validation",
            )
            val_metrics = compute_metrics(
                val_targets, val_probabilities, args.num_classes
            )
            history_row = {
                "epoch": epoch + 1,
                "seconds": time.time() - started,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                "val_loss": val_loss,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            with history_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(history_row) + "\n")
            print(json.dumps(history_row, indent=2))

            current_top1 = val_metrics["top1_accuracy"]
            if current_top1 > best_val_top1:
                best_val_top1 = current_top1
                save_checkpoint(
                    args.output_dir / "best.pt",
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    best_val_top1,
                    args,
                )
            save_checkpoint(
                args.output_dir / "last.pt",
                model,
                optimizer,
                scaler,
                epoch,
                best_val_top1,
                args,
            )

        best_path = args.output_dir / "best.pt"
        if not best_path.is_file():
            raise RuntimeError("Training finished without creating best.pt")
        load_checkpoint(model, best_path, device)

    test_loss, targets, probabilities, indices = predict(
        model,
        test_loader,
        device,
        args.amp_dtype,
        "Test",
    )
    save_test_outputs(
        args.output_dir,
        test_set,
        targets,
        probabilities,
        indices,
        test_loss,
        args.num_classes,
    )


if __name__ == "__main__":
    main()
