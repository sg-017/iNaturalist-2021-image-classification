#!/usr/bin/env python3
"""Full fine-tuning of ImageNet-1K ResNet-50 with GAP + iSQRT-COV fusion."""

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
        description=(
            "Full fine-tuning of Hugging Face ResNet-50 with "
            "GAP + iSQRT-COV fusion"
        )
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--num-classes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Use one LR for all parameters instead of the grouped LRs",
    )
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--cov-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument(
        "--fusion-scale-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the learnable GAP/iSQRT-COV fusion scale",
    )
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA AMP")
    parser.add_argument(
        "--resume", type=Path, default=None, help="Resume a training checkpoint"
    )
    parser.add_argument(
        "--test-only", action="store_true", help="Only evaluate --checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--cov-dim",
        type=int,
        default=256,
        help="Channels after the 1x1 reduction before covariance pooling",
    )
    parser.add_argument(
        "--sqrt-iters",
        type=int,
        default=5,
        help="Number of Newton-Schulz matrix-square-root iterations",
    )
    parser.add_argument(
        "--cov-eps",
        type=float,
        default=1e-5,
        help="Minimum trace used when normalizing a covariance matrix",
    )
    parser.add_argument(
        "--cov-initial-scale",
        type=float,
        default=0.05,
        help="Initial iSQRT-COV logit contribution; must be in (0, 1)",
    )
    parser.add_argument(
        "--cov-aux-loss-weight",
        type=float,
        default=0.3,
        help="Weight of the iSQRT-COV-only auxiliary cross-entropy loss",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_classes < 2:
        raise ValueError("--num-classes must be >= 2")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("Batch sizes must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.image_size < 1:
        raise ValueError("--image-size must be >= 1")
    if args.cov_dim < 2:
        raise ValueError("--cov-dim must be >= 2")
    if args.sqrt_iters < 1:
        raise ValueError("--sqrt-iters must be >= 1")
    if args.cov_eps <= 0:
        raise ValueError("--cov-eps must be > 0")
    if not 0.0 < args.cov_initial_scale < 1.0:
        raise ValueError("--cov-initial-scale must be strictly between 0 and 1")
    if args.cov_aux_loss_weight < 0:
        raise ValueError("--cov-aux-loss-weight must be >= 0")
    rates = [
        args.backbone_lr,
        args.cov_lr,
        args.classifier_lr,
        args.fusion_scale_lr,
    ]
    if args.lr is not None:
        rates.append(args.lr)
    if any(rate <= 0 for rate in rates):
        raise ValueError("All learning rates must be > 0")
    if args.weight_decay < 0 or args.warmup_epochs < 0:
        raise ValueError("Weight decay and warmup epochs must be >= 0")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.grad_clip < 0:
        raise ValueError("--grad-clip must be >= 0")
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


class INatJsonDataset(Dataset):
    def __init__(
        self, data_dir: Path, split: str, transform: Any, num_classes: int
    ) -> None:
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
            if (
                not isinstance(item, dict)
                or "file_name" not in item
                or "label" not in item
            ):
                raise ValueError(
                    f"Record {index} in {json_path} lacks file_name or label"
                )
            label = int(item["label"])
            if not 0 <= label < num_classes:
                raise ValueError(
                    f"Label {label} in record {index} is outside "
                    f"[0, {num_classes - 1}]"
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
        # Also accept annotations beginning with "processed_dataset/".
        if path.parts and path.parts[0] == self.data_dir.name:
            return self.data_dir.parent / path
        return candidate

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        item = self.records[index]
        image_path = self._image_path(str(item["file_name"]))
        try:
            with Image.open(image_path) as image:
                pixel_values = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {image_path}") from exc
        return pixel_values, int(item["label"]), index


class ISqrtCovariancePool(nn.Module):
    """Fast MPN-COV (iSQRT-COV) representation layer.

    This follows the official layer's operations but relies on native autograd:
    channel reduction, biased sample covariance (division by H*W), trace
    normalization, Newton-Schulz square-root iterations, and upper-triangular
    vectorization.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        reduction_dim: int = 256,
        num_iterations: int = 5,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.reduction_dim = reduction_dim
        self.num_iterations = num_iterations
        self.eps = eps
        self.reduction = nn.Sequential(
            nn.Conv2d(input_dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.ReLU(inplace=True),
        )
        triangle = torch.triu_indices(reduction_dim, reduction_dim)
        self.register_buffer("triangle_rows", triangle[0], persistent=False)
        self.register_buffer("triangle_cols", triangle[1], persistent=False)
        self.output_dim = reduction_dim * (reduction_dim + 1) // 2
        self.reset_parameters()

    def reset_parameters(self) -> None:
        convolution = self.reduction[0]
        normalization = self.reduction[1]
        nn.init.kaiming_normal_(
            convolution.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.ones_(normalization.weight)
        nn.init.zeros_(normalization.bias)

    def _covariance_pool(self, features: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = features.shape
        spatial_size = height * width
        if spatial_size < 2:
            raise ValueError(
                "iSQRT-COV requires at least two spatial feature positions"
            )
        flattened = features.reshape(batch_size, channels, spatial_size)
        centered = flattened - flattened.mean(dim=2, keepdim=True)
        return centered.bmm(centered.transpose(1, 2)) / float(spatial_size)

    def _matrix_square_root(self, covariance: torch.Tensor) -> torch.Tensor:
        batch_size, dimension, _ = covariance.shape
        identity = torch.eye(
            dimension, dtype=covariance.dtype, device=covariance.device
        ).unsqueeze(0)
        identity = identity.expand(batch_size, -1, -1)

        # For a positive semidefinite matrix, division by the trace places all
        # eigenvalues in [0, 1], the convergence region of Newton-Schulz.
        trace = covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        trace = trace.clamp_min(self.eps)
        normalized = covariance / trace[:, None, None]
        y = normalized
        z = identity
        for _ in range(self.num_iterations):
            update = 0.5 * (3.0 * identity - z.bmm(y))
            y = y.bmm(update)
            z = update.bmm(z)
        square_root = y * trace.sqrt()[:, None, None]
        # Remove tiny asymmetry accumulated by floating-point batched matmuls.
        return 0.5 * (square_root + square_root.transpose(1, 2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected Bx{self.input_dim}xHxW features, got "
                f"{tuple(features.shape)}"
            )

        # Newton-Schulz iteration is much less stable in float16.  Explicitly
        # leave autocast for the entire newly initialized covariance head.
        with torch.autocast(device_type=features.device.type, enabled=False):
            reduced = self.reduction(features.float())
            covariance = self._covariance_pool(reduced)
            square_root = self._matrix_square_root(covariance)
            return square_root[:, self.triangle_rows, self.triangle_cols]


class ResNet50GAPISqrtCovClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        cov_dim: int,
        sqrt_iters: int,
        cov_eps: float,
        cov_initial_scale: float,
    ) -> None:
        super().__init__()
        self.backbone = ResNetModel.from_pretrained(model_name)
        input_dim = int(self.backbone.config.hidden_sizes[-1])
        self.cov_pool = ISqrtCovariancePool(
            input_dim=input_dim,
            reduction_dim=cov_dim,
            num_iterations=sqrt_iters,
            eps=cov_eps,
        )
        self.cov_dropout = nn.Dropout(p=0.2)
        self.gap_classifier = nn.Linear(input_dim, num_classes)
        self.cov_classifier = nn.Linear(self.cov_pool.output_dim, num_classes)
        initial_logit = math.log(
            cov_initial_scale / (1.0 - cov_initial_scale)
        )
        self.cov_scale_logit = nn.Parameter(torch.tensor(initial_logit))
        nn.init.normal_(self.gap_classifier.weight, std=0.01)
        nn.init.zeros_(self.gap_classifier.bias)
        nn.init.normal_(self.cov_classifier.weight, std=0.01)
        nn.init.zeros_(self.cov_classifier.bias)

    @property
    def cov_scale(self) -> torch.Tensor:
        """Current constrained contribution of the iSQRT-COV branch."""
        return torch.sigmoid(self.cov_scale_logit)

    def forward(
        self, pixel_values: torch.Tensor, return_branch_logits: bool = False
    ):
        features = self.backbone(pixel_values=pixel_values).last_hidden_state
        gap_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        gap_logits = self.gap_classifier(gap_features)

        cov_features = self.cov_pool(features)
        # The covariance representation is FP32; keeping this large classifier
        # in FP32 avoids silently re-entering autocast in the caller.
        with torch.autocast(device_type=features.device.type, enabled=False):
            cov_features = self.cov_dropout(cov_features.float())
            cov_logits = self.cov_classifier(cov_features)
            fused_logits = gap_logits.float() + self.cov_scale * cov_logits
        if return_branch_logits:
            return fused_logits, gap_logits, cov_logits
        return fused_logits

    def optimizer_parameter_groups(
        self, args: argparse.Namespace
    ) -> list[dict[str, Any]]:
        if args.lr is not None:
            return [
                {
                    "params": self.parameters(),
                    "lr": args.lr,
                    "initial_lr": args.lr,
                    "group_name": "all",
                }
            ]
        return [
            {
                "params": self.backbone.parameters(),
                "lr": args.backbone_lr,
                "initial_lr": args.backbone_lr,
                "group_name": "backbone",
            },
            {
                "params": self.cov_pool.parameters(),
                "lr": args.cov_lr,
                "initial_lr": args.cov_lr,
                "group_name": "covariance_head",
            },
            {
                "params": list(self.gap_classifier.parameters())
                + list(self.cov_classifier.parameters()),
                "lr": args.classifier_lr,
                "initial_lr": args.classifier_lr,
                "group_name": "classifiers",
            },
            {
                "params": [self.cov_scale_logit],
                "lr": args.fusion_scale_lr,
                "initial_lr": args.fusion_scale_lr,
                "weight_decay": 0.0,
                "group_name": "fusion_scale",
            },
        ]


def make_transforms(model_name: str, image_size: int):
    processor = AutoImageProcessor.from_pretrained(model_name)
    mean, std = processor.image_mean, processor.image_std
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(round(image_size / 0.875))),
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
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
    )


def cosine_lr(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> None:
    if warmup_steps > 0 and step < warmup_steps:
        factor = float(step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        # Compatibility with PyTorch versions using the older CUDA AMP API.
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    total_steps: int,
    warmup_steps: int,
) -> dict[str, float]:
    model.train()
    loss_sum, fused_loss_sum, cov_aux_loss_sum = 0.0, 0.0, 0.0
    correct, count = 0, 0
    amp_enabled = device.type == "cuda" and not args.no_amp
    progress = tqdm(loader, desc=f"Train {epoch + 1}/{args.epochs}")
    for batch_index, (images, targets, _) in enumerate(progress):
        global_step = epoch * len(loader) + batch_index
        cosine_lr(optimizer, global_step, total_steps, warmup_steps)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            logits, _, cov_logits = model(images, return_branch_logits=True)
            fused_loss = F.cross_entropy(
                logits, targets, label_smoothing=args.label_smoothing
            )
            cov_aux_loss = F.cross_entropy(
                cov_logits, targets, label_smoothing=args.label_smoothing
            )
            loss = fused_loss + args.cov_aux_loss_weight * cov_aux_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        current_batch_size = targets.size(0)
        count += current_batch_size
        loss_sum += loss.item() * current_batch_size
        fused_loss_sum += fused_loss.item() * current_batch_size
        cov_aux_loss_sum += cov_aux_loss.item() * current_batch_size
        correct += logits.argmax(dim=1).eq(targets).sum().item()
        progress.set_postfix(
            loss=f"{loss_sum / count:.4f}",
            fused=f"{fused_loss_sum / count:.4f}",
            acc=f"{correct / count:.4f}",
        )
    if count == 0:
        raise ValueError("Training dataset is empty")
    return {
        "loss": loss_sum / count,
        "fused_loss": fused_loss_sum / count,
        "cov_aux_loss": cov_aux_loss_sum / count,
        "top1_accuracy": correct / count,
    }


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_targets, all_probabilities, all_indices = [], [], []
    loss_sum, count = 0.0, 0
    for images, targets, indices in tqdm(loader, desc=description):
        images = images.to(device, non_blocking=True)
        device_targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += F.cross_entropy(
            logits, device_targets, reduction="sum"
        ).item()
        count += targets.size(0)
        all_targets.append(targets)
        all_probabilities.append(logits.softmax(dim=1).cpu())
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
    if probabilities.ndim != 2 or probabilities.shape[0] != targets.shape[0]:
        raise ValueError("Targets and probabilities have incompatible shapes")
    if probabilities.shape[1] != num_classes:
        raise ValueError("Probability class dimension does not match num_classes")

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
        true_positive = int(
            np.sum((targets == class_id) & (predictions == class_id))
        )
        false_positive = int(
            np.sum((targets != class_id) & (predictions == class_id))
        )
        false_negative = int(
            np.sum((targets == class_id) & (predictions != class_id))
        )
        class_precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        class_recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        class_f1 = (
            2.0 * class_precision * class_recall
            / (class_precision + class_recall)
            if class_precision + class_recall
            else 0.0
        )
        precision.append(class_precision)
        recall.append(class_recall)
        f1.append(class_f1)
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
        "Test classification report",
        f"test_loss: {test_loss:.6f}",
        f"num_samples: {len(targets)}",
    ]
    lines.extend(
        f"{name}: {value:.6f} ({100.0 * value:.2f}%)"
        for name, value in metrics.items()
    )
    report = "\n".join(lines) + "\n"
    (output_dir / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"test_loss": test_loss, "num_samples": len(targets), **metrics},
            file,
            indent=2,
        )

    with (output_dir / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "image_id",
                "file_name",
                "ground_truth",
                "predicted_label",
                "predicted_probability",
                "top5_labels",
                "top5_probabilities",
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
                    json.dumps(
                        [round(float(probability[label]), 8) for label in top]
                    ),
                ]
            )
    print(report, end="")


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
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
            "args": serializable_args(args),
        },
        path,
    )


def load_model_weights(
    model: nn.Module, path: Path, device: torch.device
) -> dict[str, Any] | Any:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    # Checkpoints are produced locally by this script.  weights_only=False is
    # explicit for compatibility with PyTorch 2.6 and older checkpoint metadata.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state)
    return checkpoint


def save_fusion_summary(
    output_dir: Path, model: ResNet50GAPISqrtCovClassifier
) -> None:
    """Save the learned residual contribution of the covariance branch."""
    payload = {
        "formula": "fused_logits = gap_logits + cov_scale * cov_logits",
        "cov_scale": float(model.cov_scale.detach().float().cpu().item()),
        "cov_scale_logit": float(
            model.cov_scale_logit.detach().float().cpu().item()
        ),
    }
    with (output_dir / "fusion_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(payload, file, indent=2)


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(serializable_args(args), file, indent=2)

    train_transform, eval_transform = make_transforms(
        args.model_name, args.image_size
    )
    test_set = INatJsonDataset(
        args.data_dir, "test", eval_transform, args.num_classes
    )
    test_loader = make_loader(
        test_set,
        args.eval_batch_size,
        args.num_workers,
        shuffle=False,
        device=device,
    )
    model = ResNet50GAPISqrtCovClassifier(
        model_name=args.model_name,
        num_classes=args.num_classes,
        cov_dim=args.cov_dim,
        sqrt_iters=args.sqrt_iters,
        cov_eps=args.cov_eps,
        cov_initial_scale=args.cov_initial_scale,
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
            train_set,
            args.batch_size,
            args.num_workers,
            shuffle=True,
            device=device,
        )
        val_loader = make_loader(
            val_set,
            args.eval_batch_size,
            args.num_workers,
            shuffle=False,
            device=device,
        )
        optimizer = torch.optim.AdamW(
            model.optimizer_parameter_groups(args),
            weight_decay=args.weight_decay,
        )
        amp_enabled = device.type == "cuda" and not args.no_amp
        scaler = make_grad_scaler(amp_enabled)
        start_epoch, best_val = 0, -1.0
        if args.resume is not None:
            checkpoint = load_model_weights(model, args.resume, device)
            if not isinstance(checkpoint, dict):
                raise ValueError("--resume requires a full training checkpoint")
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val = float(checkpoint.get("best_val_top1", -1.0))

        total_steps = args.epochs * len(train_loader)
        warmup_steps = int(args.warmup_epochs * len(train_loader))
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
                total_steps,
                warmup_steps,
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
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    best_val,
                    args,
                )
            save_checkpoint(
                args.output_dir / "last.pt",
                model,
                optimizer,
                scaler,
                epoch,
                best_val,
                args,
            )

        best_checkpoint = args.output_dir / "best.pt"
        load_model_weights(model, best_checkpoint, device)

    save_fusion_summary(args.output_dir, model)
    test_loss, targets, probabilities, indices = predict(
        model, test_loader, device, "Test"
    )
    save_test_outputs(
        args.output_dir,
        test_set,
        targets,
        probabilities,
        indices,
        test_loss,
    )


if __name__ == "__main__":
    main()
