#!/usr/bin/env python3
"""Full fine-tuning of ImageNet-1K-pretrained ResNet-50 with CBAM and MC-Loss."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import re
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
    parser = argparse.ArgumentParser(description=("Full fine-tuning of Hugging Face ResNet-50 with CBAM and MC-Loss"))
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
        help="Use one LR for all parameters, overriding the grouped LRs",
    )
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--cbam-lr", type=float, default=1e-4)
    parser.add_argument(
        "--cbam-scale-lr",
        type=float,
        default=5e-3,
        help="Learning rate for constrained CBAM scale logits",
    )
    parser.add_argument(
        "--classifier-lr",
        type=float,
        default=1e-4,
        help="Learning rate for the classifier and MC head",
    )
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA AMP")
    parser.add_argument(
        "--resume", type=Path, default=None, help="Resume a training checkpoint"
    )
    parser.add_argument(
        "--init-from-mc-checkpoint",
        type=Path,
        default=None,
        help=(
            "Initialize backbone/classifier/optional MC head from an MC-only "
            "checkpoint, leave CBAM newly initialized, and start a fresh optimizer"
        ),
    )
    parser.add_argument(
        "--init-from-cbam-checkpoint",
        type=Path,
        default=None,
        help=(
            "Initialize backbone/CBAM/classifier from a CBAM-only checkpoint, "
            "leave the MC head newly initialized, and start a fresh optimizer"
        ),
    )
    parser.add_argument(
        "--test-only", action="store_true", help="Only evaluate --checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)

    cbam = parser.add_mutually_exclusive_group()
    cbam.add_argument("--use-cbam", dest="use_cbam", action="store_true")
    cbam.add_argument("--no-cbam", dest="use_cbam", action="store_false")
    parser.set_defaults(use_cbam=True)
    parser.add_argument(
        "--freeze-cbam",
        action="store_true",
        help=(
            "Keep attached CBAM attention parameters, scales, and BatchNorm "
            "statistics fixed while training the remaining model"
        ),
    )
    parser.add_argument("--cbam-reduction", type=int, default=16)
    parser.add_argument(
        "--cbam-spatial-kernel", type=int, choices=(3, 7), default=7
    )
    parser.add_argument(
        "--cbam-initial-scale",
        type=float,
        default=0.05,
        help="Initial CBAM mixing coefficient; must be in (0, 1)",
    )
    parser.add_argument(
        "--cbam-stages",
        type=int,
        nargs="+",
        choices=(1, 2, 3, 4),
        default=[3],
        help="1-indexed ResNet stages receiving CBAM (default: stage 4)",
    )

    mc = parser.add_mutually_exclusive_group()
    mc.add_argument("--use-mc-loss", dest="use_mc_loss", action="store_true")
    mc.add_argument("--no-mc-loss", dest="use_mc_loss", action="store_false")
    parser.set_defaults(use_mc_loss=True)
    parser.add_argument("--mc-channels-per-class", type=int, default=3)
    parser.add_argument("--mc-keep-channels", type=int, default=2)
    parser.add_argument("--mc-alpha", type=float, default=1.0)
    parser.add_argument("--mc-beta", type=float, default=20.0)
    parser.add_argument(
        "--mc-delay-epochs",
        type=int,
        default=3,
        help="Number of initial epochs trained with CE only (default: 3)",
    )
    parser.add_argument(
        "--mc-warmup-epochs",
        type=int,
        default=5,
        help=(
            "Linearly ramp MC-Loss from 0 to its full weight after the CE-only "
            "delay (default: 5); use 0 for an immediate transition"
        ),
    )

    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Test a probability ensemble of MC-only and CBAM-only checkpoints",
    )
    parser.add_argument(
        "--mc-checkpoint",
        type=Path,
        default=None,
        help="MC-only checkpoint used by --ensemble",
    )
    parser.add_argument(
        "--cbam-checkpoint",
        type=Path,
        default=None,
        help="CBAM-only checkpoint used by --ensemble",
    )
    parser.add_argument(
        "--ensemble-mc-weight",
        type=float,
        default=0.5,
        help=(
            "Weight of MC-only probabilities; CBAM receives 1-weight "
            "(default: 0.5)"
        ),
    )
    return parser.parse_args()


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
            if "file_name" not in item or "label" not in item:
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

    def __getitem__(self, index: int):
        item = self.records[index]
        image_path = self._image_path(str(item["file_name"]))
        try:
            with Image.open(image_path) as image:
                pixel_values = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {image_path}") from exc
        return pixel_values, int(item["label"]), index


class ChannelAttention(nn.Module):
    """Shared MLP applied to global average- and max-pooled features."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        average = self.mlp(F.adaptive_avg_pool2d(features, 1))
        maximum = self.mlp(F.adaptive_max_pool2d(features, 1))
        return features * torch.sigmoid(average + maximum)


class SpatialAttention(nn.Module):
    """Spatial attention computed from channel-wise max and mean maps."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        # This BatchNorm matches the reference implementation's BasicConv.
        self.normalization = nn.BatchNorm2d(1, eps=1e-5, momentum=0.01)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        compressed = torch.cat(
            (
                features.max(dim=1, keepdim=True).values,
                features.mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
        attention = self.normalization(self.convolution(compressed))
        return features * torch.sigmoid(attention)


class CBAM(nn.Module):
    def __init__(
        self, channels: int, reduction: int = 16, spatial_kernel: int = 7
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Reference code initializes SpatialGate BN scale to zero.
        nn.init.zeros_(self.spatial_attention.normalization.weight)
        nn.init.zeros_(self.spatial_attention.normalization.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.spatial_attention(self.channel_attention(features))


class CBAMResidualBlock(nn.Module):
    """Add CBAM to an already-loaded Hugging Face ResNet residual block."""

    def __init__(
        self,
        block: nn.Module,
        channels: int,
        reduction: int,
        spatial_kernel: int,
        initial_scale: float,
    ) -> None:
        super().__init__()
        self.block = block
        self.cbam = CBAM(channels, reduction, spatial_kernel)
        initial_logit = math.log(initial_scale / (1.0 - initial_scale))
        self.cbam_scale_logit = nn.Parameter(torch.tensor([initial_logit]))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = self.block.shortcut(hidden_state)
        hidden_state = self.block.layer(hidden_state)
        attended = self.cbam(hidden_state)
        scale = torch.sigmoid(self.cbam_scale_logit)
        hidden_state = hidden_state + scale * (attended - hidden_state)
        hidden_state = hidden_state + residual
        return self.block.activation(hidden_state)


class ResNet50Classifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        use_cbam: bool,
        cbam_reduction: int,
        cbam_spatial_kernel: int,
        cbam_stages: list[int],
        cbam_initial_scale: float,
        use_mc_loss: bool,
        mc_channels_per_class: int,
    ) -> None:
        super().__init__()
        # Load all ImageNet-1K weights before replacing residual block wrappers.
        self.backbone = ResNetModel.from_pretrained(model_name)
        hidden_size = int(self.backbone.config.hidden_sizes[-1])
        if use_cbam:
            self._attach_cbam(
                cbam_reduction,
                cbam_spatial_kernel,
                cbam_stages,
                cbam_initial_scale,
            )
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.mc_head = (
            nn.Conv2d(
                hidden_size,
                num_classes * mc_channels_per_class,
                kernel_size=1,
            )
            if use_mc_loss
            else None
        )
        self.num_classes = num_classes
        self.mc_channels_per_class = mc_channels_per_class

        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)
        if self.mc_head is not None:
            nn.init.kaiming_normal_(
                self.mc_head.weight, mode="fan_out", nonlinearity="relu"
            )
            nn.init.zeros_(self.mc_head.bias)

    def _attach_cbam(
        self,
        reduction: int,
        spatial_kernel: int,
        selected_stages: list[int],
        initial_scale: float,
    ) -> None:
        stages = self.backbone.encoder.stages
        hidden_sizes = self.backbone.config.hidden_sizes
        if len(stages) != len(hidden_sizes):
            raise RuntimeError("Unexpected Hugging Face ResNet encoder structure")
        selected = set(selected_stages)
        for stage_number, (stage, channels) in enumerate(
            zip(stages, hidden_sizes), start=1
        ):
            if stage_number not in selected:
                continue
            stage.layers = nn.Sequential(
                *[
                    CBAMResidualBlock(
                        block,
                        int(channels),
                        reduction,
                        spatial_kernel,
                        initial_scale,
                    )
                    for block in stage.layers
                ]
            )

    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.backbone(pixel_values=pixel_values).last_hidden_state
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        logits = self.classifier(pooled)
        # MC maps are only an auxiliary training signal.
        mc_maps = (
            self.mc_head(features)
            if self.training and self.mc_head is not None
            else None
        )
        return logits, mc_maps

    def optimizer_parameter_groups(
        self, args: argparse.Namespace
    ) -> list[dict[str, Any]]:
        """Build disjoint LR groups for pretrained and newly added parameters."""
        if args.lr is not None:
            return [
                {"params": self.parameters(), "lr": args.lr, "initial_lr": args.lr}
            ]

        backbone_parameters: list[nn.Parameter] = []
        cbam_parameters: list[nn.Parameter] = []
        scale_parameters: list[nn.Parameter] = []
        for name, parameter in self.backbone.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.endswith(".cbam_scale_logit"):
                scale_parameters.append(parameter)
            elif ".cbam." in name:
                cbam_parameters.append(parameter)
            else:
                backbone_parameters.append(parameter)

        new_parameters = list(self.classifier.parameters())
        if self.mc_head is not None:
            new_parameters.extend(self.mc_head.parameters())

        groups: list[dict[str, Any]] = [
            {
                "params": backbone_parameters,
                "lr": args.backbone_lr,
                "initial_lr": args.backbone_lr,
                "group_name": "backbone",
            },
            {
                "params": new_parameters,
                "lr": args.classifier_lr,
                "initial_lr": args.classifier_lr,
                "group_name": "classifier_and_mc_head",
            },
        ]
        if cbam_parameters:
            groups.append(
                {
                    "params": cbam_parameters,
                    "lr": args.cbam_lr,
                    "initial_lr": args.cbam_lr,
                    "group_name": "cbam",
                }
            )
        if scale_parameters:
            groups.append(
                {
                    "params": scale_parameters,
                    "lr": args.cbam_scale_lr,
                    "initial_lr": args.cbam_scale_lr,
                    "group_name": "cbam_scale",
                }
            )
        return groups


def mutual_channel_loss(
    maps: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    channels_per_class: int,
    keep_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the MC discriminative and diversity losses."""
    if maps.shape[1] != num_classes * channels_per_class:
        raise ValueError(
            "MC feature channel count must equal "
            "num_classes * mc_channels_per_class"
        )

    batch, _, height, width = maps.shape
    # Diversity component from resnet-mc-loss.py.
    spatial_prob = F.softmax(maps.flatten(2), dim=2).view(
        batch, num_classes, channels_per_class, height * width
    )
    group_max = spatial_prob.max(dim=2).values
    diversity_loss = (
        1.0 - group_max.sum(dim=2).mean() / float(channels_per_class)
    )

    # Discriminative component from resnet-mc-loss.py.
    noise = torch.rand(
        batch, num_classes, channels_per_class, device=maps.device
    )
    keep_indices = noise.topk(keep_channels, dim=2).indices
    mask = (
        torch.zeros_like(noise)
        .scatter_(2, keep_indices, 1.0)
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
    grouped = maps.view(
        batch, num_classes, channels_per_class, height, width
    )
    negative_infinity = torch.finfo(maps.dtype).min
    masked = grouped.masked_fill(mask == 0, negative_infinity)
    auxiliary_logits = masked.max(dim=2).values.mean(dim=(2, 3))
    discriminative_loss = F.cross_entropy(auxiliary_logits.float(), targets)
    return discriminative_loss, diversity_loss


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


def mc_loss_weight(epoch: int, delay_epochs: int, warmup_epochs: int) -> float:
    """Return the MC multiplier for a zero-indexed training epoch.

    For example, delay=3 and warmup=5 gives weights 0, 0, 0, 0.2, 0.4,
    0.6, 0.8, 1.0 for epochs 1 through 8, followed by 1.0 thereafter.
    """
    epoch_number = epoch + 1
    if epoch_number <= delay_epochs:
        return 0.0
    if warmup_epochs == 0:
        return 1.0
    return min(1.0, (epoch_number - delay_epochs) / float(warmup_epochs))


def freeze_cbam_parameters(model: nn.Module) -> int:
    """Freeze only CBAM attention/gate parameters, not residual backbones."""
    frozen = 0
    for module in model.modules():
        if not isinstance(module, CBAMResidualBlock):
            continue
        for parameter in module.cbam.parameters():
            if parameter.requires_grad:
                parameter.requires_grad_(False)
                frozen += parameter.numel()
        if module.cbam_scale_logit.requires_grad:
            module.cbam_scale_logit.requires_grad_(False)
            frozen += module.cbam_scale_logit.numel()
    return frozen


def keep_frozen_cbam_in_eval_mode(model: nn.Module) -> None:
    """Prevent frozen CBAM BatchNorm running statistics from changing."""
    for module in model.modules():
        if isinstance(module, CBAMResidualBlock):
            module.cbam.eval()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    total_steps: int,
    warmup_steps: int,
) -> dict[str, float]:
    model.train()
    if args.freeze_cbam:
        keep_frozen_cbam_in_eval_mode(model)
    current_mc_weight = (
        mc_loss_weight(epoch, args.mc_delay_epochs, args.mc_warmup_epochs)
        if args.use_mc_loss
        else 0.0
    )
    totals = {
        "loss": 0.0,
        "ce": 0.0,
        "discriminative": 0.0,
        "diversity": 0.0,
        "correct": 0.0,
        "count": 0.0,
    }
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
            logits, mc_maps = model(images)
            cross_entropy = F.cross_entropy(
                logits, targets, label_smoothing=args.label_smoothing
            )
            if args.use_mc_loss and current_mc_weight > 0.0:
                if mc_maps is None:
                    raise RuntimeError("MC-Loss is enabled but the MC head returned None")
                discriminative, diversity = mutual_channel_loss(
                    mc_maps,
                    targets,
                    args.num_classes,
                    args.mc_channels_per_class,
                    args.mc_keep_channels,
                )
                loss = (
                    cross_entropy
                    + current_mc_weight
                    * (
                        args.mc_alpha * discriminative
                        + args.mc_beta * diversity
                    )
                )
            else:
                discriminative = diversity = cross_entropy.new_zeros(())
                loss = cross_entropy

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        totals["count"] += batch_size
        totals["loss"] += loss.item() * batch_size
        totals["ce"] += cross_entropy.item() * batch_size
        totals["discriminative"] += discriminative.item() * batch_size
        totals["diversity"] += diversity.item() * batch_size
        totals["correct"] += logits.argmax(dim=1).eq(targets).sum().item()
        progress.set_postfix(
            loss=f"{totals['loss'] / totals['count']:.4f}",
            acc=f"{totals['correct'] / totals['count']:.4f}",
            mc_weight=f"{current_mc_weight:.2f}",
        )

    count = totals.pop("count")
    correct = totals.pop("correct")
    result = {key: value / count for key, value in totals.items()}
    result["top1_accuracy"] = correct / count
    result["mc_weight"] = current_mc_weight
    return result


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str,
):
    model.eval()
    all_targets: list[torch.Tensor] = []
    all_probabilities: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    loss_sum, count = 0.0, 0
    for images, targets, indices in tqdm(loader, desc=description):
        images = images.to(device, non_blocking=True)
        device_targets = targets.to(device, non_blocking=True)
        logits, _ = model(images)
        loss_sum += F.cross_entropy(
            logits, device_targets, reduction="sum"
        ).item()
        count += targets.size(0)
        all_targets.append(targets)
        all_probabilities.append(logits.float().softmax(dim=1).cpu())
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
    top_order = np.argsort(-probabilities, axis=1)[:, : min(5, num_classes)]
    metrics = {
        f"top{k}_accuracy": float(
            np.mean(np.any(top_order[:, :k] == targets[:, None], axis=1))
        )
        for k in range(1, min(5, num_classes) + 1)
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
        p = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        r = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
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


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val: float,
    args: argparse.Namespace,
) -> None:
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_top1": best_val,
            "args": serializable_args,
        },
        path,
    )


def load_model_weights(
    model: nn.Module, checkpoint_path: Path, device: torch.device
):
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except (pickle.UnpicklingError, TypeError):
        # Compatibility for trusted checkpoints made by older versions.
        print(
            "Warning: loading a legacy checkpoint with weights_only=False. "
            "Only use checkpoints from a trusted source."
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state)
    return checkpoint


def initialize_from_mc_checkpoint(
    model: nn.Module, checkpoint_path: Path, device: torch.device
) -> None:
    """Initialize a CBAM-wrapped model from an MC-only training checkpoint.

    MC-only residual-block keys such as ``...layers.0.layer...`` are mapped to
    the inner pretrained block of the CBAM wrapper, whose key is
    ``...layers.0.block.layer...``. Newly introduced CBAM parameters keep their
    near-identity initialization. Optimizer/scaler state is intentionally not
    loaded; this is initialization, not training resumption.
    """
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except (pickle.UnpicklingError, TypeError):
        print(
            "Warning: loading a legacy MC checkpoint with weights_only=False. "
            "Only use checkpoints from a trusted source."
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    source_state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    if not isinstance(source_state, dict):
        raise ValueError(f"Invalid MC checkpoint: {checkpoint_path}")

    target_state = model.state_dict()
    adapted_state: dict[str, torch.Tensor] = {}
    skipped_source_keys: list[str] = []
    block_pattern = re.compile(
        r"^(backbone\.encoder\.stages\.\d+\.layers\.\d+)(\..+)$"
    )

    for source_key, value in source_state.items():
        candidates = [source_key]
        match = block_pattern.match(source_key)
        if match is not None:
            candidates.append(f"{match.group(1)}.block{match.group(2)}")

        target_key = next(
            (
                candidate
                for candidate in candidates
                if candidate in target_state
                and target_state[candidate].shape == value.shape
            ),
            None,
        )
        if target_key is None:
            skipped_source_keys.append(source_key)
        else:
            adapted_state[target_key] = value

    incompatible = model.load_state_dict(adapted_state, strict=False)
    allowed_missing = [
        key
        for key in incompatible.missing_keys
        if ".cbam." in key or key.endswith(".cbam_scale_logit")
    ]
    unexpected_missing = [
        key for key in incompatible.missing_keys if key not in allowed_missing
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "MC checkpoint is not compatible with this model configuration. "
            f"Missing non-CBAM keys: {unexpected_missing}; "
            f"unexpected keys: {incompatible.unexpected_keys}. "
            "If MC-Loss remains enabled, ensure --mc-channels-per-class matches "
            "the MC-only training run."
        )

    ignored_mc_head = [
        key for key in skipped_source_keys if key.startswith("mc_head.")
    ]
    other_skipped = [
        key for key in skipped_source_keys if not key.startswith("mc_head.")
    ]
    if other_skipped:
        raise RuntimeError(
            "Some MC checkpoint parameters could not be mapped: "
            f"{other_skipped[:20]}"
        )
    print(
        f"Initialized {len(adapted_state)} tensors from MC checkpoint: "
        f"{checkpoint_path}"
    )
    print(
        f"Kept {len(allowed_missing)} CBAM tensors newly initialized; "
        f"ignored {len(ignored_mc_head)} MC-head tensors not present in the "
        "target model."
    )


def initialize_from_cbam_checkpoint(
    model: nn.Module, checkpoint_path: Path, device: torch.device
) -> None:
    """Initialize a combined model from a CBAM-only training checkpoint.

    Backbone, CBAM, learned CBAM scales, and classifier parameters are loaded.
    The new MC head keeps its random initialization. Optimizer/scaler state is
    intentionally not loaded because this starts a new training phase.
    """
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except (pickle.UnpicklingError, TypeError):
        print(
            "Warning: loading a legacy CBAM checkpoint with "
            "weights_only=False. Only use checkpoints from a trusted source."
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    source_state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    if not isinstance(source_state, dict):
        raise ValueError(f"Invalid CBAM checkpoint: {checkpoint_path}")

    incompatible = model.load_state_dict(source_state, strict=False)
    allowed_missing = [
        key for key in incompatible.missing_keys if key.startswith("mc_head.")
    ]
    unexpected_missing = [
        key for key in incompatible.missing_keys if key not in allowed_missing
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "CBAM checkpoint is not compatible with this model configuration. "
            f"Missing non-MC-head keys: {unexpected_missing}; "
            f"unexpected keys: {incompatible.unexpected_keys}. Ensure "
            "--cbam-stages, --cbam-reduction, --cbam-spatial-kernel, and "
            "--num-classes match the CBAM-only training run."
        )
    if not allowed_missing:
        raise RuntimeError(
            "No new MC-head parameters were found. "
            "--init-from-cbam-checkpoint requires --use-mc-loss."
        )
    print(
        f"Initialized {len(source_state)} tensors from CBAM checkpoint: "
        f"{checkpoint_path}"
    )
    print(
        f"Kept {len(allowed_missing)} MC-head tensors newly initialized; "
        "starting with a fresh optimizer and scaler."
    )


def save_cbam_scales(output_dir: Path, model: ResNet50Classifier) -> None:
    """Export the learned identity-gate scale of every attached CBAM block."""
    blocks = []
    for stage_index, stage in enumerate(model.backbone.encoder.stages, start=1):
        for block_index, block in enumerate(stage.layers, start=1):
            if not isinstance(block, CBAMResidualBlock):
                continue
            scale_logit = float(
                block.cbam_scale_logit.detach().float().cpu().item()
            )
            scale = float(
                torch.sigmoid(block.cbam_scale_logit.detach().float()).cpu().item()
            )
            blocks.append(
                {
                    "stage": stage_index,
                    "block": block_index,
                    "module": (
                        f"backbone.encoder.stages.{stage_index - 1}."
                        f"layers.{block_index - 1}"
                    ),
                    "cbam_scale": scale,
                    "cbam_scale_logit": scale_logit,
                    "absolute_scale": abs(scale),
                }
            )

    absolute_scales = [item["absolute_scale"] for item in blocks]
    payload = {
        "description": (
            "The block uses x + sigmoid(cbam_scale_logit) * (CBAM(x) - x). "
            "A scale near zero means the pretrained block is nearly unchanged."
        ),
        "num_cbam_blocks": len(blocks),
        "summary": {
            "mean_absolute_scale": (
                float(np.mean(absolute_scales)) if absolute_scales else None
            ),
            "min_absolute_scale": (
                float(np.min(absolute_scales)) if absolute_scales else None
            ),
            "max_absolute_scale": (
                float(np.max(absolute_scales)) if absolute_scales else None
            ),
        },
        "blocks": blocks,
    }
    with (output_dir / "cbam_scales.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(payload, file, indent=2)


def build_model(
    args: argparse.Namespace,
    device: torch.device,
    *,
    use_cbam: bool,
    use_mc_loss: bool,
) -> ResNet50Classifier:
    return ResNet50Classifier(
        model_name=args.model_name,
        num_classes=args.num_classes,
        use_cbam=use_cbam,
        cbam_reduction=args.cbam_reduction,
        cbam_spatial_kernel=args.cbam_spatial_kernel,
        cbam_stages=args.cbam_stages,
        cbam_initial_scale=args.cbam_initial_scale,
        use_mc_loss=use_mc_loss,
        mc_channels_per_class=args.mc_channels_per_class,
    ).to(device)


def probability_nll(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean negative log-likelihood for already-normalized probabilities."""
    target_probabilities = probabilities[np.arange(len(targets)), targets]
    return float(-np.log(np.clip(target_probabilities, 1e-12, 1.0)).mean())


def run_probability_ensemble(
    args: argparse.Namespace,
    device: torch.device,
    test_loader: DataLoader,
    test_set: INatJsonDataset,
) -> None:
    """Evaluate MC-only and CBAM-only checkpoints and average probabilities."""
    mc_model = build_model(args, device, use_cbam=False, use_mc_loss=True)
    load_model_weights(mc_model, args.mc_checkpoint, device)
    mc_loss, mc_targets, mc_probabilities, mc_indices = predict(
        mc_model, test_loader, device, "Test (MC-only)"
    )
    del mc_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cbam_model = build_model(args, device, use_cbam=True, use_mc_loss=False)
    load_model_weights(cbam_model, args.cbam_checkpoint, device)
    cbam_loss, cbam_targets, cbam_probabilities, cbam_indices = predict(
        cbam_model, test_loader, device, "Test (CBAM-only)"
    )
    save_cbam_scales(args.output_dir, cbam_model)

    if not np.array_equal(mc_targets, cbam_targets):
        raise RuntimeError("MC and CBAM predictions have different target order")
    if not np.array_equal(mc_indices, cbam_indices):
        raise RuntimeError("MC and CBAM predictions have different sample order")
    if mc_probabilities.shape != cbam_probabilities.shape:
        raise RuntimeError("MC and CBAM probability arrays have different shapes")

    weight = args.ensemble_mc_weight
    probabilities = (
        weight * mc_probabilities + (1.0 - weight) * cbam_probabilities
    )
    ensemble_loss = probability_nll(mc_targets, probabilities)
    save_test_outputs(
        args.output_dir,
        test_set,
        mc_targets,
        probabilities,
        mc_indices,
        ensemble_loss,
    )

    summary = {
        "method": "weighted_probability_average",
        "mc_checkpoint": str(args.mc_checkpoint),
        "cbam_checkpoint": str(args.cbam_checkpoint),
        "mc_weight": weight,
        "cbam_weight": 1.0 - weight,
        "mc_test_loss": mc_loss,
        "cbam_test_loss": cbam_loss,
        "ensemble_test_loss": ensemble_loss,
        "mc_metrics": compute_metrics(
            mc_targets, mc_probabilities, args.num_classes
        ),
        "cbam_metrics": compute_metrics(
            cbam_targets, cbam_probabilities, args.num_classes
        ),
        "ensemble_metrics": compute_metrics(
            mc_targets, probabilities, args.num_classes
        ),
    }
    with (args.output_dir / "ensemble_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)


def validate_args(args: argparse.Namespace) -> None:
    if args.num_classes < 2:
        raise ValueError("--num-classes must be >= 2")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("Batch sizes must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.cbam_reduction < 1:
        raise ValueError("--cbam-reduction must be >= 1")
    if not 0.0 < args.cbam_initial_scale < 1.0:
        raise ValueError("--cbam-initial-scale must be strictly between 0 and 1")
    if args.freeze_cbam and not args.use_cbam:
        raise ValueError("--freeze-cbam requires --use-cbam")
    if args.mc_channels_per_class < 1:
        raise ValueError("--mc-channels-per-class must be >= 1")
    if not 1 <= args.mc_keep_channels <= args.mc_channels_per_class:
        raise ValueError(
            "--mc-keep-channels must be in [1, --mc-channels-per-class]"
        )
    if args.mc_alpha < 0 or args.mc_beta < 0:
        raise ValueError("--mc-alpha and --mc-beta must be >= 0")
    if args.mc_delay_epochs < 0 or args.mc_warmup_epochs < 0:
        raise ValueError("MC delay and warm-up epochs must be >= 0")
    learning_rates = [
        args.backbone_lr,
        args.cbam_lr,
        args.cbam_scale_lr,
        args.classifier_lr,
    ]
    if args.lr is not None:
        learning_rates.append(args.lr)
    if any(rate <= 0 for rate in learning_rates):
        raise ValueError("All learning rates must be > 0")
    if args.test_only and not args.ensemble and args.checkpoint is None:
        raise ValueError("--test-only requires --checkpoint")
    if args.init_from_mc_checkpoint is not None:
        if not args.use_cbam:
            raise ValueError("--init-from-mc-checkpoint requires --use-cbam")
        if args.init_from_cbam_checkpoint is not None:
            raise ValueError(
                "Choose only one of --init-from-mc-checkpoint and "
                "--init-from-cbam-checkpoint"
            )
        if args.resume is not None or args.test_only or args.ensemble:
            raise ValueError(
                "--init-from-mc-checkpoint is a fresh training initialization "
                "and cannot be combined with --resume, --test-only, or --ensemble"
            )
    if args.init_from_cbam_checkpoint is not None:
        if not args.use_cbam or not args.use_mc_loss:
            raise ValueError(
                "--init-from-cbam-checkpoint requires --use-cbam and --use-mc-loss"
            )
        if args.resume is not None or args.test_only or args.ensemble:
            raise ValueError(
                "--init-from-cbam-checkpoint is a fresh training initialization "
                "and cannot be combined with --resume, --test-only, or --ensemble"
            )
    if args.ensemble:
        if args.mc_checkpoint is None or args.cbam_checkpoint is None:
            raise ValueError(
                "--ensemble requires --mc-checkpoint and --cbam-checkpoint"
            )
        if not 0.0 <= args.ensemble_mc_weight <= 1.0:
            raise ValueError("--ensemble-mc-weight must be in [0, 1]")
        if args.resume is not None or args.checkpoint is not None:
            raise ValueError(
                "--ensemble cannot be combined with --resume or --checkpoint"
            )


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
        json.dump(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            file,
            indent=2,
        )

    train_transform, eval_transform = make_transforms(
        args.model_name, args.image_size
    )
    test_set = INatJsonDataset(
        args.data_dir, "test", eval_transform, args.num_classes
    )
    test_loader = make_loader(
        test_set, args.eval_batch_size, args.num_workers, False, device
    )
    if args.ensemble:
        run_probability_ensemble(args, device, test_loader, test_set)
        return

    model = build_model(
        args,
        device,
        use_cbam=args.use_cbam,
        use_mc_loss=args.use_mc_loss,
    )
    if args.init_from_mc_checkpoint is not None:
        initialize_from_mc_checkpoint(
            model, args.init_from_mc_checkpoint, device
        )
    elif args.init_from_cbam_checkpoint is not None:
        initialize_from_cbam_checkpoint(
            model, args.init_from_cbam_checkpoint, device
        )
    if args.freeze_cbam:
        frozen_parameters = freeze_cbam_parameters(model)
        if frozen_parameters == 0:
            raise RuntimeError("--freeze-cbam found no attached CBAM parameters")
        print(
            f"Froze {frozen_parameters} CBAM attention/gate parameters; "
            "the residual backbone, classifier, and MC head remain trainable."
        )

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
            train_set, args.batch_size, args.num_workers, True, device
        )
        val_loader = make_loader(
            val_set, args.eval_batch_size, args.num_workers, False, device
        )
        optimizer = torch.optim.AdamW(
            model.optimizer_parameter_groups(args),
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

        total_steps = args.epochs * len(train_loader)
        warmup_steps = int(args.warmup_epochs * len(train_loader))
        history_path = args.output_dir / "training_history.jsonl"
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
                **{
                    f"train_{key}": value
                    for key, value in train_metrics.items()
                },
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

        load_model_weights(model, args.output_dir / "best.pt", device)

    save_cbam_scales(args.output_dir, model)
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
