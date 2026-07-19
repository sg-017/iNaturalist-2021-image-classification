#!/usr/bin/env python3
"""Full fine-tuning of ImageNet-1K ResNet-50 with Compact Bilinear Pooling."""

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
        description="Full fine-tuning of Hugging Face ResNet-50 with CBP"
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
        help="Use one LR for every trainable parameter, overriding grouped LRs",
    )
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument(
        "--fusion-gate-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the scalar GAP+CBP residual gate",
    )
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--gap-aux-loss-weight",
        type=float,
        default=0,
        help="Weight of the auxiliary GAP-branch cross-entropy loss",
    )
    parser.add_argument(
        "--cbp-aux-loss-weight",
        type=float,
        default=0,
        help="Weight of the auxiliary CBP-branch cross-entropy loss",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA AMP")
    parser.add_argument(
        "--resume", type=Path, default=None, help="Resume a complete training checkpoint"
    )
    parser.add_argument(
        "--test-only", action="store_true", help="Skip training and evaluate --checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--flip-tta",
        action="store_true",
        help=(
            "At test time, average logits from each image and its horizontal "
            "flip. This does not affect training or validation checkpoint selection."
        ),
    )
    parser.add_argument(
        "--cbp-output-dim",
        type=int,
        default=8192,
        help="Tensor Sketch dimension; a power of two is efficient for FFT",
    )
    parser.add_argument(
        "--cbp-seed",
        type=int,
        default=1,
        help=(
            "Base seed for fixed hashes/signs. The default gives DeepInsight's "
            "four seeds 1, 3, 5, and 7"
        ),
    )
    parser.add_argument(
        "--cbp-spatial-chunk-size",
        type=int,
        default=0,
        help="Spatial positions per FFT chunk; 0 processes all H*W positions",
    )
    parser.add_argument(
        "--no-signed-sqrt",
        action="store_true",
        help="Disable signed square-root normalization after spatial aggregation",
    )
    parser.add_argument(
        "--no-l2-normalize",
        action="store_true",
        help="Disable L2 normalization before the classifier",
    )
    parser.add_argument(
        "--cbp-dropout",
        type=float,
        default=0.3,
        help="Dropout applied to the normalized CBP descriptor",
    )
    parser.add_argument(
        "--fusion-initial-gate",
        type=float,
        default=0.1,
        help="Initial CBP residual coefficient; must be strictly between 0 and 1",
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
    if args.cbp_output_dim < 2:
        raise ValueError("--cbp-output-dim must be >= 2")
    if args.cbp_spatial_chunk_size < 0:
        raise ValueError("--cbp-spatial-chunk-size must be >= 0")
    rates = [args.backbone_lr, args.classifier_lr, args.fusion_gate_lr]
    if args.lr is not None:
        rates.append(args.lr)
    if any(rate <= 0 for rate in rates):
        raise ValueError("All learning rates must be > 0")
    if args.weight_decay < 0 or args.warmup_epochs < 0 or args.grad_clip < 0:
        raise ValueError("Weight decay, warmup epochs, and grad clip must be >= 0")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.gap_aux_loss_weight < 0 or args.cbp_aux_loss_weight < 0:
        raise ValueError("Auxiliary loss weights must be >= 0")
    if not 0.0 <= args.cbp_dropout < 1.0:
        raise ValueError("--cbp-dropout must be in [0, 1)")
    if not 0.0 < args.fusion_initial_gate < 1.0:
        raise ValueError("--fusion-initial-gate must be strictly between 0 and 1")
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
    """iNaturalist subset described by one JSON list per split."""

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
            if not isinstance(item, dict):
                raise ValueError(f"Record {index} in {json_path} is not an object")
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

        # Normal case: file_name is "train/000/image.jpg".
        candidate = self.data_dir / path
        if candidate.is_file():
            return candidate

        # Also accept "processed_dataset/train/000/image.jpg".
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


class CountSketch(nn.Module):
    """Fixed Count Sketch projection over the last input dimension.

    The hash and sign vectors are buffers rather than parameters: they move with
    the module, are included in checkpoints, and are never optimized.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hash_seed: int,
        sign_seed: int,
    ):
        super().__init__()
        if input_dim < 1 or output_dim < 1:
            raise ValueError("CountSketch dimensions must be positive")

        # RandomState reproduces the independent seeded NumPy construction used
        # by DeepInsight-PCALab while avoiding changes to global RNG state.
        hashes = np.random.RandomState(hash_seed).randint(
            0, output_dim, size=input_dim
        )
        signs = 2 * np.random.RandomState(sign_seed).randint(0, 2, size=input_dim) - 1

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.register_buffer(
            "hashes", torch.from_numpy(hashes.astype(np.int64)), persistent=True
        )
        self.register_buffer(
            "signs", torch.from_numpy(signs.astype(np.float32)), persistent=True
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dimension {self.input_dim}, got {inputs.shape[-1]}"
            )

        output = inputs.new_zeros(*inputs.shape[:-1], self.output_dim)
        vector_shape = (1,) * (inputs.ndim - 1) + (self.input_dim,)
        indices = self.hashes.view(vector_shape).expand_as(inputs)
        signs = self.signs.to(dtype=inputs.dtype).view(vector_shape)
        return output.scatter_add_(-1, indices, inputs * signs)


class CompactBilinearPooling(nn.Module):
    """Tensor-Sketch compact bilinear pooling for two feature maps.

    Inputs have shape BxC1xHxW and BxC2xHxW. With ``sum_pool=True``,
    output has shape BxD. With ``sum_pool=False``, output has shape BxHxWxD.
    Tensor Sketch is applied at each spatial location before any spatial sum,
    following the feature-map interface in DeepInsight-PCALab.
    """

    def __init__(
        self,
        input_dim1: int,
        input_dim2: int,
        output_dim: int,
        *,
        sum_pool: bool = True,
        seed: int = 1,
        spatial_chunk_size: int = 0,
    ):
        super().__init__()
        if spatial_chunk_size < 0:
            raise ValueError("spatial_chunk_size must be >= 0")
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sum_pool = sum_pool
        self.spatial_chunk_size = spatial_chunk_size

        # seed=1 gives the original independent seeds (1, 3, 5, 7).
        self.sketch1 = CountSketch(input_dim1, output_dim, seed, seed + 2)
        self.sketch2 = CountSketch(input_dim2, output_dim, seed + 4, seed + 6)

    def _tensor_sketch(
        self, bottom1: torch.Tensor, bottom2: torch.Tensor
    ) -> torch.Tensor:
        sketch1 = self.sketch1(bottom1)
        sketch2 = self.sketch2(bottom2)
        fft1 = torch.fft.rfft(sketch1, n=self.output_dim, dim=-1)
        fft2 = torch.fft.rfft(sketch2, n=self.output_dim, dim=-1)
        return torch.fft.irfft(fft1 * fft2, n=self.output_dim, dim=-1)

    def forward(
        self, bottom1: torch.Tensor, bottom2: torch.Tensor | None = None
    ) -> torch.Tensor:
        if bottom2 is None:
            bottom2 = bottom1
        if bottom1.ndim != 4 or bottom2.ndim != 4:
            raise ValueError("CBP inputs must both have shape BxCxHxW")
        if bottom1.shape[0] != bottom2.shape[0]:
            raise ValueError("CBP inputs must have the same batch size")
        if bottom1.shape[2:] != bottom2.shape[2:]:
            raise ValueError("CBP inputs must have the same spatial dimensions")
        if bottom1.shape[1] != self.input_dim1:
            raise ValueError(
                f"First CBP input has {bottom1.shape[1]} channels; "
                f"expected {self.input_dim1}"
            )
        if bottom2.shape[1] != self.input_dim2:
            raise ValueError(
                f"Second CBP input has {bottom2.shape[1]} channels; "
                f"expected {self.input_dim2}"
            )

        # BxCxHxW -> Bx(HW)xC. Every row is one spatial feature vector.
        first = bottom1.flatten(2).transpose(1, 2).contiguous()
        second = bottom2.flatten(2).transpose(1, 2).contiguous()
        positions = first.shape[1]
        chunk_size = self.spatial_chunk_size or positions

        if not self.sum_pool:
            chunks = []
            for start in range(0, positions, chunk_size):
                end = min(start + chunk_size, positions)
                chunks.append(
                    self._tensor_sketch(first[:, start:end], second[:, start:end])
                )
            output = torch.cat(chunks, dim=1)
            height, width = bottom1.shape[2:]
            return output.reshape(
                bottom1.shape[0], height, width, self.output_dim
            )

        pooled = first.new_zeros(first.shape[0], self.output_dim)
        for start in range(0, positions, chunk_size):
            end = min(start + chunk_size, positions)
            local_cbp = self._tensor_sketch(
                first[:, start:end], second[:, start:end]
            )
            pooled = pooled + local_cbp.sum(dim=1)
        return pooled


class ResNet50CBPClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        cbp_output_dim: int,
        cbp_seed: int,
        cbp_spatial_chunk_size: int,
        signed_sqrt: bool,
        l2_normalize: bool,
        cbp_dropout: float,
        fusion_initial_gate: float,
    ):
        super().__init__()
        # ResNetModel loads the ImageNet-1K pretrained backbone but omits its
        # original 1000-class head. All backbone parameters remain trainable.
        self.backbone = ResNetModel.from_pretrained(model_name)
        hidden_size = int(self.backbone.config.hidden_sizes[-1])
        self.cbp = CompactBilinearPooling(
            hidden_size,
            hidden_size,
            cbp_output_dim,
            sum_pool=True,
            seed=cbp_seed,
            spatial_chunk_size=cbp_spatial_chunk_size,
        )
        self.signed_sqrt = signed_sqrt
        self.l2_normalize = l2_normalize
        self.cbp_dropout = nn.Dropout(p=cbp_dropout)

        # The GAP branch is the vanilla ResNet classification path. The CBP
        # branch is a gated residual, so the model does not have to discard
        # useful first-order features in order to learn second-order features.
        self.gap_classifier = nn.Linear(hidden_size, num_classes)
        self.cbp_classifier = nn.Linear(cbp_output_dim, num_classes)
        initial_logit = math.log(fusion_initial_gate / (1.0 - fusion_initial_gate))
        self.cbp_gate_logit = nn.Parameter(torch.tensor(initial_logit))

        nn.init.normal_(self.gap_classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.gap_classifier.bias)
        # Start exactly at the GAP baseline. The non-zero gate still gives the
        # CBP classifier a gradient; the gate itself starts adapting once that
        # classifier has learned a non-zero residual.
        nn.init.zeros_(self.cbp_classifier.weight)
        nn.init.zeros_(self.cbp_classifier.bias)

    def fusion_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.cbp_gate_logit)

    def forward_branches(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(pixel_values=pixel_values).last_hidden_state
        gap_descriptor = F.adaptive_avg_pool2d(features, 1).flatten(1)
        gap_logits = self.gap_classifier(gap_descriptor)

        # CUDA FFT support for fp16 depends on signal size and hardware. CBP in
        # fp32 is stable under AMP, and .float() still propagates gradients back
        # through the fully trainable backbone.
        with torch.autocast(device_type=features.device.type, enabled=False):
            pooled = self.cbp(features.float(), features.float())
            if self.signed_sqrt:
                pooled = torch.sign(pooled) * torch.sqrt(torch.abs(pooled) + 1e-8)
            if self.l2_normalize:
                pooled = F.normalize(pooled, p=2, dim=1, eps=1e-12)
        cbp_logits = self.cbp_classifier(self.cbp_dropout(pooled))
        fused_logits = gap_logits + self.fusion_gate() * cbp_logits
        return {
            "fused_logits": fused_logits,
            "gap_logits": gap_logits,
            "cbp_logits": cbp_logits,
        }

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.forward_branches(pixel_values)["fused_logits"]

    def optimizer_parameter_groups(
        self, args: argparse.Namespace
    ) -> list[dict[str, Any]]:
        if args.lr is not None:
            return [
                {"params": self.parameters(), "lr": args.lr, "initial_lr": args.lr}
            ]

        # CBP has fixed buffers and therefore needs no optimizer group.
        classifier_parameters = [
            *self.gap_classifier.parameters(),
            *self.cbp_classifier.parameters(),
        ]
        return [
            {
                "params": self.backbone.parameters(),
                "lr": args.backbone_lr,
                "initial_lr": args.backbone_lr,
                "group_name": "backbone",
            },
            {
                "params": classifier_parameters,
                "lr": args.classifier_lr,
                "initial_lr": args.classifier_lr,
                "group_name": "classifiers",
            },
            {
                "params": [self.cbp_gate_logit],
                "lr": args.fusion_gate_lr,
                "initial_lr": args.fusion_gate_lr,
                "weight_decay": 0.0,
                "group_name": "fusion_gate",
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
    loss_sum, fused_loss_sum, gap_loss_sum, cbp_loss_sum = 0.0, 0.0, 0.0, 0.0
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
            branch_outputs = model.forward_branches(images)
            logits = branch_outputs["fused_logits"]
            fused_loss = F.cross_entropy(
                logits, targets, label_smoothing=args.label_smoothing
            )
            gap_loss = F.cross_entropy(
                branch_outputs["gap_logits"],
                targets,
                label_smoothing=args.label_smoothing,
            )
            cbp_loss = F.cross_entropy(
                branch_outputs["cbp_logits"],
                targets,
                label_smoothing=args.label_smoothing,
            )
            loss = (
                fused_loss
                + args.gap_aux_loss_weight * gap_loss
                + args.cbp_aux_loss_weight * cbp_loss
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        size = targets.size(0)
        count += size
        loss_sum += loss.item() * size
        fused_loss_sum += fused_loss.item() * size
        gap_loss_sum += gap_loss.item() * size
        cbp_loss_sum += cbp_loss.item() * size
        correct += logits.argmax(dim=1).eq(targets).sum().item()
        progress.set_postfix(
            loss=f"{loss_sum / count:.4f}", acc=f"{correct / count:.4f}"
        )

    if count == 0:
        raise ValueError("Training dataset is empty")
    return {
        "loss": loss_sum / count,
        "fused_loss": fused_loss_sum / count,
        "gap_aux_loss": gap_loss_sum / count,
        "cbp_aux_loss": cbp_loss_sum / count,
        "top1_accuracy": correct / count,
    }


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str,
    flip_tta: bool = False,
):
    model.eval()
    all_targets, all_probabilities, all_indices = [], [], []
    loss_sum, count = 0.0, 0

    for images, targets, indices in tqdm(loader, desc=description):
        images = images.to(device, non_blocking=True)
        device_targets = targets.to(device, non_blocking=True)
        logits = model(images)
        if flip_tta:
            flipped_logits = model(torch.flip(images, dims=[3]))
            logits = 0.5 * (logits + flipped_logits)
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
    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    csv_path = output_dir / "test_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "image_id",
                "file_name",
                "ground_truth",
                "predicted_label",
                "predicted_probability",
                "correct",
                "top5_labels",
                "top5_probabilities",
            ]
        )
        for target, probability, index in zip(targets, probabilities, indices):
            record = dataset.records[int(index)]
            top = np.argsort(-probability)[: min(5, len(probability))]
            prediction = int(top[0])
            writer.writerow(
                [
                    record.get("image_id", ""),
                    record["file_name"],
                    f"{int(target):03d}",
                    f"{prediction:03d}",
                    f"{float(probability[prediction]):.8f}",
                    int(prediction == int(target)),
                    json.dumps([f"{int(label):03d}" for label in top]),
                    json.dumps(
                        [round(float(probability[label]), 8) for label in top]
                    ),
                ]
            )

    print(report, end="")
    print(f"Saved report to {report_path}")
    print(f"Saved predictions to {csv_path}")


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
    state = {
        "epoch": epoch,
        "best_val_top1": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": serializable_args(args),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_model_weights(model: nn.Module, path: Path, device: torch.device):
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Compatibility with PyTorch versions that predate weights_only.
        checkpoint = torch.load(path, map_location=device)
    state_dict = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def make_grad_scaler(enabled: bool):
    # torch.amp.GradScaler is the current API. torch.cuda.amp.GradScaler keeps
    # this script usable with older PyTorch 2.x installations.
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "GradScaler"):
        try:
            return amp_module.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return amp_module.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def build_model(
    args: argparse.Namespace, device: torch.device
) -> ResNet50CBPClassifier:
    model = ResNet50CBPClassifier(
        model_name=args.model_name,
        num_classes=args.num_classes,
        cbp_output_dim=args.cbp_output_dim,
        cbp_seed=args.cbp_seed,
        cbp_spatial_chunk_size=args.cbp_spatial_chunk_size,
        signed_sqrt=not args.no_signed_sqrt,
        l2_normalize=not args.no_l2_normalize,
        cbp_dropout=args.cbp_dropout,
        fusion_initial_gate=args.fusion_initial_gate,
    )
    return model.to(device)


def save_fusion_state(output_dir: Path, model: ResNet50CBPClassifier) -> None:
    gate_logit = float(model.cbp_gate_logit.detach().float().cpu().item())
    gate = float(model.fusion_gate().detach().float().cpu().item())
    payload = {
        "fusion": "gap_logits + sigmoid(cbp_gate_logit) * cbp_logits",
        "cbp_gate": gate,
        "cbp_gate_logit": gate_logit,
        "interpretation": (
            "A gate near zero means the trained model relies mostly on GAP; "
            "a larger gate means the CBP residual contributes more strongly."
        ),
    }
    with (output_dir / "fusion_state.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(serializable_args(args), file, indent=2)

    train_transform, eval_transform = make_transforms(
        args.model_name, args.image_size
    )
    test_set = INatJsonDataset(
        args.data_dir, "test", eval_transform, args.num_classes
    )
    test_loader = make_loader(
        test_set, args.eval_batch_size, args.num_workers, False, device
    )

    model = build_model(args, device)
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {trainable:,}/{total:,} (full fine-tuning)")

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
        scaler = make_grad_scaler(amp_enabled)
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
                "cbp_fusion_gate": float(
                    model.fusion_gate().detach().float().cpu().item()
                ),
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

        best_path = args.output_dir / "best.pt"
        if not best_path.is_file():
            raise RuntimeError(
                "No best checkpoint was produced; check --epochs and --resume"
            )
        load_model_weights(model, best_path, device)

    test_description = "Test (flip TTA)" if args.flip_tta else "Test"
    test_loss, targets, probabilities, indices = predict(
        model, test_loader, device, test_description, flip_tta=args.flip_tta
    )
    save_fusion_state(args.output_dir, model)
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
