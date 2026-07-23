#!/usr/bin/env python3
"""ResNet-50 + PMG + iSQRT-COV for a 500-class iNaturalist subset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable

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
        description="Full fine-tuning and checkpoint ensembling for ResNet-50 PMG+iSQRT-COV"
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--num-classes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument(
        "--resize-size",
        type=int,
        default=550,
        help="Square resize before the PMG crop",
    )
    parser.add_argument("--feature-size", type=int, default=512)
    parser.add_argument("--cov-dim", type=int, default=256)
    parser.add_argument("--sqrt-iters", type=int, default=5)
    parser.add_argument("--cov-eps", type=float, default=1e-5)
    parser.add_argument("--cov-dropout", type=float, default=0.2)

    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--backbone-lr", type=float, default=2e-4)
    parser.add_argument("--head-lr", type=float, default=2e-3)
    parser.add_argument("--cov-lr", type=float, default=1e-3)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument(
        "--fusion-scale-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the learned GAP/iSQRT-COV scale",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--concat-loss-weight", type=float, default=2.0)
    parser.add_argument(
        "--gap-cov-loss-weight",
        "--cov-loss-weight",
        dest="gap_cov_loss_weight",
        type=float,
        default=1.0,
        help="Weight for the fused GAP+iSQRT-COV branch loss",
    )
    parser.add_argument(
        "--cov-aux-loss-weight",
        type=float,
        default=0.3,
        help="Weight for the raw iSQRT-COV classifier auxiliary loss",
    )
    parser.add_argument("--fusion-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--hybrid-cov-logit-weight",
        type=float,
        default=1.0,
        help="Weight of the fused GAP+iSQRT-COV logits added to PMG logits",
    )
    parser.add_argument(
        "--cov-initial-scale",
        type=float,
        default=0.05,
        help="Initial learned COV contribution inside the GAP+COV branch",
    )
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--pmg-pretrained-checkpoint",
        type=Path,
        default=None,
        help=(
            "Initialize the hybrid backbone and all PMG heads from a "
            "resnet-pmg.py checkpoint; GAP/iSQRT-COV modules stay new"
        ),
    )
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)

    parser.add_argument(
        "--ensemble-only",
        action="store_true",
        help="Ensemble resnet_pmg.py and resnet_isqrt-cov.py checkpoints",
    )
    parser.add_argument("--pmg-checkpoint", type=Path, default=None)
    parser.add_argument("--cov-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--ensemble-space",
        choices=("logits", "probabilities"),
        default="logits",
        help="Search alpha using weighted logits (default) or probabilities",
    )
    parser.add_argument(
        "--ensemble-metric",
        choices=("top1", "macro_f1", "nll"),
        default="top1",
        help="Validation objective used to select alpha",
    )
    parser.add_argument(
        "--alpha-step",
        type=float,
        default=0.01,
        help="Grid spacing for alpha in [0, 1]",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_classes < 2:
        raise ValueError("--num-classes must be at least 2")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 because PMG uses BatchNorm1d")
    if args.eval_batch_size < 1 or args.num_workers < 0:
        raise ValueError("--eval-batch-size must be positive and workers non-negative")
    if args.image_size < 8 or args.image_size % 8:
        raise ValueError("--image-size must be at least 8 and divisible by 8")
    if args.resize_size < args.image_size:
        raise ValueError("--resize-size must be >= --image-size")
    if args.feature_size < 1 or args.cov_dim < 2 or args.sqrt_iters < 1:
        raise ValueError("Feature dimensions and --sqrt-iters must be positive")
    if args.cov_eps <= 0 or not 0.0 <= args.cov_dropout < 1.0:
        raise ValueError("--cov-eps must be > 0 and --cov-dropout must be in [0, 1)")
    rates = (
        args.backbone_lr,
        args.head_lr,
        args.cov_lr,
        args.classifier_lr,
        args.fusion_scale_lr,
    )
    if any(rate <= 0 for rate in rates):
        raise ValueError("All learning rates must be positive")
    if args.momentum < 0 or args.weight_decay < 0 or args.warmup_epochs < 0:
        raise ValueError("Momentum, weight decay, and warmup must be non-negative")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if min(
        args.concat_loss_weight,
        args.gap_cov_loss_weight,
        args.cov_aux_loss_weight,
        args.fusion_loss_weight,
    ) < 0:
        raise ValueError("Loss weights must be non-negative")
    if (
        args.concat_loss_weight
        + args.gap_cov_loss_weight
        + args.cov_aux_loss_weight
        + args.fusion_loss_weight
        <= 0
    ):
        raise ValueError("At least one original-image loss weight must be positive")
    if args.hybrid_cov_logit_weight < 0 or args.grad_clip < 0:
        raise ValueError("Logit weight and gradient clipping must be non-negative")
    if not 0.0 < args.cov_initial_scale < 1.0:
        raise ValueError("--cov-initial-scale must be strictly between 0 and 1")
    if args.test_only and args.checkpoint is None:
        raise ValueError("--test-only requires --checkpoint")
    if args.test_only and args.resume is not None:
        raise ValueError("--test-only cannot be combined with --resume")
    if args.pmg_pretrained_checkpoint is not None:
        if args.resume is not None:
            raise ValueError(
                "--pmg-pretrained-checkpoint is an initialization option and "
                "cannot be combined with --resume"
            )
        if args.test_only:
            raise ValueError(
                "--pmg-pretrained-checkpoint cannot be combined with --test-only"
            )
    if args.ensemble_only:
        if args.pmg_checkpoint is None or args.cov_checkpoint is None:
            raise ValueError(
                "--ensemble-only requires --pmg-checkpoint and --cov-checkpoint"
            )
        if (
            args.test_only
            or args.resume is not None
            or args.checkpoint is not None
            or args.pmg_pretrained_checkpoint is not None
        ):
            raise ValueError(
                "--ensemble-only cannot be combined with hybrid checkpoint options"
            )
    if not 0.0 < args.alpha_step <= 1.0:
        raise ValueError("--alpha-step must be in (0, 1]")


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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        item = self.records[index]
        image_path = self._image_path(str(item["file_name"]))
        try:
            with Image.open(image_path) as image:
                pixel_values = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {image_path}") from exc
        return pixel_values, int(item["label"]), index


class BasicConv(nn.Module):
    """Conv-BN-ReLU block used by PMG."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(features)))


def make_pmg_conv_blocks(feature_size: int) -> tuple[nn.Sequential, ...]:
    branch_channels = 1024
    return tuple(
        nn.Sequential(
            BasicConv(channels, feature_size, kernel_size=1, padding=0),
            BasicConv(feature_size, branch_channels, kernel_size=3, padding=1),
        )
        for channels in (512, 1024, 2048)
    )


def make_branch_classifier(
    in_features: int, feature_size: int, num_classes: int
) -> nn.Sequential:
    return nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, feature_size),
        nn.BatchNorm1d(feature_size),
        nn.ELU(inplace=True),
        nn.Linear(feature_size, num_classes),
    )


def initialize_new_modules(modules: list[nn.Module]) -> None:
    for root in modules:
        for module in root.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


def resnet_stage_features(
    backbone: ResNetModel, pixel_values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = backbone(pixel_values=pixel_values, output_hidden_states=True)
    if outputs.hidden_states is None or len(outputs.hidden_states) < 4:
        raise RuntimeError("ResNet did not return the required stage hidden states")
    stage2, stage3, stage4 = outputs.hidden_states[-3:]
    actual = (stage2.shape[1], stage3.shape[1], stage4.shape[1])
    if actual != (512, 1024, 2048):
        raise RuntimeError(f"Unexpected ResNet stage channels: {actual}")
    return stage2, stage3, stage4


def pmg_branch_features(
    stages: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    blocks: tuple[nn.Module, nn.Module, nn.Module] | list[nn.Module],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(
        F.adaptive_max_pool2d(block(stage), 1).flatten(1)
        for stage, block in zip(stages, blocks)
    )


class ISqrtCovariancePool(nn.Module):
    """MPN-COV/iSQRT-COV representation implemented with native autograd."""

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
            raise ValueError("iSQRT-COV requires at least two spatial positions")
        flattened = features.reshape(batch_size, channels, spatial_size)
        centered = flattened - flattened.mean(dim=2, keepdim=True)
        return centered.bmm(centered.transpose(1, 2)) / float(spatial_size)

    def _matrix_square_root(self, covariance: torch.Tensor) -> torch.Tensor:
        batch_size, dimension, _ = covariance.shape
        identity = torch.eye(
            dimension, dtype=covariance.dtype, device=covariance.device
        ).unsqueeze(0)
        identity = identity.expand(batch_size, -1, -1)
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
        return 0.5 * (square_root + square_root.transpose(1, 2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected Bx{self.input_dim}xHxW features, got {tuple(features.shape)}"
            )
        with torch.autocast(device_type=features.device.type, enabled=False):
            reduced = self.reduction(features.float())
            covariance = self._covariance_pool(reduced)
            square_root = self._matrix_square_root(covariance)
            return square_root[:, self.triangle_rows, self.triangle_cols]


class PMGClassifier(nn.Module):
    """Checkpoint-compatible copy of the model in resnet-pmg.py."""

    def __init__(self, model_name: str, num_classes: int, feature_size: int) -> None:
        super().__init__()
        self.backbone = ResNetModel.from_pretrained(model_name)
        blocks = make_pmg_conv_blocks(feature_size)
        self.conv_block1, self.conv_block2, self.conv_block3 = blocks
        self.classifier1 = make_branch_classifier(1024, feature_size, num_classes)
        self.classifier2 = make_branch_classifier(1024, feature_size, num_classes)
        self.classifier3 = make_branch_classifier(1024, feature_size, num_classes)
        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(3072),
            nn.Linear(3072, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, num_classes),
        )
        initialize_new_modules(
            [
                self.conv_block1,
                self.conv_block2,
                self.conv_block3,
                self.classifier1,
                self.classifier2,
                self.classifier3,
                self.classifier_concat,
            ]
        )

    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stages = resnet_stage_features(self.backbone, pixel_values)
        features = pmg_branch_features(
            stages, (self.conv_block1, self.conv_block2, self.conv_block3)
        )
        logits1 = self.classifier1(features[0])
        logits2 = self.classifier2(features[1])
        logits3 = self.classifier3(features[2])
        concat_logits = self.classifier_concat(torch.cat(features, dim=1))
        return logits1, logits2, logits3, concat_logits


class ResNet50GAPISqrtCovClassifier(nn.Module):
    """Checkpoint-compatible copy of the model in resnet_isqrt-cov.py."""

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
        initial_logit = math.log(cov_initial_scale / (1.0 - cov_initial_scale))
        self.cov_scale_logit = nn.Parameter(torch.tensor(initial_logit))
        nn.init.normal_(self.gap_classifier.weight, std=0.01)
        nn.init.zeros_(self.gap_classifier.bias)
        nn.init.normal_(self.cov_classifier.weight, std=0.01)
        nn.init.zeros_(self.cov_classifier.bias)

    @property
    def cov_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.cov_scale_logit)

    def forward(
        self, pixel_values: torch.Tensor, return_branch_logits: bool = False
    ):
        features = self.backbone(pixel_values=pixel_values).last_hidden_state
        gap_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        gap_logits = self.gap_classifier(gap_features)
        cov_features = self.cov_pool(features)
        with torch.autocast(device_type=features.device.type, enabled=False):
            cov_features = self.cov_dropout(cov_features.float())
            cov_logits = self.cov_classifier(cov_features)
            fused_logits = gap_logits.float() + self.cov_scale * cov_logits
        if return_branch_logits:
            return fused_logits, gap_logits, cov_logits
        return fused_logits


class PMGISqrtCovClassifier(nn.Module):
    """A shared-backbone hybrid of PMG and iSQRT-COV."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        feature_size: int,
        cov_dim: int,
        sqrt_iters: int,
        cov_eps: float,
        cov_dropout: float,
        cov_initial_scale: float,
    ) -> None:
        super().__init__()
        self.backbone = ResNetModel.from_pretrained(model_name)
        hidden_sizes = [int(value) for value in self.backbone.config.hidden_sizes]
        if hidden_sizes[-3:] != [512, 1024, 2048]:
            raise ValueError(
                "PMG ResNet-50 requires final hidden sizes [512, 1024, 2048], "
                f"received {hidden_sizes[-3:]}"
            )

        blocks = make_pmg_conv_blocks(feature_size)
        self.conv_block1, self.conv_block2, self.conv_block3 = blocks
        self.classifier1 = make_branch_classifier(1024, feature_size, num_classes)
        self.classifier2 = make_branch_classifier(1024, feature_size, num_classes)
        self.classifier3 = make_branch_classifier(1024, feature_size, num_classes)
        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(3072),
            nn.Linear(3072, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, num_classes),
        )
        self.cov_pool = ISqrtCovariancePool(
            input_dim=2048,
            reduction_dim=cov_dim,
            num_iterations=sqrt_iters,
            eps=cov_eps,
        )
        self.cov_dropout = nn.Dropout(p=cov_dropout)
        self.gap_classifier = nn.Linear(2048, num_classes)
        self.cov_classifier = nn.Linear(self.cov_pool.output_dim, num_classes)
        initial_logit = math.log(cov_initial_scale / (1.0 - cov_initial_scale))
        self.cov_scale_logit = nn.Parameter(torch.tensor(initial_logit))
        initialize_new_modules(
            [
                self.conv_block1,
                self.conv_block2,
                self.conv_block3,
                self.classifier1,
                self.classifier2,
                self.classifier3,
                self.classifier_concat,
                self.gap_classifier,
                self.cov_classifier,
            ]
        )

    @property
    def pmg_blocks(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        return self.conv_block1, self.conv_block2, self.conv_block3

    @property
    def pmg_classifiers(self) -> tuple[nn.Module, nn.Module, nn.Module]:
        return self.classifier1, self.classifier2, self.classifier3

    @property
    def cov_scale(self) -> torch.Tensor:
        """Learned iSQRT-COV contribution, constrained to (0, 1)."""
        return torch.sigmoid(self.cov_scale_logit)

    def forward_pmg_branch(
        self, pixel_values: torch.Tensor, branch_index: int
    ) -> torch.Tensor:
        if branch_index not in (0, 1, 2):
            raise ValueError("branch_index must be 0, 1, or 2")
        stages = resnet_stage_features(self.backbone, pixel_values)
        feature = F.adaptive_max_pool2d(
            self.pmg_blocks[branch_index](stages[branch_index]), 1
        ).flatten(1)
        return self.pmg_classifiers[branch_index](feature)

    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        stages = resnet_stage_features(self.backbone, pixel_values)
        features = pmg_branch_features(stages, self.pmg_blocks)
        branch_logits = tuple(
            classifier(feature)
            for classifier, feature in zip(self.pmg_classifiers, features)
        )
        concat_logits = self.classifier_concat(torch.cat(features, dim=1))
        gap_features = F.adaptive_avg_pool2d(stages[2], 1).flatten(1)
        gap_logits = self.gap_classifier(gap_features)
        cov_features = self.cov_pool(stages[2])
        with torch.autocast(device_type=stages[2].device.type, enabled=False):
            cov_features = self.cov_dropout(cov_features.float())
            cov_logits = self.cov_classifier(cov_features)
            gap_cov_logits = gap_logits.float() + self.cov_scale * cov_logits
        return (
            *branch_logits,
            concat_logits,
            gap_cov_logits,
            gap_logits,
            cov_logits,
        )

    def optimizer_parameter_groups(
        self, args: argparse.Namespace
    ) -> list[dict[str, Any]]:
        pmg_modules = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.classifier1,
            self.classifier2,
            self.classifier3,
            self.classifier_concat,
        ]
        pmg_parameters = [
            parameter for module in pmg_modules for parameter in module.parameters()
        ]
        return [
            {
                "params": self.backbone.parameters(),
                "lr": args.backbone_lr,
                "initial_lr": args.backbone_lr,
                "group_name": "backbone",
            },
            {
                "params": pmg_parameters,
                "lr": args.head_lr,
                "initial_lr": args.head_lr,
                "group_name": "pmg_heads",
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
                "group_name": "gap_cov_classifiers",
            },
            {
                "params": [self.cov_scale_logit],
                "lr": args.fusion_scale_lr,
                "initial_lr": args.fusion_scale_lr,
                "weight_decay": 0.0,
                "group_name": "gap_cov_fusion_scale",
            },
        ]


def make_pmg_transforms(model_name: str, image_size: int, resize_size: int):
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


def make_cov_eval_transform(model_name: str, image_size: int):
    processor = AutoImageProcessor.from_pretrained(model_name)
    return transforms.Compose(
        [
            transforms.Resize(int(round(image_size / 0.875))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )


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
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    height, width = images.shape[-2:]
    if height % grid_size or width % grid_size:
        raise ValueError(
            f"Image shape {(height, width)} must be divisible by grid {grid_size}"
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


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def optimizer_step(
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    grad_clip: float,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()


def combine_hybrid_logits(
    outputs: tuple[torch.Tensor, ...], cov_logit_weight: float
) -> torch.Tensor:
    return outputs[0] + outputs[1] + outputs[2] + outputs[3] + (
        cov_logit_weight * outputs[4]
    )


def train_one_epoch(
    model: PMGISqrtCovClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    model.train()
    set_cosine_lr(optimizer, epoch, args.epochs, args.warmup_epochs)
    amp_enabled = device.type == "cuda" and not args.no_amp
    # 3 jigsaw losses + PMG concat + fused GAP/COV + raw COV auxiliary + final PMG/GAP/COV fusion.
    loss_sums = np.zeros(7, dtype=np.float64)
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
                logits = model.forward_pmg_branch(jigsaw_images, branch_index)
                branch_loss = F.cross_entropy(
                    logits, targets, label_smoothing=args.label_smoothing
                )
            optimizer_step(
                branch_loss, model, optimizer, scaler, args.grad_clip
            )
            step_losses.append(float(branch_loss.detach()))

        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            outputs = model(images)
            fused_logits = combine_hybrid_logits(
                outputs, args.hybrid_cov_logit_weight
            )
            concat_loss = F.cross_entropy(
                outputs[3], targets, label_smoothing=args.label_smoothing
            )
            gap_cov_loss = F.cross_entropy(
                outputs[4], targets, label_smoothing=args.label_smoothing
            )
            cov_aux_loss = F.cross_entropy(
                outputs[6], targets, label_smoothing=args.label_smoothing
            )
            fusion_loss = F.cross_entropy(
                fused_logits, targets, label_smoothing=args.label_smoothing
            )
            original_loss = (
                args.concat_loss_weight * concat_loss
                + args.gap_cov_loss_weight * gap_cov_loss
                + args.cov_aux_loss_weight * cov_aux_loss
                + args.fusion_loss_weight * fusion_loss
            )
        optimizer_step(original_loss, model, optimizer, scaler, args.grad_clip)

        batch_size = targets.size(0)
        count += batch_size
        measured = step_losses + [
            float(concat_loss.detach()),
            float(gap_cov_loss.detach()),
            float(cov_aux_loss.detach()),
            float(fusion_loss.detach()),
        ]
        loss_sums += np.asarray(measured) * batch_size
        correct += fused_logits.detach().argmax(dim=1).eq(targets).sum().item()
        weighted_loss = (
            loss_sums[0]
            + loss_sums[1]
            + loss_sums[2]
            + args.concat_loss_weight * loss_sums[3]
            + args.gap_cov_loss_weight * loss_sums[4]
            + args.cov_aux_loss_weight * loss_sums[5]
            + args.fusion_loss_weight * loss_sums[6]
        ) / count
        progress.set_postfix(
            loss=f"{weighted_loss:.4f}",
            fused_acc=f"{correct / count:.4f}",
        )

    if count == 0:
        raise ValueError("Training loader produced no batches; reduce --batch-size")
    return {
        "loss": float(
            (
                loss_sums[0]
                + loss_sums[1]
                + loss_sums[2]
                + args.concat_loss_weight * loss_sums[3]
                + args.gap_cov_loss_weight * loss_sums[4]
                + args.cov_aux_loss_weight * loss_sums[5]
                + args.fusion_loss_weight * loss_sums[6]
            )
            / count
        ),
        "jigsaw_8x8_loss": float(loss_sums[0] / count),
        "jigsaw_4x4_loss": float(loss_sums[1] / count),
        "jigsaw_2x2_loss": float(loss_sums[2] / count),
        "concat_loss": float(loss_sums[3] / count),
        "gap_cov_fused_loss": float(loss_sums[4] / count),
        "cov_aux_loss": float(loss_sums[5] / count),
        "fusion_loss": float(loss_sums[6] / count),
        "fused_top1_accuracy": correct / count,
    }


@torch.inference_mode()
def predict_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str,
    combine: Callable[[Any], torch.Tensor],
    amp_enabled: bool,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_targets, all_logits, all_indices = [], [], []
    loss_sum, count = 0.0, 0
    for images, targets, indices in tqdm(loader, desc=description):
        images = images.to(device, non_blocking=True)
        device_targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            logits = combine(model(images)).float()
        loss_sum += F.cross_entropy(logits, device_targets, reduction="sum").item()
        count += targets.size(0)
        all_targets.append(targets)
        all_logits.append(logits.cpu())
        all_indices.append(indices)
    if count == 0:
        raise ValueError(f"{description} dataset is empty")
    return (
        loss_sum / count,
        torch.cat(all_targets).numpy(),
        torch.cat(all_logits).numpy(),
        torch.cat(all_indices).numpy(),
    )


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)


def nll_from_probabilities(
    targets: np.ndarray, probabilities: np.ndarray
) -> float:
    chosen = probabilities[np.arange(len(targets)), targets]
    return float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean())


def per_class_statistics(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
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
        support = int(np.sum(targets == class_id))
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        rows.append(
            {
                "class_id": class_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return rows


def compute_metrics(
    targets: np.ndarray, probabilities: np.ndarray, num_classes: int
) -> dict[str, float]:
    if probabilities.shape != (len(targets), num_classes):
        raise ValueError("Targets and probabilities have incompatible shapes")
    max_k = min(5, num_classes)
    top_order = np.argsort(-probabilities, axis=1)[:, :max_k]
    metrics = {
        f"top{k}_accuracy": float(
            np.mean(np.any(top_order[:, :k] == targets[:, None], axis=1))
        )
        for k in range(1, max_k + 1)
    }
    class_rows = per_class_statistics(targets, top_order[:, 0], num_classes)
    metrics.update(
        macro_precision=float(np.mean([row["precision"] for row in class_rows])),
        macro_recall=float(np.mean([row["recall"] for row in class_rows])),
        macro_f1=float(np.mean([row["f1"] for row in class_rows])),
    )
    return metrics


def save_prediction_archive(
    path: Path,
    targets: np.ndarray,
    indices: np.ndarray,
    **arrays: np.ndarray,
) -> None:
    np.savez_compressed(path, targets=targets, indices=indices, **arrays)


def save_evaluation_outputs(
    output_dir: Path,
    dataset: INatJsonDataset,
    targets: np.ndarray,
    probabilities: np.ndarray,
    indices: np.ndarray,
    loss: float,
    title: str,
    report_name: str = "classification_report.txt",
    predictions_name: str = "test_predictions.csv",
    metrics_name: str = "test_metrics.json",
    metadata: dict[str, Any] | None = None,
) -> dict[str, float]:
    metrics = compute_metrics(targets, probabilities, probabilities.shape[1])
    predictions = probabilities.argmax(axis=1)
    class_rows = per_class_statistics(
        targets, predictions, probabilities.shape[1]
    )
    lines = [title, f"loss: {loss:.6f}", f"num_samples: {len(targets)}"]
    if metadata:
        lines.extend(f"{key}: {value}" for key, value in metadata.items())
    lines.extend(
        f"{name}: {value:.6f} ({100.0 * value:.2f}%)"
        for name, value in metrics.items()
    )
    lines.extend(
        [
            "",
            "Per-class metrics",
            "class  precision  recall  f1  support",
        ]
    )
    lines.extend(
        f"{int(row['class_id']):03d}  {float(row['precision']):.6f}  "
        f"{float(row['recall']):.6f}  {float(row['f1']):.6f}  "
        f"{int(row['support'])}"
        for row in class_rows
    )
    report = "\n".join(lines) + "\n"
    (output_dir / report_name).write_text(report, encoding="utf-8")

    metrics_payload: dict[str, Any] = {
        "loss": loss,
        "num_samples": len(targets),
        **metrics,
        "per_class": class_rows,
    }
    if metadata:
        metrics_payload["metadata"] = metadata
    with (output_dir / metrics_name).open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    with (output_dir / predictions_name).open(
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
    return metrics


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
            "architecture": "pmg-gap-isqrt-cov",
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_top1": best_val,
            "args": serializable_args(args),
        },
        path,
    )


def read_checkpoint(path: Path, device: torch.device | str = "cpu") -> Any:
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a state_dict")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    return state


def checkpoint_arguments(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("args"), dict):
        return checkpoint["args"]
    return {}


def load_model_weights(
    model: nn.Module, path: Path, device: torch.device
) -> Any:
    del device
    checkpoint = read_checkpoint(path, "cpu")
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    return checkpoint


PMG_PRETRAINED_PREFIXES = (
    "backbone.",
    "conv_block1.",
    "conv_block2.",
    "conv_block3.",
    "classifier1.",
    "classifier2.",
    "classifier3.",
    "classifier_concat.",
)


def load_pmg_pretrained_weights(
    model: PMGISqrtCovClassifier,
    path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Load only backbone and PMG tensors from a resnet-pmg.py checkpoint."""
    checkpoint = read_checkpoint(path, "cpu")
    source_state = checkpoint_state(checkpoint)
    source_args = checkpoint_arguments(checkpoint)

    if "num_classes" in source_args:
        source_num_classes = int(source_args["num_classes"])
        if source_num_classes != args.num_classes:
            raise ValueError(
                "PMG pretrained checkpoint num_classes mismatch: "
                f"checkpoint={source_num_classes}, current={args.num_classes}"
            )
    if "feature_size" in source_args:
        source_feature_size = int(source_args["feature_size"])
        if source_feature_size != args.feature_size:
            raise ValueError(
                "PMG pretrained checkpoint feature_size mismatch: "
                f"checkpoint={source_feature_size}, current={args.feature_size}"
            )

    target_state = model.state_dict()
    pmg_target_keys = {
        key
        for key in target_state
        if key.startswith(PMG_PRETRAINED_PREFIXES)
    }
    missing = sorted(pmg_target_keys.difference(source_state))
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            "PMG pretrained checkpoint is missing required tensors: "
            f"{preview}{' ...' if len(missing) > 10 else ''}"
        )

    shape_mismatches = []
    for key in sorted(pmg_target_keys):
        source_shape = tuple(source_state[key].shape)
        target_shape = tuple(target_state[key].shape)
        if source_shape != target_shape:
            shape_mismatches.append(
                f"{key}: checkpoint={source_shape}, current={target_shape}"
            )
    if shape_mismatches:
        raise ValueError(
            "PMG pretrained checkpoint has incompatible tensor shapes:\n"
            + "\n".join(shape_mismatches[:10])
        )

    selected_state = {key: source_state[key] for key in pmg_target_keys}
    incompatible = model.load_state_dict(selected_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys while loading PMG initialization: "
            + ", ".join(incompatible.unexpected_keys)
        )

    summary = {
        "initialization": "ImageNet-1K ResNet-50, then PMG checkpoint overlay",
        "pmg_checkpoint": str(path),
        "source_epoch": (
            int(checkpoint["epoch"])
            if isinstance(checkpoint, dict) and "epoch" in checkpoint
            else None
        ),
        "source_best_val_top1": (
            float(checkpoint["best_val_top1"])
            if isinstance(checkpoint, dict) and "best_val_top1" in checkpoint
            else None
        ),
        "loaded_tensor_count": len(selected_state),
        "loaded_parameter_and_buffer_values": int(
            sum(tensor.numel() for tensor in selected_state.values())
        ),
        "loaded_modules": [
            "backbone",
            "conv_block1/2/3",
            "classifier1/2/3",
            "classifier_concat",
        ],
        "new_modules": [
            "gap_classifier",
            "cov_pool",
            "cov_classifier",
            "cov_scale_logit",
        ],
    }
    print(json.dumps(summary, indent=2))
    return summary


def metadata_value(
    metadata: dict[str, Any], key: str, fallback: Any, cast: Callable[[Any], Any]
) -> Any:
    value = metadata.get(key, fallback)
    return cast(value)


def build_optimizer(
    model: PMGISqrtCovClassifier, args: argparse.Namespace
) -> torch.optim.Optimizer:
    groups = model.optimizer_parameter_groups(args)
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def save_hybrid_fusion_summary(
    output_dir: Path,
    model: PMGISqrtCovClassifier,
    hybrid_branch_weight: float,
) -> None:
    payload = {
        "gap_cov_formula": (
            "gap_cov_logits = gap_logits + cov_scale * cov_logits"
        ),
        "hybrid_formula": (
            "final_logits = sum(pmg_logits) + hybrid_gap_cov_weight * "
            "gap_cov_logits"
        ),
        "cov_scale": float(model.cov_scale.detach().float().cpu().item()),
        "cov_scale_logit": float(
            model.cov_scale_logit.detach().float().cpu().item()
        ),
        "hybrid_gap_cov_weight": hybrid_branch_weight,
    }
    with (output_dir / "fusion_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(payload, file, indent=2)


def run_hybrid(args: argparse.Namespace, device: torch.device) -> None:
    train_transform, eval_transform = make_pmg_transforms(
        args.model_name, args.image_size, args.resize_size
    )
    test_set = INatJsonDataset(
        args.data_dir, "test", eval_transform, args.num_classes
    )
    test_loader = make_loader(
        test_set,
        args.eval_batch_size,
        args.num_workers,
        False,
        device,
        args.seed,
    )
    model = PMGISqrtCovClassifier(
        model_name=args.model_name,
        num_classes=args.num_classes,
        feature_size=args.feature_size,
        cov_dim=args.cov_dim,
        sqrt_iters=args.sqrt_iters,
        cov_eps=args.cov_eps,
        cov_dropout=args.cov_dropout,
        cov_initial_scale=args.cov_initial_scale,
    ).to(device)

    if args.pmg_pretrained_checkpoint is not None:
        initialization_summary = load_pmg_pretrained_weights(
            model, args.pmg_pretrained_checkpoint, args
        )
        with (args.output_dir / "initialization_summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(initialization_summary, file, indent=2)

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
            True,
            device,
            args.seed,
            drop_last=True,
        )
        val_loader = make_loader(
            val_set,
            args.eval_batch_size,
            args.num_workers,
            False,
            device,
            args.seed,
        )
        optimizer = build_optimizer(model, args)
        amp_enabled = device.type == "cuda" and not args.no_amp
        scaler = make_grad_scaler(amp_enabled)
        start_epoch, best_val = 0, -1.0
        if args.resume is not None:
            checkpoint = load_model_weights(model, args.resume, device)
            if not isinstance(checkpoint, dict):
                raise ValueError("--resume requires a complete training checkpoint")
            if "optimizer" not in checkpoint or "scaler" not in checkpoint:
                raise ValueError("Resume checkpoint lacks optimizer/scaler state")
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val = float(checkpoint.get("best_val_top1", -1.0))

        history_path = args.output_dir / "training_history.jsonl"
        if args.resume is None:
            history_path.write_text("", encoding="utf-8")
        for epoch in range(start_epoch, args.epochs):
            started = time.time()
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scaler, device, args, epoch
            )
            val_loss, val_targets, val_logits, _ = predict_logits(
                model,
                val_loader,
                device,
                "Validation",
                lambda outputs: combine_hybrid_logits(
                    outputs, args.hybrid_cov_logit_weight
                ),
                amp_enabled,
            )
            val_metrics = compute_metrics(
                val_targets, softmax_numpy(val_logits), args.num_classes
            )
            row = {
                "epoch": epoch + 1,
                "seconds": time.time() - started,
                "cov_scale": float(
                    model.cov_scale.detach().float().cpu().item()
                ),
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
            raise RuntimeError("No best checkpoint was produced")
        load_model_weights(model, best_path, device)

    save_hybrid_fusion_summary(
        args.output_dir, model, args.hybrid_cov_logit_weight
    )
    test_loss, targets, logits, indices = predict_logits(
        model,
        test_loader,
        device,
        "Test",
        lambda outputs: combine_hybrid_logits(
            outputs, args.hybrid_cov_logit_weight
        ),
        device.type == "cuda" and not args.no_amp,
    )
    probabilities = softmax_numpy(logits)
    save_prediction_archive(
        args.output_dir / "test_logits_probabilities.npz",
        targets,
        indices,
        logits=logits,
        probabilities=probabilities,
    )
    save_evaluation_outputs(
        args.output_dir,
        test_set,
        targets,
        probabilities,
        indices,
        test_loss,
        "Test classification report (hybrid PMG + GAP + iSQRT-COV logits)",
        metadata={
            "learned_cov_scale": float(
                model.cov_scale.detach().float().cpu().item()
            ),
            "hybrid_gap_cov_logit_weight": args.hybrid_cov_logit_weight,
            "checkpoint": str(args.checkpoint) if args.test_only else "best.pt",
        },
    )


def run_separate_prediction(
    kind: str,
    checkpoint_path: Path,
    split: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[INatJsonDataset, float, np.ndarray, np.ndarray, np.ndarray]:
    checkpoint = read_checkpoint(checkpoint_path, "cpu")
    metadata = checkpoint_arguments(checkpoint)
    model_name = str(metadata.get("model_name", args.model_name))
    num_classes = metadata_value(metadata, "num_classes", args.num_classes, int)
    if num_classes != args.num_classes:
        raise ValueError(
            f"{kind} checkpoint has {num_classes} classes, expected {args.num_classes}"
        )

    if kind == "pmg":
        feature_size = metadata_value(
            metadata, "feature_size", args.feature_size, int
        )
        image_size = metadata_value(metadata, "image_size", args.image_size, int)
        resize_size = metadata_value(
            metadata, "resize_size", args.resize_size, int
        )
        _, transform = make_pmg_transforms(
            model_name, image_size, resize_size
        )
        model: nn.Module = PMGClassifier(
            model_name, num_classes, feature_size
        )
        combine: Callable[[Any], torch.Tensor] = lambda outputs: (
            outputs[0] + outputs[1] + outputs[2] + outputs[3]
        )
    elif kind == "cov":
        cov_dim = metadata_value(metadata, "cov_dim", args.cov_dim, int)
        sqrt_iters = metadata_value(
            metadata, "sqrt_iters", args.sqrt_iters, int
        )
        cov_eps = metadata_value(metadata, "cov_eps", args.cov_eps, float)
        cov_initial_scale = metadata_value(
            metadata, "cov_initial_scale", args.cov_initial_scale, float
        )
        image_size = metadata_value(metadata, "image_size", args.image_size, int)
        transform = make_cov_eval_transform(model_name, image_size)
        model = ResNet50GAPISqrtCovClassifier(
            model_name,
            num_classes,
            cov_dim,
            sqrt_iters,
            cov_eps,
            cov_initial_scale,
        )
        combine = lambda output: output
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    del checkpoint
    model.to(device)
    dataset = INatJsonDataset(args.data_dir, split, transform, num_classes)
    loader = make_loader(
        dataset,
        args.eval_batch_size,
        args.num_workers,
        False,
        device,
        args.seed,
    )
    loss, targets, logits, indices = predict_logits(
        model,
        loader,
        device,
        f"{split.capitalize()} ({kind})",
        combine,
        device.type == "cuda" and not args.no_amp,
    )
    del model, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return dataset, loss, targets, logits, indices


def verify_alignment(
    first_targets: np.ndarray,
    first_indices: np.ndarray,
    second_targets: np.ndarray,
    second_indices: np.ndarray,
    split: str,
) -> None:
    if not np.array_equal(first_indices, second_indices):
        raise RuntimeError(f"{split} prediction indices are not aligned")
    if not np.array_equal(first_targets, second_targets):
        raise RuntimeError(f"{split} targets differ between PMG and COV loaders")


def ensemble_probabilities(
    pmg_logits: np.ndarray,
    cov_logits: np.ndarray,
    alpha: float,
    space: str,
) -> tuple[np.ndarray, np.ndarray]:
    weighted_logits = alpha * pmg_logits + (1.0 - alpha) * cov_logits
    if space == "logits":
        return softmax_numpy(weighted_logits), weighted_logits.astype(np.float32)
    probabilities = (
        alpha * softmax_numpy(pmg_logits)
        + (1.0 - alpha) * softmax_numpy(cov_logits)
    )
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32), weighted_logits.astype(np.float32)


def search_alpha(
    targets: np.ndarray,
    pmg_logits: np.ndarray,
    cov_logits: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, list[dict[str, float]], dict[str, float]]:
    alphas = np.arange(0.0, 1.0 + args.alpha_step * 0.5, args.alpha_step)
    alphas = np.unique(np.clip(np.append(alphas, 1.0), 0.0, 1.0))
    rows: list[dict[str, float]] = []
    for alpha_value in alphas:
        alpha = float(alpha_value)
        probabilities, _ = ensemble_probabilities(
            pmg_logits, cov_logits, alpha, args.ensemble_space
        )
        metrics = compute_metrics(targets, probabilities, args.num_classes)
        nll = nll_from_probabilities(targets, probabilities)
        rows.append(
            {
                "alpha": alpha,
                "top1_accuracy": metrics["top1_accuracy"],
                "top5_accuracy": metrics.get("top5_accuracy", 0.0),
                "macro_f1": metrics["macro_f1"],
                "nll": nll,
            }
        )

    def ranking(row: dict[str, float]) -> tuple[float, float, float, float]:
        if args.ensemble_metric == "top1":
            primary = row["top1_accuracy"]
        elif args.ensemble_metric == "macro_f1":
            primary = row["macro_f1"]
        else:
            primary = -row["nll"]
        return (
            primary,
            row["top1_accuracy"],
            row["macro_f1"],
            -abs(row["alpha"] - 0.5),
        )

    best = max(rows, key=ranking)
    return best["alpha"], rows, best


def save_alpha_search(
    output_dir: Path,
    rows: list[dict[str, float]],
    selection: dict[str, Any],
) -> None:
    with (output_dir / "ensemble_alpha_search.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "ensemble_selection.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(selection, file, indent=2)


def run_ensemble(args: argparse.Namespace, device: torch.device) -> None:
    # Validation is completed for both models before alpha is selected.
    val_set, _, val_targets, pmg_val_logits, val_indices = run_separate_prediction(
        "pmg", args.pmg_checkpoint, "val", args, device
    )
    _, _, cov_val_targets, cov_val_logits, cov_val_indices = run_separate_prediction(
        "cov", args.cov_checkpoint, "val", args, device
    )
    verify_alignment(
        val_targets, val_indices, cov_val_targets, cov_val_indices, "validation"
    )
    pmg_val_probabilities = softmax_numpy(pmg_val_logits)
    cov_val_probabilities = softmax_numpy(cov_val_logits)
    save_prediction_archive(
        args.output_dir / "validation_model_outputs.npz",
        val_targets,
        val_indices,
        pmg_logits=pmg_val_logits,
        pmg_probabilities=pmg_val_probabilities,
        cov_logits=cov_val_logits,
        cov_probabilities=cov_val_probabilities,
    )

    alpha, search_rows, best_row = search_alpha(
        val_targets, pmg_val_logits, cov_val_logits, args
    )
    val_probabilities, val_weighted_logits = ensemble_probabilities(
        pmg_val_logits, cov_val_logits, alpha, args.ensemble_space
    )
    val_metrics = compute_metrics(
        val_targets, val_probabilities, args.num_classes
    )
    val_nll = nll_from_probabilities(val_targets, val_probabilities)
    selection = {
        "alpha": alpha,
        "formula": "alpha * pmg + (1 - alpha) * cov",
        "ensemble_space": args.ensemble_space,
        "selection_split": "validation",
        "selection_metric": args.ensemble_metric,
        "alpha_step": args.alpha_step,
        "validation_nll": val_nll,
        "validation_metrics": val_metrics,
        "selected_grid_row": best_row,
        "pmg_checkpoint": str(args.pmg_checkpoint),
        "cov_checkpoint": str(args.cov_checkpoint),
    }
    save_alpha_search(args.output_dir, search_rows, selection)
    save_prediction_archive(
        args.output_dir / "validation_ensemble_outputs.npz",
        val_targets,
        val_indices,
        weighted_logits=val_weighted_logits,
        final_probabilities=val_probabilities,
    )
    save_evaluation_outputs(
        args.output_dir,
        val_set,
        val_targets,
        val_probabilities,
        val_indices,
        val_nll,
        "Validation classification report (checkpoint ensemble)",
        report_name="validation_classification_report.txt",
        predictions_name="validation_predictions.csv",
        metrics_name="validation_metrics.json",
        metadata={
            "selected_alpha": alpha,
            "ensemble_space": args.ensemble_space,
            "selection_metric": args.ensemble_metric,
        },
    )

    test_set, _, test_targets, pmg_test_logits, test_indices = run_separate_prediction(
        "pmg", args.pmg_checkpoint, "test", args, device
    )
    _, _, cov_test_targets, cov_test_logits, cov_test_indices = run_separate_prediction(
        "cov", args.cov_checkpoint, "test", args, device
    )
    verify_alignment(
        test_targets, test_indices, cov_test_targets, cov_test_indices, "test"
    )
    test_probabilities, test_weighted_logits = ensemble_probabilities(
        pmg_test_logits, cov_test_logits, alpha, args.ensemble_space
    )
    test_nll = nll_from_probabilities(test_targets, test_probabilities)
    save_prediction_archive(
        args.output_dir / "test_model_and_ensemble_outputs.npz",
        test_targets,
        test_indices,
        pmg_logits=pmg_test_logits,
        pmg_probabilities=softmax_numpy(pmg_test_logits),
        cov_logits=cov_test_logits,
        cov_probabilities=softmax_numpy(cov_test_logits),
        weighted_logits=test_weighted_logits,
        final_probabilities=test_probabilities,
    )
    save_evaluation_outputs(
        args.output_dir,
        test_set,
        test_targets,
        test_probabilities,
        test_indices,
        test_nll,
        "Test classification report (validation-selected checkpoint ensemble)",
        metadata={
            "alpha_selected_on_validation": alpha,
            "ensemble_space": args.ensemble_space,
            "selection_metric": args.ensemble_metric,
            "pmg_checkpoint": str(args.pmg_checkpoint),
            "cov_checkpoint": str(args.cov_checkpoint),
        },
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
        json.dump(serializable_args(args), file, indent=2)

    if args.ensemble_only:
        run_ensemble(args, device)
    else:
        run_hybrid(args, device)


if __name__ == "__main__":
    main()
