#!/usr/bin/env python3
"""ResNet-50 + Compact Bilinear Pooling + API-Net for iNat2021-mini."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, ResNetModel


MODEL_NAME = "microsoft/resnet-50"
DEFAULT_DATA_DIR = None
DEFAULT_OUTPUT_DIR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full fine-tuning of ResNet-50 with CBP and API-Net"
    )
    parser.add_argument("--mode", choices=("train", "ensemble"), default="train")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--num-classes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")

    # Optimisation.  The backbone is never frozen: this is full fine-tuning.
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--api-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # CBP descriptor.
    parser.add_argument("--cbp-output-dim", type=int, default=8192)
    parser.add_argument("--cbp-seed", type=int, default=1)
    parser.add_argument(
        "--cbp-spatial-chunk-size",
        type=int,
        default=0,
        help="Spatial positions per FFT chunk; 0 processes all H*W positions",
    )
    parser.add_argument("--no-signed-sqrt", action="store_true")
    parser.add_argument("--no-l2-normalize", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.3)

    # API-Net interaction and loss.
    parser.add_argument("--api-hidden-size", type=int, default=512)
    parser.add_argument("--plain-ce-weight", type=float, default=1.0)
    parser.add_argument("--api-ce-weight", type=float, default=0.25)
    parser.add_argument("--rank-margin", type=float, default=0.2)
    parser.add_argument("--rank-weight", type=float, default=0.1)
    parser.add_argument(
        "--supcon-weight",
        type=float,
        default=0.05,
        help="Weight of supervised contrastive loss on the inference-time CBP descriptor",
    )
    parser.add_argument(
        "--supcon-temperature",
        type=float,
        default=0.1,
        help="Temperature for in-batch supervised contrastive loss",
    )

    # Optional second-stage training from a standalone CBP best checkpoint.
    parser.add_argument(
        "--init-cbp-checkpoint",
        type=Path,
        default=None,
        help=(
            "Initialise the combined model's backbone, CBP sketch buffers and "
            "classifier from a best.pt produced by local cbp.py"
        ),
    )
    parser.add_argument(
        "--stage2-train-scope",
        choices=("full", "api-heads"),
        default="full",
        help=(
            "With --init-cbp-checkpoint: 'full' fine-tunes backbone+CBP classifier+API; "
            "'api-heads' freezes ResNet and trains the shared classifier/API maps"
        ),
    )
    parser.add_argument(
        "--no-init-cbp-classifier",
        action="store_true",
        help="Load the CBP backbone/sketch state but leave the combined classifier random",
    )

    # Confusing classes are preferentially co-located in API training batches.
    parser.add_argument("--class-neighbors-json", type=Path, default=None)
    parser.add_argument("--neighbor-batch-fraction", type=float, default=0.25)
    parser.add_argument("--max-neighbors-per-class", type=int, default=5)

    # Training checkpoint workflow.
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)

    # Validation-only model-level fusion of standalone CBP and API checkpoints.
    parser.add_argument("--cbp-checkpoint", type=Path, default=None)
    parser.add_argument("--api-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--ensemble-model",
        action="append",
        nargs=3,
        metavar=("NAME", "TYPE", "CHECKPOINT"),
        help=(
            "Repeat for a multi-model ensemble. TYPE is cbp, api, or combined. "
            "Example: --ensemble-model full combined /path/best.pt"
        ),
    )
    parser.add_argument(
        "--ensemble-search-trials",
        type=int,
        default=500,
        help="Dirichlet weight samples for ensembles with more than two models",
    )
    parser.add_argument("--alpha-steps", type=int, default=101)
    parser.add_argument(
        "--alpha-metric", choices=("top1", "macro_f1"), default="top1"
    )
    parser.add_argument(
        "--fusion-space",
        choices=("logits", "probabilities"),
        default="logits",
        help="The requested formula uses logits; probability fusion is optional",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_classes < 2:
        raise ValueError("--num-classes must be at least 2")
    if args.eval_batch_size < 1 or args.num_workers < 0 or args.image_size < 1:
        raise ValueError("Invalid evaluation/data-loader argument")
    if args.mode == "ensemble":
        if args.ensemble_model is None and (
            args.cbp_checkpoint is None or args.api_checkpoint is None
        ):
            raise ValueError(
                "--mode ensemble requires repeated --ensemble-model arguments, "
                "or the legacy --cbp-checkpoint/--api-checkpoint pair"
            )
        if args.ensemble_model is not None:
            if len(args.ensemble_model) < 2:
                raise ValueError("A multi-model ensemble requires at least two models")
            names = [entry[0] for entry in args.ensemble_model]
            kinds = [entry[1] for entry in args.ensemble_model]
            if len(set(names)) != len(names):
                raise ValueError("Every --ensemble-model NAME must be unique")
            if any(kind not in ("cbp", "api", "combined") for kind in kinds):
                raise ValueError("--ensemble-model TYPE must be cbp, api, or combined")
        if args.alpha_steps < 2:
            raise ValueError("--alpha-steps must be at least 2")
        if args.ensemble_search_trials < 0:
            raise ValueError("--ensemble-search-trials must be non-negative")
        return

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.samples_per_class < 2:
        raise ValueError("--samples-per-class must be at least 2 for API-Net")
    if args.batch_size < 2 * args.samples_per_class:
        raise ValueError("Each training batch must contain at least two classes")
    if args.batch_size % args.samples_per_class:
        raise ValueError("--batch-size must be divisible by --samples-per-class")
    if args.cbp_output_dim < 2 or args.cbp_spatial_chunk_size < 0:
        raise ValueError("Invalid CBP dimensions")
    if args.api_hidden_size < 1 or not 0.0 <= args.dropout < 1.0:
        raise ValueError("Invalid API hidden size or dropout")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    nonnegative = (
        args.weight_decay,
        args.warmup_epochs,
        args.grad_clip,
        args.plain_ce_weight,
        args.api_ce_weight,
        args.rank_margin,
        args.rank_weight,
        args.supcon_weight,
    )
    if any(value < 0 for value in nonnegative):
        raise ValueError("Loss/schedule/regularisation values must be non-negative")
    if (
        args.plain_ce_weight
        + args.api_ce_weight
        + args.rank_weight
        + args.supcon_weight
        == 0
    ):
        raise ValueError("At least one loss weight must be positive")
    if args.supcon_temperature <= 0:
        raise ValueError("--supcon-temperature must be positive")
    if not 0.0 <= args.neighbor_batch_fraction <= 1.0:
        raise ValueError("--neighbor-batch-fraction must be in [0, 1]")
    if args.max_neighbors_per_class < 1:
        raise ValueError("--max-neighbors-per-class must be positive")
    rates = [args.backbone_lr, args.classifier_lr, args.api_lr]
    if args.lr is not None:
        rates.append(args.lr)
    if any(rate <= 0 for rate in rates):
        raise ValueError("All learning rates must be positive")
    if args.test_only and args.checkpoint is None:
        raise ValueError("--test-only requires --checkpoint")
    if args.test_only and args.resume is not None:
        raise ValueError("--test-only cannot be combined with --resume")
    if args.resume is not None and args.init_cbp_checkpoint is not None:
        raise ValueError("--resume and --init-cbp-checkpoint are mutually exclusive")
    if args.test_only and args.init_cbp_checkpoint is not None:
        raise ValueError("--test-only cannot also use --init-cbp-checkpoint")
    if (
        args.stage2_train_scope == "api-heads"
        and args.init_cbp_checkpoint is None
        and args.resume is None
    ):
        raise ValueError(
            "--stage2-train-scope api-heads requires --init-cbp-checkpoint or --resume"
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(_: int) -> None:
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
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"{json_path} must contain a non-empty JSON list")

        self.records = raw
        self.labels: list[int] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict) or "file_name" not in item or "label" not in item:
                raise ValueError(f"Invalid record {index} in {json_path}")
            label = int(item["label"])
            if not 0 <= label < num_classes:
                raise ValueError(f"Label {label} is outside [0, {num_classes - 1}]")
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.records)

    def _image_path(self, file_name: str) -> Path:
        path = Path(file_name)
        if path.is_absolute():
            return path
        direct = self.data_dir / path
        if direct.is_file():
            return direct
        # Also accept processed_dataset/train/000/image.jpg.
        if path.parts and path.parts[0] == self.data_dir.name:
            return self.data_dir.parent / path
        return direct

    def __getitem__(self, index: int):
        item = self.records[index]
        image_path = self._image_path(str(item["file_name"]))
        try:
            with Image.open(image_path) as image:
                pixel_values = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {image_path}") from exc
        return pixel_values, self.labels[index], index


class ClassBalancedBatchSampler(Sampler):
    """Every API batch has same-class and different-class candidates."""

    def __init__(
        self,
        labels: list[int],
        batch_size: int,
        samples_per_class: int,
        seed: int,
        class_neighbors: dict[int, list[int]] | None = None,
        neighbor_batch_fraction: float = 0.0,
    ):
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.classes_per_batch = batch_size // samples_per_class
        self.seed = seed
        self.class_neighbors = class_neighbors or {}
        self.neighbor_batch_fraction = neighbor_batch_fraction
        self.epoch = 0
        self.num_batches = len(labels) // batch_size
        self.class_indices: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            self.class_indices.setdefault(label, []).append(index)
        self.class_ids = sorted(self.class_indices)
        if len(self.class_ids) < self.classes_per_batch:
            raise ValueError("Not enough classes for one balanced batch")
        small = [key for key, values in self.class_indices.items() if len(values) < samples_per_class]
        if small:
            raise ValueError(f"Classes with too few samples: {small[:10]}")
        present_classes = set(self.class_ids)
        self.class_neighbors = {
            class_id: [
                neighbor
                for neighbor in neighbors
                if neighbor in present_classes and neighbor != class_id
            ]
            for class_id, neighbors in self.class_neighbors.items()
            if class_id in present_classes
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @staticmethod
    def _shuffle(values: list[int], generator: torch.Generator) -> list[int]:
        order = torch.randperm(len(values), generator=generator).tolist()
        return [values[i] for i in order]

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        pools = {key: self._shuffle(values, generator) for key, values in self.class_indices.items()}
        cursors = {key: 0 for key in self.class_ids}
        usage = {key: 0 for key in self.class_ids}
        for _ in range(self.num_batches):
            jitter = torch.rand(len(self.class_ids), generator=generator).tolist()
            ordered = sorted(
                zip(self.class_ids, jitter), key=lambda pair: (usage[pair[0]], pair[1])
            )
            anchor = ordered[0][0]
            chosen = [anchor]

            # Fill a configurable fraction of remaining class slots with the
            # anchor's CBP-confusing neighbours.  Feature-distance mining in
            # APIMixin then chooses the actual negative image inside the batch.
            neighbor_slots = min(
                self.classes_per_batch - 1,
                int(round((self.classes_per_batch - 1) * self.neighbor_batch_fraction)),
            )
            neighbor_ids = self.class_neighbors.get(anchor, [])
            neighbor_rank = {class_id: rank for rank, class_id in enumerate(neighbor_ids)}
            neighbor_jitter = torch.rand(len(neighbor_ids), generator=generator).tolist()
            ordered_neighbors = sorted(
                zip(neighbor_ids, neighbor_jitter),
                key=lambda pair: (usage[pair[0]], neighbor_rank[pair[0]], pair[1]),
            )
            chosen.extend(class_id for class_id, _ in ordered_neighbors[:neighbor_slots])
            if len(chosen) < self.classes_per_batch:
                chosen_set = set(chosen)
                chosen.extend(
                    class_id for class_id, _ in ordered if class_id not in chosen_set
                )
                chosen = chosen[: self.classes_per_batch]
            batch: list[int] = []
            for class_id in chosen:
                start = cursors[class_id]
                end = start + self.samples_per_class
                if end > len(pools[class_id]):
                    pools[class_id] = self._shuffle(self.class_indices[class_id], generator)
                    start, end = 0, self.samples_per_class
                batch.extend(pools[class_id][start:end])
                cursors[class_id] = end
                usage[class_id] += 1
            permutation = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[i] for i in permutation]

    def __len__(self) -> int:
        return self.num_batches


class CountSketch(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hash_seed: int, sign_seed: int):
        super().__init__()
        hashes = np.random.RandomState(hash_seed).randint(0, output_dim, size=input_dim)
        signs = 2 * np.random.RandomState(sign_seed).randint(0, 2, size=input_dim) - 1
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.register_buffer("hashes", torch.from_numpy(hashes.astype(np.int64)))
        self.register_buffer("signs", torch.from_numpy(signs.astype(np.float32)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(f"Expected last dimension {self.input_dim}")
        output = inputs.new_zeros(*inputs.shape[:-1], self.output_dim)
        shape = (1,) * (inputs.ndim - 1) + (self.input_dim,)
        indices = self.hashes.view(shape).expand_as(inputs)
        signs = self.signs.to(dtype=inputs.dtype).view(shape)
        return output.scatter_add_(-1, indices, inputs * signs)


class CompactBilinearPooling(nn.Module):
    """Feature-map-level Tensor Sketch followed by spatial sum pooling."""

    def __init__(
        self,
        input_dim1: int,
        input_dim2: int,
        output_dim: int,
        *,
        seed: int = 1,
        spatial_chunk_size: int = 0,
    ):
        super().__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.spatial_chunk_size = spatial_chunk_size
        # Same four deterministic seeds as local cbp.py.
        self.sketch1 = CountSketch(input_dim1, output_dim, seed, seed + 2)
        self.sketch2 = CountSketch(input_dim2, output_dim, seed + 4, seed + 6)

    def _tensor_sketch(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        sketch1 = self.sketch1(first)
        sketch2 = self.sketch2(second)
        fft1 = torch.fft.rfft(sketch1, n=self.output_dim, dim=-1)
        fft2 = torch.fft.rfft(sketch2, n=self.output_dim, dim=-1)
        return torch.fft.irfft(fft1 * fft2, n=self.output_dim, dim=-1)

    def forward(self, first_map: torch.Tensor, second_map: torch.Tensor | None = None) -> torch.Tensor:
        if second_map is None:
            second_map = first_map
        if first_map.ndim != 4 or second_map.ndim != 4:
            raise ValueError("CBP inputs must be BxCxHxW")
        if first_map.shape[0] != second_map.shape[0] or first_map.shape[2:] != second_map.shape[2:]:
            raise ValueError("CBP input shapes are incompatible")
        if first_map.shape[1] != self.input_dim1 or second_map.shape[1] != self.input_dim2:
            raise ValueError("CBP input channel count is incorrect")
        first = first_map.flatten(2).transpose(1, 2).contiguous()
        second = second_map.flatten(2).transpose(1, 2).contiguous()
        positions = first.shape[1]
        chunk = self.spatial_chunk_size or positions
        pooled = first.new_zeros(first.shape[0], self.output_dim)
        for start in range(0, positions, chunk):
            end = min(start + chunk, positions)
            pooled = pooled + self._tensor_sketch(
                first[:, start:end], second[:, start:end]
            ).sum(1)
        return pooled


def normalize_cbp(descriptor: torch.Tensor, signed_sqrt: bool, l2_normalize: bool) -> torch.Tensor:
    if signed_sqrt:
        descriptor = torch.sign(descriptor) * torch.sqrt(torch.abs(descriptor) + 1e-8)
    if l2_normalize:
        descriptor = F.normalize(descriptor, p=2, dim=1, eps=1e-12)
    return descriptor


class APIMixin:
    """API-Net pair mining/interaction shared by combined and standalone models."""

    map1: nn.Linear
    map2: nn.Linear
    classifier: nn.Linear
    dropout: nn.Dropout

    @staticmethod
    def mine_pairs(features: torch.Tensor, targets: torch.Tensor):
        with torch.no_grad():
            detached = features.detach().float()
            norms = detached.square().sum(1, keepdim=True)
            distances = (norms + norms.T - 2.0 * detached @ detached.T).clamp_min_(0.0)
            same = targets[:, None].eq(targets[None, :])
            same.fill_diagonal_(False)
            different = targets[:, None].ne(targets[None, :])
            if not bool(same.any(1).all()):
                raise ValueError("Every API anchor needs another sample of the same class")
            if not bool(different.any(1).all()):
                raise ValueError("Every API batch needs at least two classes")
            same_indices = distances.masked_fill(~same, float("inf")).argmin(1)
            different_indices = distances.masked_fill(~different, float("inf")).argmin(1)
            rows = torch.arange(features.size(0), device=features.device)
        return same_indices, different_indices, distances[rows, same_indices], distances[rows, different_indices]

    def interact(self, features: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        same_indices, different_indices, same_distances, different_distances = self.mine_pairs(features, targets)
        anchors = torch.arange(features.size(0), device=features.device)
        first_indices = torch.cat((anchors, anchors))
        second_indices = torch.cat((same_indices, different_indices))
        features1, features2 = features[first_indices], features[second_indices]
        labels1, labels2 = targets[first_indices], targets[second_indices]
        mutual = self.map2(self.dropout(self.map1(torch.cat((features1, features2), dim=1))))
        gate1 = torch.sigmoid(mutual * features1)
        gate2 = torch.sigmoid(mutual * features2)
        features1_self = gate1 * features1 + features1
        features1_other = gate2 * features1 + features1
        features2_self = gate2 * features2 + features2
        features2_other = gate1 * features2 + features2
        return {
            "logit1_self": self.classifier(self.dropout(features1_self)),
            "logit1_other": self.classifier(self.dropout(features1_other)),
            "logit2_self": self.classifier(self.dropout(features2_self)),
            "logit2_other": self.classifier(self.dropout(features2_other)),
            "labels1": labels1,
            "labels2": labels2,
            "gate_difference": (gate1.float() - gate2.float()).abs().mean(),
            "same_pair_squared_distance": same_distances.mean(),
            "different_pair_squared_distance": different_distances.mean(),
        }


class ResNet50APICBP(nn.Module, APIMixin):
    """The trainable combined model: ResNet feature map -> CBP -> API-Net."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        cbp_output_dim: int,
        cbp_seed: int,
        cbp_spatial_chunk_size: int,
        signed_sqrt: bool,
        l2_normalize: bool,
        api_hidden_size: int,
        dropout: float,
    ):
        super().__init__()
        self.backbone = ResNetModel.from_pretrained(model_name)
        channels = int(self.backbone.config.hidden_sizes[-1])
        self.cbp = CompactBilinearPooling(
            channels,
            channels,
            cbp_output_dim,
            seed=cbp_seed,
            spatial_chunk_size=cbp_spatial_chunk_size,
        )
        self.signed_sqrt = signed_sqrt
        self.l2_normalize = l2_normalize
        self.map1 = nn.Linear(2 * cbp_output_dim, api_hidden_size)
        self.map2 = nn.Linear(api_hidden_size, cbp_output_dim)
        self.classifier = nn.Linear(cbp_output_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        for layer in (self.map1, self.map2, self.classifier):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            nn.init.zeros_(layer.bias)
        # Explicit full fine-tuning.
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(pixel_values=pixel_values).last_hidden_state
        # CUDA fp16 FFT has shape/hardware restrictions.  FP32 CBP is stable and
        # remains differentiable all the way into the backbone under AMP.
        with torch.autocast(device_type=feature_map.device.type, enabled=False):
            descriptor = self.cbp(feature_map.float(), feature_map.float())
            return normalize_cbp(descriptor, self.signed_sqrt, self.l2_normalize)

    def forward(self, pixel_values: torch.Tensor, targets: torch.Tensor | None = None):
        features = self.extract_features(pixel_values)
        plain_logits = self.classifier(features)
        if targets is None:
            return plain_logits
        return plain_logits, self.interact(features, targets), features

    def optimizer_parameter_groups(self, args: argparse.Namespace) -> list[dict[str, Any]]:
        if args.lr is not None:
            parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
            return [{"params": parameters, "lr": args.lr, "initial_lr": args.lr, "group_name": "all"}]
        candidates = [
            ("backbone", self.backbone.parameters(), args.backbone_lr),
            ("classifier", self.classifier.parameters(), args.classifier_lr),
            ("api", list(self.map1.parameters()) + list(self.map2.parameters()), args.api_lr),
        ]
        groups = []
        for name, parameters, learning_rate in candidates:
            trainable = [parameter for parameter in parameters if parameter.requires_grad]
            if trainable:
                groups.append({
                    "params": trainable,
                    "lr": learning_rate,
                    "initial_lr": learning_rate,
                    "group_name": name,
                })
        return groups


class StandaloneAPIModel(nn.Module, APIMixin):
    """State-dict-compatible inference model for local resnet-api.py."""

    def __init__(self, model_name: str, num_classes: int, api_hidden_size: int, dropout: float):
        super().__init__()
        self.backbone = ResNetModel.from_pretrained(model_name)
        self.feature_size = int(self.backbone.config.hidden_sizes[-1])
        self.map1 = nn.Linear(2 * self.feature_size, api_hidden_size)
        self.map2 = nn.Linear(api_hidden_size, self.feature_size)
        self.classifier = nn.Linear(self.feature_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(pixel_values=pixel_values).last_hidden_state
        features = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        return self.classifier(features)


class StandaloneCBPModel(nn.Module):
    """State-dict-compatible inference model for local cbp.py."""

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
        self.backbone = ResNetModel.from_pretrained(model_name)
        channels = int(self.backbone.config.hidden_sizes[-1])
        self.cbp = CompactBilinearPooling(
            channels, channels, cbp_output_dim, seed=cbp_seed,
            spatial_chunk_size=cbp_spatial_chunk_size,
        )
        self.signed_sqrt = signed_sqrt
        self.l2_normalize = l2_normalize
        self.cbp_dropout = nn.Dropout(cbp_dropout)
        self.gap_classifier = nn.Linear(channels, num_classes)
        self.cbp_classifier = nn.Linear(cbp_output_dim, num_classes)
        initial_logit = math.log(fusion_initial_gate / (1.0 - fusion_initial_gate))
        self.cbp_gate_logit = nn.Parameter(torch.tensor(initial_logit))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pixel_values=pixel_values).last_hidden_state
        gap_logits = self.gap_classifier(F.adaptive_avg_pool2d(features, 1).flatten(1))
        with torch.autocast(device_type=features.device.type, enabled=False):
            descriptor = self.cbp(features.float(), features.float())
            descriptor = normalize_cbp(descriptor, self.signed_sqrt, self.l2_normalize)
        cbp_logits = self.cbp_classifier(self.cbp_dropout(descriptor))
        return gap_logits + torch.sigmoid(self.cbp_gate_logit) * cbp_logits


def supervised_contrastive_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Supervised contrastive loss on the exact descriptor used at inference.

    The class-balanced sampler supplies multiple images of every class, so each
    anchor has at least one positive without requiring a second augmented view.
    Similarities and log-sum-exp are evaluated in fp32 for stability under AMP.
    """
    if features.ndim != 2 or targets.ndim != 1 or features.size(0) != targets.size(0):
        raise ValueError("SupCon expects features [batch, dim] and targets [batch]")
    embeddings = F.normalize(features.float(), p=2, dim=1, eps=1e-12)
    logits = embeddings @ embeddings.transpose(0, 1) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    batch_size = targets.size(0)
    self_mask = torch.eye(batch_size, device=targets.device, dtype=torch.bool)
    valid_mask = ~self_mask
    positive_mask = targets[:, None].eq(targets[None, :]) & valid_mask
    positive_count = positive_mask.sum(dim=1)
    if not bool((positive_count > 0).all()):
        raise ValueError(
            "Every SupCon anchor needs another sample of the same class; use the "
            "class-balanced sampler with --samples-per-class >= 2"
        )

    exp_logits = torch.exp(logits) * valid_mask.to(logits.dtype)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_positive_log_prob = (
        (positive_mask.to(log_prob.dtype) * log_prob).sum(dim=1)
        / positive_count.to(log_prob.dtype)
    )
    return -mean_positive_log_prob.mean()


def api_net_loss(
    outputs: dict[str, torch.Tensor],
    plain_logits: torch.Tensor,
    features: torch.Tensor,
    targets: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    self_logits = torch.cat((outputs["logit1_self"], outputs["logit2_self"]))
    other_logits = torch.cat((outputs["logit1_other"], outputs["logit2_other"]))
    pair_targets = torch.cat((outputs["labels1"], outputs["labels2"]))
    interaction_logits = torch.cat((self_logits, other_logits))
    interaction_targets = torch.cat((pair_targets, pair_targets))
    plain_ce = F.cross_entropy(plain_logits, targets, label_smoothing=args.label_smoothing)
    api_ce = F.cross_entropy(interaction_logits, interaction_targets, label_smoothing=args.label_smoothing)

    rows = torch.arange(pair_targets.size(0), device=pair_targets.device)
    self_float, other_float = self_logits.float(), other_logits.float()
    self_target = self_float[rows, pair_targets]
    other_target = other_float[rows, pair_targets]
    self_non_target, other_non_target = self_float.clone(), other_float.clone()
    self_non_target[rows, pair_targets] = -torch.inf
    other_non_target[rows, pair_targets] = -torch.inf
    self_log_odds = self_target - torch.logsumexp(self_non_target, dim=1)
    other_log_odds = other_target - torch.logsumexp(other_non_target, dim=1)
    rank_loss = F.margin_ranking_loss(
        self_log_odds, other_log_odds, torch.ones_like(self_log_odds), margin=args.rank_margin
    )
    supcon_loss = supervised_contrastive_loss(
        features, targets, args.supcon_temperature
    )
    objective = (
        args.plain_ce_weight * plain_ce
        + args.api_ce_weight * api_ce
        + args.rank_weight * rank_loss
        + args.supcon_weight * supcon_loss
    )
    return {
        "objective": objective,
        "plain_cross_entropy": plain_ce,
        "api_cross_entropy": api_ce,
        "rank_loss": rank_loss,
        "supervised_contrastive_loss": supcon_loss,
        "plain_top1": plain_logits.argmax(1).eq(targets).float().mean(),
        "interaction_top1": interaction_logits.argmax(1).eq(interaction_targets).float().mean(),
        "rank_satisfaction": (self_log_odds - other_log_odds >= args.rank_margin).float().mean(),
        "gate_difference": outputs["gate_difference"],
        "same_pair_squared_distance": outputs["same_pair_squared_distance"],
        "different_pair_squared_distance": outputs["different_pair_squared_distance"],
    }


def make_transforms(model_name: str, image_size: int):
    processor = AutoImageProcessor.from_pretrained(model_name)
    resize_size = int(round(image_size / 0.875))
    train_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    return train_transform, eval_transform


def load_class_neighbors(
    path: Path, num_classes: int, max_neighbors_per_class: int
) -> dict[int, list[int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Class-neighbor JSON not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")

    metadata = raw.get("metadata", {})
    source_splits = (
        set(metadata.get("source_splits", [])) if isinstance(metadata, dict) else set()
    )
    test_warning = bool(metadata.get("test_data_warning", False)) if isinstance(metadata, dict) else False
    if "test" in source_splits or test_warning:
        raise ValueError(
            "Refusing class neighbors derived from test data: this would leak test "
            "labels/predictions into training"
        )
    if "val" in source_splits:
        print(
            "WARNING: class neighbors were derived from validation predictions. "
            "Training with them makes that validation split no longer fully held out."
        )

    neighbor_data = raw.get("neighbors", raw)
    if not isinstance(neighbor_data, dict):
        raise ValueError(f"{path} has no valid neighbors object")
    neighbors: dict[int, list[int]] = {}
    for raw_class_id, entries in neighbor_data.items():
        try:
            class_id = int(raw_class_id)
        except (TypeError, ValueError):
            continue
        if not 0 <= class_id < num_classes or not isinstance(entries, list):
            continue
        parsed: list[int] = []
        for entry in entries:
            neighbor = entry.get("class_id") if isinstance(entry, dict) else entry
            try:
                neighbor_id = int(neighbor)
            except (TypeError, ValueError):
                continue
            if 0 <= neighbor_id < num_classes and neighbor_id != class_id:
                parsed.append(neighbor_id)
        neighbors[class_id] = list(dict.fromkeys(parsed))[:max_neighbors_per_class]
    if not any(neighbors.values()):
        raise ValueError(f"{path} contains no usable class neighbors")
    return neighbors


def make_train_loader(dataset: INatJsonDataset, args: argparse.Namespace, device: torch.device):
    class_neighbors = None
    if args.class_neighbors_json is not None:
        class_neighbors = load_class_neighbors(
            args.class_neighbors_json, args.num_classes, args.max_neighbors_per_class
        )
    sampler = ClassBalancedBatchSampler(
        dataset.labels,
        args.batch_size,
        args.samples_per_class,
        args.seed,
        class_neighbors=class_neighbors,
        neighbor_batch_fraction=(
            args.neighbor_batch_fraction if class_neighbors is not None else 0.0
        ),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return loader, sampler


def make_eval_loader(dataset: Dataset, args: argparse.Namespace, device: torch.device, seed: int):
    return DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def cosine_lr(optimizer: torch.optim.Optimizer, step: int, total_steps: int, warmup_steps: int) -> None:
    if warmup_steps and step < warmup_steps:
        factor = (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model: ResNet50APICBP,
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
    names = (
        "objective", "plain_cross_entropy", "api_cross_entropy", "rank_loss",
        "supervised_contrastive_loss",
        "plain_top1", "interaction_top1", "rank_satisfaction", "gate_difference",
        "same_pair_squared_distance", "different_pair_squared_distance",
    )
    totals = {name: 0.0 for name in names}
    batches = 0
    amp_enabled = device.type == "cuda" and not args.no_amp
    progress = tqdm(loader, desc=f"Train {epoch + 1}/{args.epochs}")
    for batch_index, (images, targets, _) in enumerate(progress):
        cosine_lr(optimizer, epoch * len(loader) + batch_index, total_steps, warmup_steps)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            plain_logits, interaction, features = model(images, targets)
            losses = api_net_loss(interaction, plain_logits, features, targets, args)
        scaler.scale(losses["objective"]).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        batches += 1
        for name in names:
            totals[name] += float(losses[name].item())
        progress.set_postfix(
            loss=f"{totals['objective'] / batches:.4f}",
            acc=f"{totals['plain_top1'] / batches:.4f}",
            rank=f"{totals['rank_satisfaction'] / batches:.3f}",
        )
    if batches == 0:
        raise ValueError("Training loader is empty")
    return {name: value / batches for name, value in totals.items()}


@torch.inference_mode()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device, description: str):
    model.eval()
    all_logits, all_targets, all_indices = [], [], []
    for images, targets, indices in tqdm(loader, desc=description):
        images = images.to(device, non_blocking=True)
        logits = model(images).float().cpu()
        all_logits.append(logits)
        all_targets.append(targets)
        all_indices.append(indices)
    if not all_targets:
        raise ValueError(f"{description} dataset is empty")
    return (
        torch.cat(all_targets).numpy(),
        torch.cat(all_logits).numpy(),
        torch.cat(all_indices).numpy(),
    )


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def compute_metrics(targets: np.ndarray, probabilities: np.ndarray, num_classes: int):
    if probabilities.shape != (len(targets), num_classes):
        raise ValueError("Probability shape does not match targets/classes")
    top_order = np.argsort(-probabilities, axis=1)[:, : min(5, num_classes)]
    metrics = {
        f"top{k}_accuracy": float(np.mean(np.any(top_order[:, :k] == targets[:, None], axis=1)))
        for k in range(1, min(5, num_classes) + 1)
    }
    predictions = top_order[:, 0]
    per_class = []
    for class_id in range(num_classes):
        tp = int(np.sum((targets == class_id) & (predictions == class_id)))
        fp = int(np.sum((targets != class_id) & (predictions == class_id)))
        fn = int(np.sum((targets == class_id) & (predictions != class_id)))
        support = int(np.sum(targets == class_id))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"class_id": class_id, "precision": precision, "recall": recall, "f1": f1, "support": support})
    metrics.update(
        macro_precision=float(np.mean([row["precision"] for row in per_class])),
        macro_recall=float(np.mean([row["recall"] for row in per_class])),
        macro_f1=float(np.mean([row["f1"] for row in per_class])),
    )
    return metrics, per_class


def save_classification_outputs(
    output_dir: Path,
    dataset: INatJsonDataset,
    targets: np.ndarray,
    probabilities: np.ndarray,
    indices: np.ndarray,
    report_title: str,
    extra_lines: list[str] | None = None,
) -> None:
    metrics, per_class = compute_metrics(targets, probabilities, probabilities.shape[1])
    lines = [report_title, f"num_samples: {len(targets)}"]
    if extra_lines:
        lines.extend(extra_lines)
    lines.extend(["", "Overall metrics"])
    lines.extend(f"{name}: {value:.6f} ({100 * value:.2f}%)" for name, value in metrics.items())
    lines.extend(["", "Per-class metrics", "class  precision  recall  f1_score  support"])
    for row in per_class:
        lines.append(
            f"{row['class_id']:03d}    {row['precision']:.6f}   {row['recall']:.6f}  "
            f"{row['f1']:.6f}  {row['support']}"
        )
    report = "\n".join(lines) + "\n"
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image_id", "file_name", "ground_truth", "predicted_label",
            "predicted_probability", "correct", "top5_labels", "top5_probabilities",
            "original_category_id",
        ])
        for target, probability, index in zip(targets, probabilities, indices):
            record = dataset.records[int(index)]
            top = np.argsort(-probability)[: min(5, len(probability))]
            prediction = int(top[0])
            writer.writerow([
                record.get("image_id", ""),
                record["file_name"],
                f"{int(target):03d}",
                f"{prediction:03d}",
                f"{float(probability[prediction]):.8f}",
                prediction == int(target),
                json.dumps([f"{int(label):03d}" for label in top]),
                json.dumps([round(float(probability[label]), 8) for label in top]),
                record.get("original_category_id", ""),
            ])
    print(report, end="")


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    best_val: float,
    args: argparse.Namespace,
) -> None:
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_val_top1": best_val,
        "args": serializable_args(args),
    }, path)


def torch_load(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if not isinstance(state, dict):
        raise ValueError("Checkpoint has no valid model state_dict")
    # Be friendly to checkpoints saved through DataParallel/torch.compile.
    cleaned = {}
    for key, value in state.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    return cleaned


def load_combined_checkpoint(model: nn.Module, path: Path, device: torch.device):
    payload = torch_load(path, device)
    model.load_state_dict(extract_state_dict(payload), strict=True)
    return payload


def _config_bool(config: dict[str, Any], name: str, default: bool) -> bool:
    value = config.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def initialise_from_cbp_checkpoint(
    model: ResNet50APICBP, path: Path, args: argparse.Namespace
) -> dict[str, Any]:
    """Transfer the compatible parts of a standalone cbp.py checkpoint.

    The standalone CBP model has a GAP classifier plus a gated CBP residual,
    whereas this combined model intentionally classifies the CBP descriptor
    directly.  Therefore the transferable classification head is
    ``cbp_classifier``; ``gap_classifier`` and ``cbp_gate_logit`` have no
    corresponding parameters here.
    """
    payload = torch_load(path, torch.device("cpu"))
    state = extract_state_dict(payload)
    saved = payload.get("args", {}) if isinstance(payload, dict) else {}

    saved_classes = int(saved.get("num_classes", args.num_classes))
    saved_cbp_dim = int(saved.get("cbp_output_dim", args.cbp_output_dim))
    if saved_classes != args.num_classes:
        raise ValueError(
            f"CBP checkpoint has {saved_classes} classes; expected {args.num_classes}"
        )
    if saved_cbp_dim != args.cbp_output_dim:
        raise ValueError(
            f"CBP checkpoint uses output dim {saved_cbp_dim}; pass "
            f"--cbp-output-dim {saved_cbp_dim}"
        )
    saved_signed_sqrt = not _config_bool(saved, "no_signed_sqrt", False)
    saved_l2 = not _config_bool(saved, "no_l2_normalize", False)
    if saved_signed_sqrt != (not args.no_signed_sqrt):
        raise ValueError("CBP checkpoint and current --no-signed-sqrt setting differ")
    if saved_l2 != (not args.no_l2_normalize):
        raise ValueError("CBP checkpoint and current --no-l2-normalize setting differ")

    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    cbp_state = {
        key.removeprefix("cbp."): value
        for key, value in state.items()
        if key.startswith("cbp.")
    }
    if not backbone_state or not cbp_state:
        raise ValueError(
            "--init-cbp-checkpoint must be produced by local cbp.py and contain "
            "backbone.* and cbp.* state"
        )
    model.backbone.load_state_dict(backbone_state, strict=True)
    model.cbp.load_state_dict(cbp_state, strict=True)

    classifier_loaded = False
    if not args.no_init_cbp_classifier:
        classifier_state = {
            "weight": state.get("cbp_classifier.weight"),
            "bias": state.get("cbp_classifier.bias"),
        }
        if any(value is None for value in classifier_state.values()):
            raise ValueError("CBP checkpoint has no cbp_classifier state")
        model.classifier.load_state_dict(classifier_state, strict=True)
        classifier_loaded = True

    information = {
        "source_checkpoint": str(path),
        "source_epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "source_best_val_top1": (
            payload.get("best_val_top1") if isinstance(payload, dict) else None
        ),
        "loaded_backbone": True,
        "loaded_cbp_sketch_buffers": True,
        "loaded_cbp_classifier": classifier_loaded,
        "ignored_source_components": ["gap_classifier", "cbp_gate_logit"],
        "stage2_train_scope": args.stage2_train_scope,
    }
    print(json.dumps({"cbp_initialisation": information}, indent=2))
    return information


def configure_stage2_train_scope(
    model: ResNet50APICBP, scope: str
) -> dict[str, int | str]:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if scope == "api-heads":
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    elif scope != "full":
        raise ValueError(f"Unknown stage-2 train scope: {scope}")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    information: dict[str, int | str] = {
        "scope": scope,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "total_parameters": total,
    }
    print(json.dumps({"stage2_training": information}, indent=2))
    return information


def load_standalone_model(kind: str, path: Path, args: argparse.Namespace, device: torch.device) -> nn.Module:
    payload = torch_load(path, torch.device("cpu"))
    saved = payload.get("args", {}) if isinstance(payload, dict) else {}
    model_name = str(saved.get("model_name", args.model_name))
    num_classes = int(saved.get("num_classes", args.num_classes))
    if num_classes != args.num_classes:
        raise ValueError(f"{kind} checkpoint has {num_classes} classes, expected {args.num_classes}")
    if kind == "cbp":
        model = StandaloneCBPModel(
            model_name=model_name,
            num_classes=num_classes,
            cbp_output_dim=int(saved.get("cbp_output_dim", args.cbp_output_dim)),
            cbp_seed=int(saved.get("cbp_seed", args.cbp_seed)),
            cbp_spatial_chunk_size=int(saved.get("cbp_spatial_chunk_size", args.cbp_spatial_chunk_size)),
            signed_sqrt=not bool(saved.get("no_signed_sqrt", args.no_signed_sqrt)),
            l2_normalize=not bool(saved.get("no_l2_normalize", args.no_l2_normalize)),
            cbp_dropout=float(saved.get("cbp_dropout", saved.get("dropout", args.dropout))),
            fusion_initial_gate=float(saved.get("fusion_initial_gate", 0.1)),
        )
    elif kind == "api":
        model = StandaloneAPIModel(
            model_name=model_name,
            num_classes=num_classes,
            api_hidden_size=int(saved.get("api_hidden_size", args.api_hidden_size)),
            dropout=float(saved.get("dropout", args.dropout)),
        )
    elif kind == "combined":
        model = ResNet50APICBP(
            model_name=model_name,
            num_classes=num_classes,
            cbp_output_dim=int(saved.get("cbp_output_dim", args.cbp_output_dim)),
            cbp_seed=int(saved.get("cbp_seed", args.cbp_seed)),
            cbp_spatial_chunk_size=int(
                saved.get("cbp_spatial_chunk_size", args.cbp_spatial_chunk_size)
            ),
            signed_sqrt=not _config_bool(saved, "no_signed_sqrt", args.no_signed_sqrt),
            l2_normalize=not _config_bool(
                saved, "no_l2_normalize", args.no_l2_normalize
            ),
            api_hidden_size=int(saved.get("api_hidden_size", args.api_hidden_size)),
            dropout=float(saved.get("dropout", args.dropout)),
        )
    else:
        raise ValueError(f"Unknown standalone model kind: {kind}")
    model.load_state_dict(extract_state_dict(payload), strict=True)
    return model.to(device)


def run_training(args: argparse.Namespace, device: torch.device) -> None:
    train_transform, eval_transform = make_transforms(args.model_name, args.image_size)
    test_set = INatJsonDataset(args.data_dir, "test", eval_transform, args.num_classes)
    test_loader = make_eval_loader(test_set, args, device, args.seed + 2)
    model = ResNet50APICBP(
        args.model_name,
        args.num_classes,
        args.cbp_output_dim,
        args.cbp_seed,
        args.cbp_spatial_chunk_size,
        not args.no_signed_sqrt,
        not args.no_l2_normalize,
        args.api_hidden_size,
        args.dropout,
    ).to(device)

    if args.test_only:
        load_combined_checkpoint(model, args.checkpoint, device)
    else:
        initialization_information = None
        if args.init_cbp_checkpoint is not None:
            initialization_information = initialise_from_cbp_checkpoint(
                model, args.init_cbp_checkpoint, args
            )
        scope_information = configure_stage2_train_scope(
            model, args.stage2_train_scope
        )
        with (args.output_dir / "stage2_setup.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                {
                    "cbp_initialisation": initialization_information,
                    "training_scope": scope_information,
                    "class_neighbors_json": (
                        str(args.class_neighbors_json)
                        if args.class_neighbors_json is not None
                        else None
                    ),
                    "neighbor_batch_fraction": args.neighbor_batch_fraction,
                    "max_neighbors_per_class": args.max_neighbors_per_class,
                },
                file,
                indent=2,
            )
        train_set = INatJsonDataset(args.data_dir, "train", train_transform, args.num_classes)
        val_set = INatJsonDataset(args.data_dir, "val", eval_transform, args.num_classes)
        train_loader, train_sampler = make_train_loader(train_set, args, device)
        val_loader = make_eval_loader(val_set, args, device, args.seed + 1)
        optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(args), weight_decay=args.weight_decay)
        scaler = make_grad_scaler(device.type == "cuda" and not args.no_amp)
        start_epoch, best_val = 0, -1.0
        if args.resume is not None:
            payload = load_combined_checkpoint(model, args.resume, device)
            if not isinstance(payload, dict) or "optimizer" not in payload:
                raise ValueError("--resume requires a full training checkpoint")
            optimizer.load_state_dict(payload["optimizer"])
            scaler.load_state_dict(payload["scaler"])
            start_epoch = int(payload["epoch"]) + 1
            best_val = float(payload.get("best_val_top1", -1.0))

        total_steps = args.epochs * len(train_loader)
        warmup_steps = int(args.warmup_epochs * len(train_loader))
        history_path = args.output_dir / "training_history.jsonl"
        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            started = time.time()
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scaler, device, args,
                epoch, total_steps, warmup_steps,
            )
            val_targets, val_logits, _ = predict_logits(model, val_loader, device, "Validation")
            val_probabilities = probabilities_from_logits(val_logits)
            val_metrics, _ = compute_metrics(val_targets, val_probabilities, args.num_classes)
            row = {
                "epoch": epoch + 1,
                "seconds": time.time() - started,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
                **{f"{group.get('group_name', i)}_lr": group["lr"] for i, group in enumerate(optimizer.param_groups)},
            }
            with history_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row) + "\n")
            print(json.dumps(row, indent=2))
            current = val_metrics["top1_accuracy"]
            if current > best_val:
                best_val = current
                save_checkpoint(args.output_dir / "best.pt", model, optimizer, scaler, epoch, best_val, args)
            save_checkpoint(args.output_dir / "last.pt", model, optimizer, scaler, epoch, best_val, args)

        best_path = args.output_dir / "best.pt"
        if not best_path.is_file():
            raise RuntimeError("No best.pt was produced; use --test-only if training already finished")
        load_combined_checkpoint(model, best_path, device)

    targets, logits, indices = predict_logits(model, test_loader, device, "Test")
    probabilities = probabilities_from_logits(logits)
    np.savez_compressed(
        args.output_dir / "test_outputs.npz",
        targets=targets,
        indices=indices,
        logits=logits,
        probabilities=probabilities,
    )
    save_classification_outputs(
        args.output_dir, test_set, targets, probabilities, indices,
        "Test classification report: ResNet-50 + CBP + API-Net",
    )


def fuse_predictions(cbp_logits: np.ndarray, api_logits: np.ndarray, alpha: float, space: str):
    if cbp_logits.shape != api_logits.shape:
        raise ValueError("CBP and API output shapes differ")
    if space == "logits":
        fused_logits = alpha * cbp_logits + (1.0 - alpha) * api_logits
        return fused_logits, probabilities_from_logits(fused_logits)
    cbp_probabilities = probabilities_from_logits(cbp_logits)
    api_probabilities = probabilities_from_logits(api_logits)
    probabilities = alpha * cbp_probabilities + (1.0 - alpha) * api_probabilities
    # Log probabilities are saved only as a score equivalent to probability fusion.
    return np.log(np.clip(probabilities, 1e-300, None)), probabilities


def alpha_selection_metrics(targets: np.ndarray, probabilities: np.ndarray, num_classes: int):
    predictions = probabilities.argmax(1)
    top1 = float(np.mean(predictions == targets))
    f1_values = []
    for class_id in range(num_classes):
        tp = np.sum((targets == class_id) & (predictions == class_id))
        fp = np.sum((targets != class_id) & (predictions == class_id))
        fn = np.sum((targets == class_id) & (predictions != class_id))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return top1, float(np.mean(f1_values))


def assert_same_examples(
    first_targets: np.ndarray,
    first_indices: np.ndarray,
    second_targets: np.ndarray,
    second_indices: np.ndarray,
    split: str,
) -> None:
    if not np.array_equal(first_targets, second_targets) or not np.array_equal(first_indices, second_indices):
        raise RuntimeError(f"The two models saw different {split} example ordering")


def standalone_split_outputs(
    kind: str,
    checkpoint: Path,
    loader: DataLoader,
    description: str,
    args: argparse.Namespace,
    device: torch.device,
):
    model = load_standalone_model(kind, checkpoint, args, device)
    outputs = predict_logits(model, loader, device, description)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs


def _run_legacy_ensemble(args: argparse.Namespace, device: torch.device) -> None:
    # Both standalone reference scripts use the same HF preprocessing.  Saved
    # checkpoint architecture arguments are used for model construction; input
    # resolution is intentionally controlled here and must match both runs.
    _, eval_transform = make_transforms(args.model_name, args.image_size)
    val_set = INatJsonDataset(args.data_dir, "val", eval_transform, args.num_classes)
    test_set = INatJsonDataset(args.data_dir, "test", eval_transform, args.num_classes)
    val_loader = make_eval_loader(val_set, args, device, args.seed + 1)
    test_loader = make_eval_loader(test_set, args, device, args.seed + 2)

    val_targets, cbp_val_logits, val_indices = standalone_split_outputs(
        "cbp", args.cbp_checkpoint, val_loader, "Validation (CBP)", args, device
    )
    api_val_targets, api_val_logits, api_val_indices = standalone_split_outputs(
        "api", args.api_checkpoint, val_loader, "Validation (API-Net)", args, device
    )
    assert_same_examples(val_targets, val_indices, api_val_targets, api_val_indices, "validation")

    # Persist raw validation outputs, making the alpha choice auditable/reusable.
    np.savez_compressed(
        args.output_dir / "validation_outputs.npz",
        targets=val_targets,
        indices=val_indices,
        cbp_logits=cbp_val_logits,
        api_logits=api_val_logits,
        cbp_probabilities=probabilities_from_logits(cbp_val_logits),
        api_probabilities=probabilities_from_logits(api_val_logits),
    )

    rows = []
    best_alpha, best_key = None, None
    for alpha in np.linspace(0.0, 1.0, args.alpha_steps):
        _, probabilities = fuse_predictions(cbp_val_logits, api_val_logits, float(alpha), args.fusion_space)
        top1, macro_f1 = alpha_selection_metrics(val_targets, probabilities, args.num_classes)
        row = {"alpha": float(alpha), "val_top1_accuracy": top1, "val_macro_f1": macro_f1}
        rows.append(row)
        primary = top1 if args.alpha_metric == "top1" else macro_f1
        secondary = macro_f1 if args.alpha_metric == "top1" else top1
        # Deterministic tie-break: secondary metric, then alpha closest to 0.5.
        key = (primary, secondary, -abs(float(alpha) - 0.5))
        if best_key is None or key > best_key:
            best_key, best_alpha = key, float(alpha)
    assert best_alpha is not None

    with (args.output_dir / "alpha_search.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    chosen_row = next(row for row in rows if row["alpha"] == best_alpha)
    selection = {
        "selected_on": "validation only",
        "test_used_for_alpha_selection": False,
        "alpha": best_alpha,
        "formula": "alpha * cbp_output + (1 - alpha) * api_output",
        "fusion_space": args.fusion_space,
        "selection_metric": args.alpha_metric,
        "alpha_steps": args.alpha_steps,
        "validation_metrics_at_selected_alpha": chosen_row,
        "cbp_checkpoint": str(args.cbp_checkpoint),
        "api_checkpoint": str(args.api_checkpoint),
    }
    with (args.output_dir / "ensemble_selection.json").open("w", encoding="utf-8") as file:
        json.dump(selection, file, indent=2)
    print(json.dumps(selection, indent=2))

    # Test inference starts only after validation has selected and persisted
    # alpha.  No test target, prediction, or metric can influence the search.
    test_targets, cbp_test_logits, test_indices = standalone_split_outputs(
        "cbp", args.cbp_checkpoint, test_loader, "Test (CBP; alpha fixed)", args, device
    )
    api_test_targets, api_test_logits, api_test_indices = standalone_split_outputs(
        "api", args.api_checkpoint, test_loader, "Test (API-Net; alpha fixed)", args, device
    )
    assert_same_examples(test_targets, test_indices, api_test_targets, api_test_indices, "test")
    final_logits, final_probabilities = fuse_predictions(
        cbp_test_logits, api_test_logits, best_alpha, args.fusion_space
    )
    np.savez_compressed(
        args.output_dir / "test_outputs.npz",
        targets=test_targets,
        indices=test_indices,
        cbp_logits=cbp_test_logits,
        api_logits=api_test_logits,
        final_logits=final_logits,
        final_probabilities=final_probabilities,
        alpha=np.asarray(best_alpha),
    )
    save_classification_outputs(
        args.output_dir,
        test_set,
        test_targets,
        final_probabilities,
        test_indices,
        "Test classification report: validation-selected CBP/API ensemble",
        extra_lines=[
            f"alpha: {best_alpha:.6f}",
            f"fusion_space: {args.fusion_space}",
            "alpha_selected_on: validation only",
        ],
    )


def ensemble_specs(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    if args.ensemble_model is not None:
        return [
            (str(name), str(kind), Path(checkpoint))
            for name, kind, checkpoint in args.ensemble_model
        ]
    # Backwards-compatible two-checkpoint interface.
    return [
        ("cbp", "cbp", args.cbp_checkpoint),
        ("api", "api", args.api_checkpoint),
    ]


def weighted_fusion(
    model_logits: list[np.ndarray], weights: np.ndarray, space: str
) -> tuple[np.ndarray, np.ndarray]:
    if len(model_logits) != len(weights):
        raise ValueError("The number of ensemble outputs and weights differs")
    reference_shape = model_logits[0].shape
    if any(logits.shape != reference_shape for logits in model_logits):
        raise ValueError("All ensemble model output shapes must match")
    if not np.isclose(float(weights.sum()), 1.0) or np.any(weights < 0):
        raise ValueError("Ensemble weights must be non-negative and sum to one")
    if space == "logits":
        scores = np.zeros(reference_shape, dtype=np.float64)
        for weight, logits in zip(weights, model_logits):
            scores += float(weight) * logits
        return scores, probabilities_from_logits(scores)
    probabilities = np.zeros(reference_shape, dtype=np.float64)
    for weight, logits in zip(weights, model_logits):
        probabilities += float(weight) * probabilities_from_logits(logits)
    return np.log(np.clip(probabilities, 1e-300, None)), probabilities


def candidate_ensemble_weights(
    num_models: int, args: argparse.Namespace
) -> Iterator[np.ndarray]:
    seen: set[tuple[float, ...]] = set()

    def unique(weights: np.ndarray):
        normalised = np.asarray(weights, dtype=np.float64)
        normalised /= normalised.sum()
        key = tuple(np.round(normalised, 12))
        if key in seen:
            return None
        seen.add(key)
        return normalised

    if num_models == 2:
        for alpha in np.linspace(0.0, 1.0, args.alpha_steps):
            weights = unique(np.asarray((alpha, 1.0 - alpha)))
            if weights is not None:
                yield weights
        return

    initial = [np.full(num_models, 1.0 / num_models)]
    initial.extend(np.eye(num_models))
    # Exhaustive two-model edges are useful when extra models add no value.
    for first in range(num_models):
        for second in range(first + 1, num_models):
            for alpha in np.linspace(0.0, 1.0, args.alpha_steps):
                weights = np.zeros(num_models)
                weights[first], weights[second] = alpha, 1.0 - alpha
                initial.append(weights)
    for raw in initial:
        weights = unique(raw)
        if weights is not None:
            yield weights

    generator = np.random.default_rng(args.seed)
    for raw in generator.dirichlet(np.ones(num_models), size=args.ensemble_search_trials):
        weights = unique(raw)
        if weights is not None:
            yield weights


def run_ensemble(args: argparse.Namespace, device: torch.device) -> None:
    specs = ensemble_specs(args)
    _, eval_transform = make_transforms(args.model_name, args.image_size)
    val_set = INatJsonDataset(args.data_dir, "val", eval_transform, args.num_classes)
    test_set = INatJsonDataset(args.data_dir, "test", eval_transform, args.num_classes)
    val_loader = make_eval_loader(val_set, args, device, args.seed + 1)
    test_loader = make_eval_loader(test_set, args, device, args.seed + 2)

    val_targets = val_indices = None
    val_logits: list[np.ndarray] = []
    validation_archive: dict[str, Any] = {
        "model_names": np.asarray([name for name, _, _ in specs]),
        "model_types": np.asarray([kind for _, kind, _ in specs]),
    }
    for model_index, (name, kind, checkpoint) in enumerate(specs):
        targets, logits, indices = standalone_split_outputs(
            kind,
            checkpoint,
            val_loader,
            f"Validation ({name}: {kind})",
            args,
            device,
        )
        if val_targets is None:
            val_targets, val_indices = targets, indices
        else:
            assert_same_examples(
                val_targets, val_indices, targets, indices, "validation"
            )
        val_logits.append(logits)
        validation_archive[f"logits_{model_index}"] = logits
        validation_archive[f"probabilities_{model_index}"] = probabilities_from_logits(logits)
    assert val_targets is not None and val_indices is not None
    validation_archive["targets"] = val_targets
    validation_archive["indices"] = val_indices
    np.savez_compressed(args.output_dir / "validation_outputs.npz", **validation_archive)

    rows: list[dict[str, float]] = []
    best_weights = None
    best_key = None
    equal = np.full(len(specs), 1.0 / len(specs))
    for weights in candidate_ensemble_weights(len(specs), args):
        _, probabilities = weighted_fusion(val_logits, weights, args.fusion_space)
        top1, macro_f1 = alpha_selection_metrics(
            val_targets, probabilities, args.num_classes
        )
        row = {
            **{f"weight_{name}": float(weight) for (name, _, _), weight in zip(specs, weights)},
            "val_top1_accuracy": top1,
            "val_macro_f1": macro_f1,
        }
        rows.append(row)
        primary = top1 if args.alpha_metric == "top1" else macro_f1
        secondary = macro_f1 if args.alpha_metric == "top1" else top1
        key = (primary, secondary, -float(np.abs(weights - equal).sum()))
        if best_key is None or key > best_key:
            best_key, best_weights = key, weights.copy()
    if best_weights is None:
        raise RuntimeError("No ensemble weight candidates were generated")

    with (args.output_dir / "ensemble_weight_search.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected_metrics = next(
        row
        for row in rows
        if all(
            np.isclose(row[f"weight_{name}"], weight)
            for (name, _, _), weight in zip(specs, best_weights)
        )
    )
    selection = {
        "selected_on": "validation only",
        "test_used_for_weight_selection": False,
        "fusion_space": args.fusion_space,
        "selection_metric": args.alpha_metric,
        "models": [
            {
                "name": name,
                "type": kind,
                "checkpoint": str(checkpoint),
                "weight": float(weight),
            }
            for (name, kind, checkpoint), weight in zip(specs, best_weights)
        ],
        "validation_metrics_at_selected_weights": selected_metrics,
        "num_weight_candidates": len(rows),
    }
    with (args.output_dir / "ensemble_selection.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(selection, file, indent=2)
    print(json.dumps(selection, indent=2))

    # Test inference begins only after validation weights are fixed and saved.
    test_targets = test_indices = None
    test_logits: list[np.ndarray] = []
    test_archive: dict[str, Any] = {
        "model_names": np.asarray([name for name, _, _ in specs]),
        "weights": best_weights,
    }
    for model_index, (name, kind, checkpoint) in enumerate(specs):
        targets, logits, indices = standalone_split_outputs(
            kind,
            checkpoint,
            test_loader,
            f"Test ({name}: {kind}; weights fixed)",
            args,
            device,
        )
        if test_targets is None:
            test_targets, test_indices = targets, indices
        else:
            assert_same_examples(test_targets, test_indices, targets, indices, "test")
        test_logits.append(logits)
        test_archive[f"logits_{model_index}"] = logits
    assert test_targets is not None and test_indices is not None
    final_logits, final_probabilities = weighted_fusion(
        test_logits, best_weights, args.fusion_space
    )
    test_archive.update(
        targets=test_targets,
        indices=test_indices,
        final_logits=final_logits,
        final_probabilities=final_probabilities,
    )
    np.savez_compressed(args.output_dir / "test_outputs.npz", **test_archive)
    weight_text = ", ".join(
        f"{name}={weight:.6f}"
        for (name, _, _), weight in zip(specs, best_weights)
    )
    save_classification_outputs(
        args.output_dir,
        test_set,
        test_targets,
        final_probabilities,
        test_indices,
        "Test classification report: validation-selected multi-model ensemble",
        extra_lines=[
            f"weights: {weight_text}",
            f"fusion_space: {args.fusion_space}",
            "weights_selected_on: validation only",
        ],
    )


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
    if args.mode == "ensemble":
        run_ensemble(args, device)
    else:
        run_training(args, device)


if __name__ == "__main__":
    main()
