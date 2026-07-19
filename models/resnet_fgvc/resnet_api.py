#!/usr/bin/env python3
"""Full fine-tuning of ImageNet-pretrained ResNet-50 with API-Net."""

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Full fine-tuning of Hugging Face ResNet-50 with API-Net on an iNaturalist JSON subset")
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--num-classes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size. It must be divisible by --samples-per-class.",
    )
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Use one LR for all parameters, overriding the grouped LRs",
    )
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument(
        "--classifier-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the shared plain/API classifier",
    )
    parser.add_argument("--api-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument(
        "--api-hidden-size",
        type=int,
        default=512,
        help="Hidden width of API-Net's 2D -> hidden -> D interaction mapping",
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--plain-ce-weight",
        type=float,
        default=1.0,
        help="Weight for CE on the raw pooled feature used at inference",
    )
    parser.add_argument(
        "--api-ce-weight",
        type=float,
        default=0.25,
        help="Weight for CE on API-Net's four interaction logits",
    )
    parser.add_argument(
        "--rank-margin",
        type=float,
        default=0.2,
        help="Margin between correct-class self/other target log-odds",
    )
    parser.add_argument("--rank-weight", type=float, default=0.1)
    parser.add_argument(
        "--class-neighbors-json",
        type=Path,
        default=None,
        help=(
            "Optional class-neighbour JSON produced by build-class-neighbors.py; "
            "neighbouring/confused classes are preferentially placed in one batch"
        ),
    )
    parser.add_argument(
        "--neighbor-batch-fraction",
        type=float,
        default=0.25,
        help="Fraction of non-anchor class slots filled from the anchor's neighbours",
    )
    parser.add_argument(
        "--max-neighbors-per-class",
        type=int,
        default=5,
        help="Use only the top-N entries per class from --class-neighbors-json",
    )
    parser.add_argument(
        "--allow-test-neighbors",
        action="store_true",
        help="Allow a neighbour JSON built from test predictions (data leakage; analysis only)",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-size",
        type=int,
        default=448,
        help="API-Net uses 448 crops; use 224 if GPU memory is limited",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    parser.add_argument("--resume", type=Path, default=None, help="Resume a training checkpoint")
    parser.add_argument("--test-only", action="store_true", help="Only evaluate --checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=None)
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
        if not raw:
            raise ValueError(f"{json_path} contains no records")

        self.records = raw
        self.labels: list[int] = []
        for index, item in enumerate(self.records):
            if not isinstance(item, dict):
                raise ValueError(f"Record {index} in {json_path} is not a JSON object")
            if "file_name" not in item or "label" not in item:
                raise ValueError(f"Record {index} in {json_path} lacks file_name or label")
            label = int(item["label"])
            if not 0 <= label < num_classes:
                raise ValueError(
                    f"Label {label} in record {index} is outside [0, {num_classes - 1}]"
                )
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.records)

    def _image_path(self, file_name: str) -> Path:
        path = Path(file_name)
        if path.is_absolute():
            return path

        direct_path = self.data_dir / path
        if direct_path.is_file():
            return direct_path

        # Also support processed_dataset/train/000/image.jpg in annotations.
        if path.parts and path.parts[0] == self.data_dir.name:
            return self.data_dir.parent / path
        return direct_path

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
    """Yield batches containing a fixed number of images from each class.

    API-Net requires at least one same-class and one different-class neighbour
    for every anchor. Class usage is kept as even as possible over each epoch,
    and samples are drawn without replacement until a class pool is exhausted.
    """

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
            raise ValueError(
                f"Training data has {len(self.class_ids)} classes, but each batch needs "
                f"{self.classes_per_batch}"
            )
        too_small = [
            class_id
            for class_id, indices in self.class_indices.items()
            if len(indices) < samples_per_class
        ]
        if too_small:
            preview = ", ".join(str(class_id) for class_id in too_small[:10])
            raise ValueError(
                f"Classes [{preview}] have fewer than {samples_per_class} training images"
            )

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
    def _shuffled(values: list[int], generator: torch.Generator) -> list[int]:
        order = torch.randperm(len(values), generator=generator).tolist()
        return [values[position] for position in order]

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        pools = {
            class_id: self._shuffled(indices, generator)
            for class_id, indices in self.class_indices.items()
        }
        cursors = {class_id: 0 for class_id in self.class_ids}
        usage = {class_id: 0 for class_id in self.class_ids}

        for _ in range(self.num_batches):
            # Pick an under-used anchor, then preferentially add its known
            # confusing neighbours. Remaining slots are filled globally. Usage
            # remains the primary key so class frequency stays near-uniform.
            jitter = torch.rand(len(self.class_ids), generator=generator).tolist()
            ordered = sorted(
                zip(self.class_ids, jitter), key=lambda pair: (usage[pair[0]], pair[1])
            )
            anchor = ordered[0][0]
            chosen_classes = [anchor]

            neighbor_slots = min(
                self.classes_per_batch - 1,
                int(round((self.classes_per_batch - 1) * self.neighbor_batch_fraction)),
            )
            neighbor_ids = self.class_neighbors.get(anchor, [])
            neighbor_rank = {class_id: rank for rank, class_id in enumerate(neighbor_ids)}
            neighbor_jitter = torch.rand(
                len(neighbor_ids), generator=generator
            ).tolist()
            ordered_neighbors = sorted(
                zip(neighbor_ids, neighbor_jitter),
                key=lambda pair: (
                    usage[pair[0]],
                    neighbor_rank[pair[0]],
                    pair[1],
                ),
            )
            chosen_classes.extend(
                pair[0] for pair in ordered_neighbors[:neighbor_slots]
            )

            if len(chosen_classes) < self.classes_per_batch:
                chosen_set = set(chosen_classes)
                chosen_classes.extend(
                    class_id
                    for class_id, _ in ordered
                    if class_id not in chosen_set
                )
                chosen_classes = chosen_classes[: self.classes_per_batch]

            batch: list[int] = []
            for class_id in chosen_classes:
                start = cursors[class_id]
                end = start + self.samples_per_class
                if end > len(pools[class_id]):
                    pools[class_id] = self._shuffled(
                        self.class_indices[class_id], generator
                    )
                    start = 0
                    end = self.samples_per_class
                batch.extend(pools[class_id][start:end])
                cursors[class_id] = end
                usage[class_id] += 1

            permutation = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[position] for position in permutation]

    def __len__(self) -> int:
        return self.num_batches


class ResNet50APINet(nn.Module):
    """ResNet-50 plus API-Net's attentive pairwise interaction head."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        api_hidden_size: int,
        dropout: float,
    ):
        super().__init__()
        self.backbone = ResNetModel.from_pretrained(model_name)
        self.feature_size = int(self.backbone.config.hidden_sizes[-1])
        self.map1 = nn.Linear(2 * self.feature_size, api_hidden_size)
        self.map2 = nn.Linear(api_hidden_size, self.feature_size)
        self.classifier = nn.Linear(self.feature_size, num_classes)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.map1.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.map1.bias)
        nn.init.normal_(self.map2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.map2.bias)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

        # Explicitly document full fine-tuning: no backbone parameter is frozen.
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(pixel_values=pixel_values).last_hidden_state
        return F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1)

    @staticmethod
    def mine_pairs(
        features: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mined indices and their squared feature-space distances."""
        if features.ndim != 2 or targets.ndim != 1 or features.size(0) != targets.size(0):
            raise ValueError("Expected features [batch, dim] and targets [batch]")

        with torch.no_grad():
            detached = features.detach().float()
            squared_norm = detached.square().sum(dim=1, keepdim=True)
            distances = (
                squared_norm + squared_norm.transpose(0, 1)
                - 2.0 * detached @ detached.transpose(0, 1)
            ).clamp_min_(0.0)

            same_class = targets[:, None].eq(targets[None, :])
            same_class.fill_diagonal_(False)
            different_class = targets[:, None].ne(targets[None, :])
            if not bool(same_class.any(dim=1).all()):
                raise ValueError(
                    "Every API-Net anchor needs another image of the same class. "
                    "Use the class-balanced training sampler."
                )
            if not bool(different_class.any(dim=1).all()):
                raise ValueError("Every API-Net batch must contain at least two classes")

            same_indices = distances.masked_fill(~same_class, float("inf")).argmin(dim=1)
            different_indices = distances.masked_fill(
                ~different_class, float("inf")
            ).argmin(dim=1)
            rows = torch.arange(features.size(0), device=features.device)
            same_distances = distances[rows, same_indices]
            different_distances = distances[rows, different_indices]
        return same_indices, different_indices, same_distances, different_distances

    def interact(
        self, features: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        (
            same_indices,
            different_indices,
            same_distances,
            different_distances,
        ) = self.mine_pairs(features, targets)
        anchor_indices = torch.arange(features.size(0), device=features.device)

        first_indices = torch.cat((anchor_indices, anchor_indices), dim=0)
        second_indices = torch.cat((same_indices, different_indices), dim=0)
        features1 = features[first_indices]
        features2 = features[second_indices]
        labels1 = targets[first_indices]
        labels2 = targets[second_indices]

        mutual = torch.cat((features1, features2), dim=1)
        mutual = self.map2(self.dropout(self.map1(mutual)))
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

    def forward(
        self, pixel_values: torch.Tensor, targets: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = self.extract_features(pixel_values)
        if targets is None:
            return self.classifier(features)
        # This is exactly the single-image path used at validation/test time.
        # Direct CE supervision here removes API-Net's train/inference mismatch.
        plain_logits = self.classifier(features)
        return plain_logits, self.interact(features, targets)

    def optimizer_parameter_groups(self, args: argparse.Namespace) -> list[dict[str, Any]]:
        if args.lr is not None:
            return [
                {
                    "params": self.parameters(),
                    "lr": args.lr,
                    "initial_lr": args.lr,
                    "group_name": "all_parameters",
                }
            ]
        api_parameters = list(self.map1.parameters()) + list(self.map2.parameters())
        return [
            {
                "params": self.backbone.parameters(),
                "lr": args.backbone_lr,
                "initial_lr": args.backbone_lr,
                "group_name": "pretrained_backbone",
            },
            {
                "params": self.classifier.parameters(),
                "lr": args.classifier_lr,
                "initial_lr": args.classifier_lr,
                "group_name": "shared_classifier",
            },
            {
                "params": api_parameters,
                "lr": args.api_lr,
                "initial_lr": args.api_lr,
                "group_name": "api_mapping",
            },
        ]


def api_net_loss(
    outputs: dict[str, torch.Tensor],
    plain_logits: torch.Tensor,
    original_targets: torch.Tensor,
    label_smoothing: float,
    plain_ce_weight: float,
    api_ce_weight: float,
    rank_margin: float,
    rank_weight: float,
) -> dict[str, torch.Tensor]:
    """Compute plain/API classification losses, ranking loss and diagnostics."""
    logit1_self = outputs["logit1_self"]
    logit1_other = outputs["logit1_other"]
    logit2_self = outputs["logit2_self"]
    logit2_other = outputs["logit2_other"]
    labels1 = outputs["labels1"]
    labels2 = outputs["labels2"]
    self_logits = torch.cat((logit1_self, logit2_self), dim=0)
    other_logits = torch.cat((logit1_other, logit2_other), dim=0)
    pair_targets = torch.cat((labels1, labels2), dim=0)

    combined_logits = torch.cat((self_logits, other_logits), dim=0)
    combined_targets = torch.cat((pair_targets, pair_targets), dim=0)
    api_cross_entropy = F.cross_entropy(
        combined_logits, combined_targets, label_smoothing=label_smoothing
    )
    plain_cross_entropy = F.cross_entropy(
        plain_logits, original_targets, label_smoothing=label_smoothing
    )

    row_indices = torch.arange(pair_targets.size(0), device=pair_targets.device)
    self_float_logits = self_logits.float()
    other_float_logits = other_logits.float()
    self_target_logits = self_float_logits[row_indices, pair_targets]
    other_target_logits = other_float_logits[row_indices, pair_targets]
    logit_margin = self_target_logits - other_target_logits

    # Use target log-odds as the ranking score:
    #   z_y - logsumexp(z_j, j != y) = log(p_y / (1 - p_y)).
    # Unlike raw target logits, this score is invariant to shifting every logit
    # by the same value, so the model cannot satisfy ranking by merely moving an
    # entire self/other logit vector. Unlike probability ranking, it is unbounded
    # and therefore does not suffer from softmax saturation near one.
    self_non_target_logits = self_float_logits.clone()
    other_non_target_logits = other_float_logits.clone()
    self_non_target_logits[row_indices, pair_targets] = -torch.inf
    other_non_target_logits[row_indices, pair_targets] = -torch.inf
    self_target_log_odds = self_target_logits - torch.logsumexp(
        self_non_target_logits, dim=1
    )
    other_target_log_odds = other_target_logits - torch.logsumexp(
        other_non_target_logits, dim=1
    )
    log_odds_margin = self_target_log_odds - other_target_log_odds
    rank_loss = F.margin_ranking_loss(
        self_target_log_odds,
        other_target_log_odds,
        torch.ones_like(self_target_log_odds),
        margin=rank_margin,
    )

    # Probabilities are retained only as interpretable diagnostics; they are no
    # longer used by the ranking objective.
    self_probabilities = F.softmax(self_float_logits, dim=1)
    other_probabilities = F.softmax(other_float_logits, dim=1)
    self_target_probabilities = self_probabilities[row_indices, pair_targets]
    other_target_probabilities = other_probabilities[row_indices, pair_targets]
    objective = (
        plain_ce_weight * plain_cross_entropy
        + api_ce_weight * api_cross_entropy
        + rank_weight * rank_loss
    )
    return {
        "objective": objective,
        "plain_cross_entropy": plain_cross_entropy,
        "api_cross_entropy": api_cross_entropy,
        "rank_loss": rank_loss,
        "plain_logits": plain_logits,
        "plain_targets": original_targets,
        "combined_logits": combined_logits,
        "combined_targets": combined_targets,
        "self_logits": self_logits,
        "other_logits": other_logits,
        "pair_targets": pair_targets,
        "self_target_logit": self_target_logits.mean(),
        "other_target_logit": other_target_logits.mean(),
        "self_minus_other_logit": logit_margin.mean(),
        "self_target_log_odds": self_target_log_odds.mean(),
        "other_target_log_odds": other_target_log_odds.mean(),
        "self_minus_other_log_odds": log_odds_margin.mean(),
        "self_probability": self_target_probabilities.mean(),
        "other_probability": other_target_probabilities.mean(),
        "self_minus_other_probability": (
            self_target_probabilities - other_target_probabilities
        ).mean(),
        "rank_satisfaction_rate": (
            log_odds_margin >= rank_margin
        ).float().mean(),
        "gate_difference": outputs["gate_difference"],
        "same_pair_squared_distance": outputs["same_pair_squared_distance"],
        "different_pair_squared_distance": outputs[
            "different_pair_squared_distance"
        ],
    }


def make_transforms(model_name: str, image_size: int):
    processor = AutoImageProcessor.from_pretrained(model_name)
    mean, std = processor.image_mean, processor.image_std
    resize_size = int(round(image_size / 0.875))
    train_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def load_class_neighbors(
    path: Path,
    num_classes: int,
    allow_test_neighbors: bool,
    max_neighbors_per_class: int,
) -> dict[int, list[int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Class-neighbour JSON not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")

    metadata = raw.get("metadata", {})
    source_splits = set(metadata.get("source_splits", [])) if isinstance(metadata, dict) else set()
    if "test" in source_splits and not allow_test_neighbors:
        raise ValueError(
            "The class-neighbour JSON was built from test predictions. Using it "
            "for training leaks test labels/errors. Build neighbours from validation "
            "predictions, or pass --allow-test-neighbors for analysis only."
        )

    neighbor_data = raw.get("neighbors", raw)
    if not isinstance(neighbor_data, dict):
        raise ValueError(f"{path} has no valid 'neighbors' object")
    neighbors: dict[int, list[int]] = {}
    for raw_class_id, entries in neighbor_data.items():
        class_id = int(raw_class_id)
        if not 0 <= class_id < num_classes or not isinstance(entries, list):
            continue
        parsed: list[int] = []
        for entry in entries:
            neighbor = entry.get("class_id") if isinstance(entry, dict) else entry
            neighbor_id = int(neighbor)
            if 0 <= neighbor_id < num_classes and neighbor_id != class_id:
                parsed.append(neighbor_id)
        neighbors[class_id] = list(dict.fromkeys(parsed))[:max_neighbors_per_class]
    if not neighbors:
        raise ValueError(f"{path} contains no usable class neighbours")
    return neighbors


def make_balanced_loader(
    dataset: INatJsonDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, ClassBalancedBatchSampler]:
    class_neighbors = None
    if args.class_neighbors_json is not None:
        class_neighbors = load_class_neighbors(
            args.class_neighbors_json,
            args.num_classes,
            args.allow_test_neighbors,
            args.max_neighbors_per_class,
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
    generator = torch.Generator()
    generator.manual_seed(args.seed)
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


def make_eval_loader(
    dataset: Dataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def cosine_lr(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> None:
    if warmup_steps > 0 and step < warmup_steps:
        factor = float(step + 1) / float(warmup_steps)
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
    model: ResNet50APINet,
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
    totals = {
        "objective": 0.0,
        "plain_cross_entropy": 0.0,
        "api_cross_entropy": 0.0,
        "rank_loss": 0.0,
        "self_target_logit": 0.0,
        "other_target_logit": 0.0,
        "self_minus_other_logit": 0.0,
        "self_target_log_odds": 0.0,
        "other_target_log_odds": 0.0,
        "self_minus_other_log_odds": 0.0,
        "self_probability": 0.0,
        "other_probability": 0.0,
        "self_minus_other_probability": 0.0,
        "rank_satisfaction_rate": 0.0,
        "gate_difference": 0.0,
        "same_pair_squared_distance": 0.0,
        "different_pair_squared_distance": 0.0,
        "plain_correct": 0.0,
        "plain_predictions": 0.0,
        "interaction_correct": 0.0,
        "interaction_predictions": 0.0,
        "self_correct": 0.0,
        "other_correct": 0.0,
        "pair_predictions": 0.0,
        "batches": 0.0,
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
            plain_logits, outputs = model(images, targets)
            loss_outputs = api_net_loss(
                outputs,
                plain_logits,
                targets,
                args.label_smoothing,
                args.plain_ce_weight,
                args.api_ce_weight,
                args.rank_margin,
                args.rank_weight,
            )
            objective = loss_outputs["objective"]

        scaler.scale(objective).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        combined_logits = loss_outputs["combined_logits"]
        combined_targets = loss_outputs["combined_targets"]
        self_logits = loss_outputs["self_logits"]
        other_logits = loss_outputs["other_logits"]
        pair_targets = loss_outputs["pair_targets"]
        totals["batches"] += 1
        for name in (
            "objective",
            "plain_cross_entropy",
            "api_cross_entropy",
            "rank_loss",
            "self_target_logit",
            "other_target_logit",
            "self_minus_other_logit",
            "self_target_log_odds",
            "other_target_log_odds",
            "self_minus_other_log_odds",
            "self_probability",
            "other_probability",
            "self_minus_other_probability",
            "rank_satisfaction_rate",
            "gate_difference",
            "same_pair_squared_distance",
            "different_pair_squared_distance",
        ):
            totals[name] += loss_outputs[name].item()

        totals["plain_predictions"] += targets.size(0)
        totals["interaction_predictions"] += combined_targets.size(0)
        totals["pair_predictions"] += pair_targets.size(0)
        totals["plain_correct"] += plain_logits.argmax(dim=1).eq(targets).sum().item()
        totals["interaction_correct"] += (
            combined_logits.argmax(dim=1).eq(combined_targets).sum().item()
        )
        totals["self_correct"] += self_logits.argmax(dim=1).eq(pair_targets).sum().item()
        totals["other_correct"] += other_logits.argmax(dim=1).eq(pair_targets).sum().item()
        progress.set_postfix(
            loss=f"{totals['objective'] / totals['batches']:.4f}",
            plain_ce=f"{totals['plain_cross_entropy'] / totals['batches']:.4f}",
            api_ce=f"{totals['api_cross_entropy'] / totals['batches']:.4f}",
            rank=f"{totals['rank_loss'] / totals['batches']:.4f}",
            plain_acc=f"{totals['plain_correct'] / totals['plain_predictions']:.4f}",
            rank_ok=f"{totals['rank_satisfaction_rate'] / totals['batches']:.3f}",
        )

    if totals["batches"] == 0:
        raise ValueError("Training loader is empty; reduce --batch-size")
    batch_averages = {
        name: totals[name] / totals["batches"]
        for name in (
            "objective",
            "plain_cross_entropy",
            "api_cross_entropy",
            "rank_loss",
            "self_target_logit",
            "other_target_logit",
            "self_minus_other_logit",
            "self_target_log_odds",
            "other_target_log_odds",
            "self_minus_other_log_odds",
            "self_probability",
            "other_probability",
            "self_minus_other_probability",
            "rank_satisfaction_rate",
            "gate_difference",
            "same_pair_squared_distance",
            "different_pair_squared_distance",
        )
    }
    return {
        **batch_averages,
        "plain_top1_accuracy": totals["plain_correct"] / totals["plain_predictions"],
        "interaction_top1_accuracy": (
            totals["interaction_correct"] / totals["interaction_predictions"]
        ),
        "self_top1_accuracy": totals["self_correct"] / totals["pair_predictions"],
        "other_top1_accuracy": totals["other_correct"] / totals["pair_predictions"],
    }


@torch.inference_mode()
def predict(
    model: ResNet50APINet,
    loader: DataLoader,
    device: torch.device,
    description: str,
):
    model.eval()
    all_targets, all_probabilities, all_indices = [], [], []
    loss_sum, count = 0.0, 0
    for images, targets, indices in tqdm(loader, desc=description):
        images = images.to(device, non_blocking=True)
        device_targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += F.cross_entropy(logits.float(), device_targets, reduction="sum").item()
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
    targets: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    if probabilities.ndim != 2 or probabilities.shape[1] != num_classes:
        raise ValueError("Probability array does not match --num-classes")

    top_order = np.argsort(-probabilities, axis=1)[:, : min(5, num_classes)]
    metrics = {
        f"top{k}_accuracy": float(
            np.mean(np.any(top_order[:, :k] == targets[:, None], axis=1))
        )
        for k in range(1, min(5, num_classes) + 1)
    }
    predictions = top_order[:, 0]
    per_class: list[dict[str, float | int]] = []
    for class_id in range(num_classes):
        true_positive = int(np.sum((targets == class_id) & (predictions == class_id)))
        false_positive = int(np.sum((targets != class_id) & (predictions == class_id)))
        false_negative = int(np.sum((targets == class_id) & (predictions != class_id)))
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
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "class_id": class_id,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    metrics.update(
        macro_precision=float(np.mean([float(row["precision"]) for row in per_class])),
        macro_recall=float(np.mean([float(row["recall"]) for row in per_class])),
        macro_f1=float(np.mean([float(row["f1"]) for row in per_class])),
    )
    return metrics, per_class


def save_test_outputs(
    output_dir: Path,
    dataset: INatJsonDataset,
    targets: np.ndarray,
    probabilities: np.ndarray,
    indices: np.ndarray,
    test_loss: float,
) -> None:
    metrics, per_class = compute_metrics(targets, probabilities, probabilities.shape[1])
    lines = [
        "Test classification report",
        f"test_loss: {test_loss:.6f}",
        f"num_samples: {len(targets)}",
        "",
        "Overall metrics",
    ]
    lines.extend(
        f"{name}: {value:.6f} ({100.0 * value:.2f}%)"
        for name, value in metrics.items()
    )
    lines.extend(
        ["", "Per-class metrics", "class  precision  recall  f1_score  support"]
    )
    for row in per_class:
        lines.append(
            f"{int(row['class_id']):03d}    "
            f"{float(row['precision']):.6f}   "
            f"{float(row['recall']):.6f}  "
            f"{float(row['f1']):.6f}  "
            f"{int(row['support'])}"
        )
    report = "\n".join(lines) + "\n"
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

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
                "correct",
                "top5_labels",
                "top5_probabilities",
                "original_category_id",
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
                    prediction == int(target),
                    json.dumps([f"{int(label):03d}" for label in top]),
                    json.dumps([round(float(probability[label]), 8) for label in top]),
                    record.get("original_category_id", ""),
                ]
            )
    print(report, end="")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
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


def load_checkpoint(model: nn.Module, path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    state = (
        checkpoint["model"]
        if isinstance(checkpoint, dict) and "model" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state)
    return checkpoint


def validate_args(args: argparse.Namespace) -> None:
    if args.num_classes < 2:
        raise ValueError("--num-classes must be >= 2")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.samples_per_class < 2:
        raise ValueError("--samples-per-class must be >= 2 for same-class pairing")
    if args.batch_size < 2 * args.samples_per_class:
        raise ValueError("--batch-size must contain at least two classes")
    if args.batch_size % args.samples_per_class != 0:
        raise ValueError("--batch-size must be divisible by --samples-per-class")
    if args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.image_size < 1 or args.api_hidden_size < 1:
        raise ValueError("--image-size and --api-hidden-size must be >= 1")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be >= 0")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if any(
        value < 0
        for value in (
            args.plain_ce_weight,
            args.api_ce_weight,
            args.rank_margin,
            args.rank_weight,
            args.grad_clip,
        )
    ):
        raise ValueError("Loss weights, rank margin, and --grad-clip must be >= 0")
    if args.plain_ce_weight + args.api_ce_weight + args.rank_weight == 0:
        raise ValueError("At least one loss weight must be > 0")
    if not 0.0 <= args.neighbor_batch_fraction <= 1.0:
        raise ValueError("--neighbor-batch-fraction must be in [0, 1]")
    if args.max_neighbors_per_class < 1:
        raise ValueError("--max-neighbors-per-class must be >= 1")

    learning_rates = [args.backbone_lr, args.classifier_lr, args.api_lr]
    if args.lr is not None:
        learning_rates.append(args.lr)
    if any(rate <= 0 for rate in learning_rates):
        raise ValueError("All learning rates must be > 0")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0")
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
        json.dump(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            file,
            indent=2,
        )

    train_transform, eval_transform = make_transforms(args.model_name, args.image_size)
    test_set = INatJsonDataset(args.data_dir, "test", eval_transform, args.num_classes)
    test_loader = make_eval_loader(
        test_set, args.eval_batch_size, args.num_workers, device, args.seed + 2
    )
    model = ResNet50APINet(
        args.model_name,
        args.num_classes,
        args.api_hidden_size,
        args.dropout,
    ).to(device)

    if args.test_only:
        load_checkpoint(model, args.checkpoint, device)
    else:
        train_set = INatJsonDataset(
            args.data_dir, "train", train_transform, args.num_classes
        )
        val_set = INatJsonDataset(args.data_dir, "val", eval_transform, args.num_classes)
        train_loader, train_sampler = make_balanced_loader(train_set, args, device)
        val_loader = make_eval_loader(
            val_set, args.eval_batch_size, args.num_workers, device, args.seed + 1
        )

        optimizer = torch.optim.AdamW(
            model.optimizer_parameter_groups(args), weight_decay=args.weight_decay
        )
        amp_enabled = device.type == "cuda" and not args.no_amp
        scaler = make_grad_scaler(amp_enabled)
        start_epoch, best_val = 0, -1.0

        if args.resume is not None:
            checkpoint = load_checkpoint(model, args.resume, device)
            if not isinstance(checkpoint, dict):
                raise ValueError("--resume requires a full training checkpoint")
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val = float(checkpoint.get("best_val_top1", -1.0))

        total_steps = args.epochs * len(train_loader)
        warmup_steps = int(args.warmup_epochs * len(train_loader))
        history_path = args.output_dir / "training_history.jsonl"

        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
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
            val_metrics, _ = compute_metrics(
                val_targets, val_probabilities, args.num_classes
            )
            learning_rates = {
                f"{group.get('group_name', index)}_lr": group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            }
            row = {
                "epoch": epoch + 1,
                "seconds": time.time() - started,
                **learning_rates,
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
        if not best_checkpoint.is_file():
            raise RuntimeError(
                "No best.pt was produced. If --resume already reached --epochs, "
                "use --test-only --checkpoint instead."
            )
        load_checkpoint(model, best_checkpoint, device)

    test_loss, targets, probabilities, indices = predict(
        model, test_loader, device, "Test"
    )
    save_test_outputs(
        args.output_dir, test_set, targets, probabilities, indices, test_loss
    )


if __name__ == "__main__":
    main()
