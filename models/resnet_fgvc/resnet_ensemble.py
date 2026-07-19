#!/usr/bin/env python3
"""Validation-selected ensemble for the project's multiple ResNet classifiers."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


MODEL_NAME = "microsoft/resnet-50"
MODEL_SPECS = (
    ("resnet_api", "resnet_api.py", "resnet_api_checkpoint"),
    ("resnet_cbp", "resnet_cbp.py", "resnet_cbp_checkpoint"),
    ("resnet_api_cbp", "resnet_api_cbp.py", "resnet_api_cbp_checkpoint"),
    ("resnet_pmg", "resnet_pmg.py", "resnet_pmg_checkpoint"),
    ("resnet_isqrtcov", "resnet_isqrt-cov.py", "resnet_isqrtcov_checkpoint"),
)
_MODULE_CACHE: dict[str, ModuleType] = {}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select five-model ensemble weights on validation, then evaluate test"
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resnet-api-checkpoint", type=Path, default=None)
    parser.add_argument("--cbp-checkpoint", type=Path, default=None)
    parser.add_argument("--resnet-api-cbp-checkpoint", type=Path, default=None)
    parser.add_argument("--resnet-pmg-checkpoint", type=Path, default=None)
    parser.add_argument("--resnet-cov-gap-checkpoint", type=Path, default=None)

    parser.add_argument("--num-classes", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument(
        "--fusion-space",
        choices=("logits", "probabilities"),
        default="logits",
        help="Weighted-average raw logits (default) or per-model probabilities",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("top1", "macro_f1"),
        default="top1",
        help="Primary validation metric used to select weights",
    )
    parser.add_argument(
        "--pair-alpha-steps",
        type=int,
        default=21,
        help="Grid points on every two-model edge of the weight simplex",
    )
    parser.add_argument(
        "--weight-search-trials",
        type=int,
        default=2000,
        help="Additional deterministic-seed Dirichlet weight candidates",
    )
    parser.add_argument(
        "--search-batch-size",
        type=int,
        default=8,
        help="Weight candidates evaluated together; reduce if search runs out of memory",
    )

    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pmg-resize-size", type=int, default=550)
    parser.add_argument("--cbp-seed", type=int, default=1)
    parser.add_argument("--cbp-spatial-chunk-size", type=int, default=0)
    parser.add_argument("--cbp-dropout", type=float, default=0.3)
    parser.add_argument("--fusion-initial-gate", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--cov-sqrt-iters", type=int, default=5)
    parser.add_argument("--cov-eps", type=float, default=1e-5)
    parser.add_argument("--cov-initial-scale", type=float, default=0.05)
    parser.add_argument(
        "--no-signed-sqrt",
        action="store_true",
        help="Fallback only for raw CBP/API-CBP state_dict checkpoints",
    )
    parser.add_argument(
        "--no-l2-normalize",
        action="store_true",
        help="Fallback only for raw CBP/API-CBP state_dict checkpoints",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_classes < 2:
        raise ValueError("--num-classes must be at least 2")
    if args.eval_batch_size < 1 or args.search_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.pair_alpha_steps < 2:
        raise ValueError("--pair-alpha-steps must be at least 2")
    if args.weight_search_trials < 0:
        raise ValueError("--weight-search-trials cannot be negative")
    if args.image_size < 1 or args.pmg_resize_size < args.image_size:
        raise ValueError("Invalid fallback image/PMG resize size")
    if not 0.0 < args.fusion_initial_gate < 1.0:
        raise ValueError("--fusion-initial-gate must be in (0, 1)")
    if not 0.0 < args.cov_initial_scale < 1.0:
        raise ValueError("--cov-initial-scale must be in (0, 1)")

    for model_name, _, checkpoint_attribute in MODEL_SPECS:
        checkpoint = Path(getattr(args, checkpoint_attribute)).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing {model_name} checkpoint: {checkpoint}")
    for split in ("val", "test"):
        annotation = args.data_dir.expanduser() / f"{split}.json"
        if not annotation.is_file():
            raise FileNotFoundError(f"Missing annotation file: {annotation}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class INatJsonDataset(Dataset):
    """One split from the processed iNaturalist subset."""

    def __init__(
        self, data_dir: Path, split: str, transform: Any, num_classes: int
    ) -> None:
        self.data_dir = data_dir.expanduser().resolve()
        self.transform = transform
        self.json_path = self.data_dir / f"{split}.json"
        with self.json_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"{self.json_path} must contain a non-empty JSON list")
        self.records: list[dict[str, Any]] = raw
        for index, record in enumerate(self.records):
            if not isinstance(record, dict) or not {"file_name", "label"} <= record.keys():
                raise ValueError(
                    f"Record {index} in {self.json_path} lacks file_name or label"
                )
            label = int(record["label"])
            if not 0 <= label < num_classes:
                raise ValueError(
                    f"Label {label} in record {index} is outside [0, {num_classes - 1}]"
                )

    def __len__(self) -> int:
        return len(self.records)

    def image_path(self, file_name: str) -> Path:
        path = Path(file_name)
        if path.is_absolute():
            return path
        candidate = self.data_dir / path
        if candidate.is_file():
            return candidate
        # Also accept "processed_dataset/val/000/image.jpg" in the JSON.
        if path.parts and path.parts[0] == self.data_dir.name:
            return self.data_dir.parent / path
        return candidate

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        record = self.records[index]
        path = self.image_path(str(record["file_name"]))
        try:
            with Image.open(path) as image:
                pixels = self.transform(image.convert("RGB"))
        except Exception as error:
            raise RuntimeError(f"Failed to read image: {path}") from error
        return pixels, int(record["label"]), index


def load_source_module(source_filename: str) -> ModuleType:
    if source_filename in _MODULE_CACHE:
        return _MODULE_CACHE[source_filename]
    path = Path(__file__).resolve().with_name(source_filename)
    if not path.is_file():
        raise FileNotFoundError(f"Required model source file not found: {path}")
    module_name = "ensemble_source_" + source_filename.replace("-", "_").replace(".", "_")
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not create an import specification for {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    _MODULE_CACHE[source_filename] = module
    return module


def torch_load_checkpoint(path: Path) -> Any:
    """Load a locally produced project checkpoint onto CPU.

    ``weights_only=False`` is explicit because the five training scripts save
    ordinary Python scalar metadata in addition to tensors.  Checkpoints are
    executable pickle data and therefore must come from a trusted source.
    """

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint does not contain a valid model state_dict")
    cleaned: dict[str, torch.Tensor] = {}
    for original_key, value in state.items():
        key = str(original_key)
        while key.startswith("module.") or key.startswith("_orig_mod."):
            key = key.split(".", 1)[1]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"State-dict entry {original_key!r} is not a tensor")
        cleaned[key] = value
    return cleaned


def state_shape(state: dict[str, torch.Tensor], key: str, dimension: int) -> int:
    if key not in state:
        raise KeyError(
            f"Cannot infer a raw checkpoint's architecture: state key {key!r} is missing"
        )
    return int(state[key].shape[dimension])


def saved_bool(saved: dict[str, Any], key: str, fallback: bool) -> bool:
    value = saved.get(key, fallback)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def build_model(
    model_key: str,
    source_filename: str,
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any], ModuleType]:
    module = load_source_module(source_filename)
    payload = torch_load_checkpoint(checkpoint_path)
    state = extract_state_dict(payload)
    saved = payload.get("args", {}) if isinstance(payload, dict) else {}
    if not isinstance(saved, dict):
        saved = {}
    if not saved and model_key in {"cbp", "resnet_api_cbp"}:
        print(
            f"Warning: {checkpoint_path} has no saved args; CBP normalization uses CLI fallbacks."
        )

    model_name = str(saved.get("model_name", args.model_name))
    if model_key == "resnet_api":
        inferred_classes = state_shape(state, "classifier.weight", 0)
        model = module.ResNet50APINet(
            model_name=model_name,
            num_classes=inferred_classes,
            api_hidden_size=int(
                saved.get("api_hidden_size", state_shape(state, "map1.weight", 0))
            ),
            dropout=float(saved.get("dropout", args.dropout)),
        )
    elif model_key == "resnet_cbp":
        inferred_classes = state_shape(state, "gap_classifier.weight", 0)
        cbp_output_dim = state_shape(state, "cbp_classifier.weight", 1)
        model = module.ResNet50CBPClassifier(
            model_name=model_name,
            num_classes=inferred_classes,
            cbp_output_dim=int(saved.get("cbp_output_dim", cbp_output_dim)),
            cbp_seed=int(saved.get("cbp_seed", args.cbp_seed)),
            cbp_spatial_chunk_size=int(
                saved.get("cbp_spatial_chunk_size", args.cbp_spatial_chunk_size)
            ),
            signed_sqrt=not saved_bool(
                saved, "no_signed_sqrt", args.no_signed_sqrt
            ),
            l2_normalize=not saved_bool(
                saved, "no_l2_normalize", args.no_l2_normalize
            ),
            cbp_dropout=float(saved.get("cbp_dropout", args.cbp_dropout)),
            fusion_initial_gate=float(
                saved.get("fusion_initial_gate", args.fusion_initial_gate)
            ),
        )
    elif model_key == "resnet_api_cbp":
        inferred_classes = state_shape(state, "classifier.weight", 0)
        cbp_output_dim = state_shape(state, "classifier.weight", 1)
        model = module.ResNet50APICBP(
            model_name=model_name,
            num_classes=inferred_classes,
            cbp_output_dim=int(saved.get("cbp_output_dim", cbp_output_dim)),
            cbp_seed=int(saved.get("cbp_seed", args.cbp_seed)),
            cbp_spatial_chunk_size=int(
                saved.get("cbp_spatial_chunk_size", args.cbp_spatial_chunk_size)
            ),
            signed_sqrt=not saved_bool(
                saved, "no_signed_sqrt", args.no_signed_sqrt
            ),
            l2_normalize=not saved_bool(
                saved, "no_l2_normalize", args.no_l2_normalize
            ),
            api_hidden_size=int(
                saved.get("api_hidden_size", state_shape(state, "map1.weight", 0))
            ),
            dropout=float(saved.get("dropout", args.dropout)),
        )
    elif model_key == "resnet_pmg":
        inferred_classes = state_shape(state, "classifier_concat.4.weight", 0)
        feature_size = state_shape(state, "classifier_concat.1.weight", 0)
        model = module.PMGClassifier(
            model_name=model_name,
            num_classes=inferred_classes,
            feature_size=int(saved.get("feature_size", feature_size)),
        )
    elif model_key == "resnet_isqrtcov":
        inferred_classes = state_shape(state, "gap_classifier.weight", 0)
        cov_dim = state_shape(state, "cov_pool.reduction.0.weight", 0)
        model = module.ResNet50GAPISqrtCovClassifier(
            model_name=model_name,
            num_classes=inferred_classes,
            cov_dim=int(saved.get("cov_dim", cov_dim)),
            sqrt_iters=int(saved.get("sqrt_iters", args.cov_sqrt_iters)),
            cov_eps=float(saved.get("cov_eps", args.cov_eps)),
            cov_initial_scale=float(
                saved.get("cov_initial_scale", args.cov_initial_scale)
            ),
        )
    else:
        raise ValueError(f"Unknown model key: {model_key}")

    saved_classes = int(saved.get("num_classes", inferred_classes))
    if saved_classes != inferred_classes:
        raise ValueError(
            f"{model_key} checkpoint metadata says {saved_classes} classes, "
            f"but its classifier contains {inferred_classes}"
        )
    if inferred_classes != args.num_classes:
        raise ValueError(
            f"{model_key} checkpoint has {inferred_classes} classes; "
            f"--num-classes is {args.num_classes}"
        )
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    effective = {
        "model_name": model_name,
        "num_classes": inferred_classes,
        "image_size": int(saved.get("image_size", args.image_size)),
    }
    if model_key == "resnet_pmg":
        effective["resize_size"] = int(
            saved.get("resize_size", args.pmg_resize_size)
        )
    return model, effective, module


def make_eval_transform(
    model_key: str, module: ModuleType, effective: dict[str, Any]
) -> Any:
    if model_key == "resnet_pmg":
        _, transform = module.make_transforms(
            effective["model_name"],
            effective["image_size"],
            effective["resize_size"],
        )
    else:
        _, transform = module.make_transforms(
            effective["model_name"], effective["image_size"]
        )
    return transform


def make_loader(
    dataset: Dataset, args: argparse.Namespace, device: torch.device
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def inference_logits(
    model_key: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    description: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_targets: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for images, targets, indices in tqdm(loader, desc=description):
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                output = model(images)
                if model_key == "resnet_pmg":
                    if not isinstance(output, (tuple, list)) or len(output) != 4:
                        raise RuntimeError("PMG model must return four logit tensors")
                    logits = output[0] + output[1] + output[2] + output[3]
                else:
                    logits = output
            if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                raise RuntimeError(
                    f"{model_key} returned invalid logits of type {type(logits).__name__}"
                )
            all_targets.append(targets.cpu())
            all_logits.append(logits.float().cpu())
            all_indices.append(indices.cpu())
    if not all_targets:
        raise ValueError(f"{description} dataset is empty")
    return (
        torch.cat(all_targets).numpy().astype(np.int64, copy=False),
        torch.cat(all_logits).numpy().astype(np.float32, copy=False),
        torch.cat(all_indices).numpy().astype(np.int64, copy=False),
    )


def run_checkpoint_on_split(
    model_key: str,
    source_filename: str,
    checkpoint: Path,
    split: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    model, effective, module = build_model(
        model_key, source_filename, checkpoint, args, device
    )
    transform = make_eval_transform(model_key, module, effective)
    dataset = INatJsonDataset(args.data_dir, split, transform, args.num_classes)
    loader = make_loader(dataset, args, device)
    outputs = inference_logits(
        model_key,
        model,
        loader,
        device,
        amp_enabled=device.type == "cuda" and not args.no_amp,
        description=f"{split.capitalize()} ({model_key})",
    )
    records = dataset.records
    del loader, dataset, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (*outputs, records, effective)


def assert_same_examples(
    reference_targets: np.ndarray,
    reference_indices: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    split: str,
    model_key: str,
) -> None:
    if not np.array_equal(reference_targets, targets):
        raise RuntimeError(f"{model_key} saw different {split} labels/order")
    if not np.array_equal(reference_indices, indices):
        raise RuntimeError(f"{model_key} saw different {split} sample indices/order")


def probabilities_from_logits(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64, copy=False)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return (exponentials / exponentials.sum(axis=1, keepdims=True)).astype(
        np.float32
    )


def candidate_weights(
    num_models: int, pair_steps: int, random_trials: int, seed: int
) -> np.ndarray:
    candidates: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()

    def add(raw: Sequence[float] | np.ndarray) -> None:
        weights = np.asarray(raw, dtype=np.float64)
        if weights.shape != (num_models,) or np.any(weights < 0):
            raise ValueError("Invalid generated ensemble weights")
        total = float(weights.sum())
        if total <= 0:
            return
        weights = weights / total
        key = tuple(np.round(weights, 12))
        if key not in seen:
            seen.add(key)
            candidates.append(weights)

    equal = np.full(num_models, 1.0 / num_models)
    add(equal)
    for row in np.eye(num_models):
        add(row)
    grid = np.linspace(0.0, 1.0, pair_steps)
    for first in range(num_models):
        for second in range(first + 1, num_models):
            for alpha in grid:
                weights = np.zeros(num_models)
                weights[first] = alpha
                weights[second] = 1.0 - alpha
                add(weights)
    # Also search paths between equal weighting and every single model.
    for model_index in range(num_models):
        vertex = np.eye(num_models)[model_index]
        for alpha in grid:
            add(alpha * vertex + (1.0 - alpha) * equal)

    generator = np.random.default_rng(seed)
    concentrations = (0.3, 1.0, 3.0)
    base_count, remainder = divmod(random_trials, len(concentrations))
    for concentration_index, concentration in enumerate(concentrations):
        count = base_count + (concentration_index < remainder)
        if count:
            samples = generator.dirichlet(
                np.full(num_models, concentration, dtype=np.float64), size=count
            )
            for sample in samples:
                add(sample)
    return np.stack(candidates)


def metrics_from_predictions(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> tuple[float, float]:
    correct = predictions == targets
    top1 = float(np.mean(correct))
    true_positives = np.bincount(targets[correct], minlength=num_classes).astype(
        np.float64
    )
    predicted_counts = np.bincount(predictions, minlength=num_classes).astype(
        np.float64
    )
    target_counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    precision = np.divide(
        true_positives,
        predicted_counts,
        out=np.zeros(num_classes, dtype=np.float64),
        where=predicted_counts != 0,
    )
    recall = np.divide(
        true_positives,
        target_counts,
        out=np.zeros(num_classes, dtype=np.float64),
        where=target_counts != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(precision + recall) != 0,
    )
    return top1, float(f1.mean())


def search_validation_weights(
    logits: list[np.ndarray],
    targets: np.ndarray,
    candidates: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    shapes = {output.shape for output in logits}
    expected_shape = (len(targets), args.num_classes)
    if shapes != {expected_shape}:
        raise ValueError(
            f"Validation logits must all have shape {expected_shape}; received {shapes}"
        )

    source = torch.from_numpy(np.stack(logits)).to(device=device, dtype=torch.float32)
    if args.fusion_space == "probabilities":
        source = source.softmax(dim=2)
    flattened = source.reshape(len(logits), -1)
    rows: list[dict[str, float]] = []
    equal = np.full(len(logits), 1.0 / len(logits))
    best_key: tuple[float, float, float] | None = None
    best_weights: np.ndarray | None = None

    for start in tqdm(
        range(0, len(candidates), args.search_batch_size),
        desc="Validation weight search",
    ):
        batch = candidates[start : start + args.search_batch_size]
        batch_tensor = torch.as_tensor(batch, dtype=torch.float32, device=device)
        fused = (batch_tensor @ flattened).reshape(
            len(batch), len(targets), args.num_classes
        )
        batch_predictions = fused.argmax(dim=2).cpu().numpy()
        del fused, batch_tensor
        for weights, predictions in zip(batch, batch_predictions):
            top1, macro_f1 = metrics_from_predictions(
                targets, predictions, args.num_classes
            )
            row = {
                "candidate": float(len(rows)),
                **{
                    f"weight_{name}": float(weight)
                    for (name, _, _), weight in zip(MODEL_SPECS, weights)
                },
                "val_top1_accuracy": top1,
                "val_macro_f1": macro_f1,
            }
            rows.append(row)
            primary = top1 if args.selection_metric == "top1" else macro_f1
            secondary = macro_f1 if args.selection_metric == "top1" else top1
            # Deterministic final tie-break: closest to equal weighting.
            key = (primary, secondary, -float(np.abs(weights - equal).sum()))
            if best_key is None or key > best_key:
                best_key = key
                best_weights = weights.copy()
    del flattened, source
    if best_weights is None:
        raise RuntimeError("Weight search produced no candidates")
    return best_weights, rows


def weighted_fusion(
    logits: list[np.ndarray], weights: np.ndarray, space: str
) -> tuple[np.ndarray, np.ndarray]:
    if len(logits) != len(weights):
        raise ValueError("Number of outputs and weights differs")
    if any(output.shape != logits[0].shape for output in logits):
        raise ValueError("All ensemble output shapes must match")
    if np.any(weights < 0) or not np.isclose(float(weights.sum()), 1.0):
        raise ValueError("Weights must be non-negative and sum to one")
    if space == "logits":
        fused = np.zeros_like(logits[0], dtype=np.float64)
        for weight, output in zip(weights, logits):
            fused += float(weight) * output
        probabilities = probabilities_from_logits(fused)
        return fused.astype(np.float32), probabilities

    probabilities = np.zeros_like(logits[0], dtype=np.float64)
    for weight, output in zip(weights, logits):
        probabilities += float(weight) * probabilities_from_logits(output)
    probabilities = probabilities.astype(np.float32)
    equivalent_scores = np.log(np.clip(probabilities, 1e-30, None))
    return equivalent_scores.astype(np.float32), probabilities


def full_classification_metrics(
    targets: np.ndarray, probabilities: np.ndarray, num_classes: int
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    if probabilities.shape != (len(targets), num_classes):
        raise ValueError("Probability array has an unexpected shape")
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
        macro_precision=float(np.mean([row["precision"] for row in per_class])),
        macro_recall=float(np.mean([row["recall"] for row in per_class])),
        macro_f1=float(np.mean([row["f1"] for row in per_class])),
    )
    true_probabilities = probabilities[np.arange(len(targets)), targets]
    metrics["test_nll"] = float(
        -np.log(np.clip(true_probabilities.astype(np.float64), 1e-300, None)).mean()
    )
    return metrics, per_class


def save_prediction_outputs(
    output_dir: Path,
    records: list[dict[str, Any]],
    targets: np.ndarray,
    probabilities: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    args: argparse.Namespace,
) -> None:
    metrics, per_class = full_classification_metrics(
        targets, probabilities, args.num_classes
    )
    weight_text = ", ".join(
        f"{name}={weight:.8f}"
        for (name, _, _), weight in zip(MODEL_SPECS, weights)
    )
    lines = [
        "Test classification report: validation-selected five-model ensemble",
        f"num_samples: {len(targets)}",
        f"fusion_space: {args.fusion_space}",
        f"selection_metric: validation_{args.selection_metric}",
        "weights_selected_on: validation only",
        "test_used_for_weight_selection: false",
        f"weights: {weight_text}",
        "",
        "Overall test metrics",
    ]
    for k in range(1, min(5, args.num_classes) + 1):
        value = metrics[f"top{k}_accuracy"]
        lines.append(f"top{k}_accuracy: {value:.6f} ({100.0 * value:.2f}%)")
    for name in ("macro_precision", "macro_recall", "macro_f1"):
        value = metrics[name]
        lines.append(f"{name}: {value:.6f} ({100.0 * value:.2f}%)")
    lines.extend(
        [
            f"test_nll: {metrics['test_nll']:.6f}",
            "",
            "Per-class test metrics",
            "class  precision  recall  f1_score  support",
        ]
    )
    for row in per_class:
        lines.append(
            f"{int(row['class_id']):03d}    {float(row['precision']):.6f}   "
            f"{float(row['recall']):.6f}  {float(row['f1']):.6f}  "
            f"{int(row['support'])}"
        )
    report = "\n".join(lines) + "\n"
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    metrics_payload = {
        "num_samples": len(targets),
        "fusion_space": args.fusion_space,
        "selection_metric": f"validation_{args.selection_metric}",
        "weights": {
            name: float(weight)
            for (name, _, _), weight in zip(MODEL_SPECS, weights)
        },
        **metrics,
        "per_class": per_class,
    }
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

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
            record = records[int(index)]
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
                    json.dumps(
                        [round(float(probability[label]), 8) for label in top]
                    ),
                    record.get("original_category_id", ""),
                ]
            )
    print(report, end="")


def write_search_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        raise ValueError("Cannot save an empty weight search")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def selected_validation_row(
    rows: list[dict[str, float]], weights: np.ndarray
) -> dict[str, float]:
    for row in rows:
        if all(
            np.isclose(row[f"weight_{name}"], weight)
            for (name, _, _), weight in zip(MODEL_SPECS, weights)
        ):
            return row
    raise RuntimeError("Selected validation weights are missing from search rows")


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def run_validation(
    args: argparse.Namespace, device: torch.device
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[dict[str, Any]],
]:
    reference_targets: np.ndarray | None = None
    reference_indices: np.ndarray | None = None
    all_logits: list[np.ndarray] = []
    effective_configs: list[dict[str, Any]] = []
    for model_key, source_filename, checkpoint_attribute in MODEL_SPECS:
        checkpoint = Path(getattr(args, checkpoint_attribute)).expanduser()
        targets, logits, indices, _, effective = run_checkpoint_on_split(
            model_key,
            source_filename,
            checkpoint,
            "val",
            args,
            device,
        )
        if reference_targets is None or reference_indices is None:
            reference_targets, reference_indices = targets, indices
        else:
            assert_same_examples(
                reference_targets,
                reference_indices,
                targets,
                indices,
                "validation",
                model_key,
            )
        all_logits.append(logits)
        effective_configs.append(effective)
    if reference_targets is None or reference_indices is None:
        raise RuntimeError("No validation outputs were generated")
    return reference_targets, reference_indices, all_logits, effective_configs


def run_test(
    args: argparse.Namespace, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[dict[str, Any]]]:
    reference_targets: np.ndarray | None = None
    reference_indices: np.ndarray | None = None
    reference_records: list[dict[str, Any]] | None = None
    all_logits: list[np.ndarray] = []
    for model_key, source_filename, checkpoint_attribute in MODEL_SPECS:
        checkpoint = Path(getattr(args, checkpoint_attribute)).expanduser()
        targets, logits, indices, records, _ = run_checkpoint_on_split(
            model_key,
            source_filename,
            checkpoint,
            "test",
            args,
            device,
        )
        if reference_targets is None or reference_indices is None:
            reference_targets, reference_indices = targets, indices
            reference_records = records
        else:
            assert_same_examples(
                reference_targets,
                reference_indices,
                targets,
                indices,
                "test",
                model_key,
            )
        all_logits.append(logits)
    if (
        reference_targets is None
        or reference_indices is None
        or reference_records is None
    ):
        raise RuntimeError("No test outputs were generated")
    return reference_targets, reference_indices, all_logits, reference_records


def save_split_archive(
    path: Path,
    targets: np.ndarray,
    indices: np.ndarray,
    logits: list[np.ndarray],
    extra: dict[str, Any] | None = None,
) -> None:
    archive: dict[str, Any] = {
        "targets": targets,
        "indices": indices,
        "model_names": np.asarray([name for name, _, _ in MODEL_SPECS]),
    }
    for (name, _, _), output in zip(MODEL_SPECS, logits):
        archive[f"{name}_logits"] = output.astype(np.float32, copy=False)
        archive[f"{name}_probabilities"] = probabilities_from_logits(output)
    if extra:
        archive.update(extra)
    np.savez_compressed(path, **archive)


def final_fusion_archive_fields(
    scores: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    fusion_space: str,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "weights": weights.astype(np.float64),
        "fusion_space": np.asarray(fusion_space),
        "final_probabilities": probabilities.astype(np.float32, copy=False),
    }
    if fusion_space == "logits":
        fields["final_logits"] = scores.astype(np.float32, copy=False)
    else:
        fields["final_log_probabilities"] = scores.astype(np.float32, copy=False)
    return fields


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

    # Phase 1: validation only.  No test image or label is read in this phase.
    val_targets, val_indices, val_logits, effective_configs = run_validation(
        args, device
    )
    save_split_archive(
        args.output_dir / "validation_outputs.npz",
        val_targets,
        val_indices,
        val_logits,
    )
    candidates = candidate_weights(
        len(MODEL_SPECS),
        args.pair_alpha_steps,
        args.weight_search_trials,
        args.seed,
    )
    best_weights, search_rows = search_validation_weights(
        val_logits, val_targets, candidates, args, device
    )
    selected_val_scores, selected_val_probabilities = weighted_fusion(
        val_logits, best_weights, args.fusion_space
    )
    # Save the chosen validation fusion separately, avoiding recompression of
    # every model's already-persisted validation outputs.
    np.savez_compressed(
        args.output_dir / "validation_selected_fusion.npz",
        targets=val_targets,
        indices=val_indices,
        **final_fusion_archive_fields(
            selected_val_scores,
            selected_val_probabilities,
            best_weights,
            args.fusion_space,
        ),
    )
    write_search_csv(args.output_dir / "ensemble_weight_search.csv", search_rows)
    chosen_row = selected_validation_row(search_rows, best_weights)
    selection = {
        "selected_on": "validation only",
        "test_used_for_weight_selection": False,
        "fusion_space": args.fusion_space,
        "selection_metric": args.selection_metric,
        "num_weight_candidates": len(search_rows),
        "validation_metrics_at_selected_weights": chosen_row,
        "models": [
            {
                "name": model_key,
                "source": source_filename,
                "checkpoint": str(Path(getattr(args, checkpoint_attribute))),
                "weight": float(weight),
                "effective_preprocessing": effective,
            }
            for (
                model_key,
                source_filename,
                checkpoint_attribute,
            ), weight, effective in zip(
                MODEL_SPECS, best_weights, effective_configs
            )
        ],
        "formula": "final_output = sum_i(weight_i * model_i_output)",
    }
    # This file is persisted before test inference.  It is the fixed, auditable
    # input to the test phase below.
    with (args.output_dir / "ensemble_selection.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(selection, file, indent=2)
    print(json.dumps(selection, indent=2))

    del val_logits
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Phase 2: weights are fixed. Test labels are used only for the final report.
    test_targets, test_indices, test_logits, test_records = run_test(args, device)
    final_scores, final_probabilities = weighted_fusion(
        test_logits, best_weights, args.fusion_space
    )
    save_split_archive(
        args.output_dir / "test_outputs.npz",
        test_targets,
        test_indices,
        test_logits,
        extra=final_fusion_archive_fields(
            final_scores,
            final_probabilities,
            best_weights,
            args.fusion_space,
        ),
    )
    save_prediction_outputs(
        args.output_dir,
        test_records,
        test_targets,
        final_probabilities,
        test_indices,
        best_weights,
        args,
    )


if __name__ == "__main__":
    main()
