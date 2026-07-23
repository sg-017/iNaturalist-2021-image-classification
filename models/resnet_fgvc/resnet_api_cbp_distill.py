#!/usr/bin/env python3
"""Offline probability distillation into one ResNet-50 + CBP + API-Net student."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_base_module() -> ModuleType:
    path = Path(__file__).with_name("resnet_api_cbp.py")
    spec = importlib.util.spec_from_file_location("resnet_api_cbp_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import base implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distil calibrated CBP/API/CBP+API teachers into one CBP+API student"
    )
    parser.add_argument("--data-dir", type=Path, default=base.DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base.DEFAULT_OUTPUT_DIR.parent / "resnet-api-cbp-distill",
    )
    parser.add_argument("--model-name", default=base.MODEL_NAME)
    parser.add_argument("--num-classes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--cbp-teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--api-teacher-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--supcon-teacher-checkpoint",
        type=Path,
        default=None,
        help="best.pt from a combined CBP+API model trained with descriptor SupCon",
    )
    parser.add_argument(
        "--init-cbp-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional standalone cbp.py best.pt used to initialise the student. "
            "If omitted, keep the ImageNet-pretrained ResNet backbone and the "
            "randomly initialised CBP classifier/API heads."
        ),
    )
    parser.add_argument(
        "--teacher-weights",
        type=float,
        nargs=3,
        metavar=("W_CBP", "W_API", "W_SUPCON"),
        default=None,
        help="Fixed non-negative teacher weights (normalised to sum to one).",
    )
    parser.add_argument(
        "--teacher-weight-steps",
        type=int,
        default=51,
        help="Simplex grid resolution; 51 evaluates weights in increments of 0.02",
    )
    parser.add_argument(
        "--teacher-weight-metric",
        choices=("nll", "top1", "macro_f1"),
        default="nll",
        help="NLL is the recommended smooth objective for a probability teacher",
    )
    parser.add_argument(
        "--teacher-calibration",
        choices=("temperature", "none"),
        default="temperature",
        help="Calibrate each teacher's logit scale on validation before probability fusion",
    )
    parser.add_argument("--teacher-temperature-min", type=float, default=0.05)
    parser.add_argument("--teacher-temperature-max", type=float, default=10.0)
    parser.add_argument("--distill-weight", type=float, default=0.25)
    parser.add_argument("--distill-final-weight", type=float, default=0.0)
    parser.add_argument(
        "--distill-schedule",
        choices=("cosine", "constant"),
        default="cosine",
    )
    parser.add_argument("--distill-temperature", type=float, default=3.0)
    parser.add_argument(
        "--teacher-cache",
        type=Path,
        default=None,
        help="Optional .npz path; defaults to OUTPUT_DIR/teacher_train_probabilities.npz",
    )
    parser.add_argument("--rebuild-teacher-cache", action="store_true")

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--api-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--cbp-output-dim", type=int, default=8192)
    parser.add_argument("--cbp-seed", type=int, default=1)
    parser.add_argument("--cbp-spatial-chunk-size", type=int, default=0)
    parser.add_argument("--no-signed-sqrt", action="store_true")
    parser.add_argument("--no-l2-normalize", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--api-hidden-size", type=int, default=512)

    parser.add_argument("--plain-ce-weight", type=float, default=1.0)
    parser.add_argument("--api-ce-weight", type=float, default=0.15)
    parser.add_argument("--rank-margin", type=float, default=0.2)
    parser.add_argument("--rank-weight", type=float, default=0.05)
    parser.add_argument("--supcon-weight", type=float, default=0.05)
    parser.add_argument("--supcon-temperature", type=float, default=0.1)

    parser.add_argument("--stage2-train-scope", choices=("full", "api-heads"), default="full")
    parser.add_argument("--no-init-cbp-classifier", action="store_true")
    parser.add_argument("--class-neighbors-json", type=Path, default=None)
    parser.add_argument("--neighbor-batch-fraction", type=float, default=0.25)
    parser.add_argument("--max-neighbors-per-class", type=int, default=5)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_classes < 2 or args.epochs < 1:
        raise ValueError("--num-classes must be >= 2 and --epochs must be positive")
    if args.batch_size < 2 * args.samples_per_class:
        raise ValueError("A distillation/API batch must contain at least two classes")
    if args.samples_per_class < 2 or args.batch_size % args.samples_per_class:
        raise ValueError("Batch size must be divisible by samples-per-class >= 2")
    if args.eval_batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid evaluation batch size or worker count")
    if args.teacher_weights is not None:
        weights = np.asarray(args.teacher_weights, dtype=np.float64)
        if np.any(weights < 0.0) or not np.isfinite(weights).all() or weights.sum() <= 0.0:
            raise ValueError("--teacher-weights must be finite, non-negative, and sum above zero")
    if args.teacher_weight_steps < 2:
        raise ValueError("--teacher-weight-steps must be at least 2")
    if not 0.0 < args.teacher_temperature_min < args.teacher_temperature_max:
        raise ValueError("Teacher temperature bounds must satisfy 0 < min < max")
    if args.distill_weight < 0 or args.distill_final_weight < 0:
        raise ValueError("Distillation weights must be non-negative")
    if args.distill_final_weight > args.distill_weight:
        raise ValueError("--distill-final-weight cannot exceed --distill-weight")
    if args.distill_temperature <= 0:
        raise ValueError("Distillation temperature must be positive")
    if args.supcon_weight < 0 or args.supcon_temperature <= 0:
        raise ValueError("Invalid supervised contrastive configuration")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if not 0.0 <= args.neighbor_batch_fraction <= 1.0:
        raise ValueError("--neighbor-batch-fraction must be in [0, 1]")
    if args.max_neighbors_per_class < 1:
        raise ValueError("--max-neighbors-per-class must be positive")
    rates = [args.backbone_lr, args.classifier_lr, args.api_lr]
    if args.lr is not None:
        rates.append(args.lr)
    if any(rate <= 0 for rate in rates):
        raise ValueError("All learning rates must be positive")
    for path in (
        args.cbp_teacher_checkpoint,
        args.api_teacher_checkpoint,
        args.supcon_teacher_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
    if args.init_cbp_checkpoint is not None and not args.init_cbp_checkpoint.is_file():
        raise FileNotFoundError(
            f"Initialisation checkpoint not found: {args.init_cbp_checkpoint}"
        )


def make_plain_eval_loader(
    dataset: Any, args: argparse.Namespace, device: torch.device, seed: int
) -> DataLoader:
    return base.make_eval_loader(dataset, args, device, seed)


def teacher_outputs(
    kind: str,
    checkpoint: Path,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    description: str,
):
    model = base.load_standalone_model(kind, checkpoint, args, device)
    outputs = base.predict_logits(model, loader, device, description)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs


def get_teacher_specs(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    return [
        ("cbp", "cbp", args.cbp_teacher_checkpoint),
        ("api", "api", args.api_teacher_checkpoint),
        ("supcon", "combined", args.supcon_teacher_checkpoint),
    ]


def negative_log_likelihood(
    targets: np.ndarray, probabilities: np.ndarray
) -> float:
    selected = probabilities[np.arange(len(targets)), targets]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def fit_teacher_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    """Fit one positive temperature by validation NLL only."""
    uncalibrated = base.probabilities_from_logits(logits)
    before_nll = negative_log_likelihood(targets, uncalibrated)
    if args.teacher_calibration == "none":
        return 1.0, before_nll, before_nll

    logits_tensor = torch.from_numpy(logits).double()
    targets_tensor = torch.from_numpy(targets).long()
    minimum = math.log(args.teacher_temperature_min)
    maximum = math.log(args.teacher_temperature_max)
    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature], max_iter=50, tolerance_grad=1e-9, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        bounded_log_temperature = log_temperature.clamp(minimum, maximum)
        loss = F.cross_entropy(
            logits_tensor / bounded_log_temperature.exp(), targets_tensor
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        log_temperature.clamp_(minimum, maximum)
        temperature = float(log_temperature.exp().item())
    calibrated = base.probabilities_from_logits(logits / temperature)
    after_nll = negative_log_likelihood(targets, calibrated)
    return temperature, before_nll, after_nll


def teacher_weight_candidates(args: argparse.Namespace):
    if args.teacher_weights is not None:
        weights = np.asarray(args.teacher_weights, dtype=np.float64)
        yield weights / weights.sum()
        return
    denominator = args.teacher_weight_steps - 1
    for cbp_step in range(args.teacher_weight_steps):
        for api_step in range(args.teacher_weight_steps - cbp_step):
            supcon_step = denominator - cbp_step - api_step
            yield np.asarray(
                [cbp_step, api_step, supcon_step], dtype=np.float64
            ) / denominator


def select_teacher_weights(
    model_probabilities: list[np.ndarray],
    targets: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    names = ("cbp", "api", "supcon")
    equal = np.full(3, 1.0 / 3.0)
    rows: list[dict[str, float]] = []
    best_weights = None
    best_key = None
    for weights in teacher_weight_candidates(args):
        probabilities = sum(
            float(weight) * model_probabilities[index]
            for index, weight in enumerate(weights)
        )
        top1, macro_f1 = base.alpha_selection_metrics(
            targets, probabilities, args.num_classes
        )
        row = {
            **{f"weight_{name}": float(weight) for name, weight in zip(names, weights)},
            "val_top1_accuracy": top1,
            "val_macro_f1": macro_f1,
            "val_nll": negative_log_likelihood(targets, probabilities),
        }
        rows.append(row)
        if args.teacher_weight_metric == "nll":
            key = (-row["val_nll"], top1, macro_f1, -float(np.abs(weights - equal).sum()))
        elif args.teacher_weight_metric == "top1":
            key = (top1, macro_f1, -row["val_nll"], -float(np.abs(weights - equal).sum()))
        else:
            key = (macro_f1, top1, -row["val_nll"], -float(np.abs(weights - equal).sum()))
        if best_key is None or key > best_key:
            best_key, best_weights = key, weights.copy()
    if best_weights is None:
        raise RuntimeError("No teacher weight candidate was evaluated")
    return best_weights, rows


def prepare_teacher_cache(
    train_eval_set: Any,
    val_set: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    cache_path = args.teacher_cache or (
        args.output_dir / "teacher_train_probabilities.npz"
    )
    val_loader = make_plain_eval_loader(val_set, args, device, args.seed + 101)
    specs = get_teacher_specs(args)
    val_targets = val_indices = None
    val_logits: list[np.ndarray] = []
    val_probabilities: list[np.ndarray] = []
    temperatures: list[float] = []
    calibration_rows: list[dict[str, Any]] = []
    validation_archive: dict[str, Any] = {
        "model_names": np.asarray([name for name, _, _ in specs]),
        "model_types": np.asarray([kind for _, kind, _ in specs]),
    }
    for model_index, (name, kind, checkpoint) in enumerate(specs):
        targets, logits, indices = teacher_outputs(
            kind,
            checkpoint,
            val_loader,
            args,
            device,
            f"Teacher validation ({name})",
        )
        if val_targets is None:
            val_targets, val_indices = targets, indices
        else:
            base.assert_same_examples(
                val_targets, val_indices, targets, indices, "teacher validation"
            )
        temperature, nll_before, nll_after = fit_teacher_temperature(
            logits, targets, args
        )
        probabilities = base.probabilities_from_logits(logits / temperature)
        val_logits.append(logits)
        val_probabilities.append(probabilities)
        temperatures.append(temperature)
        calibration_rows.append(
            {
                "name": name,
                "temperature": temperature,
                "validation_nll_before": nll_before,
                "validation_nll_after": nll_after,
            }
        )
        validation_archive[f"logits_{model_index}"] = logits
        validation_archive[f"calibrated_probabilities_{model_index}"] = probabilities
    assert val_targets is not None and val_indices is not None
    validation_archive["targets"] = val_targets
    validation_archive["indices"] = val_indices
    validation_archive["temperatures"] = np.asarray(temperatures)

    weights, weight_rows = select_teacher_weights(
        val_probabilities, val_targets, args
    )
    chosen = next(
        row
        for row in weight_rows
        if all(
            np.isclose(row[f"weight_{name}"], weight)
            for (name, _, _), weight in zip(specs, weights)
        )
    )
    validation_archive["selected_weights"] = weights
    np.savez_compressed(
        args.output_dir / "teacher_validation_outputs.npz", **validation_archive
    )
    teacher_selection = {
        "selected_on": (
            "fixed command-line weights"
            if args.teacher_weights is not None
            else "validation only"
        ),
        "test_used": False,
        "fusion_space": "calibrated probabilities",
        "formula": "w_cbp*p_cbp + w_api*p_api + w_supcon*p_supcon",
        "selection_metric": args.teacher_weight_metric,
        "calibration": args.teacher_calibration,
        "calibration_models": calibration_rows,
        "models": [
            {
                "name": name,
                "type": kind,
                "checkpoint": str(checkpoint),
                "temperature": temperature,
                "weight": float(weight),
            }
            for (name, kind, checkpoint), temperature, weight in zip(
                specs, temperatures, weights
            )
        ],
        "validation_metrics": chosen,
        "num_weight_candidates": len(weight_rows),
    }
    (args.output_dir / "teacher_selection.json").write_text(
        json.dumps(teacher_selection, indent=2), encoding="utf-8"
    )
    print(json.dumps({"teacher_ensemble": teacher_selection}, indent=2))

    if cache_path.is_file() and not args.rebuild_teacher_cache:
        with np.load(cache_path) as cache:
            if "cache_version" not in cache or "fused_probabilities" not in cache:
                raise ValueError(
                    "Teacher cache is from the old logit format; pass "
                    "--rebuild-teacher-cache"
                )
            version = int(np.asarray(cache["cache_version"]).item())
            probabilities = np.asarray(
                cache["fused_probabilities"], dtype=np.float32
            )
            indices = np.asarray(cache["indices"], dtype=np.int64)
            cached_weights = np.asarray(cache["weights"], dtype=np.float64)
            cached_temperatures = np.asarray(cache["temperatures"], dtype=np.float64)
            cached_checkpoints = np.asarray(cache["teacher_checkpoints"]).astype(str).tolist()
            cached_data_dir = str(np.asarray(cache["data_dir"]).item())
            cached_image_size = int(np.asarray(cache["image_size"]).item())
        if version != 2:
            raise ValueError("Teacher cache is from an older format; pass --rebuild-teacher-cache")
        if probabilities.shape != (len(train_eval_set), args.num_classes):
            raise ValueError(
                f"Teacher cache has incompatible shape: {probabilities.shape}"
            )
        if not np.array_equal(np.sort(indices), np.arange(len(train_eval_set))):
            raise ValueError("Teacher cache indices do not cover the train dataset exactly")
        if not np.allclose(cached_weights, weights, rtol=0.0, atol=1e-12):
            raise ValueError(
                "Teacher cache weights differ from selected weights; pass "
                "--rebuild-teacher-cache"
            )
        if not np.allclose(
            cached_temperatures, temperatures, rtol=0.0, atol=1e-8
        ):
            raise ValueError(
                "Teacher cache temperatures differ from current calibration; pass "
                "--rebuild-teacher-cache"
            )
        expected_metadata = (
            [str(checkpoint) for _, _, checkpoint in specs],
            str(args.data_dir),
            args.image_size,
        )
        cached_metadata = (
            cached_checkpoints,
            cached_data_dir,
            cached_image_size,
        )
        if cached_metadata != expected_metadata:
            raise ValueError(
                "Teacher cache checkpoint/data/image-size metadata differs; pass "
                "--rebuild-teacher-cache"
            )
        ordered = np.empty_like(probabilities)
        ordered[indices] = probabilities
        print(f"Loaded teacher cache: {cache_path}")
        return torch.from_numpy(ordered), teacher_selection

    train_loader = make_plain_eval_loader(
        train_eval_set, args, device, args.seed + 102
    )
    train_targets = train_indices = None
    fused_probabilities = np.zeros(
        (len(train_eval_set), args.num_classes), dtype=np.float64
    )
    for (name, kind, checkpoint), temperature, weight in zip(
        specs, temperatures, weights
    ):
        targets, logits, indices = teacher_outputs(
            kind,
            checkpoint,
            train_loader,
            args,
            device,
            f"Cache train teacher probabilities ({name})",
        )
        if train_targets is None:
            train_targets, train_indices = targets, indices
        else:
            base.assert_same_examples(
                train_targets, train_indices, targets, indices, "teacher train"
            )
        calibrated = base.probabilities_from_logits(logits / temperature)
        fused_probabilities += float(weight) * calibrated
    assert train_targets is not None and train_indices is not None
    fused_probabilities = fused_probabilities.astype(np.float32)
    fused_probabilities /= fused_probabilities.sum(axis=1, keepdims=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        cache_version=np.asarray(2),
        fused_probabilities=fused_probabilities,
        targets=train_targets,
        indices=train_indices,
        weights=weights,
        temperatures=np.asarray(temperatures),
        teacher_names=np.asarray([name for name, _, _ in specs]),
        teacher_checkpoints=np.asarray([str(checkpoint) for _, _, checkpoint in specs]),
        data_dir=np.asarray(str(args.data_dir)),
        image_size=np.asarray(args.image_size),
    )
    ordered = np.empty_like(fused_probabilities)
    ordered[train_indices] = fused_probabilities
    print(f"Saved teacher cache: {cache_path}")
    return torch.from_numpy(ordered), teacher_selection


def distillation_kl(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student_log_probabilities = F.log_softmax(student_logits.float() / temperature, dim=1)
    # The calibrated teachers are fused as probabilities. Raising the fused
    # distribution to 1/T supplies the usual KD softening without reconstructing
    # or averaging uncalibrated teacher logits.
    teacher_probabilities = F.softmax(
        teacher_probabilities.float().clamp_min(1e-12).log() / temperature,
        dim=1,
    )
    return (
        F.kl_div(
            student_log_probabilities,
            teacher_probabilities,
            reduction="batchmean",
        )
        * temperature
        * temperature
    )


def distillation_weight_at_step(
    args: argparse.Namespace, global_step: int, total_steps: int
) -> float:
    if args.distill_schedule == "constant":
        return float(args.distill_weight)
    progress = min(max(global_step / max(total_steps - 1, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(
        args.distill_final_weight
        + (args.distill_weight - args.distill_final_weight) * cosine
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    teacher_probabilities_by_index: torch.Tensor,
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
        "objective",
        "base_objective",
        "distillation_kl",
        "distillation_weight",
        "plain_cross_entropy",
        "api_cross_entropy",
        "rank_loss",
        "supervised_contrastive_loss",
        "plain_top1",
        "interaction_top1",
        "rank_satisfaction",
    )
    totals = {name: 0.0 for name in names}
    batches = 0
    amp_enabled = device.type == "cuda" and not args.no_amp
    progress = tqdm(loader, desc=f"Distil {epoch + 1}/{args.epochs}")
    for batch_index, (images, targets, indices) in enumerate(progress):
        base.cosine_lr(
            optimizer,
            epoch * len(loader) + batch_index,
            total_steps,
            warmup_steps,
        )
        global_step = epoch * len(loader) + batch_index
        current_distill_weight = distillation_weight_at_step(
            args, global_step, total_steps
        )
        teacher_probabilities = teacher_probabilities_by_index[indices.long()].to(
            device, non_blocking=True
        )
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            plain_logits, interaction, features = model(images, targets)
            base_losses = base.api_net_loss(
                interaction, plain_logits, features, targets, args
            )
            kl_loss = distillation_kl(
                plain_logits, teacher_probabilities, args.distill_temperature
            )
            objective = (
                base_losses["objective"] + current_distill_weight * kl_loss
            )
        scaler.scale(objective).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batches += 1
        values = {
            "objective": objective,
            "base_objective": base_losses["objective"],
            "distillation_kl": kl_loss,
            "distillation_weight": torch.tensor(
                current_distill_weight, device=device
            ),
            **{name: base_losses[name] for name in names[4:]},
        }
        for name in names:
            totals[name] += float(values[name].item())
        progress.set_postfix(
            loss=f"{totals['objective'] / batches:.4f}",
            kl=f"{totals['distillation_kl'] / batches:.4f}",
            kd_w=f"{current_distill_weight:.3f}",
            acc=f"{totals['plain_top1'] / batches:.4f}",
        )
    if batches == 0:
        raise ValueError("Training loader is empty")
    return {name: value / batches for name, value in totals.items()}


def build_student(args: argparse.Namespace, device: torch.device):
    model = base.ResNet50APICBP(
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
    resume_payload = None
    if args.resume is not None:
        resume_payload = base.load_combined_checkpoint(model, args.resume, device)
    elif args.init_cbp_checkpoint is not None:
        base.initialise_from_cbp_checkpoint(model, args.init_cbp_checkpoint, args)
    scope_information = base.configure_stage2_train_scope(
        model, args.stage2_train_scope
    )
    return model, resume_payload, scope_information


def main() -> None:
    args = parse_args()
    validate_args(args)
    base.seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(base.serializable_args(args), indent=2), encoding="utf-8"
    )

    train_transform, eval_transform = base.make_transforms(
        args.model_name, args.image_size
    )
    train_set = base.INatJsonDataset(
        args.data_dir, "train", train_transform, args.num_classes
    )
    train_eval_set = base.INatJsonDataset(
        args.data_dir, "train", eval_transform, args.num_classes
    )
    val_set = base.INatJsonDataset(
        args.data_dir, "val", eval_transform, args.num_classes
    )
    test_set = base.INatJsonDataset(
        args.data_dir, "test", eval_transform, args.num_classes
    )
    teacher_probabilities_by_index, teacher_selection = prepare_teacher_cache(
        train_eval_set, val_set, args, device
    )

    model, resume_payload, scope_information = build_student(args, device)
    train_loader, train_sampler = base.make_train_loader(train_set, args, device)
    val_loader = base.make_eval_loader(val_set, args, device, args.seed + 1)
    test_loader = base.make_eval_loader(test_set, args, device, args.seed + 2)
    optimizer = torch.optim.AdamW(
        model.optimizer_parameter_groups(args), weight_decay=args.weight_decay
    )
    scaler = base.make_grad_scaler(device.type == "cuda" and not args.no_amp)
    start_epoch, best_val = 0, -1.0
    if resume_payload is not None:
        if not isinstance(resume_payload, dict) or "optimizer" not in resume_payload:
            raise ValueError("--resume requires a full distillation training checkpoint")
        optimizer.load_state_dict(resume_payload["optimizer"])
        scaler.load_state_dict(resume_payload["scaler"])
        start_epoch = int(resume_payload["epoch"]) + 1
        best_val = float(resume_payload.get("best_val_top1", -1.0))

    (args.output_dir / "distillation_setup.json").write_text(
        json.dumps(
            {
                "teacher_selection": teacher_selection,
                "teacher_probabilities_are_offline_center_crop_predictions": True,
                "teacher_probability_temperature_softening": (
                    "softmax(log(fused_calibrated_probabilities) / distill_temperature)"
                ),
                "distill_schedule": args.distill_schedule,
                "distill_initial_weight": args.distill_weight,
                "distill_final_weight": args.distill_final_weight,
                "distill_temperature": args.distill_temperature,
                "student_initialisation": (
                    {
                        "source": "standalone_cbp_checkpoint",
                        "checkpoint": str(args.init_cbp_checkpoint),
                    }
                    if args.init_cbp_checkpoint is not None
                    else {
                        "source": "imagenet_pretrained_backbone",
                        "model_name": args.model_name,
                        "cbp_classifier": "random_initialisation",
                        "api_heads": "random_initialisation",
                    }
                ),
                "training_scope": scope_information,
                "test_teacher_inference": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_epochs * len(train_loader))
    history_path = args.output_dir / "training_history.jsonl"
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        started = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            teacher_probabilities_by_index,
            optimizer,
            scaler,
            device,
            args,
            epoch,
            total_steps,
            warmup_steps,
        )
        val_targets, val_logits, _ = base.predict_logits(
            model, val_loader, device, "Student validation"
        )
        val_probabilities = base.probabilities_from_logits(val_logits)
        val_metrics, _ = base.compute_metrics(
            val_targets, val_probabilities, args.num_classes
        )
        row = {
            "epoch": epoch + 1,
            "seconds": time.time() - started,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{
                f"{group.get('group_name', index)}_lr": group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            },
        }
        with history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row) + "\n")
        print(json.dumps(row, indent=2))
        current = val_metrics["top1_accuracy"]
        if current > best_val:
            best_val = current
            base.save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                scaler,
                epoch,
                best_val,
                args,
            )
        base.save_checkpoint(
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
        raise RuntimeError("No best.pt was produced")
    base.load_combined_checkpoint(model, best_path, device)
    targets, logits, indices = base.predict_logits(
        model, test_loader, device, "Student test (teachers not used)"
    )
    probabilities = base.probabilities_from_logits(logits)
    np.savez_compressed(
        args.output_dir / "test_outputs.npz",
        targets=targets,
        indices=indices,
        logits=logits,
        probabilities=probabilities,
    )
    base.save_classification_outputs(
        args.output_dir,
        test_set,
        targets,
        probabilities,
        indices,
        "Test classification report: single distilled ResNet-50 + CBP + API student",
        extra_lines=[
            "teacher_fusion_space: calibrated probabilities",
            "teacher_weights: "
            + ", ".join(
                f"{model['name']}={model['weight']:.6f}"
                for model in teacher_selection["models"]
            ),
            "teacher_temperatures: "
            + ", ".join(
                f"{model['name']}={model['temperature']:.6f}"
                for model in teacher_selection["models"]
            ),
            f"distill_schedule: {args.distill_schedule}",
            f"distill_initial_weight: {args.distill_weight:.6f}",
            f"distill_final_weight: {args.distill_final_weight:.6f}",
            f"distill_temperature: {args.distill_temperature:.6f}",
            "teachers_used_at_test: false",
        ],
    )


if __name__ == "__main__":
    main()
