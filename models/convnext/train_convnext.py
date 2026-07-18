import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from .convnext_classifier import MODEL_NAME, NUM_CLASSES, build_optimizer, create_convnext_classifier, optimizer_stage, set_backbone_trainable, trainable_parameter_count
from .data import build_datasets, make_loader
from .engine import evaluate, synch_device, train_one_epoch

FILE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(__file__).resolve().parents[2]
MODES = ("scratch", "linear_probe", "full_finetune")

# Train ConvNeXt-Tiny on the subset of iNaturalist
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_DIR / "dataset")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--min-crop-scale", type=float, default=0.8)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--classifier-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()

def validate_args(args):
    if args.epochs <= 0:
        raise ValueError("Error: --epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("Error: --batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("Error: --num-workers cannot be negative")
    if not 0 < args.min_crop_scale <= 1:
        raise ValueError("Error: --min-crop-scale must be in (0, 1]")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("Error: --warmup-epochs must be between 0 and epochs")
    if args.mode != "full_finetune" and args.freeze_epochs is not None:
        raise ValueError("Error: --freeze-epochs is only for full_finetune")
    if args.freeze_epochs is not None and args.freeze_epochs < 0:
        raise ValueError("Error: --freeze-epochs cannot be negative")
    if args.early_stopping_patience is not None and args.early_stopping_patience < 0:
        raise ValueError("Error: --early-stopping-patience cannot be negative")
    if args.early_stopping_min_delta < 0:
        raise ValueError("Error: --early-stopping-min-delta cannot be negative")

def select_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def save_json(path, payload):
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

def save_checkpoint(path, checkpoint):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)

def schedule_multiplier(epoch, total_epochs, warmup_epochs, start_epoch=0):
    local_epoch = epoch - start_epoch
    if local_epoch < 0:
        return 0.0
    if warmup_epochs > 0 and local_epoch < warmup_epochs:
        return (local_epoch + 1) / warmup_epochs
    decay_start = start_epoch + warmup_epochs
    remaining = max(1, total_epochs - decay_start)
    progress = min(1.0, max(0.0, (epoch - decay_start) / remaining))
    return 0.5 * (1 + math.cos(math.pi * progress))

def apply_learning_rates(optimizer, epoch, total_epochs, warmup_epochs, mode, freeze_epochs):
    rates = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", index))
        start_epoch = freeze_epochs if mode == "full_finetune" and name == "backbone" else 0
        group["lr"] = group["base_lr"] * schedule_multiplier(
            epoch, total_epochs, warmup_epochs, start_epoch
        )
        rates[name] = float(group["lr"])
    return rates

def stage_for_epoch(mode, epoch, freeze_epochs):
    if mode == "scratch":
        return "unfrozen"
    if mode == "linear_probe":
        return "frozen"
    return "frozen" if epoch < freeze_epochs else "unfrozen"

def percent(value):
    return f"{value * 100:.2f}%"

def print_header(args, device, amp_enabled, stage, trainable, train_size, val_size):
    line = "=" * 88
    print(f"\n{line}")
    print(f"ConvNeXt-Tiny training | mode={args.mode} | device={device} | AMP={amp_enabled}")
    print(f"Epochs: 1-{args.epochs} | stage={stage} | trainable parameters={trainable:,}")
    print(f"Samples: train={train_size:,} | validation={val_size:,}")
    if args.early_stopping_patience == 0:
        print("Early stopping: disabled")
    else:
        print(
            f"Early stopping: patience={args.early_stopping_patience}, "
            f"min_delta={percent(args.early_stopping_min_delta)}"
        )
    print(line, flush=True)

def print_epoch(args, epoch, stage, train_metrics, val_metrics, rates, best_top1, best_loss, train_time, val_time, counter):
    line = "=" * 88
    print(f"\n{line}")
    print(f"Mode: {args.mode} | Epoch: {epoch}/{args.epochs} | Stage: {stage}")
    print("-" * 88)
    print(f"{'Split':<10}{'Loss':>11}{'Top-1':>12}{'Top-5':>12}{'Samples':>13}{'Runtime':>14}")
    for name, values, runtime in (("Train", train_metrics, train_time), ("Val", val_metrics, val_time)):
        print(
            f"{name:<10}{values['loss']:>11.4f}{percent(values['top1']):>12}"
            f"{percent(values['top5']):>12}{int(values['samples']):>13,}{runtime:>12.1f}s"
        )
    print("-" * 88)
    print("LR: " + " | ".join(f"{name}={rate:.2e}" for name, rate in rates.items()))
    print(f"Best val top-1: {percent(best_top1)} | Best val loss: {best_loss:.4f}")
    if args.early_stopping_patience > 0:
        print(f"Early stopping counter: {counter}/{args.early_stopping_patience}")
    print(line, flush=True)

def main():
    args = parse_args()
    validate_args(args)
    args.freeze_epochs = 2 if args.mode == "full_finetune" and args.freeze_epochs is None else (args.freeze_epochs or 0)
    args.early_stopping_patience = (
        (10 if args.mode == "scratch" else 5)
        if args.early_stopping_patience is None
        else args.early_stopping_patience
    )
    if args.mode == "full_finetune" and args.freeze_epochs + args.warmup_epochs >= args.epochs:
        raise ValueError("freeze epochs and warmup epochs must be smaller than the total epochs")
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else FILE_DIR / "outputs" / f"convnext_{args.mode}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = select_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    pretrained = args.mode != "scratch"
    model, data_config = create_convnext_classifier(pretrained=pretrained)
    model.to(device)

    train_dataset, val_dataset = build_datasets(
        args.dataset_dir, data_config, args.min_crop_scale
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, args.num_workers, True, device.type == "cuda", args.seed
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, args.num_workers, False, device.type == "cuda", args.seed + 1
    )
    stage = stage_for_epoch(args.mode, 0, args.freeze_epochs)
    set_backbone_trainable(model, stage == "unfrozen")
    optimizer = build_optimizer(model, args.backbone_lr, args.classifier_lr, args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    train_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_loss = nn.CrossEntropyLoss()
    run_config = {
        "mode": args.mode,
        "model_name": MODEL_NAME,
        "pretrained": pretrained,
        "pretraining_description": "ImageNet-22K pretrained" if pretrained else "random initialization",
        "num_classes": NUM_CLASSES,
        "resolved_model_data_config": json_safe(data_config),
        "loss_reporting": {
            "training_objective": f"cross_entropy_label_smoothing_{args.label_smoothing}",
            "validation_and_test": "ordinary_cross_entropy",
        },
        "arguments": json_safe(vars(args)),
    }
    save_json(args.output_dir / "config.json", run_config)
    metrics_path = args.output_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    best_top1 = -1.0
    best_epoch = 0
    best_loss = float("inf")
    counter = 0
    total_runtime = 0.0
    current_stage = optimizer_stage(model)
    print_header(
        args, device, amp_enabled, current_stage,
        trainable_parameter_count(model.parameters()), len(train_dataset), len(val_dataset)
    )

    for epoch in range(args.epochs):
        wanted_stage = stage_for_epoch(args.mode, epoch, args.freeze_epochs)
        if wanted_stage != current_stage:
            set_backbone_trainable(model, wanted_stage == "unfrozen")
            optimizer = build_optimizer(model, args.backbone_lr, args.classifier_lr, args.weight_decay)
            current_stage = optimizer_stage(model)
            counter = 0
        synch_device(device)
        epoch_start = time.perf_counter()
        rates = apply_learning_rates(
            optimizer, epoch, args.epochs, args.warmup_epochs, args.mode, args.freeze_epochs
        )
        train_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model, train_loader, train_loss, optimizer, scaler, device, amp_enabled,
            backbone_frozen=current_stage == "frozen"
        )
        synch_device(device)
        train_time = time.perf_counter() - train_start
        val_start = time.perf_counter()
        val_metrics, _ = evaluate(model, val_loader, eval_loss, device, amp_enabled)
        synch_device(device)
        val_time = time.perf_counter() - val_start
        epoch_time = time.perf_counter() - epoch_start
        total_runtime += epoch_time
        old_best = best_top1
        is_best = val_metrics["top1"] > old_best
        meaningful = val_metrics["top1"] > old_best + args.early_stopping_min_delta
        is_best_loss = val_metrics["loss"] < best_loss
        if is_best:
            best_top1 = val_metrics["top1"]
            best_epoch = epoch + 1
        if is_best_loss:
            best_loss = val_metrics["loss"]
        counter = 0 if meaningful else counter + 1
        record = {
            "mode": args.mode,
            "epoch": epoch + 1,
            "optimizer_stage": current_stage,
            "learning_rates": rates,
            "train": {**train_metrics, "loss_definition": "label_smoothed_cross_entropy"},
            "val": {**val_metrics, "loss_definition": "ordinary_cross_entropy"},
            "best_val_top1": best_top1,
            "best_val_epoch": best_epoch,
            "best_val_loss": best_loss,
            "epochs_without_improvement": counter,
            "training_runtime_seconds": train_time,
            "validation_runtime_seconds": val_time,
            "epoch_runtime_seconds": epoch_time,
            "total_training_runtime_seconds": total_runtime,
        }
        with metrics_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        checkpoint = {
            "mode": args.mode,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "best_val_top1": best_top1,
            "best_val_epoch": best_epoch,
            "best_val_loss": best_loss,
            "run_config": run_config,
        }
        save_checkpoint(args.output_dir / "latest_checkpoint.pt", checkpoint)
        if is_best:
            save_checkpoint(args.output_dir / "best_checkpoint.pt", checkpoint)
        if is_best_loss:
            save_checkpoint(args.output_dir / "best_val_loss_checkpoint.pt", checkpoint)
        should_stop = (
            args.early_stopping_patience > 0
            and counter >= args.early_stopping_patience
        )
        summary = {
            "mode": args.mode,
            "model_name": MODEL_NAME,
            "pretrained": pretrained,
            "completed_epochs": epoch + 1,
            "requested_epochs": args.epochs,
            "stopped_early": should_stop,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "best_val_top1": best_top1,
            "best_val_epoch": best_epoch,
            "best_val_loss": best_loss,
            "validation_loss_definition": "ordinary_cross_entropy",
            "total_training_runtime_seconds": total_runtime,
        }
        save_json(args.output_dir / "training_summary.json", summary)
        print_epoch(
            args, epoch + 1, current_stage, train_metrics, val_metrics, rates,
            best_top1, best_loss, train_time, val_time, counter
        )
        if should_stop:
            print(
                f"\nEarly stopping triggered after epoch {epoch + 1}: "
                f"validation top-1 did not improve by at least "
                f"{percent(args.early_stopping_min_delta)} for "
                f"{args.early_stopping_patience} consecutive epochs.",
                flush=True,
            )
            break

if __name__ == "__main__":
    main()
