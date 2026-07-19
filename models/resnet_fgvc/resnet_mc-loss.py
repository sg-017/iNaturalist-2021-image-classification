#!/usr/bin/env python3
"""Fine-tune microsoft/resnet-50 with MC-Loss on an iNaturalist 2021 subset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
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
        description="Full fine-tuning of Hugging Face ResNet-50 with optional MC-Loss"
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--num-classes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    parser.add_argument("--resume", type=Path, default=None, help="Resume a training checkpoint")
    parser.add_argument("--test-only", action="store_true", help="Only evaluate --checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=None)

    mc = parser.add_mutually_exclusive_group()
    mc.add_argument("--use-mc-loss", dest="use_mc_loss", action="store_true")
    mc.add_argument("--no-mc-loss", dest="use_mc_loss", action="store_false")
    parser.set_defaults(use_mc_loss=True)
    parser.add_argument("--mc-channels-per-class", type=int, default=3)
    parser.add_argument("--mc-keep-channels", type=int, default=2)
    parser.add_argument("--mc-alpha", type=float, default=1.0)
    parser.add_argument("--mc-beta", type=float, default=20.0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class INatJsonDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, transform: Any, num_classes: int):
        self.data_dir = data_dir.expanduser().resolve()
        self.transform = transform
        json_path = self.data_dir / f"{split}.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {json_path}")
        with json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError(f"{json_path} must contain a JSON list")
        self.records = raw
        for i, item in enumerate(self.records):
            if "file_name" not in item or "label" not in item:
                raise ValueError(f"Record {i} in {json_path} lacks file_name or label")
            label = int(item["label"])
            if not 0 <= label < num_classes:
                raise ValueError(f"Label {label} in record {i} is outside [0, {num_classes - 1}]")

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

    def __getitem__(self, index: int):
        item = self.records[index]
        image_path = self._image_path(str(item["file_name"]))
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                pixel_values = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {image_path}") from exc
        return pixel_values, int(item["label"]), index


class ResNet50Classifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, use_mc_loss: bool, cnum: int):
        super().__init__()
        self.backbone = ResNetModel.from_pretrained(model_name)
        hidden_size = int(self.backbone.config.hidden_sizes[-1])
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.mc_head = (
            nn.Conv2d(hidden_size, num_classes * cnum, kernel_size=1)
            if use_mc_loss
            else None
        )
        self.num_classes = num_classes
        self.cnum = cnum
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)
        if self.mc_head is not None:
            nn.init.kaiming_normal_(self.mc_head.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(self.mc_head.bias)

    def forward(self, pixel_values: torch.Tensor):
        features = self.backbone(pixel_values=pixel_values).last_hidden_state
        logits = self.classifier(F.adaptive_avg_pool2d(features, 1).flatten(1))
        mc_maps = self.mc_head(features) if self.training and self.mc_head is not None else None
        return logits, mc_maps


def mutual_channel_loss(
    maps: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    cnum: int,
    keep_channels: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (discriminative_loss, diversity_loss)."""
    if maps.shape[1] != num_classes * cnum:
        raise ValueError("MC feature channel count must equal num_classes * cnum")

    batch, _, height, width = maps.shape

    spatial_prob = F.softmax(maps.flatten(2), dim=2).view(batch, num_classes, cnum, height * width)
    group_max = spatial_prob.max(dim=2).values
    diversity_loss = 1.0 - group_max.sum(dim=2).mean() / float(cnum)

    noise = torch.rand(batch, num_classes, cnum, device=maps.device)
    keep_idx = noise.topk(keep_channels, dim=2).indices
    mask = torch.zeros_like(noise).scatter_(2, keep_idx, 1.0).unsqueeze(-1).unsqueeze(-1)
    grouped = maps.view(batch, num_classes, cnum, height, width)
    neg_inf = torch.finfo(maps.dtype).min
    masked = grouped.masked_fill(mask == 0, neg_inf)
    auxiliary_logits = masked.max(dim=2).values.mean(dim=(2, 3))
    discriminative_loss = F.cross_entropy(auxiliary_logits.float(), targets)
    return discriminative_loss, diversity_loss


def make_transforms(model_name: str, image_size: int):
    processor = AutoImageProcessor.from_pretrained(model_name)
    mean, std = processor.image_mean, processor.image_std
    resize_size = int(round(image_size / 0.875))
    train_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
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


def make_loader(dataset: Dataset, batch_size: int, workers: int, shuffle: bool, device: torch.device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
    )


def cosine_lr(optimizer, step: int, total_steps: int, warmup_steps: int, base_lr: float) -> None:
    if warmup_steps > 0 and step < warmup_steps:
        factor = float(step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    for group in optimizer.param_groups:
        group["lr"] = base_lr * factor


def train_one_epoch(model, loader, optimizer, scaler, device, args, epoch, total_steps, warmup_steps):
    model.train()
    totals = {"loss": 0.0, "ce": 0.0, "dis": 0.0, "div": 0.0, "correct": 0, "n": 0}
    amp_enabled = device.type == "cuda" and not args.no_amp
    progress = tqdm(loader, desc=f"Train {epoch + 1}/{args.epochs}")
    for batch_idx, (images, targets, _) in enumerate(progress):
        global_step = epoch * len(loader) + batch_idx
        cosine_lr(optimizer, global_step, total_steps, warmup_steps, args.lr)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits, mc_maps = model(images)
            ce = F.cross_entropy(logits, targets, label_smoothing=args.label_smoothing)
            if args.use_mc_loss:
                dis, div = mutual_channel_loss(
                    mc_maps, targets, args.num_classes,
                    args.mc_channels_per_class, args.mc_keep_channels,
                )
                loss = ce + args.mc_alpha * dis + args.mc_beta * div
            else:
                dis = div = ce.new_zeros(())
                loss = ce
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        n = targets.size(0)
        totals["n"] += n
        totals["loss"] += loss.item() * n
        totals["ce"] += ce.item() * n
        totals["dis"] += dis.item() * n
        totals["div"] += div.item() * n
        totals["correct"] += logits.argmax(1).eq(targets).sum().item()
        progress.set_postfix(loss=f"{totals['loss']/totals['n']:.4f}", acc=f"{totals['correct']/totals['n']:.4f}")
    return {k: (v / totals["n"] if k != "n" else v) for k, v in totals.items()}


@torch.inference_mode()
def predict(model, loader, device, num_classes: int, description: str):
    model.eval()
    all_targets, all_probs, all_indices = [], [], []
    loss_sum = 0.0
    count = 0
    for images, targets, indices in tqdm(loader, desc=description):
        images = images.to(device, non_blocking=True)
        targets_device = targets.to(device, non_blocking=True)
        logits, _ = model(images)
        loss_sum += F.cross_entropy(logits, targets_device, reduction="sum").item()
        count += targets.size(0)
        all_targets.append(targets)
        all_probs.append(logits.float().softmax(1).cpu())
        all_indices.append(indices)
    if count == 0:
        raise ValueError(f"{description} dataset is empty")
    targets = torch.cat(all_targets).numpy()
    probs = torch.cat(all_probs).numpy()
    indices = torch.cat(all_indices).numpy()
    return loss_sum / count, targets, probs, indices


def compute_metrics(targets: np.ndarray, probs: np.ndarray, num_classes: int) -> dict[str, float]:
    order = np.argsort(-probs, axis=1)[:, : min(5, num_classes)]
    metrics = {f"top{k}_accuracy": float(np.mean(np.any(order[:, :k] == targets[:, None], axis=1)))
               for k in range(1, min(5, num_classes) + 1)}
    predictions = order[:, 0]
    precision, recall, f1 = [], [], []
    for class_id in range(num_classes):
        tp = int(np.sum((targets == class_id) & (predictions == class_id)))
        fp = int(np.sum((targets != class_id) & (predictions == class_id)))
        fn = int(np.sum((targets == class_id) & (predictions != class_id)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    metrics.update(macro_precision=float(np.mean(precision)), macro_recall=float(np.mean(recall)), macro_f1=float(np.mean(f1)))
    return metrics


def save_test_outputs(output_dir: Path, dataset: INatJsonDataset, targets, probs, indices, test_loss: float) -> None:
    metrics = compute_metrics(targets, probs, probs.shape[1])
    lines = ["Test classification report", f"test_loss: {test_loss:.6f}", f"num_samples: {len(targets)}"]
    lines.extend(f"{name}: {value:.6f} ({100.0 * value:.2f}%)" for name, value in metrics.items())
    (output_dir / "classification_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "file_name", "ground_truth", "predicted_label", "predicted_probability",
                         "top5_labels", "top5_probabilities"])
        for target, probability, index in zip(targets, probs, indices):
            record = dataset.records[int(index)]
            top = np.argsort(-probability)[: min(5, len(probability))]
            writer.writerow([
                record.get("image_id", ""), record["file_name"], f"{int(target):03d}", f"{int(top[0]):03d}",
                f"{float(probability[top[0]]):.8f}",
                json.dumps([f"{int(x):03d}" for x in top]),
                json.dumps([round(float(probability[x]), 8) for x in top]),
            ])
    print("\n".join(lines))


def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int, best_val: float, args) -> None:
    checkpoint_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "best_val_top1": best_val,
                "args": checkpoint_args}, path)


def load_model_weights(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    ignore_mc_head: bool = False,
):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except pickle.UnpicklingError:
        print(
            "Warning: loading a legacy checkpoint with weights_only=False. "
            "Only continue if this checkpoint is from a trusted source."
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if ignore_mc_head:
        # MC-Loss is training-only. Test predictions use the backbone and main classifier
        state = {key: value for key, value in state.items() if not key.startswith("mc_head.")}
        incompatible = model.load_state_dict(state, strict=False)
        unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("mc_head.")]
        missing = [key for key in incompatible.missing_keys if not key.startswith("mc_head.")]
        if unexpected or missing:
            raise RuntimeError(
                "Checkpoint does not match the backbone/classifier. "
                f"Missing keys: {missing}; unexpected keys: {unexpected}"
            )
    else:
        model.load_state_dict(state)
    return checkpoint


def main() -> None:
    args = parse_args()
    if args.mc_channels_per_class < 1:
        raise ValueError("--mc-channels-per-class must be >= 1")
    if not 1 <= args.mc_keep_channels <= args.mc_channels_per_class:
        raise ValueError("--mc-keep-channels must be in [1, --mc-channels-per-class]")
    if args.test_only and args.checkpoint is None:
        raise ValueError("--test-only requires --checkpoint")

    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    train_transform, eval_transform = make_transforms(args.model_name, args.image_size)
    test_set = INatJsonDataset(args.data_dir, "test", eval_transform, args.num_classes)
    eval_batch_size = args.eval_batch_size or args.batch_size
    test_loader = make_loader(test_set, eval_batch_size, args.num_workers, False, device)
    model = ResNet50Classifier(args.model_name, args.num_classes, args.use_mc_loss,
                               args.mc_channels_per_class).to(device)

    if args.test_only:
        load_model_weights(model, args.checkpoint, device, ignore_mc_head=True)
    else:
        train_set = INatJsonDataset(args.data_dir, "train", train_transform, args.num_classes)
        val_set = INatJsonDataset(args.data_dir, "val", eval_transform, args.num_classes)
        train_loader = make_loader(train_set, args.batch_size, args.num_workers, True, device)
        val_loader = make_loader(val_set, eval_batch_size, args.num_workers, False, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        amp_enabled = device.type == "cuda" and not args.no_amp
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        start_epoch, best_val = 0, -1.0
        if args.resume is not None:
            checkpoint = load_model_weights(model, args.resume, device)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_val = float(checkpoint.get("best_val_top1", -1.0))

        total_steps = args.epochs * len(train_loader)
        warmup_steps = int(args.warmup_epochs * len(train_loader))
        history_path = args.output_dir / "training_history.jsonl"
        for epoch in range(start_epoch, args.epochs):
            started = time.time()
            train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, args,
                                            epoch, total_steps, warmup_steps)
            val_loss, val_targets, val_probs, _ = predict(model, val_loader, device, args.num_classes, "Validation")
            val_metrics = compute_metrics(val_targets, val_probs, args.num_classes)
            row = {"epoch": epoch + 1, "seconds": time.time() - started,
                   **{f"train_{k}": v for k, v in train_metrics.items()}, "val_loss": val_loss,
                   **{f"val_{k}": v for k, v in val_metrics.items()}}
            with history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row, indent=2))
            save_checkpoint(args.output_dir / "last.pt", model, optimizer, scaler, epoch, best_val, args)
            if val_metrics["top1_accuracy"] > best_val:
                best_val = val_metrics["top1_accuracy"]
                save_checkpoint(args.output_dir / "best.pt", model, optimizer, scaler, epoch, best_val, args)

        load_model_weights(model, args.output_dir / "best.pt", device)

    test_loss, targets, probs, indices = predict(model, test_loader, device, args.num_classes, "Test")
    save_test_outputs(args.output_dir, test_set, targets, probs, indices, test_loss)


if __name__ == "__main__":
    main()
