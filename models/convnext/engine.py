import time
from typing import Any
import torch
from torch import nn
from torch.utils.data import DataLoader

# Training and evaluation
def correct_counts(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    max_k = min(5, logits.shape[1])
    predictions = logits.topk(max_k, dim=1).indices
    matches = predictions.eq(targets.view(-1, 1))
    return int(matches[:, :1].sum().item()), int(matches.sum().item())

def synch_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def configure_training_mode(model: nn.Module, backbone_frozen: bool) -> None:
    if not backbone_frozen:
        model.train()
        return
    model.eval()
    head = getattr(model, "head", None)
    if isinstance(head, nn.Module):
        head.train()
        return
    get_classifier = getattr(model, "get_classifier", None)
    if get_classifier is None:
        raise TypeError("Frozen-backbone training requires model.get_classifier()")
    classifier = get_classifier()
    if not isinstance(classifier, nn.Module):
        raise TypeError("model.get_classifier() did not return an nn.Module")
    classifier.train()

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    backbone_frozen: bool = False,
) -> dict[str, float]:
    configure_training_mode(model, backbone_frozen)
    total_loss = 0.0
    total_samples = 0
    total_top1 = 0
    total_top5 = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.shape[0]
        top1, top5 = correct_counts(logits.detach(), targets)
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        total_top1 += top1
        total_top5 += top5
    if total_samples == 0:
        raise ValueError("Training loader produced no samples")
    return {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top5": total_top5 / total_samples,
        "samples": total_samples,
    }

@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    collect_predictions: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_top1 = 0
    total_top5 = 0
    inference_runtime = 0.0
    rows: list[dict[str, Any]] = []
    for batch in loader:
        images, targets = batch[:2]
        paths = batch[2] if len(batch) == 3 else None
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        synch_device(device)
        started = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
        synch_device(device)
        inference_runtime += time.perf_counter() - started
        # Compute reported cross entropy in float32 (outside autocast)
        loss = criterion(logits.float(), targets)
        batch_size = targets.shape[0]
        top1, top5 = correct_counts(logits, targets)
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        total_top1 += top1
        total_top5 += top5
        if collect_predictions:
            probabilities = logits.float().softmax(dim=1)
            confidence, predicted = probabilities.max(dim=1)
            target_values = targets.cpu().tolist()
            predicted_values = predicted.cpu().tolist()
            confidence_values = confidence.cpu().tolist()
            path_values = list(paths) if paths is not None else [""] * batch_size
            rows.extend(
                {
                    "image_path": path_values[index],
                    "true_label": target_values[index],
                    "predicted_label": predicted_values[index],
                    "confidence": confidence_values[index],
                }
                for index in range(batch_size)
            )
    if total_samples == 0:
        raise ValueError("Evaluation loader produced no samples")
    metrics = {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top5": total_top5 / total_samples,
        "samples": total_samples,
        "inference_runtime_seconds": inference_runtime,
        "average_inference_time_per_image_seconds": inference_runtime / total_samples,
        "inference_throughput_images_per_second": (
            total_samples / inference_runtime if inference_runtime > 0.0 else float("inf")
        ),
    }
    return metrics, rows
