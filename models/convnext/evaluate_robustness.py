import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any
import numpy as np
import timm
import torch
from torch import nn
from .convnext_classifier import MODEL_NAME, NUM_CLASSES, create_convnext_classifier
from .data import ImageFolderWithPaths, load_original_category_mapping, make_loader
from .engine import evaluate, synch_device

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEGRADATIONS = ("gaussian_noise", "gaussian_blur", "motion_blur", "brightness", "jpeg")
SEVERITIES = range(1, 5)
EXPECTED_CLASSES = {f"{i:03d}": i for i in range(NUM_CLASSES)}
CSV_FIELDS = (
    "degradation", "severity", "loss", "top1", "top5",
    "macro_precision", "macro_recall", "macro_f1", "samples",
    "inference_runtime_seconds",
    "average_inference_time_per_image_seconds",
    "inference_throughput_images_per_second",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a full-finetuned ConvNeXt checkpoint on 20 robustness sets."
    )
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_DIR / "dataset")
    parser.add_argument("--robustness-dir", type=Path, default=PROJECT_DIR / "dataset" / "robustness_data")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()

def select_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)

def write_results(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    with (output_dir / "robustness_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in CSV_FIELDS} for row in rows)
    with (output_dir / "robustness_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

def build_dataset(folder: Path, data_config: dict[str, Any]) -> ImageFolderWithPaths:
    transform = timm.data.create_transform(**data_config, is_training=False)
    dataset = ImageFolderWithPaths(folder, transform=transform)
    if dataset.class_to_idx != EXPECTED_CLASSES:
        raise ValueError(f"{folder} must contain class folders 000-499")

    counts = np.bincount([label for _, label in dataset.samples], minlength=NUM_CLASSES)
    invalid = np.flatnonzero(counts != 10)
    if invalid.size:
        preview = {int(i): int(counts[i]) for i in invalid[:10]}
        raise ValueError(f"{folder} must contain 10 images per class: {preview}")
    return dataset

def macro_metrics(predictions: list[dict[str, Any]]) -> dict[str, float]:
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    true = np.fromiter((row["true_label"] for row in predictions), dtype=np.int64)
    pred = np.fromiter((row["predicted_label"] for row in predictions), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    tp = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )
    active = support > 0
    return {
        "macro_precision": float(precision[active].mean()),
        "macro_recall": float(recall[active].mean()),
        "macro_f1": float(f1[active].mean()),
    }

def main():
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and num-workers cannot be negative")

    dataset_dir = args.dataset_dir.resolve()
    robustness_dir = args.robustness_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else checkpoint_path.parent / "robustness"

    for path, name in (
        (dataset_dir, "dataset directory"),
        (robustness_dir, "robustness directory"),
        (checkpoint_path, "checkpoint"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name.capitalize()} not found: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    load_original_category_mapping(dataset_dir)
    device = select_device(args.device)
    amp_enabled = args.amp and device.type == "cuda"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run_config = checkpoint["run_config"]
    mode = checkpoint.get("mode", run_config.get("mode"))
    if run_config["model_name"] != MODEL_NAME:
        raise ValueError(f"Expected {MODEL_NAME!r}, got {run_config['model_name']!r}")
    if mode != "full_finetune":
        raise ValueError(f"Expected a full-finetune checkpoint, got mode={mode!r}")
    model, data_config = create_convnext_classifier(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    seed = int(run_config.get("arguments", {}).get("seed", 42)) + 2
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    total_runs = len(DEGRADATIONS) * len(SEVERITIES)
    for run, (degradation, severity) in enumerate(
        ((d, s) for d in DEGRADATIONS for s in SEVERITIES), start=1
    ):
        folder = robustness_dir / degradation / f"severity_{severity}"
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing robustness set: {folder}")
        print(f"\n[{run}/{total_runs}] {degradation}, severity {severity}")
        dataset = build_dataset(folder, data_config)
        loader = make_loader(
            dataset,
            args.batch_size,
            args.num_workers,
            shuffle=False,
            pin_memory=device.type == "cuda",
            seed=seed,
        )
        synch_device(device)
        metrics, predictions = evaluate(model, loader, criterion, device, amp_enabled, collect_predictions=True)
        synch_device(device)
        row = {
            "degradation": degradation,
            "severity": severity,
            **metrics,
            **macro_metrics(predictions),
        }
        rows.append(row)
        write_results(output_dir, rows)

        summary_keys = ("degradation", "severity", "loss", "top1", "top5", "macro_f1")
        print(json.dumps({key: row[key] for key in summary_keys}, indent=2))
        del loader, dataset, predictions
        if device.type == "cuda":
            torch.cuda.empty_cache()
    metadata = {
        "mode": mode,
        "model_name": MODEL_NAME,
        "checkpoint": str(checkpoint_path),
        "dataset_dir": str(dataset_dir),
        "robustness_dir": str(robustness_dir),
        "number_of_runs": len(rows),
        "total_runtime_seconds": time.perf_counter() - started,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": str(device),
        "amp": amp_enabled,
    }
    with (output_dir / "robustness_run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"\nAll {len(rows)} robustness evaluations are completed successfully now")
    print(f"Results are saved: {output_dir / 'robustness_metrics.csv'}")

if __name__ == "__main__":
    main()
