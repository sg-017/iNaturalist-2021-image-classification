import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from .convnext_classifier import MODEL_NAME, NUM_CLASSES, create_convnext_classifier
from .data import build_test_dataset, load_original_category_mapping, make_loader
from .engine import evaluate, synch_device
from .visualization import (
    plot_normalized_confusion_matrix,
    plot_top_confused_pairs,
)

PROJECT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_MODES = {"scratch", "linear_probe", "full_finetune"}

# Run final test evaluation and save detailed classification
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_DIR / "dataset")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-confusions", type=int, default=100)
    return parser.parse_args()

def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)

def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def build_confusion_matrix(predictions: list[dict[str, Any]]) -> np.ndarray:
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    true_labels = np.fromiter(
        (row["true_label"] for row in predictions), dtype=np.int64
    )
    predicted_labels = np.fromiter(
        (row["predicted_label"] for row in predictions), dtype=np.int64
    )
    np.add.at(matrix, (true_labels, predicted_labels), 1)
    return matrix

def per_class_rows(
    matrix: np.ndarray, mapping: dict[int, int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in range(NUM_CLASSES):
        true_positive = int(matrix[label, label])
        support = int(matrix[label, :].sum())
        predicted_count = int(matrix[:, label].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "label": label,
                "original_category_id": mapping[label],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "predicted_count": predicted_count,
            }
        )
    return rows

def summarize_classification_metrics(
    matrix: np.ndarray,
    class_rows: list[dict[str, Any]],
    top1: float,
    top5: float,
) -> dict[str, float]:
    active_rows = [row for row in class_rows if int(row["support"]) > 0]
    if not active_rows:
        raise ValueError("Cannot summarize an empty test set")
    total = int(matrix.sum())
    overall_accuracy = float(np.trace(matrix) / total)
    return {
        "overall_accuracy": overall_accuracy,
        "top1_accuracy": float(top1),
        "top5_accuracy": float(top5),
        "macro_precision": float(np.mean([row["precision"] for row in active_rows])),
        "macro_recall": float(np.mean([row["recall"] for row in active_rows])),
        "macro_f1": float(np.mean([row["f1"] for row in active_rows])),
    }

def top_confusion_rows(
    matrix: np.ndarray,
    mapping: dict[int, int],
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for true_label, predicted_label in zip(*np.nonzero(matrix)):
        if true_label == predicted_label:
            continue
        rows.append(
            {
                "true_label": int(true_label),
                "predicted_label": int(predicted_label),
                "count": int(matrix[true_label, predicted_label]),
                "true_original_category_id": mapping[int(true_label)],
                "predicted_original_category_id": mapping[int(predicted_label)],
            }
        )
    rows.sort(key=lambda row: (-row["count"], row["true_label"], row["predicted_label"]))
    return rows[:limit]

def main():
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and num_workers cannot be negative")
    if args.top_confusions <= 0:
        raise ValueError("--top-confusions must be positive")
    args.dataset_dir = args.dataset_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = (
        args.output_dir.resolve() if args.output_dir else args.checkpoint.parent
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    run_config = checkpoint["run_config"]
    if run_config["model_name"] != MODEL_NAME:
        raise ValueError(
            f"Checkpoint model {run_config['model_name']!r} is not {MODEL_NAME!r}"
        )
    mode = checkpoint.get("mode", run_config.get("mode"))
    if mode not in EXPERIMENT_MODES:
        raise ValueError("Checkpoint does not contain a supported explicit experiment mode")

    model, data_config = create_convnext_classifier(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    test_dataset = build_test_dataset(args.dataset_dir, data_config)
    test_loader = make_loader(
        test_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
        seed=int(run_config.get("arguments", {}).get("seed", 42)) + 2,
    )
    mapping = load_original_category_mapping(args.dataset_dir)
    criterion = nn.CrossEntropyLoss()
    synch_device(device)
    test_started = time.perf_counter()
    metrics, predictions = evaluate(
        model,
        test_loader,
        criterion,
        device,
        amp_enabled,
        collect_predictions=True,
    )
    synch_device(device)
    total_test_runtime = time.perf_counter() - test_started

    for row in predictions:
        path = Path(row["image_path"])
        try:
            row["image_path"] = path.resolve().relative_to(args.dataset_dir).as_posix()
        except ValueError:
            row["image_path"] = path.as_posix()
        row["true_original_category_id"] = mapping[row["true_label"]]
        row["predicted_original_category_id"] = mapping[row["predicted_label"]]

    prediction_fields = [
        "image_path",
        "true_label",
        "predicted_label",
        "confidence",
        "true_original_category_id",
        "predicted_original_category_id",
    ]
    write_csv(args.output_dir / "test_predictions.csv", prediction_fields, predictions)
    matrix = build_confusion_matrix(predictions)
    np.save(args.output_dir / "confusion_matrix.npy", matrix)
    np.savetxt(args.output_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    class_rows = per_class_rows(matrix, mapping)
    write_csv(
        args.output_dir / "per_class_metrics.csv",
        ["label", "original_category_id", "precision", "recall", "f1", "support", "predicted_count"],
        class_rows,
    )
    confusion_rows = top_confusion_rows(matrix, mapping, args.top_confusions)
    write_csv(
        args.output_dir / "top_confusions.csv",
        [
            "true_label",
            "predicted_label",
            "count",
            "true_original_category_id",
            "predicted_original_category_id",
        ],
        confusion_rows,
    )
    plot_normalized_confusion_matrix(
        matrix, args.output_dir / "normalized_confusion_matrix.png"
    )
    plot_top_confused_pairs(
        confusion_rows, args.output_dir / "top_confused_pairs.png"
    )
    summary_metrics = summarize_classification_metrics(
        matrix, class_rows, metrics["top1"], metrics["top5"]
    )
    output_metrics = {
        **metrics,
        **summary_metrics,
        "mode": mode,
        "model_name": MODEL_NAME,
        "pretrained": bool(run_config["pretrained"]),
        "pretraining_description": (
            "ImageNet-22K pretrained"
            if run_config["pretrained"]
            else "random initialization (no pretrained weights)"
        ),
        "resolved_model_data_config": run_config["resolved_model_data_config"],
        "loss_definition": "ordinary_cross_entropy",
        "label_smoothing_used_for_training": float(
            run_config.get("arguments", {}).get("label_smoothing", 0.0)
        ),
        "evaluation_batch_size": args.batch_size,
        "total_test_runtime_seconds": total_test_runtime,
        "checkpoint": str(args.checkpoint),
    }
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(output_metrics, handle, indent=2)
    print(json.dumps(output_metrics, indent=2))

if __name__ == "__main__":
    main()
