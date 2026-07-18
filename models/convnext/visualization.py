import json
from pathlib import Path
from typing import Any
import matplotlib
import numpy as np
matplotlib.use("Agg")
from matplotlib import pyplot as plt

def deduplicate_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # Keep the final record for each epoch and return epochs in ascending order
    records_by_epoch: dict[int, dict[str, Any]] = {}
    for record in records:
        records_by_epoch[int(record["epoch"])] = record
    return [records_by_epoch[epoch] for epoch in sorted(records_by_epoch)]

def save_curve(
    records: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    epochs = [int(record["epoch"]) for record in records]
    training = [float(record["train"][metric]) for record in records]
    validation = [float(record["val"][metric]) for record in records]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, training, marker="o", markersize=3, label="Training")
    axis.plot(epochs, validation, marker="s", markersize=3, label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

# Generate graphs for loss, top-1, and top-5 accuracy based on metrics.jsonl
def plot_training_curves(metrics_path: Path, output_dir: Path) -> list[Path]:
    with metrics_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"No metric records found in {metrics_path}")
    records = deduplicate_records(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    specifications = [
        (
            "loss",
            "Cross-entropy loss",
            "Training and validation loss",
            "training_validation_loss.png",
        ),(
            "top1",
            "Top-1 accuracy",
            "Training and validation top-1 accuracy",
            "training_validation_top1.png",
        ),(
            "top5",
            "Top-5 accuracy",
            "Training and validation top-5 accuracy",
            "training_validation_top5.png",
        ),
    ]
    outputs: list[Path] = []
    for metric, ylabel, title, filename in specifications:
        output_path = output_dir / filename
        save_curve(records, metric, ylabel, title, output_path)
        outputs.append(output_path)
    return outputs

# Plot all classes and the confusion matrix
def plot_normalized_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
) -> None:
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(normalized, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xlabel("Predicted remapped class index")
    axis.set_ylabel("True remapped class index")
    axis.set_title("Row-normalized confusion matrix (500 classes)")
    axis.set_xticks([])
    axis.set_yticks([])
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Fraction of true-class samples")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

# Plot a readable subset of the most common directional confusions
def plot_top_confused_pairs(
    confusion_rows: list[dict[str, Any]],
    output_path: Path,
    maximum_pairs: int = 20,
) -> None:
    selected = confusion_rows[:maximum_pairs]
    figure_height = max(4.5, 0.38 * max(1, len(selected)) + 1.5)
    figure, axis = plt.subplots(figsize=(10, figure_height))
    if selected:
        labels = [
            (
                f"{row['true_label']:03d} → {row['predicted_label']:03d}  "
                f"({row['true_original_category_id']} → "
                f"{row['predicted_original_category_id']})"
            )
            for row in reversed(selected)
        ]
        counts = [int(row["count"]) for row in reversed(selected)]
        positions = np.arange(len(labels))
        bars = axis.barh(positions, counts)
        axis.set_yticks(positions, labels=labels)
        axis.bar_label(bars, padding=3)
        axis.set_xlabel("Number of test images")
        axis.set_ylabel("True → predicted (original category IDs in parentheses)")
    else:
        axis.text(0.5, 0.5, "No off-diagonal confusions", ha="center", va="center")
        axis.set_xticks([])
        axis.set_yticks([])
    axis.set_title("Most frequent directional class confusions")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    