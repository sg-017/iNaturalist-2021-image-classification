"""Small helper functions for the Grad-CAM notebook and script.

The functions in this file are intentionally written in a simple style.
They use the model-building code that is already in this repository.
"""

import json
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms


GRAD_CAM_DIR = Path(__file__).resolve().parent
REPO_ROOT = GRAD_CAM_DIR.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "grad_cam_explainability"
EXTENDED_RESULTS_DIR = REPO_ROOT / "results" / "grad_cam_explainability_extended"
DATA_DIR = REPO_ROOT / "dataset" / "processed_dataset"
TEST_JSON = DATA_DIR / "test.json"
SEED = 42

STRONG_MODEL_PREDICTION_FILES = {
    "convnext": (
        REPO_ROOT
        / "results"
        / "convnext_full_finetune_seed42"
        / "test_predictions.csv"
    ),
    "fgvc": (
        REPO_ROOT
        / "results"
        / "resnet50_finetune_pmg_isqrtcov_from_pmg"
        / "test_predictions.csv"
    ),
    "dinov2": (
        REPO_ROOT
        / "results"
        / "dinov2_finetune"
        / "test_predictions.csv"
    ),
}

# This image was already identified in the baseline run as a dataset-quality
# case containing warning text instead of a normal species photograph.
KNOWN_DATASET_QUALITY_CASE = (
    "test/166/2732378_f08aebf2-10c1-4bab-94f8-4bc2c7770a31.jpg"
)


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def safe_name(text):
    """Make a short string safe to use as a file name."""
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def relative_path(path):
    """Return a repository-relative path when possible."""
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def make_output_folders(results_dir=DEFAULT_RESULTS_DIR):
    """Create the Grad-CAM result folders."""
    results_dir = Path(results_dir)
    folders = {
        "root": results_dir,
        "per_model": results_dir / "figures" / "per_model",
        "comparisons": results_dir / "figures" / "comparisons",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def choose_device(device_name="cpu"):
    """Select CPU, Apple MPS, or CUDA."""
    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but MPS is not available.")
        return torch.device("mps")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is not available.")
        return torch.device("cuda")

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    raise ValueError("Device must be cpu, mps, cuda, or auto.")


def get_model_inventory():
    """Describe the deep-learning models and local checkpoints."""
    rows = [
        {
            "model_key": "resnet50_full_finetune",
            "family": "ResNet50",
            "training_method": "full_finetune",
            "source_file": "models/resnet/train_resnet50.py",
            "checkpoint": "results/resnet50_full_finetune/best_model.pth",
            "legacy_checkpoint": "models/resnet/results/full_finetune/best_model.pth",
            "test_result_file": "results/resnet50_full_finetune/full_finetune_metrics.json",
            "target_layer": "model.layer4[-1]",
            "run_grad_cam": True,
            "note": "Best standard ResNet50 model.",
        },
        {
            "model_key": "resnet50_scratch",
            "family": "ResNet50",
            "training_method": "scratch",
            "source_file": "models/resnet/train_resnet50.py",
            "checkpoint": "results/resnet50_scratch/best_model.pth",
            "legacy_checkpoint": "models/resnet/results/scratch/best_model.pth",
            "test_result_file": "results/resnet50_scratch/scratch_metrics.json",
            "target_layer": "model.layer4[-1]",
            "run_grad_cam": True,
            "note": "Used for the scratch versus pretrained comparison.",
        },
        {
            "model_key": "resnet50_linear_probe",
            "family": "ResNet50",
            "training_method": "linear_probe",
            "source_file": "models/resnet/train_resnet50.py",
            "checkpoint": "results/resnet50_linear_probe/best_model.pth",
            "legacy_checkpoint": "models/resnet/results/linear_probe/best_model.pth",
            "test_result_file": "results/resnet50_linear_probe/linear_probe_metrics.json",
            "target_layer": "model.layer4[-1]",
            "run_grad_cam": False,
            "note": "Checkpoint is available, but the best ResNet and scratch comparison are used.",
        },
        {
            "model_key": "convnext_full_finetune",
            "family": "ConvNeXt-Tiny",
            "training_method": "full_finetune",
            "source_file": "models/convnext/convnext_classifier.py",
            "checkpoint": "results/convnext_full_finetune_seed42/best_checkpoint.pt",
            "legacy_checkpoint": "models/convnext/final_full_finetune_seed42/best_checkpoint.pt",
            "test_result_file": "results/convnext_full_finetune_seed42/test_metrics.json",
            "target_layer": "model.stages[-1].blocks[-1]",
            "run_grad_cam": True,
            "note": "Best ConvNeXt-Tiny model.",
        },
        {
            "model_key": "resnet50_fgvc_pmg_isqrtcov",
            "family": "Fine-grained ResNet50",
            "training_method": "PMG plus iSQRT-COV",
            "source_file": "models/resnet_fgvc/resnet_pmg_cov.py",
            "checkpoint": "results/resnet50_finetune_pmg_isqrtcov_from_pmg/best.pt",
            "legacy_checkpoint": "models/resnet_fgvc/resnet-pmg-isqrtcov/best.pt",
            "test_result_file": (
                "results/resnet50_finetune_pmg_isqrtcov_from_pmg/"
                "classification_report.txt"
            ),
            "target_layer": "model.backbone.encoder.stages[-1].layers[-1]",
            "run_grad_cam": True,
            "note": (
                "Uses the uploaded best.pt and the same combined logits as "
                "the repository test code."
            ),
        },
    ]

    for row in rows:
        candidates = [row["checkpoint"]]
        legacy_checkpoint = row.pop("legacy_checkpoint", None)
        if legacy_checkpoint:
            candidates.append(legacy_checkpoint)
        resolved_checkpoint = next(
            (candidate for candidate in candidates if (REPO_ROOT / candidate).is_file()),
            candidates[0],
        )
        row["checkpoint"] = resolved_checkpoint
        row["checkpoint_candidates"] = ";".join(candidates)
        checkpoint_path = REPO_ROOT / resolved_checkpoint
        result_path = REPO_ROOT / row["test_result_file"]
        row["checkpoint_exists"] = checkpoint_path.is_file()
        row["test_result_exists"] = result_path.is_file()
        if row["run_grad_cam"] and row["checkpoint_exists"]:
            row["inventory_status"] = "ready"
        elif not row["checkpoint_exists"]:
            row["inventory_status"] = "skipped_missing_checkpoint"
        else:
            row["inventory_status"] = "listed_not_selected"

    return pd.DataFrame(rows)


def save_model_inventory(results_dir=DEFAULT_RESULTS_DIR):
    """Save the model inventory CSV."""
    folders = make_output_folders(results_dir)
    table = get_model_inventory()
    output_path = folders["root"] / "model_inventory.csv"
    table.to_csv(output_path, index=False)
    return table


def load_test_records():
    """Read test.json."""
    with TEST_JSON.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("test.json must contain a list.")
    if len(records) != 5000:
        raise ValueError("Expected 5000 test records.")
    return records


def select_baseline_samples(results_dir=DEFAULT_RESULTS_DIR):
    """Select the original 15 shared test images using ConvNeXt predictions."""
    prediction_path = (
        REPO_ROOT
        / "results"
        / "convnext_full_finetune_seed42"
        / "test_predictions.csv"
    )
    if not prediction_path.is_file():
        raise FileNotFoundError(
            "The ConvNeXt prediction CSV is needed for sample selection."
        )

    predictions = pd.read_csv(prediction_path)
    predictions["correct"] = (
        predictions["true_label"] == predictions["predicted_label"]
    )

    correct_rows = predictions[predictions["correct"]].copy()
    correct_rows = correct_rows.sort_values("confidence", ascending=False).head(50)
    high_correct = correct_rows.sample(n=5, random_state=SEED)
    high_correct = high_correct.sort_values("confidence", ascending=False)
    high_correct["sample_group"] = "high_confidence_correct"

    incorrect_rows = predictions[~predictions["correct"]].copy()
    incorrect = incorrect_rows.sample(n=5, random_state=SEED)
    incorrect = incorrect.sort_values("confidence", ascending=False)
    incorrect["sample_group"] = "incorrect"

    used_paths = set(high_correct["image_path"]) | set(incorrect["image_path"])
    remaining = predictions[~predictions["image_path"].isin(used_paths)].copy()
    difficult = remaining.nsmallest(5, "confidence")
    difficult["sample_group"] = "low_confidence_difficult"

    selected = pd.concat([high_correct, incorrect, difficult], ignore_index=True)
    selected.insert(
        0,
        "sample_id",
        [f"sample_{number:02d}" for number in range(1, len(selected) + 1)],
    )
    selected = selected.rename(
        columns={
            "true_label": "anchor_true_label",
            "predicted_label": "anchor_predicted_label",
            "confidence": "anchor_confidence",
            "correct": "anchor_correct",
        }
    )
    selected = selected[
        [
            "sample_id",
            "sample_group",
            "image_path",
            "anchor_true_label",
            "anchor_predicted_label",
            "anchor_confidence",
            "anchor_correct",
        ]
    ]

    output_path = Path(results_dir) / "selected_grad_cam_samples.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)
    return selected


def _load_standard_prediction_table(model_name, prediction_path):
    """Load one repository prediction CSV into a shared column format."""
    prediction_path = Path(prediction_path)
    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"Prediction CSV for {model_name} was not found: {prediction_path}"
        )

    table = pd.read_csv(prediction_path)
    if model_name == "convnext":
        rename = {
            "image_path": "image_path",
            "true_label": "true_label",
            "predicted_label": "predicted_label",
            "confidence": "confidence",
        }
    else:
        rename = {
            "file_name": "image_path",
            "ground_truth": "true_label",
            "predicted_label": "predicted_label",
            "predicted_probability": "confidence",
        }

    missing = set(rename) - set(table.columns)
    if missing:
        raise ValueError(
            f"Prediction CSV for {model_name} is missing columns: {sorted(missing)}"
        )

    table = table[list(rename)].rename(columns=rename).copy()
    table["true_label"] = table["true_label"].astype(int)
    table["predicted_label"] = table["predicted_label"].astype(int)
    table["confidence"] = table["confidence"].astype(float)
    if len(table) != 5000 or table["image_path"].nunique() != 5000:
        raise ValueError(
            f"Expected 5,000 unique predictions for {model_name}, got {len(table)}."
        )

    return table.rename(
        columns={
            "predicted_label": f"{model_name}_predicted_label",
            "confidence": f"{model_name}_confidence",
        }
    )


def load_strong_model_predictions():
    """Merge ConvNeXt, FGVC, and DINOv2 predictions for sample selection."""
    merged = None
    for model_name, prediction_path in STRONG_MODEL_PREDICTION_FILES.items():
        table = _load_standard_prediction_table(model_name, prediction_path)
        if merged is None:
            merged = table
        else:
            merged = merged.merge(
                table,
                on=["image_path", "true_label"],
                how="inner",
                validate="one_to_one",
            )

    if merged is None or len(merged) != 5000:
        raise ValueError("The strong-model prediction tables did not align.")

    model_names = list(STRONG_MODEL_PREDICTION_FILES)
    correct_columns = []
    rank_columns = []
    for model_name in model_names:
        predicted_column = f"{model_name}_predicted_label"
        confidence_column = f"{model_name}_confidence"
        correct_column = f"{model_name}_correct"
        rank_column = f"{model_name}_confidence_rank"
        merged[correct_column] = (
            merged[predicted_column] == merged["true_label"]
        )
        merged[rank_column] = merged[confidence_column].rank(
            method="average",
            pct=True,
        )
        correct_columns.append(correct_column)
        rank_columns.append(rank_column)

    merged["strong_model_correct_count"] = merged[correct_columns].sum(axis=1)
    merged["consensus_confidence_rank"] = merged[rank_columns].mean(axis=1)

    majority_labels = []
    majority_votes = []
    majority_confidences = []
    for _, row in merged.iterrows():
        wrong_predictions = []
        confidence_by_prediction = {}
        for model_name in model_names:
            predicted_label = int(row[f"{model_name}_predicted_label"])
            if predicted_label == int(row["true_label"]):
                continue
            wrong_predictions.append(predicted_label)
            confidence_by_prediction.setdefault(predicted_label, []).append(
                float(row[f"{model_name}_confidence"])
            )

        if not wrong_predictions:
            majority_labels.append(-1)
            majority_votes.append(0)
            majority_confidences.append(0.0)
            continue

        counts = Counter(wrong_predictions)
        majority_label, vote_count = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
        majority_labels.append(int(majority_label))
        majority_votes.append(int(vote_count))
        majority_confidences.append(
            float(np.mean(confidence_by_prediction[majority_label]))
        )

    merged["majority_wrong_label"] = majority_labels
    merged["majority_wrong_votes"] = majority_votes
    merged["majority_wrong_confidence"] = majority_confidences
    return merged


def _take_unique_true_labels(table, count, excluded_paths=None):
    """Take rows in their current order while avoiding repeated true labels."""
    excluded_paths = set(excluded_paths or [])
    used_labels = set()
    chosen_indices = []
    for index, row in table.iterrows():
        image_path = row["image_path"]
        true_label = int(row["true_label"])
        if image_path in excluded_paths or true_label in used_labels:
            continue
        chosen_indices.append(index)
        excluded_paths.add(image_path)
        used_labels.add(true_label)
        if len(chosen_indices) == count:
            break
    if len(chosen_indices) != count:
        raise ValueError(
            f"Could only select {len(chosen_indices)} of {count} unique-label samples."
        )
    return table.loc[chosen_indices].copy()


def select_extended_samples(results_dir=EXTENDED_RESULTS_DIR, samples_per_group=10):
    """Select a diverse 30-image study from three strong-model predictions."""
    if samples_per_group < 2:
        raise ValueError("The extended study needs at least two samples per group.")

    predictions = load_strong_model_predictions()

    consensus_correct = predictions[
        predictions["strong_model_correct_count"] == 3
    ].sort_values(
        ["consensus_confidence_rank", "image_path"],
        ascending=[False, True],
    )
    consensus_correct = _take_unique_true_labels(
        consensus_correct,
        samples_per_group,
    )
    consensus_correct["sample_group"] = "strong_model_consensus_correct"
    consensus_correct["selection_detail"] = consensus_correct.apply(
        lambda row: (
            "all three strong models correct; mean confidence percentile="
            f"{row['consensus_confidence_rank']:.4f}"
        ),
        axis=1,
    )

    confusable = predictions[
        predictions["majority_wrong_votes"] >= 2
    ].copy()
    pair_counts = (
        confusable.groupby(["true_label", "majority_wrong_label"])
        .size()
        .rename("confusion_pair_images")
        .reset_index()
    )
    confusable = confusable.merge(
        pair_counts,
        on=["true_label", "majority_wrong_label"],
        how="left",
        validate="many_to_one",
    )
    confusable = confusable.sort_values(
        [
            "confusion_pair_images",
            "majority_wrong_votes",
            "majority_wrong_confidence",
            "image_path",
        ],
        ascending=[False, False, False, True],
    )

    # Use at most two images per directed pair so the ten examples cover five
    # or more recurring confusion relationships rather than one dominant pair.
    pair_usage = Counter()
    confusable_indices = []
    for index, row in confusable.iterrows():
        pair = (int(row["true_label"]), int(row["majority_wrong_label"]))
        if pair_usage[pair] >= 2:
            continue
        confusable_indices.append(index)
        pair_usage[pair] += 1
        if len(confusable_indices) == samples_per_group:
            break
    if len(confusable_indices) != samples_per_group:
        raise ValueError("Not enough recurring confusion-pair examples were found.")
    confusable = confusable.loc[confusable_indices].copy()
    confusable["sample_group"] = "confusable_error"
    confusable["selection_detail"] = confusable.apply(
        lambda row: (
            f"recurring pair {int(row['true_label']):03d}->"
            f"{int(row['majority_wrong_label']):03d}; "
            f"votes={int(row['majority_wrong_votes'])}/3; "
            f"pair images={int(row['confusion_pair_images'])}"
        ),
        axis=1,
    )

    used_paths = set(consensus_correct["image_path"]) | set(
        confusable["image_path"]
    )
    all_wrong = predictions[
        predictions["strong_model_correct_count"] == 0
    ].copy()
    all_wrong = all_wrong[~all_wrong["image_path"].isin(used_paths)]
    all_wrong = all_wrong.sort_values(
        ["consensus_confidence_rank", "image_path"],
        ascending=[True, True],
    )

    difficult_parts = []
    known_case = all_wrong[
        all_wrong["image_path"] == KNOWN_DATASET_QUALITY_CASE
    ].copy()
    if not known_case.empty:
        known_case = known_case.head(1)
        known_case["selection_detail"] = (
            "retained previously identified dataset-quality text image"
        )
        difficult_parts.append(known_case)
        used_paths.add(KNOWN_DATASET_QUALITY_CASE)

    remaining_count = samples_per_group - sum(len(part) for part in difficult_parts)
    remaining_difficult = _take_unique_true_labels(
        all_wrong,
        remaining_count,
        excluded_paths=used_paths,
    )
    remaining_difficult["selection_detail"] = remaining_difficult.apply(
        lambda row: (
            "all three strong models incorrect; mean confidence percentile="
            f"{row['consensus_confidence_rank']:.4f}"
        ),
        axis=1,
    )
    difficult_parts.append(remaining_difficult)
    difficult = pd.concat(difficult_parts, ignore_index=True)
    difficult["sample_group"] = "consensus_difficult"

    selected = pd.concat(
        [consensus_correct, confusable, difficult],
        ignore_index=True,
    )
    selected.insert(
        0,
        "sample_id",
        [f"sample_{number:02d}" for number in range(1, len(selected) + 1)],
    )
    selected["selection_profile"] = f"extended_{len(selected)}"
    selected["anchor_true_label"] = selected["true_label"].astype(int)
    selected["anchor_predicted_label"] = selected[
        "convnext_predicted_label"
    ].astype(int)
    selected["anchor_confidence"] = selected["convnext_confidence"].astype(float)
    selected["anchor_correct"] = (
        selected["anchor_true_label"] == selected["anchor_predicted_label"]
    )

    leading_columns = [
        "sample_id",
        "sample_group",
        "selection_profile",
        "selection_detail",
        "image_path",
        "anchor_true_label",
        "anchor_predicted_label",
        "anchor_confidence",
        "anchor_correct",
        "strong_model_correct_count",
        "consensus_confidence_rank",
        "majority_wrong_label",
        "majority_wrong_votes",
    ]
    selected = selected[leading_columns]

    output_path = Path(results_dir) / "selected_grad_cam_samples.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)
    return selected


def select_shared_samples(
    results_dir=DEFAULT_RESULTS_DIR,
    profile="baseline",
    samples_per_group=None,
):
    """Select shared images for either the baseline or extended study."""
    if profile == "baseline":
        return select_baseline_samples(results_dir)
    if profile == "extended":
        group_size = 10 if samples_per_group is None else int(samples_per_group)
        return select_extended_samples(results_dir, group_size)
    raise ValueError("profile must be 'baseline' or 'extended'.")


def make_resnet_transform():
    """Make the same test transform used by train_resnet50.py."""
    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return image_transform


def load_resnet(checkpoint_path, device):
    """Build the repository ResNet50 and load a state dictionary."""
    from models.resnet.train_resnet50 import make_model

    # "scratch" avoids downloading ImageNet weights. The architecture is the same.
    model = make_model("scratch", 500)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)

    loaded = {
        "model": model,
        "transform": make_resnet_transform(),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "target_layer": model.layer4[-1],
        "target_layer_name": "model.layer4[-1]",
    }
    return loaded


def load_convnext(checkpoint_path, device):
    """Build the repository ConvNeXt model and load its checkpoint."""
    import timm
    from models.convnext.convnext_classifier import create_convnext_classifier

    model, data_config = create_convnext_classifier(pretrained=False)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "model_state" not in checkpoint:
        raise ValueError("ConvNeXt checkpoint does not contain model_state.")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    model.to(device)

    image_transform = timm.data.create_transform(
        **data_config,
        is_training=False,
    )
    loaded = {
        "model": model,
        "transform": image_transform,
        "mean": list(data_config["mean"]),
        "std": list(data_config["std"]),
        "target_layer": model.stages[-1].blocks[-1],
        "target_layer_name": "model.stages[-1].blocks[-1]",
    }
    return loaded


class FGVCOutputWrapper(torch.nn.Module):
    """Return the same combined logits used by the FGVC test code."""

    def __init__(self, model, cov_logit_weight):
        super().__init__()
        self.model = model
        self.cov_logit_weight = cov_logit_weight

    def forward(self, image_tensor):
        outputs = self.model(image_tensor)
        combined_logits = (
            outputs[0]
            + outputs[1]
            + outputs[2]
            + outputs[3]
            + self.cov_logit_weight * outputs[4]
        )
        return combined_logits


def load_fgvc(checkpoint_path, device):
    """Build and load the repository PMG + iSQRT-COV model."""
    from models.resnet_fgvc.resnet_pmg_cov import (
        PMGISqrtCovClassifier,
        make_pmg_transforms,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    arguments = checkpoint["args"]

    base_model = PMGISqrtCovClassifier(
        model_name=arguments["model_name"],
        num_classes=arguments["num_classes"],
        feature_size=arguments["feature_size"],
        cov_dim=arguments["cov_dim"],
        sqrt_iters=arguments["sqrt_iters"],
        cov_eps=arguments["cov_eps"],
        cov_dropout=arguments["cov_dropout"],
        cov_initial_scale=arguments["cov_initial_scale"],
    )
    base_model.load_state_dict(checkpoint["model"], strict=True)
    base_model.eval()

    model = FGVCOutputWrapper(
        base_model,
        arguments["hybrid_cov_logit_weight"],
    )
    model.eval()
    model.to(device)

    try:
        _, image_transform = make_pmg_transforms(
            arguments["model_name"],
            arguments["image_size"],
            arguments["resize_size"],
        )
    except OSError:
        # The repository processor configuration uses the standard ImageNet
        # statistics below. Newer transformers releases may still attempt a
        # network HEAD request despite the cached preprocessor_config.json, so
        # keep an explicit offline equivalent of make_pmg_transforms here.
        image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (arguments["resize_size"], arguments["resize_size"])
                ),
                transforms.CenterCrop(arguments["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    normalise = image_transform.transforms[-1]

    loaded = {
        "model": model,
        "transform": image_transform,
        "mean": list(normalise.mean),
        "std": list(normalise.std),
        "target_layer": model.model.backbone.encoder.stages[-1].layers[-1],
        "target_layer_name": (
            "model.model.backbone.encoder.stages[-1].layers[-1]"
        ),
    }
    return loaded


def load_model(model_key, checkpoint_path, device):
    """Load one supported model."""
    if model_key == "resnet50_fgvc_pmg_isqrtcov":
        return load_fgvc(checkpoint_path, device)
    if model_key.startswith("resnet50_"):
        return load_resnet(checkpoint_path, device)
    if model_key == "convnext_full_finetune":
        return load_convnext(checkpoint_path, device)
    raise ValueError(f"Grad-CAM loader is not available for {model_key}.")


def tensor_to_rgb(input_tensor, mean, std):
    """Undo normalization so the displayed image matches model input."""
    image = input_tensor[0].detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    mean_array = np.array(mean, dtype=np.float32)
    std_array = np.array(std, dtype=np.float32)
    image = image * std_array + mean_array
    image = np.clip(image, 0.0, 1.0)
    return image.astype(np.float32)


def prepare_image(image_path, loaded_model, device):
    """Read one image and apply the model's test transform."""
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        input_tensor = loaded_model["transform"](rgb_image)
    input_tensor = input_tensor.unsqueeze(0).to(device)
    display_image = tensor_to_rgb(
        input_tensor,
        loaded_model["mean"],
        loaded_model["std"],
    )
    return input_tensor, display_image


def predict_one(model, input_tensor):
    """Return the predicted label, confidence, and logits."""
    with torch.no_grad():
        logits = model(input_tensor)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("The model output is not a tensor.")
        if logits.ndim != 2 or logits.shape[1] != 500:
            raise ValueError(
                f"Expected logits with shape [batch, 500], got {tuple(logits.shape)}."
            )
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_label = probabilities.max(dim=1)
    return (
        int(predicted_label.item()),
        float(confidence.item()),
        logits.detach().cpu(),
    )


def make_grad_cam(cam_object, input_tensor, target_label, display_image):
    """Generate one Grad-CAM heatmap and overlay."""
    targets = [ClassifierOutputTarget(int(target_label))]
    grayscale_cam = cam_object(
        input_tensor=input_tensor,
        targets=targets,
        aug_smooth=False,
        eigen_smooth=False,
    )
    heatmap = grayscale_cam[0]
    heatmap = np.asarray(heatmap, dtype=np.float32)

    if not np.isfinite(heatmap).all():
        raise ValueError("The Grad-CAM heatmap contains invalid values.")
    if float(heatmap.max()) <= 0.0:
        raise ValueError("The Grad-CAM heatmap is empty.")

    overlay = show_cam_on_image(
        display_image,
        heatmap,
        use_rgb=True,
    )
    color_heatmap = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET,
    )
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
    return heatmap, color_heatmap, overlay


def top_fraction_mask(heatmap, fraction=0.20):
    """Return a Boolean mask containing the largest CAM values by rank."""
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
        raise ValueError("A finite two-dimensional heatmap is required.")
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("fraction must be between zero and one.")

    flat = heatmap.reshape(-1)
    selected_count = max(1, int(np.ceil(float(fraction) * flat.size)))
    selected_indices = np.argpartition(flat, -selected_count)[-selected_count:]
    mask = np.zeros(flat.size, dtype=bool)
    mask[selected_indices] = True
    return mask.reshape(heatmap.shape)


def cam_top_fraction_iou(first_heatmap, second_heatmap, fraction=0.20):
    """Compute IoU between the most activated regions of two CAMs."""
    first = np.asarray(first_heatmap, dtype=np.float32)
    second = np.asarray(second_heatmap, dtype=np.float32)
    if first.shape != second.shape:
        second = cv2.resize(
            second,
            (first.shape[1], first.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    first_mask = top_fraction_mask(first, fraction)
    second_mask = top_fraction_mask(second, fraction)
    union = np.logical_or(first_mask, second_mask).sum()
    if union == 0:
        return float("nan")
    intersection = np.logical_and(first_mask, second_mask).sum()
    return float(intersection / union)


def target_probability(model, input_tensor, target_label):
    """Return the softmax probability assigned to one target class."""
    with torch.no_grad():
        logits = model(input_tensor)
        if logits.ndim != 2 or logits.shape[1] != 500:
            raise ValueError(
                f"Expected logits with shape [batch, 500], got {tuple(logits.shape)}."
            )
        probability = torch.softmax(logits, dim=1)[0, int(target_label)]
    return float(probability.item())


def compute_deletion_metrics(
    model,
    input_tensor,
    target_label,
    heatmap,
    fraction=0.20,
    random_key="",
):
    """Compare target-probability drops after salient and control deletion.

    Deleted pixels are replaced by a local 21-by-21 mean-blurred baseline. This
    avoids introducing black or mean-colour impulses. The control is a random
    spatial shift of the CAM mask, preserving its exact area and geometry while
    changing its image location.
    """
    height, width = input_tensor.shape[-2:]
    resized_heatmap = np.asarray(heatmap, dtype=np.float32)
    if resized_heatmap.shape != (height, width):
        resized_heatmap = cv2.resize(
            resized_heatmap,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    salient_mask = top_fraction_mask(resized_heatmap, fraction)
    seed = (SEED + zlib.crc32(str(random_key).encode("utf-8"))) % (2**32)
    generator = np.random.default_rng(seed)
    control_mask = None
    best_overlap = float("inf")
    for _ in range(100):
        vertical_shift = int(generator.integers(0, height))
        horizontal_shift = int(generator.integers(0, width))
        if vertical_shift == 0 and horizontal_shift == 0:
            continue
        candidate = np.roll(
            salient_mask,
            shift=(vertical_shift, horizontal_shift),
            axis=(0, 1),
        )
        intersection = np.logical_and(salient_mask, candidate).sum()
        union = np.logical_or(salient_mask, candidate).sum()
        overlap = float(intersection / union) if union else 1.0
        if overlap < best_overlap:
            best_overlap = overlap
            control_mask = candidate
        if overlap <= 0.05:
            break
    if control_mask is None:
        raise ValueError("Could not construct a shifted CAM control mask.")

    salient_keep = torch.from_numpy(~salient_mask).to(
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )[None, None, :, :]
    control_keep = torch.from_numpy(~control_mask).to(
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )[None, None, :, :]
    blur_radius = 10
    padded = functional.pad(
        input_tensor,
        (blur_radius, blur_radius, blur_radius, blur_radius),
        mode="reflect",
    )
    blurred_baseline = functional.avg_pool2d(
        padded,
        kernel_size=2 * blur_radius + 1,
        stride=1,
    )
    salient_deleted = (
        input_tensor * salient_keep
        + blurred_baseline * (1.0 - salient_keep)
    )
    control_deleted = (
        input_tensor * control_keep
        + blurred_baseline * (1.0 - control_keep)
    )

    original_probability = target_probability(model, input_tensor, target_label)
    salient_probability = target_probability(
        model,
        salient_deleted,
        target_label,
    )
    control_probability = target_probability(
        model,
        control_deleted,
        target_label,
    )
    denominator = max(original_probability, 1e-12)
    salient_drop = original_probability - salient_probability
    control_drop = original_probability - control_probability
    return {
        "deletion_fraction": float(fraction),
        "deletion_baseline": "local_mean_blur_21_shifted_cam_control",
        "original_target_probability": original_probability,
        "salient_deleted_probability": salient_probability,
        "shifted_control_deleted_probability": control_probability,
        "salient_probability_drop": salient_drop,
        "shifted_control_probability_drop": control_drop,
        "salient_relative_drop": salient_drop / denominator,
        "shifted_control_relative_drop": control_drop / denominator,
    }


def save_png_figure(figure, png_path):
    """Save one Matplotlib figure as PNG."""
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_three_panel_figure(
    display_image,
    color_heatmap,
    overlay,
    model_name,
    true_label,
    predicted_label,
    target_label,
    target_type,
    sample_id,
    png_path,
):
    """Save Original | Heatmap | Overlay."""
    is_correct = int(true_label) == int(predicted_label)
    correctness = "correct" if is_correct else "incorrect"

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(display_image)
    axes[0].set_title("Original")
    axes[1].imshow(color_heatmap)
    axes[1].set_title("Heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")

    for axis in axes:
        axis.axis("off")

    title = (
        f"{model_name} | {sample_id}\n"
        f"true={int(true_label):03d}, predicted={int(predicted_label):03d}, "
        f"target={int(target_label):03d} ({target_type}), {correctness}"
    )
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    save_png_figure(figure, png_path)


def save_overlay_comparison(items, title, png_path):
    """Save a simple row of labelled overlay images."""
    if len(items) == 0:
        return

    figure, axes = plt.subplots(1, len(items), figsize=(4 * len(items), 4))
    if len(items) == 1:
        axes = [axes]

    for axis, item in zip(axes, items):
        axis.imshow(item["image"])
        axis.set_title(item["title"], fontsize=9)
        axis.axis("off")

    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    save_png_figure(figure, png_path)


def load_original_image(image_path):
    """Read an RGB image for a comparison panel."""
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"))
