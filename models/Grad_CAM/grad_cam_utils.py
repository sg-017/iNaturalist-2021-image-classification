"""Small helper functions for the Grad-CAM notebook and script.

The functions in this file are intentionally written in a simple style.
They use the model-building code that is already in this repository.
"""

import json
import re
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms


GRAD_CAM_DIR = Path(__file__).resolve().parent
REPO_ROOT = GRAD_CAM_DIR.parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "grad_cam"
DATA_DIR = REPO_ROOT / "dataset" / "processed_dataset"
TEST_JSON = DATA_DIR / "test.json"
SEED = 42


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
            "checkpoint": "models/resnet/results/full_finetune/best_model.pth",
            "test_result_file": "models/resnet/full_finetune_metrics.json",
            "target_layer": "model.layer4[-1]",
            "run_grad_cam": True,
            "note": "Best standard ResNet50 model.",
        },
        {
            "model_key": "resnet50_scratch",
            "family": "ResNet50",
            "training_method": "scratch",
            "source_file": "models/resnet/train_resnet50.py",
            "checkpoint": "models/resnet/results/scratch/best_model.pth",
            "test_result_file": "models/resnet/scratch_metrics.json",
            "target_layer": "model.layer4[-1]",
            "run_grad_cam": True,
            "note": "Used for the scratch versus pretrained comparison.",
        },
        {
            "model_key": "resnet50_linear_probe",
            "family": "ResNet50",
            "training_method": "linear_probe",
            "source_file": "models/resnet/train_resnet50.py",
            "checkpoint": "models/resnet/results/linear_probe/best_model.pth",
            "test_result_file": "models/resnet/linear_probe_metrics.json",
            "target_layer": "model.layer4[-1]",
            "run_grad_cam": False,
            "note": "Checkpoint is available, but the best ResNet and scratch comparison are used.",
        },
        {
            "model_key": "convnext_full_finetune",
            "family": "ConvNeXt-Tiny",
            "training_method": "full_finetune",
            "source_file": "models/convnext/convnext_classifier.py",
            "checkpoint": "models/convnext/final_full_finetune_seed42/best_checkpoint.pt",
            "test_result_file": "models/convnext/final_full_finetune_seed42/test_metrics.json",
            "target_layer": "model.stages[-1].blocks[-1]",
            "run_grad_cam": True,
            "note": "Best ConvNeXt-Tiny model.",
        },
        {
            "model_key": "resnet50_fgvc_pmg_isqrtcov",
            "family": "Fine-grained ResNet50",
            "training_method": "PMG plus iSQRT-COV",
            "source_file": "models/resnet_fgvc/resnet_pmg_cov.py",
            "checkpoint": "models/resnet_fgvc/resnet-pmg-isqrtcov/best.pt",
            "test_result_file": (
                "models/resnet_fgvc/outputs/"
                "resnet50_finetune_pmg_isqrtcov_from_pmg/classification_report.txt"
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
        checkpoint_path = REPO_ROOT / row["checkpoint"]
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


def select_shared_samples(results_dir=DEFAULT_RESULTS_DIR):
    """Select 15 shared test images using ConvNeXt predictions."""
    prediction_path = (
        REPO_ROOT
        / "models"
        / "convnext"
        / "final_full_finetune_seed42"
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

    _, image_transform = make_pmg_transforms(
        arguments["model_name"],
        arguments["image_size"],
        arguments["resize_size"],
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
