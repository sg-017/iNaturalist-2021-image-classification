"""Run the repository Grad-CAM workflow.

Examples from the repository root:

    python explainability/run_grad_cam.py --mode smoke --device auto
    python explainability/run_grad_cam.py --mode full --device auto
    python explainability/run_grad_cam.py --mode full --profile extended --device auto
"""

import argparse
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam import GradCAM

try:
    from . import grad_cam_utils as _grad_cam_utils
except ImportError:  # Support direct execution from the repository root.
    import grad_cam_utils as _grad_cam_utils


DATA_DIR = _grad_cam_utils.DATA_DIR
DEFAULT_RESULTS_DIR = _grad_cam_utils.DEFAULT_RESULTS_DIR
EXTENDED_RESULTS_DIR = _grad_cam_utils.EXTENDED_RESULTS_DIR
REPO_ROOT = _grad_cam_utils.REPO_ROOT
cam_top_fraction_iou = _grad_cam_utils.cam_top_fraction_iou
choose_device = _grad_cam_utils.choose_device
compute_deletion_metrics = _grad_cam_utils.compute_deletion_metrics
load_original_image = _grad_cam_utils.load_original_image
load_model = _grad_cam_utils.load_model
make_grad_cam = _grad_cam_utils.make_grad_cam
make_output_folders = _grad_cam_utils.make_output_folders
prepare_image = _grad_cam_utils.prepare_image
predict_one = _grad_cam_utils.predict_one
relative_path = _grad_cam_utils.relative_path
safe_name = _grad_cam_utils.safe_name
save_model_inventory = _grad_cam_utils.save_model_inventory
save_overlay_comparison = _grad_cam_utils.save_overlay_comparison
save_three_panel_figure = _grad_cam_utils.save_three_panel_figure
select_shared_samples = _grad_cam_utils.select_shared_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Simple Grad-CAM workflow")
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="full",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda", "auto"],
        default="cpu",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--profile",
        choices=["baseline", "extended"],
        default="baseline",
        help="Use the original 15 samples or the stratified extended study.",
    )
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=10,
        help="Group size for the extended profile; the default gives 30 images.",
    )
    parser.add_argument(
        "--deletion-fraction",
        type=float,
        default=0.20,
        help="Fraction of image pixels removed for the CAM deletion test.",
    )
    parser.add_argument(
        "--skip-deletion",
        action="store_true",
        help="Skip the salient-versus-shifted-control deletion metric.",
    )
    parser.add_argument(
        "--smoke-model",
        default="resnet50_full_finetune",
        help="Model key used for the one-image smoke test.",
    )
    return parser.parse_args()


def model_rows_to_run(inventory, mode, smoke_model):
    """Choose the models for a smoke or full run."""
    if mode == "smoke":
        names = [smoke_model]
    else:
        names = [
            "resnet50_full_finetune",
            "convnext_full_finetune",
            "resnet50_scratch",
            "resnet50_fgvc_pmg_isqrtcov",
        ]
    return inventory[inventory["model_key"].isin(names)].copy()


def save_status_table(status_rows, results_dir):
    """Write the model run status CSV."""
    path = Path(results_dir) / "grad_cam_run_status.csv"
    pd.DataFrame(status_rows).to_csv(path, index=False)


def save_result_table(result_rows, results_dir):
    """Write the Grad-CAM result CSV."""
    columns = [
        "model_key",
        "sample_id",
        "sample_group",
        "image_path",
        "true_label",
        "predicted_label",
        "confidence",
        "correct",
        "target_type",
        "target_label",
        "target_layer",
        "cam_min",
        "cam_max",
        "cam_mean",
        "deletion_fraction",
        "deletion_baseline",
        "original_target_probability",
        "salient_deleted_probability",
        "shifted_control_deleted_probability",
        "salient_probability_drop",
        "shifted_control_probability_drop",
        "salient_relative_drop",
        "shifted_control_relative_drop",
        "figure_png",
        "status",
        "error",
    ]
    path = Path(results_dir) / "grad_cam_results.csv"
    pd.DataFrame(result_rows, columns=columns).to_csv(path, index=False)


MODEL_ORDER = [
    "resnet50_full_finetune",
    "convnext_full_finetune",
    "resnet50_scratch",
    "resnet50_fgvc_pmg_isqrtcov",
]

MODEL_DISPLAY_NAMES = {
    "resnet50_full_finetune": "ResNet50 full",
    "convnext_full_finetune": "ConvNeXt full",
    "resnet50_scratch": "ResNet50 scratch",
    "resnet50_fgvc_pmg_isqrtcov": "PMG + iSQRT-COV",
}


def create_pair_metrics(selected, visual_cache, results_dir, fraction):
    """Measure overlap between predicted- and true-class CAMs on errors."""
    rows = []
    for model_key in MODEL_ORDER:
        for _, sample in selected.iterrows():
            predicted = visual_cache.get(
                (model_key, sample["sample_id"], "predicted")
            )
            true_target = visual_cache.get(
                (model_key, sample["sample_id"], "true")
            )
            if predicted is None or true_target is None:
                continue
            rows.append(
                {
                    "model_key": model_key,
                    "sample_id": sample["sample_id"],
                    "sample_group": sample["sample_group"],
                    "image_path": sample["image_path"],
                    "true_label": int(predicted["true_label"]),
                    "predicted_label": int(predicted["predicted_label"]),
                    "top_fraction": float(fraction),
                    "predicted_true_cam_iou": cam_top_fraction_iou(
                        predicted["heatmap"],
                        true_target["heatmap"],
                        fraction=fraction,
                    ),
                }
            )

    columns = [
        "model_key",
        "sample_id",
        "sample_group",
        "image_path",
        "true_label",
        "predicted_label",
        "top_fraction",
        "predicted_true_cam_iou",
    ]
    table = pd.DataFrame(rows, columns=columns)
    table.to_csv(Path(results_dir) / "grad_cam_pair_metrics.csv", index=False)
    return table


def create_quantitative_summaries(result_rows, pair_metrics, folders):
    """Save per-model/group summaries and a compact report plot."""
    results = pd.DataFrame(result_rows)
    valid = results[results["status"] == "completed"].copy()
    predicted = valid[valid["target_type"] == "predicted"].copy()
    numeric_columns = [
        "salient_relative_drop",
        "shifted_control_relative_drop",
    ]
    for column in numeric_columns:
        predicted[column] = pd.to_numeric(predicted[column], errors="coerce")
    predicted["salient_minus_control_relative_drop"] = (
        predicted["salient_relative_drop"]
        - predicted["shifted_control_relative_drop"]
    )

    model_rows = []
    for model_key in MODEL_ORDER:
        model_all = results[results["model_key"] == model_key]
        model_predicted = predicted[predicted["model_key"] == model_key]
        model_pairs = pair_metrics[pair_metrics["model_key"] == model_key]
        requested = len(model_all)
        completed = int((model_all["status"] == "completed").sum())
        model_rows.append(
            {
                "model_key": model_key,
                "requested_cams": requested,
                "valid_cams": completed,
                "valid_cam_rate": completed / requested if requested else np.nan,
                "valid_predicted_cams": len(model_predicted),
                "median_salient_relative_drop": model_predicted[
                    "salient_relative_drop"
                ].median(),
                "median_shifted_control_relative_drop": model_predicted[
                    "shifted_control_relative_drop"
                ].median(),
                "median_salient_minus_control_drop": model_predicted[
                    "salient_minus_control_relative_drop"
                ].median(),
                "valid_predicted_true_pairs": len(model_pairs),
                "median_predicted_true_cam_iou": model_pairs[
                    "predicted_true_cam_iou"
                ].median(),
            }
        )

    model_summary = pd.DataFrame(model_rows)
    model_summary.to_csv(
        folders["root"] / "grad_cam_model_summary.csv",
        index=False,
    )

    group_rows = []
    for (model_key, sample_group), group_all in results.groupby(
        ["model_key", "sample_group"],
        sort=False,
    ):
        group_predicted = predicted[
            (predicted["model_key"] == model_key)
            & (predicted["sample_group"] == sample_group)
        ]
        group_pairs = pair_metrics[
            (pair_metrics["model_key"] == model_key)
            & (pair_metrics["sample_group"] == sample_group)
        ]
        requested = len(group_all)
        completed = int((group_all["status"] == "completed").sum())
        group_rows.append(
            {
                "model_key": model_key,
                "sample_group": sample_group,
                "requested_cams": requested,
                "valid_cams": completed,
                "valid_cam_rate": completed / requested if requested else np.nan,
                "median_salient_relative_drop": group_predicted[
                    "salient_relative_drop"
                ].median(),
                "median_shifted_control_relative_drop": group_predicted[
                    "shifted_control_relative_drop"
                ].median(),
                "valid_predicted_true_pairs": len(group_pairs),
                "median_predicted_true_cam_iou": group_pairs[
                    "predicted_true_cam_iou"
                ].median(),
            }
        )
    pd.DataFrame(group_rows).to_csv(
        folders["root"] / "grad_cam_group_summary.csv",
        index=False,
    )

    labels = [MODEL_DISPLAY_NAMES[key] for key in MODEL_ORDER]
    x_positions = np.arange(len(MODEL_ORDER))
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    axes[0].bar(
        x_positions,
        100.0 * model_summary["valid_cam_rate"],
        color="#4C78A8",
    )
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Valid CAMs (%)")
    axes[0].set_title("CAM generation success")

    bar_width = 0.36
    axes[1].bar(
        x_positions - bar_width / 2,
        100.0 * model_summary["median_salient_relative_drop"],
        width=bar_width,
        label="CAM top 20%",
        color="#E45756",
    )
    axes[1].bar(
        x_positions + bar_width / 2,
        100.0 * model_summary["median_shifted_control_relative_drop"],
        width=bar_width,
        label="Shifted CAM mask",
        color="#72B7B2",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_ylabel("Median relative target-probability drop (%)")
    axes[1].set_title("Deletion faithfulness test")
    axes[1].legend(fontsize=8)

    axes[2].bar(
        x_positions,
        model_summary["median_predicted_true_cam_iou"],
        color="#F2CF5B",
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Median top-20% IoU")
    axes[2].set_title("Predicted vs. true CAM overlap")

    for axis in axes:
        axis.set_xticks(x_positions)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        folders["comparisons"] / "grad_cam_quantitative_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)
    return model_summary


def find_cache_item(visual_cache, model_key, sample_id, target_label=None):
    """Find a cached visual for a model and sample."""
    for key, value in visual_cache.items():
        key_model, key_sample, _ = key
        if key_model != model_key or key_sample != sample_id:
            continue
        if target_label is None or int(value["target_label"]) == int(target_label):
            return value
    return None


def create_cross_model_figures(selected, visual_cache, folders):
    """Create the report comparison figures."""
    # One comparison for the first sample in each selection group.
    for group_name in selected["sample_group"].unique():
        group_rows = selected[selected["sample_group"] == group_name]
        if len(group_rows) == 0:
            continue
        sample = group_rows.iloc[0]
        original_image = load_original_image(DATA_DIR / sample["image_path"])
        items = [
            {
                "image": original_image,
                "title": (
                    "Original image\n"
                    f"true={int(sample['anchor_true_label']):03d}"
                ),
            }
        ]
        for model_key in MODEL_ORDER:
            visual = find_cache_item(
                visual_cache,
                model_key,
                sample["sample_id"],
            )
            if visual is None:
                continue
            correctness = (
                "correct"
                if visual["predicted_label"] == visual["true_label"]
                else "incorrect"
            )
            title = (
                f"{model_key}\n"
                f"true={visual['true_label']:03d}, "
                f"pred={visual['predicted_label']:03d}\n"
                f"target={visual['target_label']:03d}, {correctness}"
            )
            items.append({"image": visual["overlay"], "title": title})

        if len(items) > 1:
            base_name = f"cross_model_{safe_name(group_name)}"
            save_overlay_comparison(
                items,
                f"Cross-model comparison: {group_name}",
                folders["comparisons"] / f"{base_name}.png",
            )

    # Correct versus incorrect using ConvNeXt.
    correct_group = (
        "strong_model_consensus_correct"
        if "strong_model_consensus_correct" in set(selected["sample_group"])
        else "high_confidence_correct"
    )
    incorrect_group = (
        "confusable_error"
        if "confusable_error" in set(selected["sample_group"])
        else "incorrect"
    )
    correct_rows = selected[selected["sample_group"] == correct_group]
    incorrect_rows = selected[selected["sample_group"] == incorrect_group]
    if len(correct_rows) > 0 and len(incorrect_rows) > 0:
        correct_id = correct_rows.iloc[0]["sample_id"]
        incorrect_id = incorrect_rows.iloc[0]["sample_id"]
        correct_visual = find_cache_item(
            visual_cache,
            "convnext_full_finetune",
            correct_id,
        )
        incorrect_visual = find_cache_item(
            visual_cache,
            "convnext_full_finetune",
            incorrect_id,
        )
        items = []
        if correct_visual is not None:
            correct_sample = correct_rows.iloc[0]
            items.append(
                {
                    "image": load_original_image(
                        DATA_DIR / correct_sample["image_path"]
                    ),
                    "title": (
                        "Original image\n"
                        f"{correct_id}\n"
                        f"true={correct_visual['true_label']:03d}"
                    ),
                }
            )
            items.append(
                {
                    "image": correct_visual["overlay"],
                    "title": (
                        "convnext_full_finetune\n"
                        f"{correct_id}: correct\n"
                        f"true={correct_visual['true_label']:03d}, "
                        f"pred={correct_visual['predicted_label']:03d}\n"
                        f"target={correct_visual['target_label']:03d}"
                    ),
                }
            )
        if incorrect_visual is not None:
            incorrect_sample = incorrect_rows.iloc[0]
            items.append(
                {
                    "image": load_original_image(
                        DATA_DIR / incorrect_sample["image_path"]
                    ),
                    "title": (
                        "Original image\n"
                        f"{incorrect_id}\n"
                        f"true={incorrect_visual['true_label']:03d}"
                    ),
                }
            )
            items.append(
                {
                    "image": incorrect_visual["overlay"],
                    "title": (
                        "convnext_full_finetune\n"
                        f"{incorrect_id}: incorrect\n"
                        f"true={incorrect_visual['true_label']:03d}, "
                        f"pred={incorrect_visual['predicted_label']:03d}\n"
                        f"target={incorrect_visual['target_label']:03d}"
                    ),
                }
            )
        if len(items) == 4:
            save_overlay_comparison(
                items,
                "ConvNeXt: correct versus incorrect",
                folders["comparisons"] / "correct_vs_incorrect.png",
            )

    # Predicted-class versus true-class for a ConvNeXt error.
    for _, sample in incorrect_rows.iterrows():
        predicted_visual = visual_cache.get(
            (
                "convnext_full_finetune",
                sample["sample_id"],
                "predicted",
            )
        )
        true_visual = visual_cache.get(
            (
                "convnext_full_finetune",
                sample["sample_id"],
                "true",
            )
        )
        if predicted_visual is None or true_visual is None:
            continue
        items = [
            {
                "image": load_original_image(
                    DATA_DIR / sample["image_path"]
                ),
                "title": (
                    "Original image\n"
                    f"{sample['sample_id']}\n"
                    f"true={predicted_visual['true_label']:03d}"
                ),
            },
            {
                "image": predicted_visual["overlay"],
                "title": (
                    "convnext_full_finetune: incorrect\n"
                    f"true={predicted_visual['true_label']:03d}, "
                    f"pred={predicted_visual['predicted_label']:03d}\n"
                    f"predicted target={predicted_visual['target_label']:03d}"
                ),
            },
            {
                "image": true_visual["overlay"],
                "title": (
                    "convnext_full_finetune: incorrect\n"
                    f"true={true_visual['true_label']:03d}, "
                    f"pred={true_visual['predicted_label']:03d}\n"
                    f"true target={true_visual['target_label']:03d}"
                ),
            },
        ]
        save_overlay_comparison(
            items,
            (
                "ConvNeXt predicted-class versus true-class Grad-CAM\n"
                f"{sample['sample_id']}"
            ),
            folders["comparisons"] / "predicted_vs_true.png",
        )
        break

    # Scratch versus pretrained ResNet with the same true-class target.
    for _, sample in selected.iterrows():
        true_label = int(sample["anchor_true_label"])
        full_visual = find_cache_item(
            visual_cache,
            "resnet50_full_finetune",
            sample["sample_id"],
            true_label,
        )
        scratch_visual = find_cache_item(
            visual_cache,
            "resnet50_scratch",
            sample["sample_id"],
            true_label,
        )
        if full_visual is None or scratch_visual is None:
            continue
        full_correctness = (
            "correct"
            if full_visual["predicted_label"] == full_visual["true_label"]
            else "incorrect"
        )
        scratch_correctness = (
            "correct"
            if scratch_visual["predicted_label"] == scratch_visual["true_label"]
            else "incorrect"
        )
        items = [
            {
                "image": load_original_image(
                    DATA_DIR / sample["image_path"]
                ),
                "title": (
                    "Original image\n"
                    f"{sample['sample_id']}\n"
                    f"true={true_label:03d}"
                ),
            },
            {
                "image": full_visual["overlay"],
                "title": (
                    f"resnet50_full_finetune: {full_correctness}\n"
                    f"true={full_visual['true_label']:03d}, "
                    f"pred={full_visual['predicted_label']:03d}\n"
                    f"target={true_label:03d}"
                ),
            },
            {
                "image": scratch_visual["overlay"],
                "title": (
                    f"resnet50_scratch: {scratch_correctness}\n"
                    f"true={scratch_visual['true_label']:03d}, "
                    f"pred={scratch_visual['predicted_label']:03d}\n"
                    f"target={true_label:03d}"
                ),
            },
        ]
        save_overlay_comparison(
            items,
            "ResNet scratch versus pretrained full fine-tuning",
            folders["comparisons"] / "resnet_scratch_vs_pretrained.png",
        )
        break


def run_one_model(
    model_row,
    selected,
    device,
    folders,
    result_rows,
    visual_cache,
    deletion_fraction,
    skip_deletion,
):
    """Load and run Grad-CAM for one model."""
    model_key = model_row["model_key"]
    checkpoint_path = REPO_ROOT / model_row["checkpoint"]
    loaded = load_model(model_key, checkpoint_path, device)
    model = loaded["model"]
    completed_cams = 0
    failed_cams = 0

    target_layers = [loaded["target_layer"]]
    with GradCAM(model=model, target_layers=target_layers) as cam_object:
        for _, sample in selected.iterrows():
            image_path = DATA_DIR / sample["image_path"]
            input_tensor, display_image = prepare_image(
                image_path,
                loaded,
                device,
            )
            true_label = int(sample["anchor_true_label"])
            predicted_label, confidence, _ = predict_one(model, input_tensor)
            is_correct = predicted_label == true_label

            targets_to_run = [("predicted", predicted_label)]
            if not is_correct:
                targets_to_run.append(("true", true_label))

            for target_type, target_label in targets_to_run:
                output_folder = folders["per_model"] / model_key
                base_name = (
                    f"{sample['sample_id']}_{target_type}_"
                    f"class_{int(target_label):03d}"
                )
                png_path = output_folder / f"{base_name}.png"

                result = {
                    "model_key": model_key,
                    "sample_id": sample["sample_id"],
                    "sample_group": sample["sample_group"],
                    "image_path": sample["image_path"],
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "correct": is_correct,
                    "target_type": target_type,
                    "target_label": int(target_label),
                    "target_layer": loaded["target_layer_name"],
                    "cam_min": "",
                    "cam_max": "",
                    "cam_mean": "",
                    "deletion_fraction": "",
                    "deletion_baseline": "",
                    "original_target_probability": "",
                    "salient_deleted_probability": "",
                    "shifted_control_deleted_probability": "",
                    "salient_probability_drop": "",
                    "shifted_control_probability_drop": "",
                    "salient_relative_drop": "",
                    "shifted_control_relative_drop": "",
                    "figure_png": relative_path(png_path),
                    "status": "failed",
                    "error": "",
                }

                try:
                    heatmap, color_heatmap, overlay = make_grad_cam(
                        cam_object,
                        input_tensor,
                        target_label,
                        display_image,
                    )
                    save_three_panel_figure(
                        display_image=display_image,
                        color_heatmap=color_heatmap,
                        overlay=overlay,
                        model_name=model_key,
                        true_label=true_label,
                        predicted_label=predicted_label,
                        target_label=target_label,
                        target_type=target_type,
                        sample_id=sample["sample_id"],
                        png_path=png_path,
                    )
                    result["cam_min"] = float(heatmap.min())
                    result["cam_max"] = float(heatmap.max())
                    result["cam_mean"] = float(heatmap.mean())
                    if not skip_deletion:
                        deletion_metrics = compute_deletion_metrics(
                            model=model,
                            input_tensor=input_tensor,
                            target_label=target_label,
                            heatmap=heatmap,
                            fraction=deletion_fraction,
                            random_key=(
                                f"{model_key}|{sample['sample_id']}|"
                                f"{target_type}|{target_label}"
                            ),
                        )
                        result.update(deletion_metrics)
                    result["status"] = "completed"
                    completed_cams += 1
                    visual_cache[
                        (model_key, sample["sample_id"], target_type)
                    ] = {
                        "display_image": display_image,
                        "heatmap": heatmap,
                        "color_heatmap": color_heatmap,
                        "overlay": overlay,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "target_label": int(target_label),
                        "target_type": target_type,
                    }
                except Exception as error:
                    result["error"] = f"{type(error).__name__}: {error}"
                    failed_cams += 1

                result_rows.append(result)

    del loaded
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()
    return completed_cams, failed_cams


def run_workflow(
    mode="full",
    device_name="cpu",
    results_dir=DEFAULT_RESULTS_DIR,
    smoke_model="resnet50_full_finetune",
    profile="baseline",
    samples_per_group=None,
    deletion_fraction=0.20,
    skip_deletion=False,
):
    """Run the smoke test or complete available-model workflow."""
    if not 0.0 < float(deletion_fraction) < 1.0:
        raise ValueError("deletion_fraction must be between zero and one.")
    folders = make_output_folders(results_dir)
    inventory = save_model_inventory(results_dir)
    selected = select_shared_samples(
        results_dir,
        profile=profile,
        samples_per_group=samples_per_group,
    )
    if mode == "smoke":
        selected_for_run = selected.head(1).copy()
    else:
        selected_for_run = selected.copy()

    device = choose_device(device_name)
    result_rows = []
    status_rows = []
    visual_cache = {}

    rows_to_run = model_rows_to_run(inventory, mode, smoke_model)
    for _, model_row in rows_to_run.iterrows():
        model_key = model_row["model_key"]
        checkpoint_path = REPO_ROOT / model_row["checkpoint"]
        status = {
            "model_key": model_key,
            "checkpoint": model_row["checkpoint"],
            "checkpoint_exists": checkpoint_path.is_file(),
            "target_layer": model_row["target_layer"],
            "device": str(device),
            "samples_requested": len(selected_for_run),
            "cams_completed": 0,
            "cams_failed": 0,
            "status": "failed",
            "error": "",
        }

        if not checkpoint_path.is_file():
            status["status"] = "skipped_missing_checkpoint"
            status["error"] = "Checkpoint file was not found."
            status_rows.append(status)
            save_status_table(status_rows, results_dir)
            continue

        print(f"Loading {model_key}")
        try:
            completed, failed = run_one_model(
                model_row=model_row,
                selected=selected_for_run,
                device=device,
                folders=folders,
                result_rows=result_rows,
                visual_cache=visual_cache,
                deletion_fraction=deletion_fraction,
                skip_deletion=skip_deletion,
            )
            status["cams_completed"] = completed
            status["cams_failed"] = failed
            if failed == 0:
                status["status"] = "completed"
            else:
                status["status"] = "completed_with_empty_cams"
                status["error"] = (
                    f"{failed} target(s) produced an empty positive Grad-CAM. "
                    "They are kept as failed result rows."
                )
        except Exception as error:
            status["error"] = f"{type(error).__name__}: {error}"
            print(status["error"])

        status_rows.append(status)
        save_result_table(result_rows, results_dir)
        save_status_table(status_rows, results_dir)

    pair_metrics = create_pair_metrics(
        selected_for_run,
        visual_cache,
        results_dir,
        deletion_fraction,
    )
    if mode == "full":
        create_cross_model_figures(selected, visual_cache, folders)
        create_quantitative_summaries(result_rows, pair_metrics, folders)

    save_result_table(result_rows, results_dir)
    save_status_table(status_rows, results_dir)

    summary = {
        "mode": mode,
        "profile": profile,
        "device": str(device),
        "selected_samples": len(selected_for_run),
        "samples_per_group": samples_per_group,
        "result_rows": len(result_rows),
        "completed_cams": sum(
            1 for row in result_rows if row["status"] == "completed"
        ),
        "failed_cams": sum(
            1 for row in result_rows if row["status"] != "completed"
        ),
        "predicted_true_pairs": len(pair_metrics),
        "deletion_fraction": deletion_fraction,
        "deletion_metrics_enabled": not skip_deletion,
        "results_dir": relative_path(results_dir),
    }
    summary_path = Path(results_dir) / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    args = parse_args()
    if args.results_dir is None:
        results_dir = (
            EXTENDED_RESULTS_DIR
            if args.profile == "extended"
            else DEFAULT_RESULTS_DIR
        )
    else:
        results_dir = args.results_dir
    run_workflow(
        mode=args.mode,
        device_name=args.device,
        results_dir=results_dir,
        smoke_model=args.smoke_model,
        profile=args.profile,
        samples_per_group=args.samples_per_group,
        deletion_fraction=args.deletion_fraction,
        skip_deletion=args.skip_deletion,
    )


if __name__ == "__main__":
    main()
