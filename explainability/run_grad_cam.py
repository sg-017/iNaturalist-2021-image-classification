"""Run the repository Grad-CAM workflow.

Examples from the repository root:

    python explainability/run_grad_cam.py --mode smoke --device auto
    python explainability/run_grad_cam.py --mode full --device auto
"""

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from pytorch_grad_cam import GradCAM

from grad_cam_utils import (
    DATA_DIR,
    DEFAULT_RESULTS_DIR,
    REPO_ROOT,
    choose_device,
    load_original_image,
    load_model,
    make_grad_cam,
    make_output_folders,
    prepare_image,
    predict_one,
    relative_path,
    safe_name,
    save_model_inventory,
    save_overlay_comparison,
    save_three_panel_figure,
    select_shared_samples,
)


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
        default=DEFAULT_RESULTS_DIR,
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
        "figure_png",
        "status",
        "error",
    ]
    path = Path(results_dir) / "grad_cam_results.csv"
    pd.DataFrame(result_rows, columns=columns).to_csv(path, index=False)


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
    model_order = [
        "resnet50_full_finetune",
        "convnext_full_finetune",
        "resnet50_scratch",
        "resnet50_fgvc_pmg_isqrtcov",
    ]

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
        for model_key in model_order:
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
    correct_rows = selected[
        selected["sample_group"] == "high_confidence_correct"
    ]
    incorrect_rows = selected[selected["sample_group"] == "incorrect"]
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
                    result["status"] = "completed"
                    completed_cams += 1
                    visual_cache[
                        (model_key, sample["sample_id"], target_type)
                    ] = {
                        "display_image": display_image,
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
):
    """Run the smoke test or complete available-model workflow."""
    folders = make_output_folders(results_dir)
    inventory = save_model_inventory(results_dir)
    selected = select_shared_samples(results_dir)
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

    if mode == "full":
        create_cross_model_figures(selected, visual_cache, folders)

    save_result_table(result_rows, results_dir)
    save_status_table(status_rows, results_dir)

    summary = {
        "mode": mode,
        "device": str(device),
        "selected_samples": len(selected_for_run),
        "result_rows": len(result_rows),
        "completed_cams": sum(
            1 for row in result_rows if row["status"] == "completed"
        ),
        "failed_cams": sum(
            1 for row in result_rows if row["status"] != "completed"
        ),
        "results_dir": relative_path(results_dir),
    }
    summary_path = Path(results_dir) / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    args = parse_args()
    run_workflow(
        mode=args.mode,
        device_name=args.device,
        results_dir=args.results_dir,
        smoke_model=args.smoke_model,
    )


if __name__ == "__main__":
    main()
