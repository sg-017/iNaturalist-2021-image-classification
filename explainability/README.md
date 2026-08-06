# Grad-CAM Explainability

This directory contains the repository's Grad-CAM workflow for comparing what different image classifiers attend to on a shared set of iNaturalist test images.

## Contents

| File | Purpose |
| --- | --- |
| `run_grad_cam.py` | Command-line smoke and full Grad-CAM workflows |
| `grad_cam_utils.py` | Model inventory, checkpoint loaders, sample selection, CAM generation, and plotting helpers |
| `grad_cam_analysis.ipynb` | Interactive analysis using the same helper functions |
| `__init__.py` | Marks the directory as a Python package |

## Supported models

The model inventory is defined by `get_model_inventory()` in `grad_cam_utils.py`.

| Model key | Architecture | Grad-CAM target layer |
| --- | --- | --- |
| `resnet50_full_finetune` | Standard fully fine-tuned ResNet-50 | Last block of `layer4` |
| `resnet50_scratch` | ResNet-50 trained from scratch | Last block of `layer4` |
| `convnext_full_finetune` | Fully fine-tuned ConvNeXt-Tiny | Last block of the final stage |
| `resnet50_fgvc_pmg_isqrtcov` | ResNet-50 PMG + iSQRT-COV | Last layer of the final backbone stage |

The linear-probe ResNet checkpoint is listed in the inventory for completeness but is not selected by the full workflow.

## Prerequisites

Run commands from the repository root. First install the root environment and prepare the dataset as described in the [main README](../README.md).

The workflow expects:

1. `dataset/processed_dataset/test.json` and its test images;
2. the model checkpoints listed in `get_model_inventory()`;
3. the prediction CSVs under `results/` used by the selected sampling profile.

Large checkpoints are normally not committed. Train/evaluate the corresponding model first, copy each checkpoint to the path listed below, or update the inventory paths to match your storage layout:

```text
results/resnet50_full_finetune/best_model.pth
results/resnet50_scratch/best_model.pth
results/convnext_full_finetune_seed42/best_checkpoint.pt
results/resnet50_finetune_pmg_isqrtcov_from_pmg/best.pt
```

The checkpoint must match the architecture and preprocessing configuration used by its training script.
For compatibility with existing local experiments, the inventory also checks the
legacy checkpoint locations under `models/` when a checkpoint is not present in
the new `results/` location.

## Run the workflow

Start with a one-image smoke test:

```bash
python explainability/run_grad_cam.py \
  --mode smoke \
  --device auto \
  --results-dir results/grad_cam_explainability
```

The default smoke model is `resnet50_full_finetune`. Select another inventory key with `--smoke-model`.

Run the complete available-model comparison after the smoke test succeeds:

```bash
python explainability/run_grad_cam.py \
  --mode full \
  --device auto \
  --results-dir results/grad_cam_explainability
```

`--device` accepts `cpu`, `cuda`, `mps`, or `auto`. Models whose checkpoints are missing are recorded as skipped instead of stopping all other models.

Run the extended 30-image study with quantitative faithfulness analysis:

```bash
python explainability/run_grad_cam.py \
  --mode full \
  --profile extended \
  --samples-per-group 10 \
  --deletion-fraction 0.20 \
  --device auto
```

The extended profile writes to `results/grad_cam_explainability_extended/` by
default. Use `--profile baseline` for the original reproducible 15-image study.

## Sampling and CAM targets

The baseline profile uses the ConvNeXt prediction file to deterministically select five examples from each of three groups using seed 42:

- high-confidence correct predictions;
- incorrect predictions;
- low-confidence difficult examples.

The extended profile jointly uses ConvNeXt, DINOv2, and PMG+iSQRT-COV predictions
to select consensus-correct, recurring confusable-error, and consensus-difficult
groups. By default, it selects ten images per group.

For every selected image, Grad-CAM targets the model's predicted class. For an incorrect prediction, a second CAM targets the ground-truth class. This permits both cross-model comparisons and predicted-versus-true-class comparisons.

## Outputs

The default output directory is `results/grad_cam_explainability/`:

```text
results/grad_cam_explainability/
├── model_inventory.csv
├── selected_grad_cam_samples.csv
├── grad_cam_run_status.csv
├── grad_cam_results.csv
├── run_summary.json
└── figures/
    ├── per_model/
    └── comparisons/
```

Per-model figures contain the original input, heatmap, and overlay. Comparison figures cover shared examples, correct versus incorrect predictions, predicted-class versus true-class targets, and scratch versus pretrained ResNet-50.

The extended workflow additionally writes predicted-versus-true CAM IoU,
salient-region deletion results with shifted-mask controls, per-model and
per-group summary CSVs, and `grad_cam_quantitative_summary.png`.

## Notebook

Open `grad_cam_analysis.ipynb` after configuring the same environment. Run the notebook with the repository root as its working directory so that imports and relative dataset/result paths resolve consistently.
