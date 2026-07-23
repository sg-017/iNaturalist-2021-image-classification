# Fine-Grained ResNet-50 Models

This directory contains ResNet-50 fine-grained image-classification experiments for the 500-class iNaturalist 2021 subset. The scripts use the Hugging Face `microsoft/resnet-50` ImageNet-1K checkpoint by default and implement CBAM, MC-Loss, Compact Bilinear Pooling (CBP), API-Net, PMG, iSQRT-COV, knowledge distillation, and validation-selected ensembles.

## Contents

| File | Model or role | Main operating modes |
| --- | --- | --- |
| `resnet_api.py` | ResNet-50 + API-Net attentive pairwise interaction | train, resume, test |
| `resnet_api_cbp.py` | ResNet-50 + CBP + API-Net + supervised contrastive loss | train, resume, test, ensemble |
| `resnet_api_cbp_distill.py` | Offline probability distillation from CBP/API/CBP+API teachers into one CBP+API student | train, resume |
| `resnet_cbam.py` | ResNet-50 with CBAM attention | train, resume, test |
| `resnet_cbp.py` | ResNet-50 with GAP + Compact Bilinear Pooling fusion | train, resume, test, optional flip TTA |
| `resnet_ensemble.py` | Validation-selected five-model ensemble | ensemble only |
| `resnet_isqrt-cov.py` | ResNet-50 with GAP + iSQRT-COV fusion | train, resume, test |
| `resnet_mc-loss.py` | ResNet-50 with Mutual-Channel Loss | train, resume, test |
| `resnet_mc_cbam.py` | ResNet-50 with CBAM and MC-Loss | train, initialize from another model, resume, test, two-model ensemble |
| `resnet_pmg.py` | ResNet-50 with Progressive Multi-Granularity training | train, resume, test |
| `resnet_pmg_cov.py` | Shared-backbone PMG + GAP/iSQRT-COV hybrid | train, resume, test, two-model ensemble |
| `outputs/` | Committed evaluation reports and prediction CSV files | generated artifacts, not executable |

All model scripts accept `--data-dir` and `--output-dir`, pass both paths explicitly. Training normally writes `run_config.json`, `training_history.jsonl`, `best.pt`, `last.pt`, `test_predictions.csv`, `classification_report.txt`, and sometimes model-specific diagnostics. After training, the best validation checkpoint is automatically evaluated on the test split.

Common execution patterns are:

```bash
# Train from ImageNet initialization.
python SCRIPT.py --data-dir DATA_DIR --output-dir OUTPUT_DIR

# Resume a complete training checkpoint. --epochs is the final total epoch count.
python SCRIPT.py --data-dir DATA_DIR --output-dir OUTPUT_DIR \
  --resume OUTPUT_DIR/last.pt --epochs 30

# Evaluate a checkpoint without training.
python SCRIPT.py --data-dir DATA_DIR --output-dir EVAL_DIR \
  --test-only --checkpoint OUTPUT_DIR/best.pt
```

`--device` defaults to `cuda` when CUDA is available and otherwise to `cpu`. Mixed precision is enabled only on CUDA; pass `--no-amp` to disable it. Boolean flags take no value: use `--no-amp`, not `--no-amp true`.

## `resnet_mc-loss.py`

This adds the Mutual-Channel auxiliary loss during training. The normal classifier is used for validation and test inference. MC-Loss is enabled by default.

```bash
python resnet_mc-loss.py --data-dir DATA_DIR --output-dir outputs/mc-loss

# Plain ResNet-50 baseline.
python resnet_mc-loss.py --data-dir DATA_DIR --output-dir outputs/plain --no-mc-loss
```

Parser inputs:

| Group | Options and defaults |
| --- | --- |
| Paths/model | `--data-dir PATH` (`None`), `--output-dir PATH` (`None`), `--model-name NAME` (`microsoft/resnet-50`), `--num-classes N` (`500`) |
| Schedule/data loader | `--epochs N` (`50`), `--batch-size N` (`32`), `--eval-batch-size N` (`None`, falls back to training batch size), `--num-workers N` (`4`), `--image-size N` (`224`) |
| Optimization | `--lr FLOAT` (`1e-4`), `--weight-decay FLOAT` (`5e-4`), `--warmup-epochs FLOAT` (`1.0`), `--label-smoothing FLOAT` (`0.1`), `--grad-clip FLOAT` (`1.0`) |
| MC-Loss | mutually exclusive `--use-mc-loss` / `--no-mc-loss` (default: enabled), `--mc-channels-per-class N` (`3`), `--mc-keep-channels N` (`2`), `--mc-alpha FLOAT` (`1.0`), `--mc-beta FLOAT` (`20.0`) |
| Runtime/checkpoint | `--seed N` (`42`), `--device DEVICE` (automatic), `--no-amp`, `--resume PATH` (`None`), `--test-only`, `--checkpoint PATH` (`None`) |

## `resnet_cbam.py`

This inserts CBAM channel and spatial attention into selected ResNet stages. CBAM is enabled by default.

```bash
python resnet_cbam.py --data-dir DATA_DIR --output-dir outputs/cbam \
  --cbam-stages 3 4

# Plain ResNet-50 baseline through the same implementation.
python resnet_cbam.py --data-dir DATA_DIR --output-dir outputs/plain --no-cbam
```

Parser inputs:

| Group | Options and defaults |
| --- | --- |
| Paths/model | `--data-dir PATH` (`None`), `--output-dir PATH` (`None`), `--model-name NAME` (`microsoft/resnet-50`), `--num-classes N` (`500`) |
| Schedule/data loader | `--epochs N` (`50`), `--batch-size N` (`32`), `--eval-batch-size N` (`64`), `--num-workers N` (`4`), `--image-size N` (`224`) |
| Optimization | `--lr FLOAT` (`None`, global override), `--backbone-lr FLOAT` (`1e-4`), `--cbam-lr FLOAT` (`1e-4`), `--cbam-scale-lr FLOAT` (`5e-3`), `--classifier-lr FLOAT` (`1e-4`), `--weight-decay FLOAT` (`5e-4`), `--warmup-epochs FLOAT` (`2.0`), `--label-smoothing FLOAT` (`0.05`), `--grad-clip FLOAT` (`1.0`) |
| CBAM | mutually exclusive `--use-cbam` / `--no-cbam` (default: enabled), `--cbam-reduction N` (`16`), `--cbam-spatial-kernel {3,7}` (`7`), `--cbam-initial-scale FLOAT` (`0.05`), `--cbam-stages {1,2,3,4} [...]` (`4`) |
| Runtime/checkpoint | `--seed N` (`42`), `--device DEVICE` (automatic), `--no-amp`, `--resume PATH` (`None`), `--test-only`, `--checkpoint PATH` (`None`) |

## `resnet_mc_cbam.py`

This jointly supports CBAM and MC-Loss, both enabled by default. It can also start a new run from an MC-only or CBAM-only checkpoint, freeze CBAM while training the remaining network, or directly average the probabilities of separate MC-only and CBAM-only models.

```bash
# Joint training.
python resnet_mc_cbam.py --data-dir DATA_DIR --output-dir outputs/mc-cbam

# Add CBAM to a trained MC model and start a fresh run.
python resnet_mc_cbam.py --data-dir DATA_DIR --output-dir outputs/mc-to-cbam \
  --init-from-mc-checkpoint outputs/mc-loss/best.pt

# Fixed-weight probability ensemble.
python resnet_mc_cbam.py --data-dir DATA_DIR --output-dir outputs/mc-cbam-ensemble \
  --ensemble --mc-checkpoint outputs/mc-loss/best.pt \
  --cbam-checkpoint outputs/cbam/best.pt --ensemble-mc-weight 0.5
```

Parser inputs:

| Group | Options and defaults |
| --- | --- |
| Paths/model | `--data-dir PATH` (`None`), `--output-dir PATH` (`None`), `--model-name NAME` (`microsoft/resnet-50`), `--num-classes N` (`500`) |
| Schedule/data loader | `--epochs N` (`50`), `--batch-size N` (`32`), `--eval-batch-size N` (`64`), `--num-workers N` (`4`), `--image-size N` (`224`) |
| Optimization | `--lr FLOAT` (`None`, global override), `--backbone-lr FLOAT` (`1e-4`), `--cbam-lr FLOAT` (`1e-4`), `--cbam-scale-lr FLOAT` (`5e-3`), `--classifier-lr FLOAT` (`1e-4`), `--weight-decay FLOAT` (`5e-4`), `--warmup-epochs FLOAT` (`2.0`), `--label-smoothing FLOAT` (`0.05`), `--grad-clip FLOAT` (`1.0`) |
| CBAM | `--use-cbam` / `--no-cbam` (default: enabled), `--freeze-cbam`, `--cbam-reduction N` (`16`), `--cbam-spatial-kernel {3,7}` (`7`), `--cbam-initial-scale FLOAT` (`0.05`), `--cbam-stages {1,2,3,4} [...]` (default: `[3]`) |
| MC-Loss | `--use-mc-loss` / `--no-mc-loss` (default: enabled), `--mc-channels-per-class N` (`3`), `--mc-keep-channels N` (`2`), `--mc-alpha FLOAT` (`1.0`), `--mc-beta FLOAT` (`20.0`), `--mc-delay-epochs N` (`3`), `--mc-warmup-epochs N` (`5`) |
| Initialization/checkpoint | `--resume PATH` (`None`), `--init-from-mc-checkpoint PATH` (`None`), `--init-from-cbam-checkpoint PATH` (`None`), `--test-only`, `--checkpoint PATH` (`None`) |
| Ensemble | `--ensemble`, `--mc-checkpoint PATH` (`None`), `--cbam-checkpoint PATH` (`None`), `--ensemble-mc-weight FLOAT` (`0.5`; CBAM receives `1 - weight`) |
| Runtime | `--seed N` (`42`), `--device DEVICE` (automatic), `--no-amp` |

The two fresh-initialization options are mutually exclusive and cannot be combined with resume, test, or ensemble mode.

## `resnet_api.py`

This model applies API-Net attentive pairwise interactions to globally pooled ResNet-50 features. Training uses class-balanced batches so that every anchor has same-class and different-class partners. Plain classification logits are used at inference.

```bash
python resnet_api.py --data-dir DATA_DIR --output-dir outputs/api

python resnet_api.py --data-dir DATA_DIR --output-dir outputs/api-eval \
  --test-only --checkpoint outputs/api/best.pt
```

Parser inputs:

| Group | Options and defaults |
| --- | --- |
| Paths/model | `--data-dir PATH` (`None`), `--output-dir PATH` (`None`), `--model-name NAME` (`microsoft/resnet-50`), `--num-classes N` (`500`) |
| Schedule/data loader | `--epochs N` (`50`), `--batch-size N` (`64`), `--samples-per-class N` (`4`), `--eval-batch-size N` (`32`), `--num-workers N` (`4`), `--image-size N` (`448`) |
| Optimization | `--lr FLOAT` (`None`, overrides all grouped learning rates), `--backbone-lr FLOAT` (`1e-4`), `--classifier-lr FLOAT` (`1e-3`), `--api-lr FLOAT` (`1e-3`), `--weight-decay FLOAT` (`5e-4`), `--warmup-epochs FLOAT` (`2.0`), `--label-smoothing FLOAT` (`0.05`), `--grad-clip FLOAT` (`1.0`) |
| API-Net/loss | `--api-hidden-size N` (`512`), `--dropout FLOAT` (`0.3`), `--plain-ce-weight FLOAT` (`1.0`), `--api-ce-weight FLOAT` (`0.25`), `--rank-margin FLOAT` (`0.2`), `--rank-weight FLOAT` (`0.1`) |
| Neighbor-aware batches | `--class-neighbors-json PATH` (`None`), `--neighbor-batch-fraction FLOAT` (`0.25`), `--max-neighbors-per-class N` (`5`), `--allow-test-neighbors` (off) |
| Runtime/checkpoint | `--seed N` (`42`), `--device DEVICE` (automatic), `--no-amp` (off), `--resume PATH` (`None`), `--test-only` (off), `--checkpoint PATH` (`None`) |

The training batch size must be divisible by `--samples-per-class`, with at least two samples per class and at least two classes per batch. `--test-only` requires `--checkpoint` and cannot be combined with `--resume`.

## `resnet_cbp.py`

This fuses the normal global-average-pooling (GAP) classifier with a Tensor-Sketch CBP classifier through a learned residual gate.

```bash
python resnet_cbp.py --data-dir DATA_DIR --output-dir outputs/cbp

python resnet_cbp.py --data-dir DATA_DIR --output-dir outputs/cbp-tta \
  --test-only --checkpoint outputs/cbp/best.pt --flip-tta
```

Parser inputs:

| Group | Options and defaults |
| --- | --- |
| Paths/model | `--data-dir PATH` (`None`), `--output-dir PATH` (`None`), `--model-name NAME` (`microsoft/resnet-50`), `--num-classes N` (`500`) |
| Schedule/data loader | `--epochs N` (`50`), `--batch-size N` (`32`), `--eval-batch-size N` (`64`), `--num-workers N` (`4`), `--image-size N` (`384`) |
| Optimization | `--lr FLOAT` (`None`, global override), `--backbone-lr FLOAT` (`1e-4`), `--classifier-lr FLOAT` (`1e-3`), `--fusion-gate-lr FLOAT` (`1e-3`), `--weight-decay FLOAT` (`5e-4`), `--warmup-epochs FLOAT` (`2.0`), `--label-smoothing FLOAT` (`0.05`), `--gap-aux-loss-weight FLOAT` (`0`), `--cbp-aux-loss-weight FLOAT` (`0`), `--grad-clip FLOAT` (`1.0`) |
| CBP/fusion | `--cbp-output-dim N` (`8192`), `--cbp-seed N` (`1`), `--cbp-spatial-chunk-size N` (`0`), `--no-signed-sqrt`, `--no-l2-normalize`, `--cbp-dropout FLOAT` (`0.3`), `--fusion-initial-gate FLOAT` (`0.1`) |
| Runtime/checkpoint | `--seed N` (`42`), `--device DEVICE` (automatic), `--no-amp`, `--resume PATH` (`None`), `--test-only`, `--checkpoint PATH` (`None`), `--flip-tta` (test-time horizontal-flip logit averaging) |

## `resnet_api_cbp.py`

This combines feature-map Tensor-Sketch CBP with API-Net. It can be trained directly, initialized from a standalone `resnet_cbp.py` checkpoint, or used to select fusion weights for standalone/combined checkpoints on validation before testing.

```bash
# Direct training.
python resnet_api_cbp.py --data-dir DATA_DIR --output-dir outputs/api-cbp

# Stage-two initialization from a CBP model.
python resnet_api_cbp.py --data-dir DATA_DIR --output-dir outputs/api-cbp-stage2 \
  --init-cbp-checkpoint outputs/cbp/best.pt --stage2-train-scope api-heads

# Evaluate the combined model.
python resnet_api_cbp.py --data-dir DATA_DIR --output-dir outputs/api-cbp-eval \
  --test-only --checkpoint outputs/api-cbp/best.pt

# General two-or-more-model ensemble. Repeat --ensemble-model for every member.
python resnet_api_cbp.py --mode ensemble --data-dir DATA_DIR \
  --output-dir outputs/api-cbp-ensemble \
  --ensemble-model cbp cbp outputs/cbp/best.pt \
  --ensemble-model api api outputs/api/best.pt \
  --ensemble-model combined combined outputs/api-cbp/best.pt
```

Parser inputs:

| Group | Options and defaults |
| --- | --- |
| Mode/paths | `--mode {train,ensemble}` (`train`), `--data-dir PATH` (`None`), `--output-dir PATH` (`None`), `--model-name NAME` (`microsoft/resnet-50`), `--num-classes N` (`500`) |
| Schedule/data loader | `--epochs N` (`50`), `--batch-size N` (`64`), `--samples-per-class N` (`4`), `--eval-batch-size N` (`128`), `--num-workers N` (`4`), `--image-size N` (`448`) |
| Runtime | `--seed N` (`42`), `--device DEVICE` (automatic), `--no-amp` (off) |
| Optimization | `--lr FLOAT` (`None`, global override), `--backbone-lr FLOAT` (`1e-4`), `--classifier-lr FLOAT` (`1e-3`), `--api-lr FLOAT` (`1e-3`), `--weight-decay FLOAT` (`5e-4`), `--warmup-epochs FLOAT` (`2.0`), `--label-smoothing FLOAT` (`0.05`), `--grad-clip FLOAT` (`1.0`) |
| CBP | `--cbp-output-dim N` (`8192`), `--cbp-seed N` (`1`), `--cbp-spatial-chunk-size N` (`0`, all spatial positions), `--no-signed-sqrt` (off), `--no-l2-normalize` (off), `--dropout FLOAT` (`0.3`) |
| API/loss | `--api-hidden-size N` (`512`), `--plain-ce-weight FLOAT` (`1.0`), `--api-ce-weight FLOAT` (`0.25`), `--rank-margin FLOAT` (`0.2`), `--rank-weight FLOAT` (`0.1`), `--supcon-weight FLOAT` (`0.05`), `--supcon-temperature FLOAT` (`0.1`) |
| Stage-two initialization | `--init-cbp-checkpoint PATH` (`None`), `--stage2-train-scope {full,api-heads}` (`full`), `--no-init-cbp-classifier` (off) |
| Neighbor-aware batches | `--class-neighbors-json PATH` (`None`), `--neighbor-batch-fraction FLOAT` (`0.25`), `--max-neighbors-per-class N` (`5`) |
| Checkpoints | `--resume PATH` (`None`), `--test-only` (off), `--checkpoint PATH` (`None`) |
| Ensemble members | `--ensemble-model NAME TYPE CHECKPOINT` (repeatable; `TYPE` must be `cbp`, `api`, or `combined`), or the legacy pair `--cbp-checkpoint PATH` and `--api-checkpoint PATH` |
| Ensemble search | `--ensemble-search-trials N` (`500`, used for more than two models), `--alpha-steps N` (`101`), `--alpha-metric {top1,macro_f1}` (`top1`), `--fusion-space {logits,probabilities}` (`logits`) |

`api-heads` scope requires a CBP initialization or resumed checkpoint. A resume and a fresh CBP initialization are mutually exclusive.

## `resnet_api_cbp_distill.py`

This performs offline knowledge distillation. It calibrates and fuses three teachers (standalone CBP, standalone API-Net, and a combined CBP+API model), caches their training-set probabilities, and trains one combined student. Teachers are not needed at student test time.

Intended invocation:

```bash
python resnet_api_cbp_distill.py --data-dir DATA_DIR \
  --output-dir outputs/api-cbp-distill \
  --cbp-teacher-checkpoint outputs/cbp/best.pt \
  --api-teacher-checkpoint outputs/api/best.pt \
  --supcon-teacher-checkpoint outputs/api-cbp/best.pt
```

Parser inputs:

| Group | Options and defaults |
| --- | --- |
| Paths/model | `--data-dir PATH` (`None`), `--output-dir PATH` (`None`), `--model-name NAME` (`microsoft/resnet-50`), `--num-classes N` (`500`) |
| Schedule/data loader | `--epochs N` (`50`), `--batch-size N` (`16`), `--samples-per-class N` (`4`), `--eval-batch-size N` (`32`), `--num-workers N` (`4`), `--image-size N` (`448`) |
| Runtime | `--seed N` (`42`), `--device DEVICE` (automatic), `--no-amp` (off), `--resume PATH` (`None`) |
| Required teachers | `--cbp-teacher-checkpoint PATH`, `--api-teacher-checkpoint PATH`, `--supcon-teacher-checkpoint PATH` (all default to `None`) |
| Teacher fusion | `--teacher-weights W_CBP W_API W_SUPCON` (`None`, search weights), `--teacher-weight-steps N` (`51`), `--teacher-weight-metric {nll,top1,macro_f1}` (`nll`), `--teacher-calibration {temperature,none}` (`temperature`), `--teacher-temperature-min FLOAT` (`0.05`), `--teacher-temperature-max FLOAT` (`10.0`) |
| Distillation | `--distill-weight FLOAT` (`0.25`), `--distill-final-weight FLOAT` (`0.0`), `--distill-schedule {cosine,constant}` (`cosine`), `--distill-temperature FLOAT` (`3.0`), `--teacher-cache PATH` (`OUTPUT_DIR/teacher_train_probabilities.npz` when omitted), `--rebuild-teacher-cache` (off) |
| Optimization | `--lr FLOAT` (`None`, global override), `--backbone-lr FLOAT` (`1e-4`), `--classifier-lr FLOAT` (`1e-3`), `--api-lr FLOAT` (`1e-3`), `--weight-decay FLOAT` (`5e-4`), `--warmup-epochs FLOAT` (`2.0`), `--label-smoothing FLOAT` (`0.05`), `--grad-clip FLOAT` (`1.0`) |
| Student CBP/API | `--cbp-output-dim N` (`8192`), `--cbp-seed N` (`1`), `--cbp-spatial-chunk-size N` (`0`), `--no-signed-sqrt`, `--no-l2-normalize`, `--dropout FLOAT` (`0.3`), `--api-hidden-size N` (`512`) |
| Student losses | `--plain-ce-weight FLOAT` (`1.0`), `--api-ce-weight FLOAT` (`0.15`), `--rank-margin FLOAT` (`0.2`), `--rank-weight FLOAT` (`0.05`), `--supcon-weight FLOAT` (`0.05`), `--supcon-temperature FLOAT` (`0.1`) |
| Initialization/batches | `--init-cbp-checkpoint PATH` (`None`), `--stage2-train-scope {full,api-heads}` (`full`), `--no-init-cbp-classifier`, `--class-neighbors-json PATH` (`None`), `--neighbor-batch-fraction FLOAT` (`0.25`), `--max-neighbors-per-class N` (`5`) |

## `resnet_pmg.py`

PMG adds classifiers to the final three ResNet stages and progressively trains them using 2x2, 4x4, and 8x8 jigsaw inputs, followed by the original image and concatenated multi-granularity prediction.

```bash
python resnet_pmg.py --data-dir DATA_DIR --output-dir outputs/pmg
```

Parser inputs:

| Group | Options and defaults |
| --- | --- |
| Paths/model | `--data-dir PATH` (`None`), `--output-dir PATH` (`None`), `--model-name NAME` (`microsoft/resnet-50`), `--num-classes N` (`500`) |
| Schedule/data loader | `--epochs N` (`100`), `--batch-size N` (`32`), `--eval-batch-size N` (`64`), `--num-workers N` (`4`), `--image-size N` (`448`), `--resize-size N` (`550`), `--feature-size N` (`512`) |
| Optimization/loss | `--backbone-lr FLOAT` (`2e-4`), `--head-lr FLOAT` (`2e-3`), `--momentum FLOAT` (`0.9`), `--weight-decay FLOAT` (`5e-4`), `--warmup-epochs FLOAT` (`0.0`), `--label-smoothing FLOAT` (`0.05`), `--concat-loss-weight FLOAT` (`2.0`), `--grad-clip FLOAT` (`0.0`, disabled) |
| Runtime/checkpoint | `--seed N` (`42`), `--device DEVICE` (automatic), `--no-amp`, `--resume PATH` (`None`), `--test-only`, `--checkpoint PATH` (`None`) |

The image size must be at least 8 and divisible by 8; the resize size must not be smaller than the crop. Training uses SGD and requires a batch size of at least 2.

## `resnet_isqrt-cov.py`

This reduces the final feature-map channel dimension, computes an iSQRT-COV second-order representation with Newton-Schulz iterations, and fuses its logits with a GAP classifier through a learned scale.

```bash
python resnet_isqrt-cov.py --data-dir DATA_DIR --output-dir outputs/isqrt-cov
```

Parser inputs:

| Group | Options and defaults |
| --- | --- |
| Paths/model | `--data-dir PATH` (`None`), `--output-dir PATH` (`None`), `--model-name NAME` (`microsoft/resnet-50`), `--num-classes N` (`500`) |
| Schedule/data loader | `--epochs N` (`50`), `--batch-size N` (`32`), `--eval-batch-size N` (`64`), `--num-workers N` (`4`), `--image-size N` (`448`) |
| Optimization | `--lr FLOAT` (`None`, global override), `--backbone-lr FLOAT` (`1e-4`), `--cov-lr FLOAT` (`1e-3`), `--classifier-lr FLOAT` (`1e-3`), `--fusion-scale-lr FLOAT` (`1e-3`), `--weight-decay FLOAT` (`5e-4`), `--warmup-epochs FLOAT` (`2.0`), `--label-smoothing FLOAT` (`0.05`), `--grad-clip FLOAT` (`1.0`) |
| Covariance branch | `--cov-dim N` (`256`), `--sqrt-iters N` (`5`), `--cov-eps FLOAT` (`1e-5`), `--cov-initial-scale FLOAT` (`0.05`), `--cov-aux-loss-weight FLOAT` (`0.3`) |
| Runtime/checkpoint | `--seed N` (`42`), `--device DEVICE` (automatic), `--no-amp`, `--resume PATH` (`None`), `--test-only`, `--checkpoint PATH` (`None`) |

## `resnet_pmg_cov.py`

This script either trains a shared-backbone hybrid containing PMG and a fused GAP/iSQRT-COV branch, or selects an alpha on validation for an ensemble of standalone PMG and iSQRT-COV checkpoints.

Intended invocations:

```bash
# Hybrid initialized from a PMG checkpoint.
python resnet_pmg_cov.py --data-dir DATA_DIR --output-dir outputs/pmg-cov \
  --pmg-pretrained-checkpoint outputs/pmg/best.pt

# Standalone-checkpoint ensemble.
python resnet_pmg_cov.py --data-dir DATA_DIR --output-dir outputs/pmg-cov-ensemble \
  --ensemble-only --pmg-checkpoint outputs/pmg/best.pt \
  --cov-checkpoint outputs/isqrt-cov/best.pt
```

Parser inputs:

| Group | Options and defaults |
| --- | --- |
| Paths/model | `--data-dir PATH` (`None`), `--output-dir PATH` (`None`), `--model-name NAME` (`microsoft/resnet-50`), `--num-classes N` (`500`) |
| Schedule/data loader | `--epochs N` (`100`), `--batch-size N` (`8`), `--eval-batch-size N` (`16`), `--num-workers N` (`4`), `--image-size N` (`448`), `--resize-size N` (`550`), `--feature-size N` (`512`) |
| Covariance branch | `--cov-dim N` (`256`), `--sqrt-iters N` (`5`), `--cov-eps FLOAT` (`1e-5`), `--cov-dropout FLOAT` (`0.2`), `--cov-initial-scale FLOAT` (`0.05`) |
| Optimization | `--optimizer {sgd,adamw}` (`sgd`), `--backbone-lr FLOAT` (`2e-4`), `--head-lr FLOAT` (`2e-3`), `--cov-lr FLOAT` (`1e-3`), `--classifier-lr FLOAT` (`1e-3`), `--fusion-scale-lr FLOAT` (`1e-3`), `--momentum FLOAT` (`0.9`), `--weight-decay FLOAT` (`5e-4`), `--warmup-epochs FLOAT` (`0.0`), `--label-smoothing FLOAT` (`0.0`), `--grad-clip FLOAT` (`0.0`) |
| Loss/fusion | `--concat-loss-weight FLOAT` (`2.0`), `--gap-cov-loss-weight FLOAT` / alias `--cov-loss-weight` (`1.0`), `--cov-aux-loss-weight FLOAT` (`0.3`), `--fusion-loss-weight FLOAT` (`1.0`), `--hybrid-cov-logit-weight FLOAT` (`1.0`) |
| Hybrid checkpoint | `--pmg-pretrained-checkpoint PATH` (`None`), `--resume PATH` (`None`), `--test-only`, `--checkpoint PATH` (`None`) |
| Standalone ensemble | `--ensemble-only`, `--pmg-checkpoint PATH` (`None`), `--cov-checkpoint PATH` (`None`), `--ensemble-space {logits,probabilities}` (`logits`), `--ensemble-metric {top1,macro_f1,nll}` (`top1`), `--alpha-step FLOAT` (`0.01`) |
| Runtime | `--seed N` (`42`), `--device DEVICE` (automatic), `--no-amp` |


## `resnet_ensemble.py`

This script is designed to load exactly five checkpoints—API-Net, CBP, combined API+CBP, PMG, and iSQRT-COV—search weights using validation only, and evaluate the selected fusion once on test.

Intended invocation:

```bash
python resnet_ensemble.py --data-dir DATA_DIR --output-dir outputs/ensemble \
  --resnet-api-checkpoint outputs/api/best.pt \
  --resnet-cbp-checkpoint outputs/cbp/best.pt \
  --resnet-api-cbp-checkpoint outputs/api-cbp/best.pt \
  --resnet-pmg-checkpoint outputs/pmg/best.pt \
  --resnet-isqrtcov-checkpoint outputs/isqrt-cov/best.pt
```

Parser inputs:

| Group | Options and defaults |
| --- | --- |
| Paths | `--data-dir PATH`, `--output-dir PATH`, `--resnet-api-checkpoint PATH`, `--resnet-cbp-checkpoint PATH`, `--resnet-api-cbp-checkpoint PATH`, `--resnet-pmg-checkpoint PATH`, `--resnet-isqrtcov-checkpoint PATH` (all default to `None`) |
| Evaluation/runtime | `--num-classes N` (`500`), `--eval-batch-size N` (`32`), `--num-workers N` (`4`), `--seed N` (`42`), `--device DEVICE` (automatic), `--no-amp` |
| Weight selection | `--fusion-space {logits,probabilities}` (`logits`), `--selection-metric {top1,macro_f1}` (`top1`), `--pair-alpha-steps N` (`21`), `--weight-search-trials N` (`2000`), `--search-batch-size N` (`8`) |
| Fallback architecture values | `--model-name NAME` (`microsoft/resnet-50`), `--image-size N` (`224`), `--pmg-resize-size N` (`550`), `--cbp-seed N` (`1`), `--cbp-spatial-chunk-size N` (`0`), `--cbp-dropout FLOAT` (`0.3`), `--fusion-initial-gate FLOAT` (`0.1`), `--dropout FLOAT` (`0.3`), `--cov-sqrt-iters N` (`5`), `--cov-eps FLOAT` (`1e-5`), `--cov-initial-scale FLOAT` (`0.05`), `--no-signed-sqrt`, `--no-l2-normalize` |

