# DINOv2 Image Classification

This directory contains a full fine-tuning pipeline for DINOv2 on the 500-class iNaturalist 2021 subset.

## Directory contents

| Path | Description |
| --- | --- |
| `dinov2_full_finetune.py` | Training, validation, checkpointing, and test evaluation for DINOv2 ViT-S/14 |

## Model

`dinov2_full_finetune.py` uses the Hugging Face `facebook/dinov2-small` checkpoint by default. This is a DINOv2 ViT-S/14 backbone with a newly initialized linear classification head.

Run the examples from this directory, or replace `dinov2_full_finetune.py` with its full path.

### Train

```bash
python dinov2_full_finetune.py \
  --data-dir ../../dataset/processed_dataset \
  --output-dir outputs/dinov2-small
```

Training loads the self-supervised DINOv2 checkpoint, initializes a new classifier, trains all model parameters, evaluates the validation split after every epoch, and finally evaluates the best checkpoint on test.

The default image size and batch size require substantial GPU memory. A memory-saving example is:

```bash
python dinov2_full_finetune.py \
  --data-dir ../../dataset/processed_dataset \
  --output-dir outputs/dinov2-small-memory-saving \
  --batch-size 4 \
  --accumulation-steps 8 \
  --eval-batch-size 8 \
  --gradient-checkpointing
```

This uses an effective training batch size of approximately `batch-size * accumulation-steps`, except for the final partial accumulation window.

### Resume training

```bash
python dinov2_full_finetune.py \
  --data-dir ../../dataset/processed_dataset \
  --output-dir outputs/dinov2-small \
  --resume outputs/dinov2-small/last.pt \
  --epochs 100
```

`--resume` expects a complete checkpoint containing model, optimizer, epoch, and optional gradient-scaler state. `--epochs` is the final total epoch count, not the number of additional epochs. It must be greater than the number of epochs already completed by the checkpoint.

### Test a checkpoint

```bash
python dinov2_full_finetune.py \
  --data-dir ../../dataset/processed_dataset \
  --output-dir outputs/dinov2-small-eval \
  --test-only \
  --checkpoint outputs/dinov2-small/best.pt
```

Test-only mode skips the train and validation datasets. The model architecture arguments, especially `--model-name` and `--num-classes`, must remain compatible with the checkpoint because the script loads its state dictionary strictly.

## Parser inputs

| Input | Type/default | Description |
| --- | --- | --- |
| `--data-dir PATH` | `Path`, default `None` | Processed dataset root. Required in practice. |
| `--output-dir PATH` | `Path`, default `None` | Run artifact directory. When omitted, it becomes `DATA_DIR.parent/outputs/dinov2-small`. |
| `--model-name NAME` | string, default `facebook/dinov2-small` | Hugging Face DINOv2 checkpoint and image-processor identifier. |
| `--num-classes N` | integer, default `500` | Number of output classes; must be at least 2. |
| `--image-size N` | integer, default `224` | Square train/evaluation crop size. It must be a multiple of the 14-pixel patch size and must equal the selected model configuration's `image_size`. |
| `--dropout FLOAT` | float, default `0.0` | Dropout probability before the linear classifier; must be in `[0, 1)`. |
| `--epochs N` | integer, default `50` | Final total number of training epochs; must be positive. |
| `--batch-size N` | integer, default `32` | Training mini-batch size; must be positive. |
| `--eval-batch-size N` | integer, default `64` | Validation and test batch size; must be positive. |
| `--accumulation-steps N` | integer, default `1` | Number of mini-batches accumulated before each optimizer update; must be positive. |
| `--num-workers N` | integer, default `4` | DataLoader worker processes; `0` loads data in the main process. |
| `--seed N` | integer, default `2026` | Seed for Python, NumPy, PyTorch, workers, and loader generators. |
| `--backbone-lr FLOAT` | float, default `5e-5` | Initial learning rate for all DINOv2 backbone parameters; must be positive. |
| `--classifier-lr FLOAT` | float, default `5e-4` | Initial learning rate for the new linear classifier; must be positive. |
| `--weight-decay FLOAT` | float, default `0.05` | AdamW weight decay. Biases and one-dimensional normalization parameters receive no decay. |
| `--warmup-epochs FLOAT` | float, default `2.0` | Linear warm-up duration measured in epochs; must be non-negative. |
| `--min-lr-ratio FLOAT` | float, default `0.01` | Final cosine-schedule LR as a fraction of each parameter group's initial LR; must be in `[0, 1]`. |
| `--label-smoothing FLOAT` | float, default `0.1` | Cross-entropy label smoothing; must be in `[0, 1)`. |
| `--grad-clip FLOAT` | float, default `1.0` | Maximum global gradient norm. Set to `0` to disable clipping. |
| `--device DEVICE` | string, default `cuda` when available, otherwise `cpu` | PyTorch device such as `cuda`, `cuda:0`, or `cpu`. |
| `--amp-dtype {float16,bfloat16,none}` | choice, default `float16` | CUDA autocast data type. `none` disables mixed precision. |
| `--gradient-checkpointing` | flag, default off | Recomputes backbone activations during backward propagation to reduce memory use at the cost of extra computation. |
| `--resume PATH` | `Path`, default `None` | Resume model, optimizer, epoch, and scaler state from a complete training checkpoint. |
| `--test-only` | flag, default off | Skip training and evaluate `--checkpoint` on the test split. |
| `--checkpoint PATH` | `Path`, default `None` | Model or complete checkpoint loaded by test-only mode. |

`--test-only` requires `--checkpoint` and cannot be combined with `--resume`. Both complete checkpoints containing a `model` entry and raw model state dictionaries can be loaded for testing. Resume mode requires the complete training-checkpoint format.

## Generated outputs

The run directory contains:

| File | Description |
| --- | --- |
| `run_config.json` | Resolved parser inputs for the run |
| `training_history.jsonl` | One JSON record per epoch with training and validation metrics |
| `best.pt` | Complete checkpoint with the highest validation top-1 accuracy |
| `last.pt` | Complete checkpoint from the most recent epoch |
| `classification_report.txt` | Test loss, top-1 through top-5 accuracy, macro precision, macro recall, and macro F1 |
| `test_predictions.csv` | Per-image ground truth, prediction, confidence, correctness, and top-five labels/probabilities |
