# ConvNeXt-Tiny
This model uses `convnext_tiny.fb_in22k` from the
[timm library](https://github.com/huggingface/pytorch-image-models).

For linear probing and full fine-tuning, the model starts from weights
pretrained on ImageNet-22K. For training from scratch, the same
ConvNeXt-Tiny architecture is used without pretrained weights.

The specific timm model used in this project is available from: https://huggingface.co/timm/convnext_tiny.fb_in22k

The original ConvNeXt implementation is available from: https://github.com/facebookresearch/ConvNeXt

## Structure

- `train_convnext.py`: scratch, linear-probe, and full fine-tuning
- `evaluate_convnext.py`: test metrics, predictions, and confusion analysis
- `plot_training_metrics.py`: training/validation curves
- `convnext_classifier.py`: model and optimizer construction
- `data.py`: transforms, dataset checks, and data loaders
- `engine.py`: training and evaluation loops
- `visualization.py`: plotting helpers
- `outputs/`: local experiment outputs; checkpoints should not be committed

### Training from Scratch

```powershell
python -m models.convnext.train_convnext `
  --mode scratch `
  --epochs 100 `
  --batch-size 64 `
  --num-workers 4 `
  --warmup-epochs 5 `
  --early-stopping-patience 15 `
  --backbone-lr 0.001 `
  --classifier-lr 0.001 `
  --seed 42 `
  --output-dir models/convnext/outputs/final_scratch_seed42 `
  --device cuda
```

### Linear probe

```powershell
python -m models.convnext.train_convnext `
  --mode linear_probe `
  --epochs 30 `
  --batch-size 64 `
  --num-workers 4 `
  --warmup-epochs 2 `
  --classifier-lr 0.0005 `
  --early-stopping-patience 5 `
  --seed 42 `
  --output-dir models/convnext/outputs/final_linear_probe_seed42 `
  --device cuda
```

### Full fine-tuning

```powershell
python -m models.convnext.train_convnext `
  --mode full_finetune `
  --epochs 30 `
  --freeze-epochs 2 `
  --warmup-epochs 2 `
  --batch-size 64 `
  --num-workers 4 `
  --backbone-lr 0.0001 `
  --classifier-lr 0.0005 `
  --early-stopping-patience 5 `
  --seed 42 `
  --output-dir models/convnext/outputs/final_full_finetune_seed42 `
  --device cuda
```

### Normal Evaluation

```powershell
python -m models.convnext.evaluate_convnext `
  --checkpoint models/convnext/outputs/final_full_finetune_seed42/best_checkpoint.pt `
  --device cuda
```
### Robustness evaluation

The ConvNeXt-Tiny full fine-tuning checkpoint was evaluated on five image degradations:

- Gaussian noise
- Gaussian blur
- Motion blur
- Brightness reduction
- JPEG compression

Each degradation contains four severity levels, therefore there are 20 robustness test sets. Each test set contains 5000 images across 500 classes.

```powershell
python -m models.convnext.evaluate_robustness `
  --dataset-dir dataset `
  --robustness-dir dataset/robustness_data `
  --checkpoint models/convnext/outputs/final_full_finetune_seed42/best_checkpoint.pt `
  --output-dir models/convnext/outputs/final_full_finetune_seed42/robustness `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda
```


### Plot training curves

```powershell
python -m models.convnext.plot_training_metrics `
  --metrics models/convnext/outputs/final_full_finetune_seed42/metrics.jsonl `
  --output-dir models/convnext/outputs/final_full_finetune_seed42
```

## Command-line Arguments

### Training

- `--mode`: training setting: `scratch`, `linear_probe`, or `full_finetune`
- `--dataset-dir`: path to the processed dataset
- `--output-dir`: directory used to save the outputs
- `--epochs`: maximum number of training epochs
- `--freeze-epochs`: number of initial frozen backbone epochs for full fine-tuning
- `--batch-size`: training batch size
- `--num-workers`: number of DataLoader workers
- `--backbone-lr`: learning rate for the ConvNeXt backbone
- `--classifier-lr`: learning rate for the classifier
- `--weight-decay`: AdamW weight decay
- `--label-smoothing`: label smoothing used during training
- `--warmup-epochs`: number of learning-rate warmup epochs
- `--early-stopping-patience`: epochs without sufficient validation improvement before it terminates
- `--early-stopping-min-delta`: minimum validation Top-1 improvement
- `--seed`: random seed
- `--device`: `auto`, `cpu`, or `cuda`
- `--amp` / `--no-amp`: enable or disable automatic mixed precision

### Evaluation

- `--dataset-dir`: path to the processed dataset
- `--checkpoint`: path to the checkpoint used for test evaluation
- `--output-dir`: directory used to save test results
- `--batch-size`: evaluation batch size
- `--num-workers`: number of DataLoader workers
- `--device`: `auto`, `cpu`, or `cuda`
- `--top-confusions`: number of class-confusion pairs saved

## Outputs

Each training experiment produces:

- `best_checkpoint.pt`: checkpoint with the highest validation Top-1 accuracy
- `best_val_loss_checkpoint.pt`: checkpoint with the lowest validation loss
- `latest_checkpoint.pt`: checkpoint from the final completed epoch
- `config.json`: model, dataset and training configuration
- `metrics.jsonl`: per-epoch training and validation metrics
- `training_summary.json`: contains the best validation result, best epoch, runtime and early-stopping information
- `training_validation_loss.png`: training and validation loss curves
- `training_validation_top1.png`: training and validation Top-1 accuracy curves
- `training_validation_top5.png`: training and validation Top-5 accuracy curves

Test evaluation also produces:

- `test_metrics.json`: overall test metrics and inference runtime
- `test_predictions.csv`: prediction and confidence for each test image
- `per_class_metrics.csv`: precision, recall, F1-score and support for each class
- `top_confusions.csv`: most frequent directional class confusions
- `confusion_matrix.csv`: full confusion matrix
- `confusion_matrix.npy`: NumPy version of the confusion matrix
- `normalized_confusion_matrix.png`: normalized confusion matrix visualization
- `top_confused_pairs.png`: visualization of the most frequent class confusions

Robustness evaluation produces:

- `robustness_metrics.csv`
- `robustness_metrics.json`
- `robustness_run_metadata.json`

The reported metrics include cross entropy loss, Top-1 accuracy, Top-5 accuracy, macro precision, macro recall, macro F1, inference time, and throughput.

Note that Gaussian blur caused the largest reduction in classification performance, while brightness reduction had the smallest effect.

## Notes

- Training uses label-smoothed cross-entropy.
- Validation and test use ordinary cross-entropy.
- `best_checkpoint.pt` is selected by validation top-1 accuracy.
- `best_val_loss_checkpoint.pt` is saved as a separate checkpoint.
- `training_summary.json` records `best_val_epoch`.
- Early stopping may terminate before the maximum epoch count.
