# ConvNeXt-Tiny

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

### Evaluation

```powershell
python -m models.convnext.evaluate_convnext `
  --checkpoint models/convnext/outputs/final_full_finetune_seed42/best_checkpoint.pt `
  --device cuda
```

### Plot training curves

```powershell
python -m models.convnext.plot_training_metrics `
  --metrics models/convnext/outputs/final_full_finetune_seed42/metrics.jsonl `
  --output-dir models/convnext/outputs/final_full_finetune_seed42
```

## Notes

- Training uses label-smoothed cross-entropy.
- Validation and test use ordinary cross-entropy.
- `best_checkpoint.pt` is selected by validation top-1 accuracy.
- `best_val_loss_checkpoint.pt` is saved as a separate checkpoint.
- `training_summary.json` records `best_val_epoch`.
- Early stopping may terminate before the maximum epoch count.
