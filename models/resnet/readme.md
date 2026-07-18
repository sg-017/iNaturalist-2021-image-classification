## Running the code
### Training from scratch

```
python train_resnet50.py \
  --data-root /path/to/processed_dataset \
  --method scratch \
  --output-dir results/scratch \
  --epochs 20 \
  --batch-size 32 \
  --seed 42
```
### Linear probing

```
python train_resnet50.py \
  --data-root /path/to/processed_dataset \
  --method linear_probe \
  --output-dir results/linear_probe \
  --epochs 20 \
  --batch-size 32 \
  --seed 42

```
### Full fine-tuning

```
python train_resnet50.py \
  --data-root /path/to/processed_dataset \
  --method full_finetune \
  --output-dir results/full_finetune \
  --epochs 20 \
  --batch-size 32 \
  --seed 42

```
## Command-line Arguments

- `--data-root`: path to the dataset
- `--method`: `scratch`, `linear_probe`, or `full_finetune`
- `--output-dir`: directory used to save outputs
- `--epochs`: number of training epochs
- `--batch-size`: training batch size
- `--learning-rate`: optional custom learning rate
- `--seed`: random seed
- `--augmentation`: enables training data augmentation

## Outputs
Each experiment produces:

- `best_model.pth`: model with the highest validation accuracy
- `history.csv`: training and validation loss and accuracy
- `metrics.json`: test metrics and running time
- `loss_curve.png`: training and validation loss curve
- `accuracy_curve.png`: training and validation accuracy curve
- `confusion_matrix.npy`: confusion matrix
- `classes.json`: ordered class list