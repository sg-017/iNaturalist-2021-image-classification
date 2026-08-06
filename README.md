<div align="center">
    <h1>iNaturalist 2021 Image Classification</h1>

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-iNaturalist%202021-74AC00)](https://github.com/visipedia/inat_comp/tree/master/2021)

</div>

This project focuses on fine-grained image classification using the iNaturalist 2021 dataset. We combine traditional handcrafted features, supervised convolutional networks, self-supervised vision transformers, and specialized fine-grained learning techniques to distinguish visually similar species. The project also explores knowledge distillation and model ensembling, evaluates robustness under common image degradations, and uses Grad-CAM to explain model predictions.

Our study provides a unified empirical comparison of traditional machine-learning methods, standard deep-learning baselines, and fine-grained recognition methods on the same 500-class benchmark. The experiments examine training from scratch, linear probing, full fine-tuning, attention mechanisms, second-order feature representations, pairwise interactions, multi-granularity learning, and ensembles selected using validation data. The results demonstrate substantial gains from pretrained representations and fine-grained modeling. DINOv2 achieves the highest Top-1 accuracy, while the FGVC ensemble produces the strongest result among the ResNet-50-based methods.


## Dataset
[iNaturalist 2021](https://github.com/visipedia/inat_comp/tree/master/2021) is a large-scale fine-grained visual classification benchmark consisting of over 2.7 million images from 10,000 species collected from real-world observations. The dataset presents substantial challenges due to high inter-class similarity and large intra-class variation. 
We use the iNaturalist2021-mini version, which provides a balanced subset with 60 images per class. For computational efficiency, our experiments are conducted on a randomly selected subset of 500 classes while preserving the original class balance. The 60 images for each selected class are partitioned into 40 training, 10 validation, and 10 test samples.


## Project design
- **Traditional machine learning**
    - **Feature extraction:** LBP (Local Binary Pattern), HOG (Histogram of Oriented Gradients), SIFT (Scale-Invariant Feature Transform), and BoVW (Bag of Visual Words).
    - **Classifiers:** Random Forest and linear SVM classifiers.
- **Deep-learning baselines**
    - **ResNet-50:** Evaluated under three settings: training from scratch, linear probing with ImageNet-1K pretraining, and full fine-tuning with ImageNet-1K pretraining.
    - **ConvNeXt-Tiny:** Evaluated under three settings: training from scratch, linear probing with ImageNet-22K pretraining, and full fine-tuning with ImageNet-22K pretraining.
    - **DINOv2 ViT-S/14:** Fully fine-tuned using weights obtained through self-supervised pretraining on LVD-142M.
- **Fine-grained ResNet-50 methods**
    - **CBAM:** Applies channel and spatial attention to emphasize discriminative visual regions.
    - **Mutual-Channel Loss:** Encourages feature channels to capture diverse and class-discriminative patterns.
    - **Compact Bilinear Pooling:** Models second-order feature interactions while avoiding the full dimensionality of bilinear representations.
    - **API-Net:** Learns subtle inter-class differences through pairwise interactions between image features.
    - **Progressive Multi-Granularity Training:** Uses progressively finer image regions to learn features at multiple levels of granularity.
    - **iSQRT-COV:** Uses iterative matrix square-root normalization of covariance features for second-order representation learning.
    - **Knowledge distillation:** Transfers knowledge from multiple teacher models to a student model.
- **Model analysis**
    - **Robustness evaluation:** Measures performance under Gaussian noise, Gaussian blur, motion blur, brightness reduction, and JPEG compression at multiple severity levels.
    - **Grad-CAM explainability:** Visualizes the image regions used by different models and compares their attention on a shared deterministic sample.

## Results

| Model | Setting | Pretrain | Acc@1 | Acc@5 | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Traditional ML** |  |  |  |  |  |  |  |
| RF (LBP + HOG) | Default | None | 2.44% | 7.52% | 1.79% | 2.44% | 1.89% |
| SVM (LBP + HOG) | Default | None | 2.52% | 8.52% | 2.31% | 2.52% | 1.99% |
| SVM (BoW-SIFT) | Default | None | 4.94% | 12.34% | 4.75% | 4.94% | 3.59% |
| **ResNet** |  |  |  |  |  |  |  |
| ResNet-50 | Training from scratch | Random | 19.80% | 41.62% | 23.92% | 19.80% | 19.20% |
| ResNet-50 | Linear probing | ImageNet-1K | 59.76% | 81.58% | 62.07% | 59.76% | 59.63% |
| ResNet-50 | Full fine-tuning | ImageNet-1K | 67.26% | 85.84% | 70.98% | 67.26% | 67.22% |
| **ResNet FGVC** |  |  |  |  |  |  |  |
| ResNet-50 | MC-Loss + CBAM | CBAM checkpoint | 75.54% | 89.30% | 76.93% | 75.54% | 75.31% |
| ResNet-50 | CBP + API-Net | CBP checkpoint | 83.28% | 94.06% | 84.48% | 83.28% | 83.22% |
| ResNet-50 | PMG + iSQRT-COV | PMG checkpoint | 84.92% | 94.74% | 86.26% | 84.92% | 84.82% |
| ResNet-50 | Distillation | CBP checkpoint | 83.28% | 93.96% | 84.55% | 83.28% | 83.28% |
| ResNet-50 | Ensemble | Multiple checkpoints | 86.70% | 95.34% | 87.76% | 86.70% | 86.63% |
| **ConvNeXt** |  |  |  |  |  |  |  |
| ConvNeXt-Tiny | Training from scratch | Random | 23.12% | 42.24% | 22.59% | 23.12% | 21.91% |
| ConvNeXt-Tiny | Linear probing | ImageNet-22K | 85.64% | 96.36% | 87.02% | 85.64% | 85.50% |
| ConvNeXt-Tiny | Full fine-tuning | ImageNet-22K | 88.38% | **96.64%** | 89.33% | 88.38% | 88.28% |
| **DINOv2** |  |  |  |  |  |  |  |
| DINOv2 ViT-S/14 | Full fine-tuning | LVD-142M | **89.02%** | 96.02% | **89.88%** | **89.02%** | **88.98%** |


## Repository structure

```text
/
├── dataset/             # Dataset preparation and generated splits
├── models/
│   ├── ml/              # Traditional machine-learning methods
│   ├── resnet/          # ResNet-50 experiments
│   ├── convnext/        # ConvNeXt-Tiny experiments
│   ├── dinov2/          # DINOv2 experiments
│   └── resnet_fgvc/     # Fine-grained models and ensembles
├── explainability/      # Grad-CAM analysis
├── utils/               # Robustness utilities
├── results/             # Reports, predictions, metrics, and figures
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

### Dataset setup

Download the iNaturalist 2021 mini training set, validation set, and annotation JSON files from the [official repository](https://github.com/visipedia/inat_comp/tree/master/2021), then arrange them as follows:

```text
dataset/source_data/
├── train_mini/
├── val/
├── train_mini.json
└── val.json
```

Create the 500-class subset:

```bash
python dataset/prepare_dataset.py
```

The generated dataset contains 40 training, 10 validation, and 10 test images for each class under `dataset/processed_dataset/`.

## Running guide

Follow the README for each model family:

- [Traditional machine learning](models/ml/README.md)
- [ResNet-50](models/resnet/readme.md)
- [ConvNeXt-Tiny](models/convnext/README.md)
- [DINOv2](models/dinov2/README.md)
- [Fine-grained ResNet-50](models/resnet_fgvc/README.md)

Use the training split for optimization, the validation split for checkpoint and ensemble-weight selection, and the test split only for final evaluation.

### Robustness evaluation

Generate the selected degraded test sets with:

```bash
python utils/robustness.py
```

Configuration and usage are documented in the [robustness guide](utils/ROBUSTNESS_GUIDE.md).

### Explainability evaluation

Run the Grad-CAM smoke test or full comparison with:

```bash
python explainability/run_grad_cam.py --mode smoke --device auto
python explainability/run_grad_cam.py --mode full --device auto
```

See the [explainability guide](explainability/README.md) for details.

## References

- **iNaturalist 2021**: [Dataset](https://github.com/visipedia/inat_comp/tree/master/2021) | [Paper](https://arxiv.org/abs/2103.16483v2) | [Survey](https://arxiv.org/abs/2111.06119v2)
- **ResNet-50**: [Hugging Face](https://huggingface.co/microsoft/resnet-50) | [Paper](https://arxiv.org/abs/1512.03385)
- **ConvNeXt-Tiny**: [Hugging Face](https://huggingface.co/timm/convnext_tiny.fb_in22k) | [Paper](https://arxiv.org/abs/2201.03545)
- **DINOv2**: [Hugging Face](https://huggingface.co/facebook/dinov2-small) | [Paper](https://arxiv.org/abs/2304.07193)
- **Grad-CAM**: [Code](https://github.com/jacobgil/pytorch-grad-cam) | [Paper](https://arxiv.org/abs/1610.02391)
- **CBAM**: [Code](https://github.com/Jongchan/attention-module) | [Paper](https://arxiv.org/abs/1807.06521)
- **MC-Loss**: [Code](https://github.com/PRIS-CV/Mutual-Channel-Loss) | [Paper](https://arxiv.org/abs/2002.04264)
- **CBP**: [Code](https://github.com/gdlg/pytorch_compact_bilinear_pooling) | [Paper](https://arxiv.org/abs/1511.06062)
- **API-Net**: [Code](https://github.com/mul-hjh/API-Net) | [Paper](https://arxiv.org/abs/2002.10191)
- **PMG**: [Code](https://github.com/PRIS-CV/PMG-Progressive-Multi-Granularity-Training) | [Paper](https://arxiv.org/abs/2003.03836)
- **iSQRT-COV**: [Code](https://github.com/jiangtaoxie/fast-MPN-COV) | [Paper](https://arxiv.org/abs/1712.01034)
