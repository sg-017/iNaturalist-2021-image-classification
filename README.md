# iNaturalist-2021-image-classification 

## Dataset
iNaturalist-2021 (iNat2021)
https://github.com/visipedia/inat_comp/tree/master/2021


## Report Link
https://www.overleaf.com/1795634125pdvrvxgzhgcn#c7d8f7


## Our Design
- **(Lily)** Traditional method
    - LBP + HOG + SVM
    - LBP + HOG + RF
    - Bag-of-Visual-Words + SIFT features + SVM
- Deep learning method
    - **(Jiani)** ResNet50
        - training from scratch
        - ImageNet-1K pretrain linear-probing
        - ImageNet-1K full fine-tuning
    - **(Rui)** ConvNeXt-Tiny
        - training from scratch
        - ImageNet-22K pretrain linear-probing
        - ImageNet-22K full fine-tuning
- **(Lanli)** Explainability with Grad-CAM
- **(Lanli)** Robustness
- **(Shutian)** Fine-grained ResNet50
    - CBAM
    - MC_Loss
    - CBP
    - API-Net
    - PMG
    - iSQRT-COV


## References

- **iNaturalist-2021**: [Dataset](https://github.com/visipedia/inat_comp/tree/master/2021) | [Paper](https://arxiv.org/abs/2103.16483v2) | [Survey](https://arxiv.org/abs/2111.06119v2)
- **ResNet50**: [Hugging Face](https://huggingface.co/microsoft/resnet-50) | [Paper](https://arxiv.org/abs/1512.03385)
- **ConvNeXt-Tiny**: [Hugging Face](https://huggingface.co/timm/convnext_tiny.fb_in22k) | [Paper](https://arxiv.org/abs/2201.03545)
- **DINOv2**: [Hugging Face](https://huggingface.co/facebook/dinov2-small) | [Paper](https://arxiv.org/abs/2304.07193)
- **Grad-CAM**: [Code](https://github.com/jacobgil/pytorch-grad-cam) | [Paper](https://arxiv.org/abs/1610.02391)
- **CBAM**: [Code](https://github.com/Jongchan/attention-module) | [Paper](https://arxiv.org/abs/1807.06521)
- **MC-Loss**: [Code](https://github.com/PRIS-CV/Mutual-Channel-Loss) | [Paper](https://arxiv.org/abs/2002.04264)
- **CBP**: [Code](https://github.com/gdlg/pytorch_compact_bilinear_pooling) | [Paper](https://arxiv.org/abs/1511.06062)
- **API-Net**: [Code](https://github.com/mul-hjh/API-Net) | [Paper](https://arxiv.org/abs/2002.10191)
- **PMG**: [Code](https://github.com/PRIS-CV/PMG-Progressive-Multi-Granularity-Training) | [Paper](https://arxiv.org/abs/2003.03836)
- **iSQRT-COV**: [Code](https://github.com/jiangtaoxie/fast-MPN-COV) | [Paper](https://arxiv.org/abs/1712.01034)

