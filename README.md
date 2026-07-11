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
        - ImageNet-21K pretrain linear-probing
        - ImageNet-21K full fine-tuning
- **(Lanli)** Explainability with Grad-CAM
- **(Lanli)** Robustness
- **(Shutian)** Fine-grained ResNet50
    - CBAM
    - MC_Loss
    - CBP
    - API-Net
    - PMG

