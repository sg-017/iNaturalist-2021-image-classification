import timm
import torch
from torch import nn 

MODEL_NAME = "convnext_tiny.fb_in22k"
NUM_CLASSES = 500
# ConvNeXt-Tiny model and optimizer
def create_convnext_classifier(pretrained, model_name=MODEL_NAME, num_classes=NUM_CLASSES):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model, timm.data.resolve_model_data_config(model)

def classifier_parameters(model):
    classifier = model.get_classifier()
    if not isinstance(classifier, nn.Module):
        raise TypeError("Model classifier is not an nn.Module")
    return list(classifier.parameters())

# Freeze/unfreeze the backbone while always leaving the classifier trainable
def set_backbone_trainable(model, trainable):
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in classifier_parameters(model):
        parameter.requires_grad = True

def build_optimizer(model, backbone_lr, classifier_lr, weight_decay):
    classifier_ids = {id(parameter) for parameter in classifier_parameters(model)}
    backbone = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in classifier_ids
    ]
    classifier = [
        parameter for parameter in classifier_parameters(model)
        if parameter.requires_grad
    ]
    groups = []
    if backbone:
        groups.append({
            "params": backbone,
            "lr": backbone_lr,
            "base_lr": backbone_lr,
            "name": "backbone",
        })
    if classifier:
        groups.append({
            "params": classifier,
            "lr": classifier_lr,
            "base_lr": classifier_lr,
            "name": "classifier",
        })
    return torch.optim.AdamW(groups, weight_decay=weight_decay)

def optimizer_stage(model):
    classifier_ids = {id(parameter) for parameter in classifier_parameters(model)}
    backbone_trainable = any(
        parameter.requires_grad
        for parameter in model.parameters()
        if id(parameter) not in classifier_ids
    )
    return "unfrozen" if backbone_trainable else "frozen"

def trainable_parameter_count(parameters):
    return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
