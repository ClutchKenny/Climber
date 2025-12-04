import torch.nn as nn
from torchvision import models

# ResNet18 based classifier for overgripping vs not overgripping
def create_model(num_classes, device):
    model = models.resnet18(weights="IMAGENET1K_V1")

    # Freeze backbone first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layers 3 and 4
    for name, param in model.named_parameters():
        if name.startswith("layer3.") or name.startswith("layer4."):
            param.requires_grad = True

    # Replace classifier head with larger dropout head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(128, num_classes),
    )

    return model.to(device)