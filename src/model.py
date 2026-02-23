"""src/model.py

Model definition for cats classification.
Uses a pretrained ResNet-18 backbone with a custom classification head.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

# Supported backbone architectures
SUPPORTED_BACKBONES = ["resnet18", "resnet34", "resnet50", "mobilenet_v3_small"]


def cats_model(
    num_classes: int = 2,
    backbone: str = "resnet18",
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Build a fine-tunable cats classifier.

    Args:
        num_classes: Number of output classes (default 2: cat / other).
        backbone: Name of the torchvision backbone to use.
        pretrained: Whether to load ImageNet-pretrained weights.
        dropout: Dropout probability before the final linear layer.

    Returns:
        nn.Module ready for training.
    """
    weights = "DEFAULT" if pretrained else None

    if backbone == "resnet18":
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    elif backbone == "resnet34":
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    elif backbone == "resnet50":
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    elif backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. Choose from {SUPPORTED_BACKBONES}")

    return model


def load_checkpoint(path: str | Path, num_classes: int = 2, backbone: str = "resnet18") -> nn.Module:
    """Load a model from a saved state dict checkpoint.

    Args:
        path: Path to the .pt checkpoint file.
        num_classes: Number of output classes.
        backbone: Backbone architecture used during training.

    Returns:
        Model with loaded weights, set to eval mode.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model = cats_model(num_classes=num_classes, backbone=backbone, pretrained=False)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint from {path}")
    return model


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
