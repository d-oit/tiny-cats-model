"""src/model.py

Model definition for cats classification.
Uses a pretrained ResNet-18 backbone with a custom classification head.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torchvision.models as models

from model_contract import (
    CLASSIFIER_LABELS,
    ContractError,
    classifier_metadata,
    metadata_from_checkpoint,
    validate_classifier_metadata,
)
from safe_checkpoint import load_torch_checkpoint

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
        raise ValueError(
            f"Unsupported backbone: {backbone}. Choose from {SUPPORTED_BACKBONES}"
        )

    return model


def _state_dict_from_checkpoint(checkpoint: object) -> dict[str, torch.Tensor]:
    """Extract a model state dict from supported checkpoint envelopes."""
    if not isinstance(checkpoint, dict):
        raise ContractError("checkpoint must contain a state dictionary")
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise ContractError("checkpoint contains no model state dictionary")


def _classifier_shape(state_dict: dict[str, torch.Tensor]) -> tuple[str, int]:
    """Infer backbone and class count from classifier tensors and backbone keys."""
    weight = state_dict.get("fc.1.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        features = int(weight.shape[1])
        if features == 2048:
            backbone = "resnet50"
        elif "layer4.2.conv1.weight" in state_dict:
            backbone = "resnet34"
        else:
            backbone = "resnet18"
        return backbone, int(weight.shape[0])
    for key in ("classifier.3.1.weight", "classifier.1.weight"):
        weight = state_dict.get(key)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            return "mobilenet_v3_small", int(weight.shape[0])
    weight = state_dict.get("fc.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        backbone = "resnet50" if weight.shape[1] == 2048 else "resnet18"
        return backbone, int(weight.shape[0])
    raise ContractError("could not infer classifier architecture from final layer")


def load_checkpoint(
    path: str | Path,
    num_classes: int | None = None,
    backbone: str | None = None,
    allow_unsafe_legacy: bool = False,
) -> nn.Module:
    """Safely load and strictly validate a classifier checkpoint."""
    return load_checkpoint_with_metadata(
        path, num_classes, backbone, allow_unsafe_legacy
    )[0]


def load_checkpoint_with_metadata(
    path: str | Path,
    num_classes: int | None = None,
    backbone: str | None = None,
    allow_unsafe_legacy: bool = False,
) -> tuple[nn.Module, dict[str, object]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = load_torch_checkpoint(
        checkpoint_path, map_location="cpu", allow_unsafe_legacy=allow_unsafe_legacy
    )
    state_dict = _state_dict_from_checkpoint(checkpoint)
    inferred_backbone, inferred_classes = _classifier_shape(state_dict)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    config = config if isinstance(config, dict) else {}
    metadata = metadata_from_checkpoint(checkpoint)
    arch = metadata.get("architecture", {}) if metadata else {}
    if metadata is not None and not isinstance(arch, dict):
        raise ContractError("classifier metadata architecture must be an object")
    config_classes = config.get("num_classes")
    config_architecture = config.get("architecture")
    config_backbone = config.get("backbone")
    if isinstance(config_architecture, dict):
        config_classes = config_classes or config_architecture.get("num_classes")
        config_backbone = config_backbone or config_architecture.get("name")
    if metadata is not None and config_classes is not None:
        if int(cast(int | str, config_classes)) != int(arch.get("num_classes", -1)):
            raise ContractError("classifier metadata and config num_classes disagree")
    if metadata is not None and config_backbone is not None:
        if str(config_backbone) != str(arch.get("name", "")):
            raise ContractError("classifier metadata and config backbone disagree")
    resolved_class_value = (
        num_classes
        if num_classes is not None
        else arch.get("num_classes", config_classes or inferred_classes)
    )
    if isinstance(resolved_class_value, bool) or not isinstance(
        resolved_class_value, (int, str)
    ):
        raise ContractError("classifier metadata is missing a valid num_classes")
    resolved_classes = int(resolved_class_value)
    resolved_backbone = str(
        backbone
        if backbone is not None
        else arch.get("name", config_backbone or inferred_backbone)
    )
    if resolved_classes != inferred_classes or resolved_backbone != inferred_backbone:
        raise ContractError(
            "classifier architecture disagrees with checkpoint tensors: "
            f"requested/configured {resolved_backbone}/{resolved_classes}, "
            f"tensors imply {inferred_backbone}/{inferred_classes}"
        )
    if metadata is not None:
        validate_classifier_metadata(
            metadata,
            expected_num_classes=resolved_classes,
            expected_backbone=resolved_backbone,
            expected_labels=(
                CLASSIFIER_LABELS
                if resolved_classes == len(CLASSIFIER_LABELS)
                else None
            ),
        )
    elif resolved_classes == len(CLASSIFIER_LABELS):
        metadata = classifier_metadata(
            backbone=resolved_backbone, num_classes=resolved_classes
        )
    else:
        metadata = {
            "architecture": {
                "name": resolved_backbone,
                "num_classes": resolved_classes,
            }
        }
    model = cats_model(
        num_classes=resolved_classes, backbone=resolved_backbone, pretrained=False
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, metadata


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
