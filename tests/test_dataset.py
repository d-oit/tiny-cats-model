"""tests/test_dataset.py

Unit tests for the dataset module.
These tests do NOT require a real dataset - they test transforms, utilities, and error handling.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch

from dataset import DEFAULT_IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_transforms
from model import SUPPORTED_BACKBONES, cats_model, count_parameters


class TestBuildTransforms:
    """Tests for build_transforms()."""

    def test_train_transforms_returns_compose(self):
        import torchvision.transforms as T

        transform = build_transforms(train=True)
        assert isinstance(transform, T.Compose)

    def test_val_transforms_returns_compose(self):
        import torchvision.transforms as T

        transform = build_transforms(train=False)
        assert isinstance(transform, T.Compose)

    def test_train_transforms_process_image(self):
        """Train transforms should process a PIL image to a tensor."""
        import numpy as np
        from PIL import Image

        transform = build_transforms(train=True, image_size=64)
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        tensor = transform(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 64, 64)

    def test_val_transforms_process_image(self):
        """Val transforms should process a PIL image to a tensor."""
        import numpy as np
        from PIL import Image

        transform = build_transforms(train=False, image_size=64)
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        tensor = transform(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 64, 64)

    def test_default_image_size(self):
        assert DEFAULT_IMAGE_SIZE == 224

    def test_imagenet_normalization_constants(self):
        assert len(IMAGENET_MEAN) == 3
        assert len(IMAGENET_STD) == 3
        assert all(0 < v < 1 for v in IMAGENET_MEAN)
        assert all(0 < v < 1 for v in IMAGENET_STD)


class TestCatsModel:
    """Tests for the cats_model() factory."""

    def test_resnet18_builds(self):
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        assert model is not None

    def test_resnet18_output_shape(self):
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)

    def test_custom_num_classes(self):
        model = cats_model(num_classes=12, backbone="resnet18", pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 12)

    def test_mobilenet_builds(self):
        model = cats_model(
            num_classes=2, backbone="mobilenet_v3_small", pretrained=False
        )
        assert model is not None

    def test_unsupported_backbone_raises(self):
        with pytest.raises(ValueError, match="Unsupported backbone"):
            cats_model(backbone="vgg16")

    def test_supported_backbones_list(self):
        assert "resnet18" in SUPPORTED_BACKBONES
        assert "mobilenet_v3_small" in SUPPORTED_BACKBONES

    def test_count_parameters_positive(self):
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        params = count_parameters(model)
        assert params > 0

    def test_count_parameters_resnet18_expected_range(self):
        """ResNet-18 has ~11M parameters."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        params = count_parameters(model)
        assert 10_000_000 < params < 15_000_000


class TestDatasetErrors:
    """Tests for error handling in the dataset module."""

    def test_cats_dataloader_missing_dir_raises(self):
        from dataset import cats_dataloader

        with pytest.raises(FileNotFoundError):
            cats_dataloader(root="/nonexistent/path/12345")
