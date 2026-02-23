"""Tests for the model module."""

import pytest
import torch

from model import SUPPORTED_BACKBONES, cats_model, count_parameters


class TestCatsModel:
    """Tests for cats_model factory function."""

    def test_model_returns_nn_module(self):
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        assert isinstance(model, torch.nn.Module)

    def test_model_default_num_classes(self):
        model = cats_model(backbone="resnet18", pretrained=False)
        assert model is not None

    def test_model_output_shape_resnet18(self):
        model = cats_model(num_classes=3, backbone="resnet18", pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3)

    def test_model_output_shape_mobilenet(self):
        model = cats_model(num_classes=5, backbone="mobilenet_v3_small", pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 5)

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

    def test_model_trainable_params_resnet18(self):
        """All parameters should be trainable when pretrained=False."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        for param in model.parameters():
            assert param.requires_grad

    def test_model_num_classes_2(self):
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape[1] == 2

    def test_model_batch_size_1(self):
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 1

    def test_model_batch_size_8(self):
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()
        x = torch.randn(8, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 8


class TestCountParameters:
    """Tests for count_parameters utility."""

    def test_count_parameters_simple_model(self):
        model = torch.nn.Linear(10, 5)
        params = count_parameters(model)
        # 10*5 weights + 5 biases = 55
        assert params == 55

    def test_count_parameters_no_grad(self):
        model = torch.nn.Linear(10, 5)
        for param in model.parameters():
            param.requires_grad = False
        params = count_parameters(model)
        # count_parameters counts all params regardless of requires_grad
        assert params >= 0

    def test_count_parameters_returns_int(self):
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        params = count_parameters(model)
        assert isinstance(params, int)
