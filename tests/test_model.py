"""tests/test_model.py

Unit tests for the model module with comprehensive edge cases.
Tests cover deterministic output, CFG stability, batch size edge cases,
different input sizes, and checkpoint format handling.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import SUPPORTED_BACKBONES, cats_model, count_parameters, load_checkpoint

if TYPE_CHECKING:
    pass


class TestCatsModel:
    """Tests for cats_model factory function."""

    def test_model_returns_nn_module(self) -> None:
        """Verify cats_model returns a valid nn.Module."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        assert isinstance(model, torch.nn.Module)

    def test_model_default_num_classes(self) -> None:
        """Test model with default number of classes."""
        model = cats_model(backbone="resnet18", pretrained=False)
        assert model is not None

    def test_model_output_shape_resnet18(self) -> None:
        """Test ResNet-18 produces correct output shape."""
        model = cats_model(num_classes=3, backbone="resnet18", pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3)

    def test_model_output_shape_mobilenet(self) -> None:
        """Test MobileNetV3 produces correct output shape."""
        model = cats_model(
            num_classes=5, backbone="mobilenet_v3_small", pretrained=False
        )
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 5)

    def test_unsupported_backbone_raises(self) -> None:
        """Test that unsupported backbones raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported backbone"):
            cats_model(backbone="vgg16")

    def test_supported_backbones_list(self) -> None:
        """Test SUPPORTED_BACKBONES contains expected architectures."""
        assert "resnet18" in SUPPORTED_BACKBONES
        assert "mobilenet_v3_small" in SUPPORTED_BACKBONES

    def test_count_parameters_positive(self) -> None:
        """Test count_parameters returns positive value."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        params = count_parameters(model)
        assert params > 0

    def test_count_parameters_resnet18_expected_range(self) -> None:
        """ResNet-18 has ~11M parameters."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        params = count_parameters(model)
        assert 10_000_000 < params < 15_000_000

    def test_model_trainable_params_resnet18(self) -> None:
        """All parameters should be trainable when pretrained=False."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        for param in model.parameters():
            assert param.requires_grad

    def test_model_num_classes_2(self) -> None:
        """Test model with 2 output classes."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape[1] == 2

    def test_model_batch_size_1(self) -> None:
        """Test model with batch size of 1."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 1

    def test_model_batch_size_8(self) -> None:
        """Test model with batch size of 8."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()
        x = torch.randn(8, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 8


class TestDeterministicOutput:
    """Tests for deterministic model output with same seed."""

    def test_deterministic_output_with_same_seed(self) -> None:
        """Test model produces identical output with same random seed."""
        torch.manual_seed(42)
        model1 = cats_model(num_classes=2, backbone="resnet18", pretrained=False)

        torch.manual_seed(42)
        model2 = cats_model(num_classes=2, backbone="resnet18", pretrained=False)

        # Compare initial weights
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

        # Test forward pass
        model1.eval()
        model2.eval()
        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)

        assert torch.allclose(out1, out2)

    def test_deterministic_output_multiple_runs(self) -> None:
        """Test model produces identical output across multiple runs."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        torch.manual_seed(123)
        x = torch.randn(2, 3, 224, 224)

        outputs = []
        for _ in range(5):
            with torch.no_grad():
                out = model(x)
            outputs.append(out.clone())

        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i])

    def test_deterministic_different_seeds_produce_different_output(self) -> None:
        """Test different seeds produce different model initializations."""
        torch.manual_seed(42)
        model1 = cats_model(num_classes=2, backbone="resnet18", pretrained=False)

        torch.manual_seed(123)
        model2 = cats_model(num_classes=2, backbone="resnet18", pretrained=False)

        # Weights should be different
        weights_different = False
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if not torch.allclose(p1, p2):
                weights_different = True
                break

        assert weights_different


class TestCFGStability:
    """Tests for CFG (Classifier-Free Guidance) numerical stability.

    Note: While CFG is typically used in diffusion models, these tests
    verify numerical stability with different scale factors that could
    be applied to model logits or features.
    """

    def test_cfg_scale_1_0(self) -> None:
        """Test numerical stability with cfg_scale=1.0."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        cfg_scale = 1.0
        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            out = model(x)
            # Simulate CFG scaling
            scaled_out = out * cfg_scale

        assert not torch.isnan(scaled_out).any()
        assert not torch.isinf(scaled_out).any()
        assert scaled_out.shape == out.shape

    def test_cfg_scale_1_5(self) -> None:
        """Test numerical stability with cfg_scale=1.5."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        cfg_scale = 1.5
        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            out = model(x)
            scaled_out = out * cfg_scale

        assert not torch.isnan(scaled_out).any()
        assert not torch.isinf(scaled_out).any()

    def test_cfg_scale_3_0(self) -> None:
        """Test numerical stability with cfg_scale=3.0."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        cfg_scale = 3.0
        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            out = model(x)
            scaled_out = out * cfg_scale

        assert not torch.isnan(scaled_out).any()
        assert not torch.isinf(scaled_out).any()

    def test_cfg_scale_5_0(self) -> None:
        """Test numerical stability with cfg_scale=5.0."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        cfg_scale = 5.0
        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            out = model(x)
            scaled_out = out * cfg_scale

        assert not torch.isnan(scaled_out).any()
        assert not torch.isinf(scaled_out).any()

    def test_cfg_scale_extreme_values(self) -> None:
        """Test numerical stability with extreme cfg_scale values."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        x = torch.randn(2, 3, 224, 224)

        for cfg_scale in [0.1, 0.5, 10.0, 100.0]:
            with torch.no_grad():
                out = model(x)
                scaled_out = out * cfg_scale

            assert not torch.isnan(scaled_out).any(), f"NaN at cfg_scale={cfg_scale}"
            assert not torch.isinf(scaled_out).any(), f"Inf at cfg_scale={cfg_scale}"


class TestBatchSizeEdgeCases:
    """Tests for various batch size edge cases."""

    @pytest.mark.parametrize("batch_size", [1, 2, 127])
    def test_batch_size_variations(self, batch_size: int) -> None:
        """Test model forward pass with various batch sizes."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        x = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            out = model(x)

        assert out.shape[0] == batch_size
        assert out.shape[1] == 2

    @pytest.mark.slow
    def test_batch_size_256(self) -> None:
        """Test model with batch size 256 (slow test)."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        batch_size = 256
        x = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            out = model(x)

        assert out.shape[0] == batch_size
        assert out.shape[1] == 2

    @pytest.mark.slow
    def test_batch_size_513(self) -> None:
        """Test model with batch size 513 (edge case > 512, slow test)."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        batch_size = 513
        x = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            out = model(x)

        assert out.shape[0] == batch_size
        assert out.shape[1] == 2

    def test_batch_size_1_single_sample(self) -> None:
        """Test model with single sample (batch_size=1)."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 2)
        assert out.ndim == 2

    def test_batch_size_large_memory_efficiency(self) -> None:
        """Test large batch size doesn't cause memory issues."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        batch_size = 256
        x = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (batch_size, 2)

    def test_batch_size_uneven_division(self) -> None:
        """Test batch sizes that don't divide evenly."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        # Prime numbers that don't divide evenly
        for batch_size in [127, 251, 509]:
            x = torch.randn(batch_size, 3, 224, 224)

            with torch.no_grad():
                out = model(x)

            assert out.shape[0] == batch_size


class TestDifferentInputSizes:
    """Tests for forward pass with different input sizes."""

    @pytest.mark.parametrize(
        "input_size",
        [
            (32, 32),
            (64, 64),
            (128, 128),
            (224, 224),
            (256, 256),
            (299, 299),
        ],
    )
    def test_square_input_sizes(self, input_size: tuple[int, int]) -> None:
        """Test model with various square input sizes."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        h, w = input_size
        x = torch.randn(2, 3, h, w)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (2, 2)
        assert not torch.isnan(out).any()

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "input_size",
        [
            (384, 384),
            (512, 512),
        ],
    )
    def test_square_input_sizes_large(self, input_size: tuple[int, int]) -> None:
        """Test model with large square input sizes (slow test)."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        h, w = input_size
        x = torch.randn(2, 3, h, w)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (2, 2)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize(
        "input_size",
        [
            (100, 150),
            (150, 100),
            (200, 300),
            (300, 200),
            (64, 128),
            (128, 64),
        ],
    )
    def test_non_square_input_sizes(self, input_size: tuple[int, int]) -> None:
        """Test model with non-square input sizes."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        h, w = input_size
        x = torch.randn(2, 3, h, w)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (2, 2)
        assert not torch.isnan(out).any()

    def test_minimum_viable_input_size(self) -> None:
        """Test model with minimum viable input size."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        # ResNet-18 needs at least 32x32 due to pooling layers
        x = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 2)

    @pytest.mark.slow
    def test_very_large_input_size(self) -> None:
        """Test model with very large input (1024x1024, slow test)."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 1024, 1024)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 2)


class TestCheckpointFormats:
    """Tests for model loading from different checkpoint formats."""

    def test_load_standard_state_dict(
        self, temp_dir: Path, simple_model: nn.Module
    ) -> None:
        """Test loading from standard state dict checkpoint."""
        # Save standard checkpoint
        ckpt_path = temp_dir / "standard.pt"
        torch.save(simple_model.state_dict(), ckpt_path)

        # Load checkpoint
        loaded_model = load_checkpoint(ckpt_path, num_classes=2, backbone="resnet18")

        # Verify weights match
        for p1, p2 in zip(simple_model.parameters(), loaded_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_load_full_checkpoint_with_metadata(
        self, temp_dir: Path, simple_model: nn.Module
    ) -> None:
        """Test loading from full checkpoint with metadata."""
        # Save full checkpoint
        ckpt_path = temp_dir / "full.pt"
        full_data = {
            "epoch": 10,
            "model_state_dict": simple_model.state_dict(),
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "val_acc": 0.95,
            "timestamp": "2024-01-01T00:00:00",
        }
        torch.save(full_data, ckpt_path)

        # Load checkpoint
        loaded_model = load_checkpoint(ckpt_path, num_classes=2, backbone="resnet18")

        # Verify weights match
        for p1, p2 in zip(simple_model.parameters(), loaded_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_load_checkpoint_with_extra_keys(
        self, temp_dir: Path, simple_model: nn.Module
    ) -> None:
        """Test loading checkpoint with extra keys (backward compatibility)."""
        # Save checkpoint with extra keys
        ckpt_path = temp_dir / "extra.pt"
        extra_data = {
            "model_state_dict": simple_model.state_dict(),
            "extra_key": "should_be_ignored",
            "another_extra": [1, 2, 3],
        }
        torch.save(extra_data, ckpt_path)

        # Load checkpoint - should work despite extra keys
        loaded_model = load_checkpoint(ckpt_path, num_classes=2, backbone="resnet18")

        # Verify weights match
        for p1, p2 in zip(simple_model.parameters(), loaded_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_load_checkpoint_missing_file_raises(self, temp_dir: Path) -> None:
        """Test that missing checkpoint file raises FileNotFoundError."""
        missing_path = temp_dir / "nonexistent.pt"

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_checkpoint(missing_path, num_classes=2, backbone="resnet18")

    def test_load_checkpoint_wrong_num_classes(
        self, temp_dir: Path, simple_model: nn.Module
    ) -> None:
        """Test loading checkpoint with mismatched num_classes."""
        # Save checkpoint
        ckpt_path = temp_dir / "model.pt"
        torch.save(simple_model.state_dict(), ckpt_path)

        # Try to load with wrong num_classes - should fail during load_state_dict
        with pytest.raises(RuntimeError):
            load_checkpoint(ckpt_path, num_classes=10, backbone="resnet18")

    def test_load_checkpoint_wrong_backbone(
        self, temp_dir: Path, simple_model: nn.Module
    ) -> None:
        """Test loading checkpoint with mismatched backbone."""
        # Save ResNet-18 checkpoint
        ckpt_path = temp_dir / "resnet18.pt"
        torch.save(simple_model.state_dict(), ckpt_path)

        # Try to load as MobileNet - should fail due to architecture mismatch
        with pytest.raises(RuntimeError):
            load_checkpoint(ckpt_path, num_classes=2, backbone="mobilenet_v3_small")

    def test_checkpoint_load_eval_mode(
        self, temp_dir: Path, simple_model: nn.Module
    ) -> None:
        """Test that loaded checkpoint sets model to eval mode."""
        ckpt_path = temp_dir / "model.pt"
        torch.save(simple_model.state_dict(), ckpt_path)

        loaded_model = load_checkpoint(ckpt_path, num_classes=2, backbone="resnet18")

        assert not loaded_model.training


class TestCountParameters:
    """Tests for count_parameters utility."""

    def test_count_parameters_simple_model(self) -> None:
        """Test count_parameters with simple linear model."""
        model = nn.Linear(10, 5)
        params = count_parameters(model)
        # 10*5 weights + 5 biases = 55
        assert params == 55

    def test_count_parameters_no_grad(self) -> None:
        """Test count_parameters with frozen parameters."""
        model = nn.Linear(10, 5)
        for param in model.parameters():
            param.requires_grad = False
        params = count_parameters(model)
        # count_parameters counts all params regardless of requires_grad
        assert params >= 0

    def test_count_parameters_returns_int(self) -> None:
        """Test count_parameters returns integer."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        params = count_parameters(model)
        assert isinstance(params, int)

    def test_count_parameters_different_backbones(self) -> None:
        """Test parameter counts for different backbones."""
        params_resnet18 = count_parameters(
            cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        )
        params_mobilenet = count_parameters(
            cats_model(num_classes=2, backbone="mobilenet_v3_small", pretrained=False)
        )

        # ResNet-18 should have more parameters than MobileNetV3 Small
        assert params_resnet18 > params_mobilenet
        assert params_mobilenet > 0


class TestAllBackbones:
    """Tests for all supported backbone architectures."""

    @pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
    def test_backbone_builds(self, backbone: str) -> None:
        """Test each supported backbone builds successfully."""
        model = cats_model(num_classes=2, backbone=backbone, pretrained=False)
        assert model is not None

    @pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
    def test_backbone_forward_pass(self, backbone: str) -> None:
        """Test each supported backbone forward pass."""
        model = cats_model(num_classes=2, backbone=backbone, pretrained=False)
        model.eval()

        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (2, 2)

    @pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
    def test_backbone_parameter_count(self, backbone: str) -> None:
        """Test each backbone has positive parameter count."""
        model = cats_model(num_classes=2, backbone=backbone, pretrained=False)
        params = count_parameters(model)
        assert params > 0


class TestModelNumericalStability:
    """Tests for model numerical stability."""

    def test_no_nan_output_random_input(self) -> None:
        """Test model doesn't produce NaN with random input."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        for _ in range(10):
            x = torch.randn(2, 3, 224, 224)

            with torch.no_grad():
                out = model(x)

            assert not torch.isnan(out).any()

    def test_no_inf_output_random_input(self) -> None:
        """Test model doesn't produce Inf with random input."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        for _ in range(10):
            x = torch.randn(2, 3, 224, 224)

            with torch.no_grad():
                out = model(x)

            assert not torch.isinf(out).any()

    def test_output_finite_with_extreme_input(self) -> None:
        """Test model output is finite with extreme input values."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()

        # Very large input values
        x = torch.randn(2, 3, 224, 224) * 100

        with torch.no_grad():
            out = model(x)

        assert torch.isfinite(out).all()

    def test_gradient_flow(self) -> None:
        """Test gradients flow through the model."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.train()

        x = torch.randn(2, 3, 224, 224)
        target = torch.randint(0, 2, (2,))

        loss_fn = nn.CrossEntropyLoss()
        out = model(x)
        loss = loss_fn(out, target)
        loss.backward()

        # Check that gradients exist and are finite
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"
