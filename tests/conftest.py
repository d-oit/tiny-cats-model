"""tests/conftest.py

Shared pytest fixtures and utilities for the tiny-cats-model test suite.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import cats_model


@pytest.fixture(scope="session")
def test_device() -> torch.device:
    """Return CPU device for all tests (no GPU required)."""
    return torch.device("cpu")


@pytest.fixture
def test_logger() -> logging.Logger:
    """Create a test logger with no handlers to avoid output pollution."""
    logger = logging.getLogger(f"test_logger_{id(logging.getLogger())}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    # Add a null handler to prevent "No handler" warnings
    logger.addHandler(logging.NullHandler())
    return logger


@pytest.fixture
def simple_model(test_device: torch.device) -> nn.Module:
    """Small resnet18-based model for testing."""
    model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
    return model.to(test_device)


@pytest.fixture
def mobilenet_model(test_device: torch.device) -> nn.Module:
    """MobileNetV3 model for testing alternative backbones."""
    model = cats_model(num_classes=2, backbone="mobilenet_v3_small", pretrained=False)
    return model.to(test_device)


@pytest.fixture
def tiny_dataloader(test_device: torch.device) -> DataLoader:
    """Tiny dataloader with synthetic data for quick tests."""
    x = torch.randn(8, 3, 224, 224)
    y = torch.randint(0, 2, (8,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=4)


@pytest.fixture
def single_batch_loader(test_device: torch.device) -> DataLoader:
    """Dataloader with exactly one batch for edge case testing."""
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 2, (4,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=4)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_image_dataset(temp_dir: Path) -> Path:
    """Create a temporary ImageFolder-compatible dataset structure.

    Creates:
        temp_dir/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                img1.jpg
    """
    # Create class directories
    class_a_dir = temp_dir / "class_a"
    class_b_dir = temp_dir / "class_b"
    class_a_dir.mkdir(parents=True, exist_ok=True)
    class_b_dir.mkdir(parents=True, exist_ok=True)

    # Create sample images
    img1 = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    img1.save(class_a_dir / "img1.jpg")

    img2 = Image.fromarray(np.random.randint(0, 255, (150, 80, 3), dtype=np.uint8))
    img2.save(class_a_dir / "img2.jpg")

    img3 = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
    img3.save(class_b_dir / "img1.jpg")

    return temp_dir


@pytest.fixture
def temp_image_dataset_13_classes(temp_dir: Path) -> Path:
    """Create a temporary dataset with all 13 cat breed classes.

    Creates directories for each of the 13 cat breeds with sample images.
    """
    # All 13 cat breed classes
    breeds = [
        "abyssinian",
        "bengal",
        "birman",
        "bombay",
        "british_shorthair",
        "egyptian_mau",
        "maine_coon",
        "persian",
        "ragdoll",
        "russian_blue",
        "siamese",
        "sphynx",
        "other",
    ]

    for breed in breeds:
        breed_dir = temp_dir / breed
        breed_dir.mkdir(parents=True, exist_ok=True)

        # Create 2 sample images per breed
        for i in range(2):
            img = Image.fromarray(
                np.random.randint(
                    0, 255, (100 + i * 20, 100 + i * 10, 3), dtype=np.uint8
                )
            )
            img.save(breed_dir / f"img{i}.jpg")

    return temp_dir


@pytest.fixture
def corrupt_image_file(temp_dir: Path) -> Path:
    """Create a corrupt image file for error handling tests."""
    corrupt_file = temp_dir / "corrupt.jpg"
    # Write invalid image data
    corrupt_file.write_bytes(b"not a valid image file content")
    return corrupt_file


@pytest.fixture
def empty_image_file(temp_dir: Path) -> Path:
    """Create an empty image file for error handling tests."""
    empty_file = temp_dir / "empty.jpg"
    empty_file.write_bytes(b"")
    return empty_file


@pytest.fixture
def extreme_aspect_ratio_images(temp_dir: Path) -> list[Path]:
    """Create images with extreme aspect ratios for testing.

    Returns list of paths to created images.
    """
    images = []

    # Very wide image (10:1 aspect ratio)
    wide_img = Image.fromarray(np.random.randint(0, 255, (50, 500, 3), dtype=np.uint8))
    wide_path = temp_dir / "wide.jpg"
    wide_img.save(wide_path)
    images.append(wide_path)

    # Very tall image (1:10 aspect ratio)
    tall_img = Image.fromarray(np.random.randint(0, 255, (500, 50, 3), dtype=np.uint8))
    tall_path = temp_dir / "tall.jpg"
    tall_img.save(tall_path)
    images.append(tall_path)

    # Extremely wide (50:1)
    extreme_wide = Image.fromarray(
        np.random.randint(0, 255, (20, 1000, 3), dtype=np.uint8)
    )
    extreme_wide_path = temp_dir / "extreme_wide.jpg"
    extreme_wide.save(extreme_wide_path)
    images.append(extreme_wide_path)

    # Extremely tall (1:50)
    extreme_tall = Image.fromarray(
        np.random.randint(0, 255, (1000, 20, 3), dtype=np.uint8)
    )
    extreme_tall_path = temp_dir / "extreme_tall.jpg"
    extreme_tall.save(extreme_tall_path)
    images.append(extreme_tall_path)

    return images


@pytest.fixture
def boundary_size_images(temp_dir: Path) -> list[tuple[Path, tuple[int, int]]]:
    """Create images at boundary sizes for testing.

    Returns list of (path, (height, width)) tuples.
    """
    images = []

    # Minimum viable size (1x1)
    min_img = Image.fromarray(np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8))
    min_path = temp_dir / "min_1x1.jpg"
    min_img.save(min_path)
    images.append((min_path, (1, 1)))

    # Very small (8x8)
    small_img = Image.fromarray(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8))
    small_path = temp_dir / "small_8x8.jpg"
    small_img.save(small_path)
    images.append((small_path, (8, 8)))

    # Standard size (224x224)
    std_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    std_path = temp_dir / "std_224x224.jpg"
    std_img.save(std_path)
    images.append((std_path, (224, 224)))

    # Large size (1024x1024)
    large_img = Image.fromarray(
        np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    )
    large_path = temp_dir / "large_1024x1024.jpg"
    large_img.save(large_path)
    images.append((large_path, (1024, 1024)))

    # Very large (2048x2048)
    very_large_img = Image.fromarray(
        np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    )
    very_large_path = temp_dir / "very_large_2048x2048.jpg"
    very_large_img.save(very_large_path)
    images.append((very_large_path, (2048, 2048)))

    return images


@pytest.fixture
def checkpoint_formats(temp_dir: Path, simple_model: nn.Module) -> dict[str, Path]:
    """Create checkpoints in different formats for testing.

    Returns dict mapping format name to checkpoint path.
    """
    checkpoints = {}

    # Standard state dict format
    standard_ckpt = temp_dir / "model_standard.pt"
    torch.save(simple_model.state_dict(), standard_ckpt)
    checkpoints["standard"] = standard_ckpt

    # Full checkpoint with metadata
    full_ckpt = temp_dir / "model_full.pt"
    full_data = {
        "epoch": 10,
        "model_state_dict": simple_model.state_dict(),
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "val_acc": 0.95,
        "timestamp": "2024-01-01T00:00:00",
    }
    torch.save(full_data, full_ckpt)
    checkpoints["full"] = full_ckpt

    # Checkpoint with extra keys (backward compatibility test)
    extra_ckpt = temp_dir / "model_extra.pt"
    extra_data = {
        "model_state_dict": simple_model.state_dict(),
        "extra_key": "should_be_ignored",
        "another_extra": [1, 2, 3],
    }
    torch.save(extra_data, extra_ckpt)
    checkpoints["extra_keys"] = extra_ckpt

    return checkpoints


@pytest.fixture
def batch_size_test_cases() -> list[int]:
    """Return batch sizes for edge case testing."""
    return [1, 2, 127, 256, 513]


@pytest.fixture
def cfg_scale_values() -> list[float]:
    """Return CFG scale values for numerical stability testing."""
    return [1.0, 1.5, 3.0, 5.0]


@pytest.fixture
def input_size_variations() -> list[tuple[int, int]]:
    """Return input size variations for forward pass testing."""
    return [
        (32, 32),  # Very small
        (64, 64),  # Small
        (128, 128),  # Medium-small
        (224, 224),  # Standard
        (256, 256),  # Medium-large
        (299, 299),  # Inception size
        (384, 384),  # Large
        (512, 512),  # Very large
        (100, 150),  # Non-square
        (150, 100),  # Non-square (rotated)
    ]


@pytest.fixture
def optimizer_factory():
    """Factory function to create optimizers for testing."""

    def _create_optimizer(
        model: nn.Module, lr: float = 1e-3, optimizer_type: str = "adam"
    ) -> torch.optim.Optimizer:
        if optimizer_type == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_type == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_type == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return _create_optimizer


@pytest.fixture
def lr_scheduler_factory():
    """Factory function to create learning rate schedulers for testing."""

    def _create_scheduler(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = "cosine",
        total_epochs: int = 10,
        warmup_epochs: int = 2,
    ):
        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs
            )
        elif scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        elif scheduler_type == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
        elif scheduler_type == "sequential":
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            main = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - warmup_epochs
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, main], milestones=[warmup_epochs]
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return _create_scheduler


@pytest.fixture
def synthetic_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a synthetic batch for testing."""
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, 2, (4,))
    return x, y


@pytest.fixture
def loss_fn() -> nn.Module:
    """Create a CrossEntropyLoss for testing."""
    return nn.CrossEntropyLoss()
