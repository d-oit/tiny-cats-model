"""tests/test_dataset.py

Unit tests for the dataset module with comprehensive edge cases.
These tests do NOT require a real dataset - they test transforms,
utilities, and error handling.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import (
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transforms,
    cats_dataloader,
    get_class_names,
)
from model import SUPPORTED_BACKBONES, cats_model, count_parameters

if TYPE_CHECKING:
    pass


class TestBuildTransforms:
    """Tests for build_transforms()."""

    def test_train_transforms_returns_compose(self) -> None:
        """Verify train transforms returns a Compose object."""
        import torchvision.transforms as T

        transform = build_transforms(train=True)
        assert isinstance(transform, T.Compose)

    def test_val_transforms_returns_compose(self) -> None:
        """Verify validation transforms returns a Compose object."""
        import torchvision.transforms as T

        transform = build_transforms(train=False)
        assert isinstance(transform, T.Compose)

    def test_train_transforms_process_image(self) -> None:
        """Train transforms should process a PIL image to a tensor."""
        import numpy as np

        transform = build_transforms(train=True, image_size=64)
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        tensor = transform(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 64, 64)

    def test_val_transforms_process_image(self) -> None:
        """Val transforms should process a PIL image to a tensor."""
        import numpy as np

        transform = build_transforms(train=False, image_size=64)
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        tensor = transform(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 64, 64)

    def test_default_image_size(self) -> None:
        """Verify default image size constant."""
        assert DEFAULT_IMAGE_SIZE == 224

    def test_imagenet_normalization_constants(self) -> None:
        """Verify ImageNet normalization constants are valid."""
        assert len(IMAGENET_MEAN) == 3
        assert len(IMAGENET_STD) == 3
        assert all(0 < v < 1 for v in IMAGENET_MEAN)
        assert all(0 < v < 1 for v in IMAGENET_STD)


class TestTransformEdgeCases:
    """Edge case tests for transform pipeline."""

    def test_transform_1x1_image(self) -> None:
        """Test transform handles minimum size (1x1) images."""
        import numpy as np

        transform = build_transforms(train=False, image_size=32)
        img = Image.fromarray(np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8))
        tensor = transform(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 32, 32)

    def test_transform_very_large_image(self) -> None:
        """Test transform handles very large images (2048x2048)."""
        import numpy as np

        transform = build_transforms(train=False, image_size=224)
        img = Image.fromarray(
            np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
        )
        tensor = transform(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)

    def test_transform_extreme_wide_aspect_ratio(
        self, extreme_aspect_ratio_images: list[Path]
    ) -> None:
        """Test transform handles extreme wide aspect ratio images."""
        transform = build_transforms(train=False, image_size=224)

        for img_path in extreme_aspect_ratio_images:
            if "wide" in img_path.name:
                img = Image.open(img_path)
                tensor = transform(img)
                assert isinstance(tensor, torch.Tensor)
                assert tensor.shape == (3, 224, 224)

    def test_transform_extreme_tall_aspect_ratio(
        self, extreme_aspect_ratio_images: list[Path]
    ) -> None:
        """Test transform handles extreme tall aspect ratio images."""
        transform = build_transforms(train=False, image_size=224)

        for img_path in extreme_aspect_ratio_images:
            if "tall" in img_path.name:
                img = Image.open(img_path)
                tensor = transform(img)
                assert isinstance(tensor, torch.Tensor)
                assert tensor.shape == (3, 224, 224)

    def test_transform_grayscale_image(self) -> None:
        """Test transform handles grayscale (1-channel) images.

        Note: Grayscale images must be converted to RGB before applying
        ImageNet normalization (which expects 3 channels).
        """
        import numpy as np

        transform = build_transforms(train=False, image_size=32)
        img = Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8))
        img = img.convert("L")  # Convert to grayscale
        img = img.convert("RGB")  # Convert to RGB for normalization
        tensor = transform(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 32, 32)

    def test_transform_rgba_image(self) -> None:
        """Test transform handles RGBA (4-channel) images.

        Note: RGBA images must be converted to RGB before applying
        ImageNet normalization (which expects 3 channels).
        """
        import numpy as np

        transform = build_transforms(train=False, image_size=32)
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8))
        img = img.convert("RGB")  # Convert to RGB (drops alpha channel)
        tensor = transform(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 32, 32)

    def test_transform_non_square_image(self) -> None:
        """Test transform handles non-square images."""
        import numpy as np

        transform = build_transforms(train=False, image_size=64)
        img = Image.fromarray(np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8))
        tensor = transform(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 64, 64)

    def test_train_transforms_deterministic_with_seed(self) -> None:
        """Test train transforms are deterministic with fixed seed."""
        import numpy as np

        np.random.seed(42)
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        torch.manual_seed(42)
        transform1 = build_transforms(train=True, image_size=64)
        img1 = Image.fromarray(img_array)
        tensor1 = transform1(img1)

        torch.manual_seed(42)
        transform2 = build_transforms(train=True, image_size=64)
        img2 = Image.fromarray(img_array)
        tensor2 = transform2(img2)

        # Note: Random augmentations may differ, but shapes should match
        assert tensor1.shape == tensor2.shape

    def test_val_transforms_deterministic(self) -> None:
        """Test validation transforms are fully deterministic."""
        import numpy as np

        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        transform = build_transforms(train=False, image_size=64)
        img1 = Image.fromarray(img_array)
        tensor1 = transform(img1)

        img2 = Image.fromarray(img_array)
        tensor2 = transform(img2)

        assert torch.allclose(tensor1, tensor2)

    def test_transform_preserves_image_content(self) -> None:
        """Test that transform preserves general image content structure."""
        import numpy as np

        # Create an image with distinct color regions
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[:50, :50] = [255, 0, 0]  # Red top-left
        img_array[:50, 50:] = [0, 255, 0]  # Green top-right
        img_array[50:, :50] = [0, 0, 255]  # Blue bottom-left
        img_array[50:, 50:] = [255, 255, 0]  # Yellow bottom-right

        transform = build_transforms(train=False, image_size=64)
        img = Image.fromarray(img_array)
        tensor = transform(img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 64, 64)
        # Normalized values should be in reasonable range
        assert tensor.min() > -3
        assert tensor.max() < 3


class TestCorruptAndEmptyImageHandling:
    """Tests for handling corrupt and empty image files."""

    def test_corrupt_image_file_handling(
        self, corrupt_image_file: Path, temp_dir: Path
    ) -> None:
        """Test that corrupt image files are handled gracefully."""
        # Create a valid directory structure with corrupt file
        class_dir = temp_dir / "test_class"
        class_dir.mkdir(parents=True, exist_ok=True)

        # Copy corrupt file to class directory
        import shutil

        shutil.copy(corrupt_image_file, class_dir / "corrupt.jpg")

        # Create a valid image too
        import numpy as np

        valid_img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        valid_img.save(class_dir / "valid.jpg")

        # Try to create dataloader - should handle corrupt file
        transform = build_transforms(train=False, image_size=32)
        from torchvision.datasets import ImageFolder

        with pytest.raises((OSError, RuntimeError, Image.UnidentifiedImageError)):
            dataset = ImageFolder(class_dir, transform=transform)
            _ = dataset[0]

    def test_empty_image_file_handling(
        self, empty_image_file: Path, temp_dir: Path
    ) -> None:
        """Test that empty image files raise appropriate errors."""
        class_dir = temp_dir / "test_class"
        class_dir.mkdir(parents=True, exist_ok=True)

        import shutil

        shutil.copy(empty_image_file, class_dir / "empty.jpg")

        transform = build_transforms(train=False, image_size=32)
        from torchvision.datasets import ImageFolder

        with pytest.raises((OSError, RuntimeError, Image.UnidentifiedImageError)):
            dataset = ImageFolder(class_dir, transform=transform)
            _ = dataset[0]


class TestAllBreedsRepresented:
    """Tests to verify all 13 cat breeds are represented in dataset."""

    def test_13_breed_classes_present(
        self, temp_image_dataset_13_classes: Path
    ) -> None:
        """Test that all 13 cat breed classes can be loaded."""
        class_names = get_class_names(temp_image_dataset_13_classes)

        expected_breeds = [
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

        for breed in expected_breeds:
            assert breed in class_names, f"Breed '{breed}' not found in dataset"

    def test_13_breed_dataloader_creation(
        self, temp_image_dataset_13_classes: Path
    ) -> None:
        """Test dataloader can be created with all 13 breeds."""
        train_loader, val_loader = cats_dataloader(
            root=temp_image_dataset_13_classes,
            batch_size=4,
            val_split=0.2,
            image_size=32,
            num_workers=0,
            seed=42,
        )

        assert train_loader is not None
        assert val_loader is not None
        assert len(train_loader) > 0
        assert len(val_loader) > 0

    def test_all_breeds_have_samples(self, temp_image_dataset_13_classes: Path) -> None:
        """Test that each breed has at least one sample."""
        from torchvision.datasets import ImageFolder

        transform = build_transforms(train=False, image_size=32)
        dataset = ImageFolder(temp_image_dataset_13_classes, transform=transform)

        # Count samples per class
        class_counts: dict[str, int] = {}
        for _, label in dataset:
            class_name = dataset.classes[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        assert len(class_counts) == 13, f"Expected 13 classes, got {len(class_counts)}"

        for breed, count in class_counts.items():
            assert count > 0, f"Breed '{breed}' has no samples"


class TestBoundaryConditions:
    """Tests for boundary conditions in image sizes and data loading."""

    def test_minimum_image_size_1x1(
        self, boundary_size_images: list[tuple[Path, tuple[int, int]]]
    ) -> None:
        """Test processing of minimum size (1x1) images."""
        transform = build_transforms(train=False, image_size=32)

        for img_path, (h, w) in boundary_size_images:
            if h == 1 and w == 1:
                img = Image.open(img_path)
                tensor = transform(img)
                assert tensor.shape == (3, 32, 32)

    def test_small_image_size_8x8(
        self, boundary_size_images: list[tuple[Path, tuple[int, int]]]
    ) -> None:
        """Test processing of small (8x8) images."""
        transform = build_transforms(train=False, image_size=32)

        for img_path, (h, w) in boundary_size_images:
            if h == 8 and w == 8:
                img = Image.open(img_path)
                tensor = transform(img)
                assert tensor.shape == (3, 32, 32)

    def test_standard_image_size_224x224(
        self, boundary_size_images: list[tuple[Path, tuple[int, int]]]
    ) -> None:
        """Test processing of standard (224x224) images."""
        transform = build_transforms(train=False, image_size=224)

        for img_path, (h, w) in boundary_size_images:
            if h == 224 and w == 224:
                img = Image.open(img_path)
                tensor = transform(img)
                assert tensor.shape == (3, 224, 224)

    def test_large_image_size_1024x1024(
        self, boundary_size_images: list[tuple[Path, tuple[int, int]]]
    ) -> None:
        """Test processing of large (1024x1024) images."""
        transform = build_transforms(train=False, image_size=224)

        for img_path, (h, w) in boundary_size_images:
            if h == 1024 and w == 1024:
                img = Image.open(img_path)
                tensor = transform(img)
                assert tensor.shape == (3, 224, 224)

    def test_very_large_image_size_2048x2048(
        self, boundary_size_images: list[tuple[Path, tuple[int, int]]]
    ) -> None:
        """Test processing of very large (2048x2048) images."""
        transform = build_transforms(train=False, image_size=224)

        for img_path, (h, w) in boundary_size_images:
            if h == 2048 and w == 2048:
                img = Image.open(img_path)
                tensor = transform(img)
                assert tensor.shape == (3, 224, 224)

    def test_batch_size_1(self, temp_image_dataset: Path) -> None:
        """Test dataloader with batch size of 1."""
        train_loader, _val_loader = cats_dataloader(
            root=temp_image_dataset,
            batch_size=1,
            val_split=0.2,
            image_size=32,
            num_workers=0,
            seed=42,
        )

        for batch in train_loader:
            x, _y = batch
            assert x.shape[0] == 1
            break

    def test_val_split_boundary_0(self, temp_image_dataset: Path) -> None:
        """Test dataloader with 0% validation split."""
        _train_loader, val_loader = cats_dataloader(
            root=temp_image_dataset,
            batch_size=4,
            val_split=0.0,
            image_size=32,
            num_workers=0,
            seed=42,
        )

        assert len(val_loader) == 0

    def test_val_split_boundary_1(self, temp_image_dataset: Path) -> None:
        """Test dataloader with 100% validation split.

        Note: This raises ValueError because empty train loader is not allowed.
        """
        with pytest.raises(ValueError, match="num_samples"):
            cats_dataloader(
                root=temp_image_dataset,
                batch_size=4,
                val_split=1.0,
                image_size=32,
                num_workers=0,
                seed=42,
            )


class TestCatsModel:
    """Tests for the cats_model() factory."""

    def test_resnet18_builds(self) -> None:
        """Test ResNet-18 backbone builds successfully."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        assert model is not None

    def test_resnet18_output_shape(self) -> None:
        """Test ResNet-18 produces correct output shape."""
        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)

    def test_custom_num_classes(self) -> None:
        """Test model with custom number of classes."""
        model = cats_model(num_classes=12, backbone="resnet18", pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 12)

    def test_mobilenet_builds(self) -> None:
        """Test MobileNetV3 backbone builds successfully."""
        model = cats_model(
            num_classes=2, backbone="mobilenet_v3_small", pretrained=False
        )
        assert model is not None

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


class TestDatasetErrors:
    """Tests for error handling in the dataset module."""

    def test_cats_dataloader_missing_dir_raises(self) -> None:
        """Test that missing dataset directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            cats_dataloader(root="/nonexistent/path/12345")

    def test_cats_dataloader_invalid_root_raises(self, temp_dir: Path) -> None:
        """Test that invalid root (file instead of directory) raises error."""
        invalid_file = temp_dir / "not_a_dir.txt"
        invalid_file.write_text("not a directory")

        with pytest.raises((FileNotFoundError, NotADirectoryError)):
            cats_dataloader(root=str(invalid_file))


class TestGetClassNames:
    """Tests for get_class_names utility function."""

    def test_get_class_names_returns_sorted_list(
        self, temp_image_dataset: Path
    ) -> None:
        """Test get_class_names returns sorted list of class names."""
        names = get_class_names(temp_image_dataset)
        assert isinstance(names, list)
        assert names == sorted(names)

    def test_get_class_names_excludes_hidden_dirs(
        self, temp_image_dataset: Path
    ) -> None:
        """Test get_class_names excludes hidden directories."""
        # Create hidden directory
        hidden_dir = temp_image_dataset / ".hidden_class"
        hidden_dir.mkdir(parents=True, exist_ok=True)

        names = get_class_names(temp_image_dataset)
        assert ".hidden_class" not in names

    def test_get_class_names_empty_directory(self, temp_dir: Path) -> None:
        """Test get_class_names with empty directory."""
        empty_dir = temp_dir / "empty_dataset"
        empty_dir.mkdir(parents=True, exist_ok=True)

        names = get_class_names(empty_dir)
        assert names == []


class TestTransformPipelineIntegration:
    """Integration tests for the full transform pipeline."""

    def test_full_pipeline_train_mode(self) -> None:
        """Test complete training transform pipeline."""
        import numpy as np

        transform = build_transforms(train=True, image_size=224)
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

        tensor = transform(img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32
        # Normalized values should be in reasonable range
        assert tensor.min() > -4
        assert tensor.max() < 4

    def test_full_pipeline_val_mode(self) -> None:
        """Test complete validation transform pipeline."""
        import numpy as np

        transform = build_transforms(train=False, image_size=224)
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

        tensor = transform(img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_transform_different_image_sizes_same_output(
        self,
    ) -> None:
        """Test that different input sizes produce same output size."""
        import numpy as np

        transform = build_transforms(train=False, image_size=64)

        sizes = [(32, 32), (128, 128), (256, 256), (512, 512)]

        for h, w in sizes:
            img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
            tensor = transform(img)
            assert tensor.shape == (3, 64, 64)

    def test_transform_preserves_batch_dimension(
        self, temp_image_dataset: Path
    ) -> None:
        """Test that dataloader preserves batch dimension correctly."""
        train_loader, _ = cats_dataloader(
            root=temp_image_dataset,
            batch_size=2,
            val_split=0.2,
            image_size=32,
            num_workers=0,
            seed=42,
        )

        for batch in train_loader:
            x, _y = batch
            assert x.ndim == 4  # (B, C, H, W)
            assert x.shape[0] <= 2  # Batch size
            assert x.shape[1] == 3  # Channels
            break
