"""src/dataset.py

DataLoader factory for the cats classification dataset.
Expects ImageFolder-compatible structure:

    data/cats/
        cat/
            img1.jpg ...
        other/
            img1.jpg ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

# ImageNet mean/std for pretrained model normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE = 224


def build_transforms(train: bool = True, image_size: int = DEFAULT_IMAGE_SIZE) -> T.Compose:
    """Return torchvision transforms for train or validation."""
    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def cats_dataloader(
    root: str | Path,
    batch_size: int = 32,
    val_split: float = 0.2,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from an ImageFolder directory.

    Args:
        root: Path to the dataset root (expects class subdirectories).
        batch_size: Batch size for both loaders.
        val_split: Fraction of data to use for validation.
        image_size: Spatial size to resize images to.
        num_workers: Number of DataLoader worker processes.
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}. Run `bash data/download.sh` first.")

    # Full dataset with train transforms (we apply different transforms after split)
    full_dataset = ImageFolder(root, transform=build_transforms(train=True, image_size=image_size))

    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    import torch

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=generator)

    # Override val transforms
    val_ds.dataset = ImageFolder(root, transform=build_transforms(train=False, image_size=image_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Dataset: {n_train} train / {n_val} val samples")
    print(f"Classes: {full_dataset.classes}")

    return train_loader, val_loader


def get_class_names(root: str | Path) -> list[str]:
    """Return sorted class names from ImageFolder root."""
    return sorted(entry.name for entry in Path(root).iterdir() if entry.is_dir() and not entry.name.startswith("."))
