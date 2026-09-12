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
from typing import Any, Literal

import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader

# ImageNet mean/std for pretrained model normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE = 224

# Supported augmentation levels
AugmentationLevel = Literal["basic", "standard", "advanced"]
EnhancedAugmentationLevel = Literal["basic", "medium", "full"]

CAT_BREEDS = (
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx",
)
OTHER_CLASS_INDEX = len(CAT_BREEDS)


class CatBreedGenerationDataset(Dataset):
    """Load the two-folder dataset with the generator's 13 class labels.

    The downloaded dataset stores all cat breeds under ``cat/`` and all dog
    breeds under ``other/``. ``ImageFolder`` therefore produces only binary
    labels, which leaves 11 of TinyDiT's 13 conditioning embeddings untrained.
    Oxford Pets filenames retain the cat breed, so map those names to indices
    0-11 and reserve index 12 for every image in ``other/``.
    """

    def __init__(self, root: str | Path, transform: T.Compose | None = None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for class_dir in ("cat", "other"):
            directory = self.root / class_dir
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Generator dataset directory not found: {directory}"
                )

            for path in sorted(directory.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMG_EXTENSIONS:
                    continue
                label = OTHER_CLASS_INDEX
                if class_dir == "cat":
                    matches = [
                        index
                        for index, breed in enumerate(CAT_BREEDS)
                        if path.name.startswith(f"{breed}_")
                    ]
                    if not matches:
                        raise ValueError(f"Unknown cat breed filename: {path.name}")
                    label = matches[0]
                self.samples.append((path, label))

        if not self.samples:
            raise RuntimeError(f"No images found in generator dataset: {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        path, label = self.samples[index]
        image = default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_enhanced_transforms(
    train: bool = True,
    image_size: int = 128,
    augmentation_level: EnhancedAugmentationLevel = "full",
) -> T.Compose:
    """Build enhanced data augmentation pipeline.

    Args:
        train: Whether to build training transforms
        image_size: Target image size
        augmentation_level: "basic", "medium", or "full"

    Returns:
        Composed transform pipeline
    """
    if not train:
        return T.Compose(
            [
                T.Resize([image_size, image_size]),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    # Basic augmentation (current)
    if augmentation_level == "basic":
        return T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.Resize([image_size, image_size]),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    # Medium augmentation
    elif augmentation_level == "medium":
        return T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.Resize([image_size, image_size]),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    # Full augmentation (recommended for production)
    else:  # augmentation_level == "full"
        return T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
                T.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=10,
                ),
                T.Resize([image_size, image_size]),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )


def build_transforms(
    train: bool = True,
    image_size: int = DEFAULT_IMAGE_SIZE,
    augmentation_level: AugmentationLevel = "standard",
) -> T.Compose:
    """Return torchvision transforms for train or validation.

    Args:
        train: Whether to use training transforms (True) or validation (False).
        image_size: Target image size for resizing/cropping.
        augmentation_level: Level of data augmentation ("basic", "standard", "advanced").
            - basic: RandomHorizontalFlip only
            - standard: RandomHorizontalFlip + ColorJitter (default)
            - advanced: All transforms including RandomRotation and RandomAffine

    Returns:
        Composed torchvision transforms.
    """
    if train:
        transforms: list[T.Transform] = [T.RandomResizedCrop(image_size)]

        # Always apply horizontal flip
        transforms.append(T.RandomHorizontalFlip(p=0.5))

        # Apply additional augmentations based on level
        if augmentation_level in ("standard", "advanced"):
            transforms.append(
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            )

        if augmentation_level == "advanced":
            transforms.append(T.RandomRotation(degrees=15))
            transforms.append(
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
            )

        transforms.extend(
            [T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
        )

        return T.Compose(transforms)

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
    augmentation_level: AugmentationLevel = "standard",
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from an ImageFolder directory.

    Args:
        root: Path to the dataset root (expects class subdirectories).
        batch_size: Batch size for both loaders.
        val_split: Fraction of data to use for validation.
        image_size: Spatial size to resize images to.
        num_workers: Number of DataLoader worker processes.
        seed: Random seed for reproducible splits.
        augmentation_level: Level of data augmentation ("basic", "standard", "advanced").

    Returns:
        Tuple of (train_loader, val_loader).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}. Run `bash data/download.sh` first."
        )

    # Full dataset with train transforms (we apply different transforms after split)
    full_dataset = ImageFolder(
        root,
        transform=build_transforms(
            train=True, image_size=image_size, augmentation_level=augmentation_level
        ),
    )

    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    import torch

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=generator)

    # Override val transforms
    val_ds.dataset = ImageFolder(
        root, transform=build_transforms(train=False, image_size=image_size)
    )

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
    return sorted(
        entry.name
        for entry in Path(root).iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )
