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

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS

from dataset_manifest import dataset_fingerprint
from model_contract import (
    CAT_BREEDS,
    CLASSIFIER_IMAGE_SIZE,
    CLASSIFIER_LABELS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    OTHER_CLASS_INDEX,
)
from preprocessing import classifier_eval_transform, load_classifier_image
from splits import ManifestDataset, read_split_manifest

DEFAULT_IMAGE_SIZE = CLASSIFIER_IMAGE_SIZE
AugmentationLevel = Literal["basic", "standard", "advanced"]
EnhancedAugmentationLevel = Literal["basic", "medium", "full"]


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
        image = load_classifier_image(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def create_train_val_dataloaders(
    root: str | Path,
    batch_size: int = 512,
    image_size: int = 128,
    num_workers: int = 4,
    augmentation_level: EnhancedAugmentationLevel = "full",
    val_split: float = 0.05,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader | None]:
    """Build the generator's train loader plus a held-out validation loader.

    The generator previously trained on 100% of the data with a running loss
    taken from augmented training batches, so checkpoint selection and early
    stopping had no unbiased signal and overfitting was unobservable. The split
    uses eval transforms (resize + normalize, no augmentation) and a fixed seed
    so the validation set is identical across resumed slices.

    Args:
        root: Dataset root containing ``cat/`` and ``other/``.
        batch_size: Batch size for both loaders.
        image_size: Target image size.
        num_workers: DataLoader workers.
        augmentation_level: Augmentation level for the train loader.
        val_split: Fraction held out for validation. 0 disables the split.
        seed: Seed for the (deterministic) train/val partition.

    Returns:
        Tuple of (train_loader, val_loader). ``val_loader`` is None when
        ``val_split`` is 0 or rounds down to fewer than one image.
    """
    train_transform = build_enhanced_transforms(
        train=True, image_size=image_size, augmentation_level=augmentation_level
    )
    dataset = CatBreedGenerationDataset(root, transform=train_transform)

    n_total = len(dataset)
    n_val = int(n_total * max(val_split, 0.0))
    if n_val <= 0 or n_total - n_val < 1:
        return (
            _weighted_generator_loader(
                dataset, batch_size=batch_size, num_workers=num_workers
            ),
            None,
        )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_loader = _weighted_generator_loader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        num_workers=num_workers,
        labels=[dataset.samples[i][1] for i in train_indices],
    )

    # A second dataset instance supplies the eval transforms; the validation
    # indices refer to the same sorted sample order, so no labels are lost.
    val_dataset = CatBreedGenerationDataset(
        root, transform=build_enhanced_transforms(train=False, image_size=image_size)
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def _weighted_generator_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    labels: list[int] | None = None,
) -> DataLoader:
    """Wrap a generator dataset in a class-balanced DataLoader.

    Args:
        dataset: Dataset (or Subset) yielding (image, label).
        batch_size: Batch size.
        num_workers: DataLoader workers.
        labels: Optional class labels aligned with ``dataset``. When omitted the
            labels are read from ``dataset.samples``.

    Returns:
        DataLoader sampling classes with equal probability.
    """
    if labels is None:
        labels = [label for _, label in dataset.samples]  # type: ignore[attr-defined]
    class_counts = Counter(labels)
    sample_weights = [1.0 / class_counts[label] for label in labels]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(labels), replacement=True
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


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

    return classifier_eval_transform(image_size)


def cats_dataloader(
    root: str | Path,
    batch_size: int = 32,
    val_split: float = 0.2,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_workers: int = 2,
    seed: int = 42,
    augmentation_level: AugmentationLevel = "standard",
    split_manifest: str | Path | None = None,
    test_split: float = 0.0,
    allow_stale_split: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create class-stratified train/validation loaders."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}. Run `bash data/download.sh` first."
        )
    train_transform = build_transforms(
        train=True, image_size=image_size, augmentation_level=augmentation_level
    )
    eval_transform = build_transforms(train=False, image_size=image_size)
    if split_manifest is None and (
        not 0 <= val_split < 1 or not 0 <= test_split < 1 or val_split + test_split >= 1
    ):
        raise ValueError("val_split and test_split must be in [0, 1) and sum below 1")
    split_seed = seed
    split_hash: str | None = None
    if split_manifest is not None:
        manifest = read_split_manifest(split_manifest)
        actual_hash = dataset_fingerprint(root)
        if manifest.get("dataset_hash") != actual_hash and not allow_stale_split:
            raise ValueError(
                "Split manifest dataset fingerprint does not match dataset root"
            )
        split_seed = int(manifest.get("seed", seed))
        split_hash = str(manifest.get("split_hash"))
        train_ds = ManifestDataset(root, manifest, "train", transform=train_transform)
        val_ds = ManifestDataset(root, manifest, "val", transform=eval_transform)
        class_names = list(CLASSIFIER_LABELS)
        train_labels = [label for _, label in train_ds.samples]
        val_labels = [label for _, label in val_ds.samples]
    else:
        train_full = ImageFolder(
            root, transform=train_transform, loader=load_classifier_image
        )
        val_full = ImageFolder(
            root, transform=eval_transform, loader=load_classifier_image
        )
        class_names = train_full.classes
        if val_full.classes != class_names or len(class_names) != 2:
            raise ValueError(
                f"Expected matching two-class ImageFolder classes, found {class_names}"
            )
        by_class: dict[int, list[int]] = defaultdict(list)
        for index, (_, label) in enumerate(train_full.samples):
            by_class[label].append(index)
        generator = torch.Generator().manual_seed(seed)
        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []
        for label in sorted(by_class):
            indices = torch.tensor(by_class[label])
            shuffled = indices[
                torch.randperm(len(indices), generator=generator)
            ].tolist()
            n_val = round(len(shuffled) * val_split)
            n_test = min(round(len(shuffled) * test_split), max(0, len(shuffled) - 1))
            available_val = max(0, len(shuffled) - n_test - 1)
            if val_split > 0 and available_val:
                n_val = max(1, n_val)
            n_val = min(max(0, n_val), available_val)
            val_indices.extend(shuffled[:n_val])
            test_indices.extend(shuffled[n_val : n_val + n_test])
            train_indices.extend(shuffled[n_val + n_test :])
        train_ds = Subset(train_full, train_indices)
        val_ds = Subset(val_full, val_indices)
        train_labels = [train_full.samples[i][1] for i in train_indices]
        split_identity = (
            f"{seed}\n"
            + "\n".join(
                f"{Path(train_full.samples[index][0]).relative_to(root).as_posix()}:train"
                for index in sorted(train_indices)
            )
            + "\n"
            + "\n".join(
                f"{Path(train_full.samples[index][0]).relative_to(root).as_posix()}:val"
                for index in sorted(val_indices)
            )
            + "\n"
            + "\n".join(
                f"{Path(train_full.samples[index][0]).relative_to(root).as_posix()}:test"
                for index in sorted(test_indices)
            )
        )
        split_hash = hashlib.sha256(split_identity.encode()).hexdigest()
        val_labels = [val_full.samples[i][1] for i in val_indices]
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
    val_loader.split_hash = split_hash
    print(
        f"Dataset: {len(train_ds)} train / {len(val_ds)} val samples; "
        f"train support={dict(Counter(train_labels))}, "
        f"val support={dict(Counter(val_labels))}; "
        f"seed={split_seed}, split_hash={split_hash or 'generated'}"
    )
    print(f"Classes: {class_names}")
    return train_loader, val_loader


def cats_eval_loader(
    root: str | Path | None = None,
    *,
    split: str = "val",
    split_manifest: str | Path | None = None,
    test_dir: str | Path | None = None,
    batch_size: int = 32,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_workers: int = 2,
    allow_stale_split: bool = False,
) -> tuple[DataLoader, list[tuple[Path, int]], list[str], dict[str, Any]]:
    """Create an evaluation loader and provenance for its exact sample set."""
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported evaluation split: {split!r}")
    transform = build_transforms(train=False, image_size=image_size)
    ds: Dataset[Any]
    split_hash = dataset_hash = None
    if test_dir is not None:
        test_root = Path(test_dir)
        if not test_root.is_dir():
            raise FileNotFoundError(f"Test directory not found: {test_root}")
        has_class_folders = any(path.is_dir() for path in test_root.iterdir())
        folder_ds = (
            ImageFolder(test_root, transform=transform, loader=load_classifier_image)
            if has_class_folders
            else None
        )
        if folder_ds is not None and folder_ds.classes:
            folder_labels = [
                "not_cat" if name == "other" else name for name in folder_ds.classes
            ]
            if any(name not in CLASSIFIER_LABELS for name in folder_labels):
                raise ValueError(
                    f"Unsupported classifier test labels: {folder_ds.classes}"
                )
            class_names = list(CLASSIFIER_LABELS)
            samples = [
                (
                    Path(path),
                    CLASSIFIER_LABELS.index(
                        "not_cat"
                        if folder_ds.classes[label] == "other"
                        else folder_ds.classes[label]
                    ),
                )
                for path, label in folder_ds.samples
            ]

            class MappedFolder(Dataset):
                def __len__(self) -> int:
                    return len(samples)

                def __getitem__(self, index: int) -> tuple[Any, int]:
                    path, label = samples[index]
                    image = load_classifier_image(path)
                    return transform(image), label

            ds = MappedFolder()
        else:
            images = sorted(
                path
                for path in test_root.iterdir()
                if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
            )
            label_name = {"other": "not_cat", "cat": "cat"}.get(
                test_root.name, test_root.name
            )
            if label_name not in CLASSIFIER_LABELS:
                raise ValueError("Flat test_dir must be the label's folder")
            label = CLASSIFIER_LABELS.index(label_name)
            samples = [(path, label) for path in images]

            class FlatDataset(Dataset):
                def __len__(self) -> int:
                    return len(samples)

                def __getitem__(self, index: int) -> tuple[Any, int]:
                    path, target = samples[index]
                    return transform(load_classifier_image(path)), target

            ds = FlatDataset()
            class_names = list(CLASSIFIER_LABELS)
        source = "test_dir"
    elif split_manifest is not None:
        manifest = read_split_manifest(split_manifest)
        if split not in {entry.get("split") for entry in manifest["entries"]}:
            raise ValueError(
                f"Split manifest does not contain requested split {split!r}"
            )
        if root is None:
            raise ValueError("root is required when evaluating a split manifest")
        dataset_hash = dataset_fingerprint(root)
        if manifest.get("dataset_hash") != dataset_hash and not allow_stale_split:
            raise ValueError(
                "Split manifest dataset fingerprint does not match dataset root"
            )
        split_hash = str(manifest.get("split_hash"))
        ds = ManifestDataset(root, manifest, split, transform=transform)
        samples = ds.samples
        class_names = list(CLASSIFIER_LABELS)
        source = split
    else:
        if split == "test":
            raise ValueError("split='test' requires split_manifest or test_dir")
        if root is None:
            raise ValueError("root is required for validation evaluation")
        _, val_loader = cats_dataloader(
            root,
            batch_size=batch_size,
            val_split=0.2,
            image_size=image_size,
            num_workers=num_workers,
        )
        split_hash = getattr(val_loader, "split_hash", None)
        subset = val_loader.dataset
        full_ds = subset.dataset
        samples = [full_ds.samples[i] for i in subset.indices]
        ds = subset
        class_names = list(CLASSIFIER_LABELS)
        dataset_hash = dataset_fingerprint(root)
        source = "val"
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    provenance = {
        "source": source,
        "split_hash": split_hash,
        "dataset_hash": dataset_hash,
        "counts": dict(Counter(label for _, label in samples)),
        "class_names": class_names,
        "evaluated_samples": len(samples),
    }
    return loader, samples, class_names, provenance


def get_class_names(root: str | Path) -> list[str]:
    """Return sorted class names from ImageFolder root."""
    return sorted(
        entry.name
        for entry in Path(root).iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )
