"""Canonical classifier image preprocessing shared by every Python runtime."""

from pathlib import Path

import torchvision.transforms as T
from PIL import Image, ImageOps

from model_contract import (
    CLASSIFIER_IMAGE_SIZE,
    CLASSIFIER_RESIZE_SHORT_SIDE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def canonicalize_classifier_image(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and composite transparency on black, then RGB."""
    image = ImageOps.exif_transpose(image)
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def load_classifier_image(path: str | Path) -> Image.Image:
    """Load a classifier image with its orientation and transparency preserved."""
    with Image.open(path) as image:
        return canonicalize_classifier_image(image).copy()


def classifier_eval_transform(image_size: int = CLASSIFIER_IMAGE_SIZE) -> T.Compose:
    """Build the canonical deterministic classifier evaluation transform."""
    return T.Compose(
        [
            T.Lambda(canonicalize_classifier_image),
            T.Resize(
                CLASSIFIER_RESIZE_SHORT_SIDE,
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
