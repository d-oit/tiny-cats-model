from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from preprocessing import (
    canonicalize_classifier_image,
    classifier_eval_transform,
    load_classifier_image,
)


def test_transparent_classifier_images_composite_on_black() -> None:
    image = Image.new("RGBA", (2, 1))
    image.putdata([(255, 0, 0, 0), (255, 0, 0, 128)])

    actual = np.asarray(canonicalize_classifier_image(image))

    np.testing.assert_array_equal(actual[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(actual[0, 1], [128, 0, 0])


def test_classifier_loader_applies_exif_orientation_before_transform(
    tmp_path: Path,
) -> None:
    image = Image.new("RGB", (2, 3))
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path / "rotated.jpg"
    image.save(path, format="JPEG", quality=100, subsampling=0, exif=exif)

    oriented = load_classifier_image(str(path))

    assert oriented.size == (3, 2)


def test_classifier_evaluation_transform_produces_contract_shape_and_dtype() -> None:
    tensor = classifier_eval_transform()(Image.new("RGB", (500, 300), (127, 127, 127)))

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_classifier_tensor_matches_public_rgb_normalization_reference() -> None:
    image = Image.new("RGB", (224, 224), (127, 89, 213))
    tensor = classifier_eval_transform()(image)
    expected = np.asarray(
        [
            (127 / 255 - 0.485) / 0.229,
            (89 / 255 - 0.456) / 0.224,
            (213 / 255 - 0.406) / 0.225,
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(tensor[:, 0, 0].numpy(), expected, rtol=0, atol=1e-6)
