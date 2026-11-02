"""tests/test_dataset_download.py

Dataset materialisation must be layout-independent: the Oxford tarball stores
JPEGs flat in ``images/`` while the Zenodo mirror (the fallback used whenever
the Oxford host is unreachable, i.e. on Kaggle) nests them by folder. Counting
only direct children made a Kaggle slice train on ~200 images of one breed
(2026-11-02), which also gave it a different dataset identity from Modal.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

import download


def _image(path: Path, payload: bytes = b"jpeg") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_nested_mirror_layout_is_discovered_and_flattened(tmp_path: Path) -> None:
    nested = tmp_path / "The Oxford-IIIT Pet" / "images"
    _image(nested / "cat" / "Abyssinian_1.jpg")
    _image(nested / "cat" / "Bengal_2.jpg")
    _image(nested / "dog" / "dog_1.jpg")

    images_dir = download.find_images_dir(tmp_path)

    assert images_dir == nested
    cats_dir = tmp_path / "out"
    cat_count, other_count = download.copy_images(images_dir, cats_dir)
    assert (cat_count, other_count) == (2, 1)
    assert (cats_dir / "cat" / "Bengal_2.jpg").is_file()
    assert (cats_dir / "other" / "dog_1.jpg").is_file()


def test_flat_tarball_layout_still_works(tmp_path: Path) -> None:
    flat = tmp_path / "images"
    _image(flat / "Abyssinian_9.jpg")
    _image(flat / "Persian_3.jpg")
    _image(flat / "dog_2.jpg")

    images_dir = download.find_images_dir(tmp_path)

    assert images_dir == flat
    cats_dir = tmp_path / "out"
    cat_count, other_count = download.copy_images(images_dir, cats_dir)
    assert (cat_count, other_count) == (2, 1)


def test_every_breed_is_recognised(tmp_path: Path) -> None:
    source = tmp_path / "images"
    for index, breed in enumerate(download.CAT_BREEDS):
        _image(source / f"{breed}_{index}.jpg")
    _image(source / "notacat_1.jpg")

    cats_dir = tmp_path / "out"
    cat_count, other_count = download.copy_images(source, cats_dir)

    assert cat_count == len(download.CAT_BREEDS)
    assert other_count == 1


def test_breed_prefix_must_be_exact(tmp_path: Path) -> None:
    source = tmp_path / "images"
    _image(source / "Abyssinian_1.jpg")
    # A longer breed-like name must not be swept into cat/.
    _image(source / "Abyssinianmix_2.jpg")

    cats_dir = tmp_path / "out"
    cat_count, other_count = download.copy_images(source, cats_dir)

    assert (cat_count, other_count) == (1, 1)


def test_other_images_are_capped(tmp_path: Path) -> None:
    source = tmp_path / "images"
    _image(source / "Abyssinian_1.jpg")
    for index in range(5):
        _image(source / f"dog_{index}.jpg")

    cats_dir = tmp_path / "out"
    cat_count, other_count = download.copy_images(source, cats_dir, max_other=3)

    assert (cat_count, other_count) == (1, 3)
