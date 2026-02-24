#!/usr/bin/env python3
"""data/download.py - Download and prepare the Oxford IIIT Pet cats dataset.

This script uses Python's built-in urllib and tarfile modules,
so it works in minimal container environments without wget/tar.

Usage:
    python data/download.py
"""

from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path


def download_file(url: str, dest: Path, show_progress: bool = True) -> None:
    """Download file using urllib with optional progress reporting.

    Args:
        url: URL to download from.
        dest: Destination file path.
        show_progress: Whether to show download progress.
    """

    def reporthook(blocknum: int, blocksize: int, totalsize: int) -> None:
        if totalsize > 0 and show_progress:
            readsofar = blocknum * blocksize
            percent = min(readsofar * 100 / totalsize, 100)
            print(f"\rProgress: {percent:.1f}%", end="", file=sys.stderr)

    print(f"Downloading {url}...", file=sys.stderr)
    urllib.request.urlretrieve(url, dest, reporthook)
    if show_progress:
        print(file=sys.stderr)  # Newline after progress


def extract_tar_gz(archive: Path, dest: Path) -> None:
    """Extract tar.gz archive.

    Args:
        archive: Path to tar.gz file.
        dest: Destination directory.
    """
    print(f"Extracting {archive} to {dest}...", file=sys.stderr)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest)


def main() -> None:
    """Main download function."""
    # Setup paths
    script_dir = Path(__file__).parent
    cats_dir = script_dir / "cats"

    print(f"==> Preparing cats dataset in: {cats_dir}", file=sys.stderr)
    cats_dir.mkdir(parents=True, exist_ok=True)
    (cats_dir / "cat").mkdir(exist_ok=True)
    (cats_dir / "other").mkdir(exist_ok=True)

    # Oxford IIIT Pet Dataset URLs
    pet_images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"

    # Create temp directory
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Download and extract images
        images_archive = tmp_path / "images.tar.gz"
        download_file(pet_images_url, images_archive)
        extract_tar_gz(images_archive, tmp_path)

        # Cat breeds (filenames start with capital letter)
        cat_breeds = [
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
        ]

        print("==> Filtering cat breeds...", file=sys.stderr)

        # Copy cat images
        images_dir = tmp_path / "images"
        cat_count = 0
        for breed in cat_breeds:
            for img_path in images_dir.glob(f"{breed}_*.jpg"):
                dest = cats_dir / "cat" / img_path.name
                dest.write_bytes(img_path.read_bytes())
                cat_count += 1

        # Copy other (dog) images as negative samples
        other_count = 0
        max_other = 500
        for img_path in images_dir.glob("*.jpg"):
            # Skip cat breeds
            is_cat = any(img_path.name.startswith(breed) for breed in cat_breeds)
            if not is_cat and other_count < max_other:
                dest = cats_dir / "other" / img_path.name
                dest.write_bytes(img_path.read_bytes())
                other_count += 1

        print("\n==> Dataset ready:", file=sys.stderr)
        print(f"    cat/   : {cat_count} images", file=sys.stderr)
        print(f"    other/ : {other_count} images", file=sys.stderr)
        print(f"    Total  : {cat_count + other_count} images", file=sys.stderr)

    print(f"\n==> Done. Dataset is at: {cats_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
