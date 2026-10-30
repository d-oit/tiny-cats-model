"""Integration coverage for atomic dataset downloads using local archives."""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tarfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent
BREEDS = (
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


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _archive(path: Path, *, revision: int = 0) -> None:
    images = path.parent / f"images-{revision}"
    images.mkdir()
    for index, breed in enumerate(BREEDS):
        (images / f"{breed}_1.jpg").write_bytes(
            _jpeg_bytes(((index * 17 + revision) % 255, index * 11 % 255, 80))
        )
    (images / f"dog_{revision}.jpg").write_bytes(_jpeg_bytes((4 + revision, 100, 20)))
    with tarfile.open(path, "w:gz") as archive:
        for image in sorted(images.iterdir()):
            archive.add(image, arcname=f"images/{image.name}")


def _annotations(path: Path) -> None:
    with tarfile.open(path, "w:gz") as archive:
        data = b"annotations fixture\n"
        info = tarfile.TarInfo("annotations/README")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))


def _run_download(tmp_path: Path, images_url: str, annotations_url: str, *flags: str):
    destination = tmp_path / "data" / "cats"
    env = os.environ.copy()
    env.update(
        {
            "DEST": str(destination),
            "PET_IMAGES_URL": images_url,
            "PET_ANNOTS_URL": annotations_url,
            "DOWNLOAD_WORKDIR": str(tmp_path / "work"),
            "PATH": f"{Path('/usr/bin')}:{Path('/bin')}:{env.get('PATH', '')}",
        }
    )
    return subprocess.run(
        ["bash", str(ROOT / "data" / "download.sh"), *flags],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    ), destination


def test_download_manifest_noop_and_replace_safety(tmp_path: Path) -> None:
    archive1 = tmp_path / "images1.tar.gz"
    archive2 = tmp_path / "images2.tar.gz"
    annotations = tmp_path / "annotations.tar.gz"
    _archive(archive1)
    _archive(archive2, revision=1)
    _annotations(annotations)

    result, destination = _run_download(
        tmp_path, archive1.as_uri(), annotations.as_uri()
    )
    assert result.returncode == 0, result.stderr
    manifest_path = destination / "dataset_manifest.json"
    assert manifest_path.exists()
    assert len(list((destination / "cat").glob("*.jpg"))) == len(BREEDS)
    assert len(list((destination / "other").glob("*.jpg"))) == 1
    manifest_text = manifest_path.read_text()
    assert hashlib.sha256(archive1.read_bytes()).hexdigest() in manifest_text
    assert "annotations" in manifest_text

    rerun, _ = _run_download(tmp_path, archive1.as_uri(), annotations.as_uri())
    assert rerun.returncode == 0, rerun.stderr
    assert "no-op" in rerun.stdout

    before = {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }
    corrupt = tmp_path / "corrupt.tar.gz"
    corrupt.write_bytes(b"not a tar archive")
    failed, _ = _run_download(tmp_path, corrupt.as_uri(), annotations.as_uri())
    assert failed.returncode != 0
    after = {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert after == before

    refused, _ = _run_download(tmp_path, archive2.as_uri(), annotations.as_uri())
    assert refused.returncode != 0
    assert "--replace" in refused.stderr
    assert {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    } == before

    replaced, _ = _run_download(
        tmp_path, archive2.as_uri(), annotations.as_uri(), "--replace"
    )
    assert replaced.returncode == 0, replaced.stderr
    new_manifest = (destination / "dataset_manifest.json").read_text()
    assert hashlib.sha256(archive2.read_bytes()).hexdigest() in new_manifest
    assert new_manifest != manifest_text
