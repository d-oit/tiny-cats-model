"""tests/test_dataset_sources.py

Dataset acquisition must survive an unreachable Oxford host: the Zenodo mirror
carries the original archive, so the extracted JPEGs keep the exact
``relpath:size`` dataset fingerprint the checkpoint was trained with.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

import download  # noqa: E402


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


class TestFindImagesDir:
    def test_zenodo_wrapped_layout(self, tmp_path: Path) -> None:
        images = tmp_path / "The Oxford-IIIT Pet" / "images"
        images.mkdir(parents=True)
        (images / "Abyssinian_1.jpg").write_bytes(b"cat")
        (images / "dog_1.jpg").write_bytes(b"dog")

        assert download.find_images_dir(tmp_path) == images

    def test_flat_layout(self, tmp_path: Path) -> None:
        images = tmp_path / "images"
        images.mkdir()
        (images / "Bengal_1.jpg").write_bytes(b"cat")

        assert download.find_images_dir(tmp_path) == images

    def test_no_images_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("nothing here")

        assert download.find_images_dir(tmp_path) is None


class TestSourceFallback:
    def test_second_source_is_used_when_the_first_is_unreachable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        def fake_download(url: str, dest: Path, show_progress: bool = True) -> None:
            calls.append(url)
            if "robots.ox.ac.uk" in url:
                raise OSError("timed out")
            _write_zip(
                dest,
                {"The Oxford-IIIT Pet/images/Abyssinian_1.jpg": b"original-bytes"},
            )

        monkeypatch.setattr(download, "download_file", fake_download)

        images = download.download_images(tmp_path)

        assert len(calls) == 2
        assert (images / "Abyssinian_1.jpg").read_bytes() == b"original-bytes"

    def test_all_sources_failing_is_loud(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_download(url: str, dest: Path, show_progress: bool = True) -> None:
            raise OSError("timed out")

        monkeypatch.setattr(download, "download_file", fake_download)

        with pytest.raises(RuntimeError, match="could not download"):
            download.download_images(tmp_path)

    def test_archive_without_images_tries_the_next_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []

        def fake_download(url: str, dest: Path, show_progress: bool = True) -> None:
            calls.append(url)
            if "robots.ox.ac.uk" in url:
                dest.write_bytes(b"not a tar")
                return
            _write_zip(dest, {"The Oxford-IIIT Pet/images/Bengal_1.jpg": b"x"})

        monkeypatch.setattr(download, "download_file", fake_download)

        images = download.download_images(tmp_path)

        assert len(calls) == 2
        assert (images / "Bengal_1.jpg").exists()
