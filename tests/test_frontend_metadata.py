"""tests/test_frontend_metadata.py

Issue #204: the browser's release manifest is generated from *verified*
artifacts, pins only immutable Hub revisions, and a stale local artifact can
never be certified by ``--check``.

Covered here (the Python half of the contract; the browser-side digest and
stale-fallback paths are exercised by ``tests/e2e/runtime.spec.ts``):

- recorded SHA-256/size match the bundled bytes,
- a branch/tag ("main", "v1") is rejected as a revision,
- a 40-hex revision is pinned and marks the manifest immutable,
- ``--package-manifest`` refuses bytes that are not in the verified package,
- ``--check`` reports a stale manifest after an artifact or the revision
  changes, so a stale fallback cannot silently ship.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import generate_frontend_metadata as metadata  # noqa: E402

REVISION = "a" * 40


@pytest.fixture(autouse=True)
def _no_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the tests independent of an installed ONNX runtime."""
    monkeypatch.setattr(metadata, "_onnx_io", lambda _path: None)


def _artifacts(
    tmp_path: Path,
    *,
    classifier: bytes = b"classifier-bytes",
    generator: bytes = b"generator-bytes",
) -> Path:
    artifact_dir = tmp_path / "models"
    artifact_dir.mkdir()
    (artifact_dir / "cats_quantized.onnx").write_bytes(classifier)
    (artifact_dir / "generator_quantized.onnx").write_bytes(generator)
    return artifact_dir


def _package_manifest(tmp_path: Path, entries: list[tuple[str, str, int]]) -> Path:
    path = tmp_path / "artifacts-manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {
                        "path": f"artifacts/generator/{name}",
                        "sha256": digest,
                        "bytes": size,
                    }
                    for name, digest, size in entries
                ],
            }
        )
    )
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestManifestDigests:
    def test_digest_and_size_match_the_bundled_bytes(self, tmp_path: Path) -> None:
        artifact_dir = _artifacts(tmp_path)
        manifest = metadata.build_manifest(artifact_dir)

        for name in ("classifier", "generator"):
            asset = manifest["models"][name]
            path = artifact_dir / asset["artifact"]
            assert asset["sha256"] == _digest(path)
            assert asset["size"] == path.stat().st_size

        # No revision was supplied: the browser must use only bundled bytes.
        assert manifest["hub"]["revision"] is None
        assert manifest["hub"]["immutable"] is False
        assert manifest["models"]["classifier"]["hubRevision"] is None
        assert manifest["policy"]["allowMutableRevision"] is False


class TestImmutableRevisions:
    @pytest.mark.parametrize("revision", ["main", "v1", "dev", "A" * 40, "a" * 39])
    def test_mutable_or_malformed_revisions_are_rejected(
        self, tmp_path: Path, revision: str
    ) -> None:
        with pytest.raises(SystemExit, match="40-hex"):
            metadata.build_manifest(_artifacts(tmp_path), revision)

    def test_full_commit_sha_is_pinned_everywhere(self, tmp_path: Path) -> None:
        manifest = metadata.build_manifest(_artifacts(tmp_path), REVISION)

        assert manifest["hub"]["revision"] == REVISION
        assert manifest["hub"]["immutable"] is True
        for asset in manifest["models"].values():
            assert asset["hubRevision"] == REVISION


class TestVerifiedPackageSource:
    def test_matching_package_entries_are_accepted(self, tmp_path: Path) -> None:
        artifact_dir = _artifacts(tmp_path)
        package = _package_manifest(
            tmp_path,
            [
                (
                    "cats_quantized.onnx",
                    _digest(artifact_dir / "cats_quantized.onnx"),
                    (artifact_dir / "cats_quantized.onnx").stat().st_size,
                ),
                (
                    "generator_quantized.onnx",
                    _digest(artifact_dir / "generator_quantized.onnx"),
                    (artifact_dir / "generator_quantized.onnx").stat().st_size,
                ),
            ],
        )

        manifest = metadata.build_manifest(artifact_dir, None, package)
        assert manifest["models"]["generator"]["size"] == len(b"generator-bytes")

    def test_bytes_outside_the_verified_package_are_refused(
        self, tmp_path: Path
    ) -> None:
        artifact_dir = _artifacts(tmp_path)
        package = _package_manifest(
            tmp_path,
            [
                (
                    "cats_quantized.onnx",
                    "0" * 64,
                    (artifact_dir / "cats_quantized.onnx").stat().st_size,
                ),
                (
                    "generator_quantized.onnx",
                    _digest(artifact_dir / "generator_quantized.onnx"),
                    (artifact_dir / "generator_quantized.onnx").stat().st_size,
                ),
            ],
        )

        with pytest.raises(SystemExit, match="verified artifact package"):
            metadata.build_manifest(artifact_dir, None, package)

    def test_unreadable_package_manifest_is_rejected(self, tmp_path: Path) -> None:
        artifact_dir = _artifacts(tmp_path)
        broken = tmp_path / "broken.json"
        broken.write_text("{not json")

        with pytest.raises(SystemExit, match="Cannot read artifact package"):
            metadata.build_manifest(artifact_dir, None, broken)


class TestCheckMode:
    """``--check`` is the gate that keeps a stale fallback out of a release."""

    def _point_generated_paths_at(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(metadata, "CONTRACT_PATH", tmp_path / "modelContract.json")
        monkeypatch.setattr(metadata, "MANIFEST_PATH", tmp_path / "modelManifest.json")

    def test_fresh_manifest_passes_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._point_generated_paths_at(tmp_path, monkeypatch)
        artifact_dir = _artifacts(tmp_path)

        assert metadata.main(["--artifact-dir", str(artifact_dir)]) == 0
        assert metadata.main(["--artifact-dir", str(artifact_dir), "--check"]) == 0

    def test_changed_artifact_fails_check(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        self._point_generated_paths_at(tmp_path, monkeypatch)
        artifact_dir = _artifacts(tmp_path)
        assert metadata.main(["--artifact-dir", str(artifact_dir)]) == 0

        # A stale/different local model is exactly what the manifest must catch.
        (artifact_dir / "cats_quantized.onnx").write_bytes(b"tampered-bytes")

        assert metadata.main(["--artifact-dir", str(artifact_dir), "--check"]) == 1
        assert "STALE" in capsys.readouterr().out

    def test_new_revision_fails_check_until_regenerated(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        self._point_generated_paths_at(tmp_path, monkeypatch)
        artifact_dir = _artifacts(tmp_path)
        assert metadata.main(["--artifact-dir", str(artifact_dir)]) == 0

        assert (
            metadata.main(
                ["--artifact-dir", str(artifact_dir), "--revision", REVISION, "--check"]
            )
            == 1
        )
        assert "STALE" in capsys.readouterr().out
