"""Tests for immutable model releases and frontend manifest generation (#204)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_frontend_metadata as generator  # noqa: E402
import promote_frontend_release as promoter  # noqa: E402

from model_release import (  # noqa: E402
    HUB_REVISION,
    ReleaseError,
    build_release_entry,
    empty_release_history,
    expected_models,
    find_package_file,
    find_release,
    hub_url,
    immutable_commit_sha,
    latest_release,
    read_package_manifest,
    read_release_history,
    record_release,
    resolve_release,
    sha256_bytes,
    validate_revision,
    verify_artifact_bytes,
    verify_hub_artifact,
    verify_package_file,
    write_release_history,
)

REVISION = "a" * 40
OTHER_REVISION = "b" * 40


def _tiny_onnx(path: Path, input_name: str, output_name: str) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", [input_name], [output_name])],
        "tiny",
        [helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 3, 8, 8])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, str(path))


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    """Bundled ONNX artifacts with deterministic graph bytes."""
    directory = tmp_path / "models"
    directory.mkdir()
    _tiny_onnx(directory / "cats_quantized.onnx", "input", "output")
    _tiny_onnx(directory / "generator_quantized.onnx", "noise", "velocity")
    return directory


def _package_manifest(artifacts: Path, root: Path) -> Path:
    files = []
    for model_dir, name in (
        ("classifier", "cats_quantized.onnx"),
        ("generator", "generator_quantized.onnx"),
    ):
        data = (artifacts / name).read_bytes()
        staged = root / "package" / model_dir / "model_quantized.onnx"
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(data)
        files.append(
            {"path": str(staged), "sha256": sha256_bytes(data), "bytes": len(data)}
        )
    manifest_path = root / "package" / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "completed_steps": 400000,
                "experiment_id": "dit-breed-conditioned-v4",
                "git_sha": "c0ffee",
                "files": files,
            }
        )
    )
    return manifest_path


@pytest.mark.parametrize("revision", ["main", "v1.0.0", "abc1234", REVISION.upper()])
def test_validate_revision_rejects_mutable_references(revision: str) -> None:
    with pytest.raises(ReleaseError, match="40-character"):
        validate_revision(revision)


def test_validate_revision_accepts_full_commit_sha() -> None:
    assert validate_revision(REVISION) == REVISION
    assert HUB_REVISION.fullmatch(REVISION)


def test_hub_url_is_immutable_and_rejects_main() -> None:
    url = hub_url("https://huggingface.co/o/r/resolve/", REVISION, "cats.onnx")
    assert url == f"https://huggingface.co/o/r/resolve/{REVISION}/cats.onnx"
    assert "/main/" not in url
    with pytest.raises(ReleaseError):
        hub_url("https://huggingface.co/o/r/resolve/", "main", "cats.onnx")


def test_read_package_manifest_validates_records(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps({"files": [{"path": "a.onnx", "sha256": "0" * 64, "bytes": 10}]})
    )
    assert read_package_manifest(good)["files"][0]["path"] == "a.onnx"

    invalid_documents: list[dict[str, object]] = [
        {"files": [{"path": "a.onnx"}]},
        {"files": [{"path": "a.onnx", "sha256": "short", "bytes": 1}]},
        {"files": [{"path": "a.onnx", "sha256": "0" * 64, "bytes": 0}]},
        {"unexpected": []},
    ]
    for document in invalid_documents:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(document))
        with pytest.raises(ReleaseError):
            read_package_manifest(path)

    with pytest.raises(ReleaseError, match="missing"):
        read_package_manifest(tmp_path / "absent.json")


def test_find_package_file_matches_suffix_and_rejects_ambiguity() -> None:
    manifest = {
        "files": [
            {"path": "/out/artifacts/generator/model_quantized.onnx"},
        ]
    }
    record = find_package_file(manifest, "generator", "model_quantized.onnx")
    assert record["path"].endswith("generator/model_quantized.onnx")
    with pytest.raises(ReleaseError, match="no file"):
        find_package_file(manifest, "classifier", "model_quantized.onnx")
    with pytest.raises(ReleaseError, match="2 files"):
        find_package_file(
            {
                "files": [
                    {"path": "a/model_quantized.onnx"},
                    {"path": "b/model_quantized.onnx"},
                ]
            },
            "model_quantized.onnx",
        )


def test_verify_artifact_bytes_detects_mismatch() -> None:
    data = b"verified model bytes"
    digest = sha256_bytes(data)
    assert verify_artifact_bytes(data, sha256=digest, size=len(data)) == digest
    with pytest.raises(ReleaseError, match="size mismatch"):
        verify_artifact_bytes(data, sha256=digest, size=len(data) + 1)
    with pytest.raises(ReleaseError, match="digest mismatch"):
        verify_artifact_bytes(data, sha256="0" * 64, size=len(data))


def test_verify_package_file_rejects_stale_bundle(tmp_path: Path) -> None:
    path = tmp_path / "cats_quantized.onnx"
    path.write_bytes(b"old release bytes")
    record = {
        "path": "artifacts/classifier/model_quantized.onnx",
        "sha256": sha256_bytes(b"new release bytes"),
        "bytes": len(b"new release bytes"),
    }
    with pytest.raises(ReleaseError, match="mismatch"):
        verify_package_file(record, path)
    with pytest.raises(ReleaseError, match="missing"):
        verify_package_file(record, tmp_path / "absent.onnx")
    record["sha256"] = sha256_bytes(b"old release bytes")
    record["bytes"] = len(b"old release bytes")
    assert verify_package_file(record, path) == record["sha256"]


def test_verify_hub_artifact_checks_downloaded_bytes() -> None:
    data = b"hub artifact bytes"
    digest = sha256_bytes(data)
    url = hub_url("https://huggingface.co/o/r/resolve/", REVISION, "cats.onnx")
    assert (
        verify_hub_artifact(url, sha256=digest, size=len(data), fetch=lambda _: data)
        == digest
    )
    with pytest.raises(ReleaseError, match="digest mismatch"):
        verify_hub_artifact(url, sha256="0" * 64, size=len(data), fetch=lambda _: data)
    with pytest.raises(ReleaseError, match="Cannot download"):

        def broken(_: str) -> bytes:
            raise OSError("boom")

        verify_hub_artifact(url, sha256=digest, size=len(data), fetch=broken)
    with pytest.raises(ReleaseError, match="40-character"):
        verify_hub_artifact(
            "https://huggingface.co/o/r/resolve/main/cats.onnx",
            sha256=digest,
            size=len(data),
            fetch=lambda _: data,
        )


def test_release_history_records_and_replaces_revisions(tmp_path: Path) -> None:
    history = empty_release_history()
    first = build_release_entry(
        REVISION,
        {
            "classifier": {
                "artifact": "cats_quantized.onnx",
                "hubPath": "cats_quantized.onnx",
                "sha256": "1" * 64,
                "size": 10,
            }
        },
        created_at="2026-01-01T00:00:00+00:00",
    )
    history = record_release(history, first)
    assert find_release(history, REVISION) == first
    assert latest_release(history) == first

    replacement = dict(first, created_at="2026-01-02T00:00:00+00:00")
    history = record_release(history, replacement)
    assert len(history["releases"]) == 1
    assert latest_release(history) == replacement

    second = build_release_entry(
        OTHER_REVISION,
        {
            "generator": {
                "artifact": "generator_quantized.onnx",
                "hubPath": "generator_quantized.onnx",
                "sha256": "2" * 64,
                "size": 20,
            }
        },
        created_at="2026-01-03T00:00:00+00:00",
    )
    history = record_release(history, second)
    assert latest_release(history) == second
    assert resolve_release(history) == second
    assert resolve_release(history, REVISION) == replacement

    path = tmp_path / "modelReleases.json"
    write_release_history(path, history)
    assert read_release_history(path) == history
    assert read_release_history(tmp_path / "absent.json") == empty_release_history()


def test_resolve_release_rejects_unknown_revision() -> None:
    with pytest.raises(ReleaseError, match="not recorded"):
        resolve_release(empty_release_history(), REVISION)
    with pytest.raises(ReleaseError, match="empty"):
        resolve_release(empty_release_history())


def test_expected_models_rejects_malformed_release() -> None:
    with pytest.raises(ReleaseError, match="models"):
        expected_models({"revision": REVISION})
    with pytest.raises(ReleaseError, match="digest"):
        expected_models({"models": {"classifier": {"sha256": "bad", "size": 1}}})


def test_build_release_entry_requires_valid_models() -> None:
    with pytest.raises(ReleaseError, match="at least one"):
        build_release_entry(REVISION, {})
    with pytest.raises(ReleaseError, match="40-character"):
        build_release_entry(
            "main",
            {"classifier": {"sha256": "0" * 64, "size": 1}},
        )


def test_build_manifest_sources_digests_from_verified_package(
    artifacts: Path, tmp_path: Path
) -> None:
    package = _package_manifest(artifacts, tmp_path)
    manifest = generator.build_manifest(artifacts, package_manifest_path=package)
    for name in ("cats_quantized.onnx", "generator_quantized.onnx"):
        expected = sha256_bytes((artifacts / name).read_bytes())
        asset = next(a for a in manifest["models"].values() if a["artifact"] == name)
        assert asset["sha256"] == expected
        assert asset["size"] == (artifacts / name).stat().st_size
        assert asset["hubRevision"] is None
        assert asset["onnx"]["opset"] == 17
    assert manifest["hub"]["immutable"] is False
    assert manifest["policy"]["allowMutableRevision"] is False
    assert manifest["source"]["completedSteps"] == 400000


def test_build_manifest_rejects_stale_bundled_artifact(
    artifacts: Path, tmp_path: Path
) -> None:
    package = _package_manifest(artifacts, tmp_path)
    (artifacts / "cats_quantized.onnx").write_bytes(b"stale local model")
    with pytest.raises(SystemExit, match="stale"):
        generator.build_manifest(artifacts, package_manifest_path=package)


def test_build_manifest_requires_packaged_frontend_artifacts(
    artifacts: Path, tmp_path: Path
) -> None:
    package = _package_manifest(artifacts, tmp_path)
    document = json.loads(package.read_text())
    document["files"] = [
        record for record in document["files"] if "classifier" not in record["path"]
    ]
    package.write_text(json.dumps(document))
    with pytest.raises(SystemExit, match="missing the frontend release files"):
        generator.build_manifest(artifacts, package_manifest_path=package)


def test_build_manifest_pins_only_verified_immutable_revision(
    artifacts: Path,
) -> None:
    seen: list[str] = []

    def verifier(url: str, **_: object) -> str:
        seen.append(url)
        return REVISION

    manifest = generator.build_manifest(artifacts, REVISION, hub_verifier=verifier)
    assert manifest["hub"] == {
        "repoId": "d4oit/tiny-cats-model",
        "baseUrl": generator.HUB_BASE_URL,
        "revision": REVISION,
        "immutable": True,
    }
    assert all(
        asset["hubRevision"] == REVISION for asset in manifest["models"].values()
    )
    assert len(seen) == 2
    assert all(f"/resolve/{REVISION}/" in url for url in seen)
    assert not any("/resolve/main/" in url for url in seen)


def test_build_manifest_refuses_unverified_hub_revision(artifacts: Path) -> None:
    def rejecting(url: str, **_: object) -> str:
        raise ReleaseError("digest mismatch")

    with pytest.raises(SystemExit, match="unverified Hub revision"):
        generator.build_manifest(artifacts, REVISION, hub_verifier=rejecting)
    with pytest.raises(SystemExit, match="40-character"):
        generator.build_manifest(artifacts, "main")


def test_build_manifest_rollback_requires_matching_bundled_bytes(
    artifacts: Path, tmp_path: Path
) -> None:
    package = _package_manifest(artifacts, tmp_path)
    manifest = generator.build_manifest(artifacts, package_manifest_path=package)
    entry = build_release_entry(REVISION, manifest["models"])

    rolled_back = generator.build_manifest(artifacts, release_entry=entry)
    assert rolled_back["hub"]["revision"] == REVISION
    assert (
        rolled_back["models"]["classifier"]["sha256"]
        == manifest["models"]["classifier"]["sha256"]
    )

    (artifacts / "generator_quantized.onnx").write_bytes(b"other release")
    with pytest.raises(SystemExit, match="stale"):
        generator.build_manifest(artifacts, release_entry=entry)


def test_cli_promotes_records_checks_and_rolls_back(
    artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path = tmp_path / "modelContract.json"
    manifest_path = tmp_path / "modelManifest.json"
    history_path = tmp_path / "modelReleases.json"
    monkeypatch.setattr(generator, "CONTRACT_PATH", contract_path)
    monkeypatch.setattr(generator, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(generator, "HISTORY_PATH", history_path)

    package = _package_manifest(artifacts, tmp_path)
    assert (
        generator.main(
            [
                "--artifact-dir",
                str(artifacts),
                "--package-manifest",
                str(package),
                "--revision",
                REVISION,
                "--no-verify-hub",
            ]
        )
        == 0
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["hub"]["revision"] == REVISION
    assert manifest["hub"]["immutable"] is True
    assert manifest["source"]["packageManifest"] == str(package)

    history = read_release_history(history_path)
    entry = find_release(history, REVISION)
    assert entry is not None
    assert entry["package_manifest"]["completed_steps"] == 400000

    assert generator.main(["--artifact-dir", str(artifacts), "--check"]) == 0

    # The bundled artifact going stale is caught before anything can load it.
    original = (artifacts / "cats_quantized.onnx").read_bytes()
    (artifacts / "cats_quantized.onnx").write_bytes(b"stale bundled model")
    assert generator.main(["--artifact-dir", str(artifacts), "--check"]) == 1
    (artifacts / "cats_quantized.onnx").write_bytes(original)

    # Rollback re-pins the recorded revision and keeps the release history.
    assert (
        generator.main(["--artifact-dir", str(artifacts), "--rollback", REVISION]) == 0
    )
    assert json.loads(manifest_path.read_text())["hub"]["revision"] == REVISION
    assert read_release_history(history_path) == history


def test_cli_rejects_mutable_revision_and_empty_rollback(
    artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generator, "CONTRACT_PATH", tmp_path / "contract.json")
    monkeypatch.setattr(generator, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(generator, "HISTORY_PATH", tmp_path / "history.json")

    with pytest.raises(SystemExit, match="40-character"):
        generator.main(["--artifact-dir", str(artifacts), "--revision", "main"])
    with pytest.raises(SystemExit, match="empty"):
        generator.main(["--artifact-dir", str(artifacts), "--rollback"])
    with pytest.raises(SystemExit, match="do not pass --revision"):
        generator.main(
            [
                "--artifact-dir",
                str(artifacts),
                "--rollback",
                REVISION,
                "--revision",
                OTHER_REVISION,
            ]
        )


def test_cli_local_bundle_writes_empty_history(
    artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generator, "CONTRACT_PATH", tmp_path / "contract.json")
    monkeypatch.setattr(generator, "MANIFEST_PATH", tmp_path / "manifest.json")
    history_path = tmp_path / "history.json"
    monkeypatch.setattr(generator, "HISTORY_PATH", history_path)

    assert generator.main(["--artifact-dir", str(artifacts)]) == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["hub"]["revision"] is None
    assert manifest["hub"]["immutable"] is False
    assert read_release_history(history_path) == empty_release_history()


def test_immutable_commit_sha_requires_full_sha() -> None:
    class Modern:
        oid = REVISION

    class Legacy:
        commit_sha = REVISION

    assert immutable_commit_sha(Modern()) == REVISION
    assert immutable_commit_sha(Legacy()) == REVISION
    with pytest.raises(ReleaseError, match="no immutable commit SHA"):
        immutable_commit_sha(object())
    with pytest.raises(ReleaseError, match="40-character"):
        immutable_commit_sha(type("BranchName", (), {"oid": "main"})())


def test_committed_frontend_manifest_matches_bundled_artifacts() -> None:
    """The checked-in manifest must stay consistent with the bundled models."""
    pytest.importorskip("onnx")
    manifest = json.loads(generator.MANIFEST_PATH.read_text())
    assert manifest["hub"]["revision"] is None or HUB_REVISION.fullmatch(
        manifest["hub"]["revision"]
    )
    assert manifest["policy"] == {
        "acceptLocalFallbackOnlyOnDigestMatch": True,
        "allowMutableRevision": False,
        "requireSha256": True,
        "requireSize": True,
    }
    assert generator.check_generated() == 0


def test_promote_release_resolves_hub_revision_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr(promoter, "resolve_hub_revision", lambda repo, token: REVISION)
    monkeypatch.setattr(
        promoter.generator, "main", lambda argv: captured.append(list(argv)) or 0
    )

    package = tmp_path / "manifest.json"
    assert (
        promoter.main(
            [
                "--artifact-dir",
                str(tmp_path),
                "--package-manifest",
                str(package),
                "--history",
                str(tmp_path / "modelReleases.json"),
            ]
        )
        == 0
    )
    argv = captured[0]
    assert argv[argv.index("--revision") + 1] == REVISION
    assert argv[argv.index("--package-manifest") + 1] == str(package)
    assert argv[argv.index("--history") + 1] == str(tmp_path / "modelReleases.json")
    assert "--no-verify-hub" not in argv


def test_promote_release_accepts_explicit_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def fail_resolve(*args: object, **kwargs: object) -> str:
        raise AssertionError("explicit --revision must not query the Hub")

    monkeypatch.setattr(promoter, "resolve_hub_revision", fail_resolve)
    monkeypatch.setattr(
        promoter.generator, "main", lambda argv: captured.append(list(argv)) or 0
    )
    assert promoter.main(["--revision", REVISION, "--no-verify-hub"]) == 0
    assert captured[0][captured[0].index("--revision") + 1] == REVISION
    assert "--no-verify-hub" in captured[0]


def test_promote_release_rejects_mutable_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(promoter, "resolve_hub_revision", lambda repo, token: "main")
    with pytest.raises(SystemExit, match="40-character"):
        promoter.main([])
