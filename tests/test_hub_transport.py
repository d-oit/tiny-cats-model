"""tests/test_hub_transport.py

Issue #163 WP3: HuggingFace Hub as the cross-provider checkpoint transport.

Covers the issue's requested matrix against an in-memory fake Hub:

- valid round trip (immutable ``step-XXXXXX`` snapshot + pointer)
- missing checkpoint
- corrupt checkpoint (validated before activation, quarantined)
- stale checkpoint (an older provider result never replaces a newer one)
- concurrent provider handoff
- interrupted upload (pointer flip is the commit boundary)
- token absence + token never logged
- retry behavior with exponential backoff
- legacy flat layout compatibility (pre-WP3 readers/writers)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpu_pool import (
    pull_checkpoint_from_hub,
    push_checkpoint_to_hub,
)
from training_state import write_training_state

_RETRY_SLEEPS: list[float] = []


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Record retry sleeps instead of actually waiting."""
    _RETRY_SLEEPS.clear()
    import retry_utils

    monkeypatch.setattr(retry_utils.time, "sleep", lambda s: _RETRY_SLEEPS.append(s))


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch):
    """In-memory replacement for the huggingface_hub SDK entry points."""
    import huggingface_hub as hh

    store: dict[str, bytes] = {}
    order: list[str] = []
    # Rules: {"match": substring|None, "remaining": int, "exc": exception type}
    failures: list[dict] = []

    def _maybe_fail(path_in_repo: str) -> None:
        for rule in failures:
            if rule["remaining"] > 0 and (
                rule["match"] is None or rule["match"] in path_in_repo
            ):
                rule["remaining"] -= 1
                raise rule["exc"](f"injected failure for {path_in_repo}")

    class FakeApi:
        def __init__(self, token: str | None = None):
            self.token = token

        def upload_file(
            self,
            path_or_fileobj=None,
            path_in_repo=None,
            repo_id=None,
            repo_type=None,
            token=None,
            commit_message=None,
        ):
            _maybe_fail(path_in_repo)
            if isinstance(path_or_fileobj, (bytes, bytearray)):
                data = bytes(path_or_fileobj)
            else:
                data = Path(path_or_fileobj).read_bytes()
            order.append(path_in_repo)
            store[path_in_repo] = data

        def upload_folder(self, **kwargs):
            raise NotImplementedError("not needed for the pool layout tests")

    def fake_create_repo(*args, **kwargs):
        return None

    def fake_download(repo_id, filename, repo_type=None, token=None, local_dir=None):
        if filename not in store:
            raise FileNotFoundError(filename)
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(store[filename])
        return str(target)

    monkeypatch.setattr(hh, "HfApi", FakeApi)
    monkeypatch.setattr(hh, "create_repo", fake_create_repo)
    monkeypatch.setattr(hh, "hf_hub_download", fake_download)
    monkeypatch.setenv("HF_TOKEN", "env-test-token")

    return SimpleNamespace(store=store, order=order, failures=failures)


def make_ckpt(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step_marker": marker}, path)


def push(
    fake_hub,
    tmp_path: Path,
    marker: str,
    step: int,
    *,
    experiment_id: str = "exp-a",
    checkpoint_name: str = "dit_model.pt",
) -> bool:
    ckpt = tmp_path / f"{checkpoint_name}.{marker}"
    make_ckpt(ckpt, marker)
    state = tmp_path / f"training_state.{marker}.json"
    write_training_state(
        state,
        manifest={"experiment_id": experiment_id, "target_steps": step},
        completed_steps=step,
    )
    return push_checkpoint_to_hub(
        checkpoint_path=ckpt,
        hub_repo="test/repo",
        checkpoint_name=checkpoint_name,
        experiment_id=experiment_id,
        completed_steps=step,
        training_state_path=state,
    )


class TestRoundTrip:
    def test_push_uses_immutable_layout_and_flips_pointer_last(
        self, fake_hub, tmp_path: Path
    ) -> None:
        assert push(fake_hub, tmp_path, "s10", 10) is True

        base = "checkpoints/pool/exp-a"
        assert f"{base}/step-000010/dit_model.pt" in fake_hub.store
        assert f"{base}/step-000010/training_state.json" in fake_hub.store
        # Pointer flip is the LAST upload = the commit boundary
        assert fake_hub.order[-1] == f"{base}/latest/manifest.json"

        pointer = json.loads(fake_hub.store[f"{base}/latest/manifest.json"])
        assert pointer["completed_steps"] == 10
        assert pointer["step_dir"] == "step-000010"
        assert pointer["experiment_id"] == "exp-a"
        assert "dit_model.pt" in pointer["files"]
        assert "training_state.json" in pointer["files"]

    def test_pull_resolves_newest_snapshot_and_travels_state(
        self, fake_hub, tmp_path: Path
    ) -> None:
        assert push(fake_hub, tmp_path, "s10", 10) is True
        assert push(fake_hub, tmp_path, "s20", 20) is True

        pulled = pull_checkpoint_from_hub(
            hub_repo="test/repo",
            checkpoint_name="dit_model.pt",
            output_dir=tmp_path / "out",
            experiment_id="exp-a",
        )

        assert pulled is not None
        assert (
            torch.load(pulled, map_location="cpu", weights_only=False)["step_marker"]
            == "s20"
        )
        assert (pulled.parent / "training_state.json").exists()

    def test_same_step_pushes_merge_file_list(self, fake_hub, tmp_path: Path) -> None:
        assert push(fake_hub, tmp_path, "s10", 10) is True
        assert (
            push(fake_hub, tmp_path, "s10ema", 10, checkpoint_name="dit_model_ema.pt")
            is True
        )

        base = "checkpoints/pool/exp-a"
        pointer = json.loads(fake_hub.store[f"{base}/latest/manifest.json"])
        assert {"dit_model.pt", "dit_model_ema.pt"} <= set(pointer["files"])

        pulled = pull_checkpoint_from_hub(
            hub_repo="test/repo",
            checkpoint_name="dit_model_ema.pt",
            output_dir=tmp_path / "out",
            experiment_id="exp-a",
        )
        assert pulled is not None

    def test_pull_falls_back_to_primary_when_requested_name_absent(
        self, fake_hub, tmp_path: Path
    ) -> None:
        # An older push uploaded only ``dit_model.pt``. A pull that asks for the
        # EMA sibling must fall back to the primary checkpoint rather than
        # refusing the whole snapshot (which sent pool slices to a stale local
        # file that then failed the manifest gate).
        assert push(fake_hub, tmp_path, "s10", 10) is True

        pulled = pull_checkpoint_from_hub(
            hub_repo="test/repo",
            checkpoint_name="dit_model_ema.pt",
            output_dir=tmp_path / "out",
            experiment_id="exp-a",
        )

        assert pulled is not None
        assert pulled.name == "dit_model.pt"
        assert (
            torch.load(pulled, map_location="cpu", weights_only=False)["step_marker"]
            == "s10"
        )


class TestMissingAndCorrupt:
    def test_missing_returns_none(self, fake_hub, tmp_path: Path) -> None:
        pulled = pull_checkpoint_from_hub(
            hub_repo="test/repo",
            checkpoint_name="dit_model.pt",
            output_dir=tmp_path / "out",
            experiment_id="exp-a",
        )
        assert pulled is None

    def test_corrupt_snapshot_quarantined_not_activated(
        self, fake_hub, tmp_path: Path
    ) -> None:
        assert push(fake_hub, tmp_path, "s10", 10) is True
        # Simulate bit-rot / a truncated transfer on the remote snapshot.
        base = "checkpoints/pool/exp-a"
        fake_hub.store[f"{base}/step-000010/dit_model.pt"] = b"not-a-zip"

        pulled = pull_checkpoint_from_hub(
            hub_repo="test/repo",
            checkpoint_name="dit_model.pt",
            output_dir=tmp_path / "out",
            experiment_id="exp-a",
        )

        assert pulled is None
        quarantined = list((tmp_path / "out").rglob("*.corrupt"))
        assert quarantined, "corrupt download must be quarantined locally"


class TestStaleAndHandoff:
    def test_stale_push_rejected_before_any_upload(
        self, fake_hub, tmp_path: Path
    ) -> None:
        assert push(fake_hub, tmp_path, "s20", 20) is True
        uploads_before = len(fake_hub.order)

        # A slow provider finishes later with an older result.
        assert push(fake_hub, tmp_path, "s10", 10) is False

        # Nothing new uploaded, pointer still at 20, step-000010 absent.
        assert len(fake_hub.order) == uploads_before
        base = "checkpoints/pool/exp-a"
        assert not any("step-000010" in key for key in fake_hub.store)
        pointer = json.loads(fake_hub.store[f"{base}/latest/manifest.json"])
        assert pointer["completed_steps"] == 20

    def test_concurrent_handoff_pulls_newest(self, fake_hub, tmp_path: Path) -> None:
        # Modal pushes 10, Lightning pushes 20, Modal's late 10 is rejected.
        assert push(fake_hub, tmp_path, "s10", 10) is True
        assert push(fake_hub, tmp_path, "s20", 20) is True
        assert push(fake_hub, tmp_path, "s10late", 10) is False

        pulled = pull_checkpoint_from_hub(
            hub_repo="test/repo",
            checkpoint_name="dit_model.pt",
            output_dir=tmp_path / "out",
            experiment_id="exp-a",
        )
        assert pulled is not None
        assert (
            torch.load(pulled, map_location="cpu", weights_only=False)["step_marker"]
            == "s20"
        )


class TestInterruptedUpload:
    def test_failed_pointer_flip_keeps_previous_snapshot(
        self, fake_hub, tmp_path: Path
    ) -> None:
        assert push(fake_hub, tmp_path, "s10", 10) is True

        # Every pointer-flip attempt fails (beyond the retry budget).
        fake_hub.failures.append(
            {"match": "latest/manifest.json", "remaining": 99, "exc": ConnectionError}
        )
        assert push(fake_hub, tmp_path, "s20", 20) is False

        # The partial step-000020 snapshot may exist, but the pointer still
        # resolves the last complete snapshot.
        pulled = pull_checkpoint_from_hub(
            hub_repo="test/repo",
            checkpoint_name="dit_model.pt",
            output_dir=tmp_path / "out-a",
            experiment_id="exp-a",
        )
        assert pulled is not None
        assert (
            torch.load(pulled, map_location="cpu", weights_only=False)["step_marker"]
            == "s10"
        )

        # Transient failure clears; the same push succeeds on retry.
        fake_hub.failures.clear()
        assert push(fake_hub, tmp_path, "s20", 20) is True
        pulled_again = pull_checkpoint_from_hub(
            hub_repo="test/repo",
            checkpoint_name="dit_model.pt",
            output_dir=tmp_path / "out-b",
            experiment_id="exp-a",
        )
        assert pulled_again is not None
        assert (
            torch.load(pulled_again, map_location="cpu", weights_only=False)[
                "step_marker"
            ]
            == "s20"
        )


class TestTokenHygiene:
    def test_push_without_token_fails_cleanly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
        ckpt = tmp_path / "dit_model.pt"
        make_ckpt(ckpt, "s10")

        ok = push_checkpoint_to_hub(
            checkpoint_path=ckpt,
            hub_repo="test/repo",
            checkpoint_name="dit_model.pt",
            experiment_id="exp-a",
            completed_steps=10,
        )
        assert ok is False

    def test_token_never_appears_in_logs(
        self, fake_hub, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        fake_hub.failures.append(
            {"match": "step-000010", "remaining": 99, "exc": ConnectionError}
        )
        ckpt = tmp_path / "dit_model.pt"
        make_ckpt(ckpt, "s10")

        with caplog.at_level(logging.DEBUG):
            ok = push_checkpoint_to_hub(
                checkpoint_path=ckpt,
                hub_repo="test/repo",
                checkpoint_name="dit_model.pt",
                token="sekret-token-value",
                experiment_id="exp-a",
                completed_steps=10,
            )

        assert ok is False
        assert "sekret-token-value" not in caplog.text
        assert "env-test-token" not in caplog.text


class TestRetries:
    def test_transient_upload_failures_retry_with_backoff(
        self, fake_hub, tmp_path: Path
    ) -> None:
        fake_hub.failures.append(
            {
                "match": "step-000010/dit_model.pt",
                "remaining": 2,
                "exc": ConnectionError,
            }
        )

        assert push(fake_hub, tmp_path, "s10", 10) is True

        assert len(_RETRY_SLEEPS) >= 2
        # Exponential growth between consecutive sleeps (ignoring jitter).
        assert max(_RETRY_SLEEPS) > min(_RETRY_SLEEPS)

    def test_exhausted_retries_fail_the_push(self, fake_hub, tmp_path: Path) -> None:
        fake_hub.failures.append(
            {"match": "step-000010", "remaining": 99, "exc": ConnectionError}
        )
        assert push(fake_hub, tmp_path, "s10", 10) is False


class TestLegacyLayout:
    def test_legacy_push_stays_flat_and_single_upload(
        self, fake_hub, tmp_path: Path
    ) -> None:
        ckpt = tmp_path / "dit_model.pt"
        make_ckpt(ckpt, "legacy")

        ok = push_checkpoint_to_hub(
            checkpoint_path=ckpt,
            hub_repo="test/repo",
            checkpoint_name="dit_model.pt",
        )

        assert ok is True
        assert fake_hub.order == ["checkpoints/pool/dit_model.pt"]

    def test_legacy_pull_untouched(self, fake_hub, tmp_path: Path) -> None:
        fake_hub.store["checkpoints/pool/model.pt"] = b"legacy-bytes"

        pulled = pull_checkpoint_from_hub(
            hub_repo="test/repo",
            checkpoint_name="model.pt",
            output_dir=tmp_path / "out",
        )

        # Legacy readers get exactly the old contract (no zip validation).
        assert pulled is not None
        assert pulled.name == "model.pt"
        assert pulled.read_bytes() == b"legacy-bytes"

    def test_experiment_pull_falls_back_to_legacy_when_pointer_missing(
        self, fake_hub, tmp_path: Path
    ) -> None:
        flat = tmp_path / "flat.pt"
        make_ckpt(flat, "flat")
        fake_hub.store["checkpoints/pool/dit_model.pt"] = flat.read_bytes()

        pulled = pull_checkpoint_from_hub(
            hub_repo="test/repo",
            checkpoint_name="dit_model.pt",
            output_dir=tmp_path / "out",
            experiment_id="exp-a",
        )

        assert pulled is not None
        assert (
            torch.load(pulled, map_location="cpu", weights_only=False)["step_marker"]
            == "flat"
        )
