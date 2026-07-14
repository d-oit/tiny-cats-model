"""tests/test_train_chain.py

Unit tests for train_chain() and train_with_fallback() in gpu_pool.py.

Covers:
- train_chain() provider ordering and display
- train_with_fallback() success and failure paths
- Checkpoint push/pull coordination
- Error handling (missing data, import failures, checkpoint sync failures)
- PoolTrainingResult state transitions
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpu_pool import (
    FALLBACK_CHAIN,
    PROVIDER_CONFIGS,
    PoolTrainingResult,
    Provider,
    iterate_fallback_chain,
    train_chain,
    train_with_fallback,
)

# ── helpers ─────────────────────────────────────────────────


def _mock_train_dit_module(loss: float = 0.01):
    """Context manager that injects a mock train_dit module.

    The lazy ``from train_dit import train_dit_local`` inside
    train_with_fallback() finds this mock and calls its train_dit_local.
    """
    mock_mod = MagicMock()
    mock_mod.train_dit_local.return_value = loss
    return patch.dict(sys.modules, {"train_dit": mock_mod})


def _mkdir_noop():
    """Context manager that mocks Path.mkdir to prevent filesystem writes."""
    return patch("pathlib.Path.mkdir")


def _success(provider: Provider = Provider.MODAL) -> PoolTrainingResult:
    return PoolTrainingResult(
        provider=provider,
        success=True,
        final_loss=0.005,
        steps_completed=100,
        total_time_seconds=10.0,
    )


# ── train_chain() ───────────────────────────────────────────


class TestTrainChain:
    """Tests for train_chain() orchestration function."""

    def test_returns_pool_training_result(self):
        with (
            patch("gpu_pool.detect_provider", return_value=Provider.MODAL),
            patch("gpu_pool.train_with_fallback", return_value=_success()),
        ):
            result = train_chain(steps=100, batch_size=64)
            assert isinstance(result, PoolTrainingResult)
            assert result.success is True
            assert result.provider == Provider.MODAL

    def test_delegates_to_train_with_fallback(self):
        with (
            patch("gpu_pool.detect_provider", return_value=Provider.LIGHTNING),
            patch("gpu_pool.train_with_fallback") as mock_twf,
        ):
            mock_twf.return_value = _success(Provider.LIGHTNING)
            train_chain(steps=500, batch_size=128, lr=1e-4, hub_repo="t/r")
            mock_twf.assert_called_once()
            kw = mock_twf.call_args.kwargs
            assert kw["steps"] == 500
            assert kw["batch_size"] == 128
            assert kw["lr"] == 1e-4
            assert kw["hub_repo"] == "t/r"

    def test_passes_extra_kwargs(self):
        with (
            patch("gpu_pool.detect_provider", return_value=Provider.MODAL),
            patch("gpu_pool.train_with_fallback") as mock_twf,
        ):
            mock_twf.return_value = _success()
            train_chain(steps=100, gradient_clip=0.5, augmentation_level="basic")
            assert mock_twf.call_args.kwargs["gradient_clip"] == 0.5
            assert mock_twf.call_args.kwargs["augmentation_level"] == "basic"

    def test_start_from_controls_display_only(self):
        with (
            patch("gpu_pool.detect_provider", return_value=Provider.LOCAL),
            patch("gpu_pool.train_with_fallback") as mock_twf,
        ):
            mock_twf.return_value = _success(Provider.LOCAL)
            result = train_chain(steps=50, start_from=Provider.COLAB)
            assert result.provider == Provider.LOCAL

    def test_start_from_none_uses_detected(self):
        with (
            patch("gpu_pool.detect_provider", return_value=Provider.KAGGLE),
            patch(
                "gpu_pool.train_with_fallback", return_value=_success(Provider.KAGGLE)
            ),
        ):
            result = train_chain(steps=50)
            assert result.provider == Provider.KAGGLE


# ── train_with_fallback() success paths ─────────────────────


class TestTrainWithFallbackSuccess:
    def test_successful_training_returns_result(self):
        with (
            patch("gpu_pool.detect_provider_and_log", return_value=Provider.MODAL),
            patch("gpu_pool.pull_checkpoint_from_hub", return_value=None),
            patch("gpu_pool._setup_data_for_provider"),
            _mock_train_dit_module(loss=0.005),
            _mkdir_noop(),
            patch("gpu_pool.push_checkpoint_to_hub", return_value=True),
            patch.dict(os.environ, {"HF_TOKEN": "fake"}, clear=True),
        ):
            result = train_with_fallback(
                data_dir="/data/cats", steps=100, batch_size=64, hub_repo="t/r"
            )
            assert result.success is True
            assert result.final_loss == 0.005
            assert result.provider == Provider.MODAL

    def test_syncs_checkpoint_on_success(self):
        with (
            patch("gpu_pool.detect_provider_and_log", return_value=Provider.MODAL),
            patch("gpu_pool.pull_checkpoint_from_hub", return_value=None),
            patch("gpu_pool._setup_data_for_provider"),
            _mock_train_dit_module(loss=0.01),
            _mkdir_noop(),
            patch("gpu_pool.push_checkpoint_to_hub") as mock_push,
            patch.dict(os.environ, {"HF_TOKEN": "fake"}, clear=True),
        ):
            train_with_fallback(data_dir="/data/cats", steps=100, hub_repo="test/repo")
            mock_push.assert_called_once()
            assert mock_push.call_args.kwargs["hub_repo"] == "test/repo"

    def test_resumes_from_hub_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pulled_path = Path(tmpdir) / "dit_model_ema.pt"
            pulled_path.write_text("fake checkpoint")
            with (
                patch("gpu_pool.detect_provider_and_log", return_value=Provider.MODAL),
                patch("gpu_pool.pull_checkpoint_from_hub", return_value=pulled_path),
                patch("gpu_pool._setup_data_for_provider"),
                _mock_train_dit_module(loss=0.01),
                _mkdir_noop(),
                patch("gpu_pool.push_checkpoint_to_hub", return_value=True),
                patch.dict(os.environ, {"HF_TOKEN": "fake"}, clear=True),
            ):
                train_with_fallback(
                    data_dir="/data/cats", steps=100, hub_repo="test/repo"
                )
                mock_mod = sys.modules["train_dit"]
                kw = mock_mod.train_dit_local.call_args.kwargs
                assert kw["resume"] == str(pulled_path)

    def test_creates_checkpoint_dir(self):
        with (
            patch("gpu_pool.detect_provider_and_log", return_value=Provider.LIGHTNING),
            patch("gpu_pool.pull_checkpoint_from_hub", return_value=None),
            patch("gpu_pool._setup_data_for_provider"),
            _mock_train_dit_module(loss=0.01),
            _mkdir_noop() as mock_mkdir,
            patch("gpu_pool.push_checkpoint_to_hub", return_value=True),
            patch.dict(os.environ, {"HF_TOKEN": "fake"}, clear=True),
        ):
            result = train_with_fallback(data_dir="/data/cats", steps=100)
            assert result.success is True
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)


# ── train_with_fallback() error paths ───────────────────────


class TestTrainWithFallbackErrors:
    def test_training_exception_returns_failure(self):
        mock_mod = MagicMock()
        mock_mod.train_dit_local.side_effect = RuntimeError("GPU OOM")
        with (
            patch("gpu_pool.detect_provider_and_log", return_value=Provider.MODAL),
            patch("gpu_pool.pull_checkpoint_from_hub", return_value=None),
            patch("gpu_pool._setup_data_for_provider"),
            patch.dict(sys.modules, {"train_dit": mock_mod}),
            _mkdir_noop(),
            patch.dict(os.environ, {}, clear=True),
        ):
            result = train_with_fallback(data_dir="/data/cats", steps=100)
            assert result.success is False
            assert "GPU OOM" in result.error
            assert result.provider == Provider.MODAL

    def test_pushes_partial_checkpoint_on_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "checkpoints"
            ckpt_dir.mkdir(parents=True)
            ckpt_path = ckpt_dir / "dit_model.pt"
            ckpt_path.write_text("partial data")
            cfg = PROVIDER_CONFIGS[Provider.MODAL]
            with (
                patch("gpu_pool.detect_provider_and_log", return_value=Provider.MODAL),
                patch("gpu_pool.pull_checkpoint_from_hub", return_value=None),
                patch("gpu_pool._setup_data_for_provider"),
                _mock_train_dit_module(),
                _mkdir_noop(),
                patch("gpu_pool.push_checkpoint_to_hub") as mock_push,
                patch.dict(os.environ, {"HF_TOKEN": "fake"}, clear=True),
            ):
                mock_mod = sys.modules["train_dit"]
                mock_mod.train_dit_local.side_effect = RuntimeError("Crashed")
                with patch.object(cfg, "checkpoint_dir", str(ckpt_dir)):
                    train_with_fallback(
                        data_dir="/data/cats",
                        steps=100,
                        hub_repo="test/repo",
                        hub_token="fake",
                    )
                assert mock_push.called

    def test_no_hub_token_skips_push(self):
        with (
            patch("gpu_pool.detect_provider_and_log", return_value=Provider.COLAB),
            patch("gpu_pool.pull_checkpoint_from_hub", return_value=None),
            patch("gpu_pool._setup_data_for_provider"),
            _mock_train_dit_module(loss=0.01),
            _mkdir_noop(),
            patch("gpu_pool.push_checkpoint_to_hub") as mock_push,
            patch.dict(os.environ, {}, clear=True),
        ):
            result = train_with_fallback(data_dir="/data/cats", steps=100)
            assert result.success is True
            mock_push.assert_not_called()

    def test_missing_data_dir_triggers_setup(self):
        with (
            patch("gpu_pool.detect_provider_and_log", return_value=Provider.MODAL),
            patch("gpu_pool.pull_checkpoint_from_hub", return_value=None),
            patch("gpu_pool._setup_data_for_provider") as mock_setup,
            _mock_train_dit_module(loss=0.01),
            _mkdir_noop(),
            patch("gpu_pool.push_checkpoint_to_hub", return_value=True),
            patch.dict(os.environ, {"HF_TOKEN": "fake"}, clear=True),
            patch("pathlib.Path.exists", return_value=False),
        ):
            train_with_fallback(data_dir="/nonexistent/path", steps=100)
            mock_setup.assert_called_once()


# ── checkpoint sync edge cases ─────────────────────────────


class TestCheckpointSyncEdgeCases:
    def test_push_fails_but_training_succeeds(self):
        with (
            patch("gpu_pool.detect_provider_and_log", return_value=Provider.MODAL),
            patch("gpu_pool.pull_checkpoint_from_hub", return_value=None),
            patch("gpu_pool._setup_data_for_provider"),
            _mock_train_dit_module(loss=0.005),
            _mkdir_noop(),
            patch("gpu_pool.push_checkpoint_to_hub", return_value=False),
            patch.dict(os.environ, {"HF_TOKEN": "fake"}, clear=True),
        ):
            result = train_with_fallback(data_dir="/data/cats", steps=100)
            assert result.success is True
            assert result.final_loss == 0.005

    def test_pull_returns_none_starts_fresh(self):
        with (
            patch("gpu_pool.detect_provider_and_log", return_value=Provider.MODAL),
            patch("gpu_pool.pull_checkpoint_from_hub", return_value=None),
            patch("gpu_pool._setup_data_for_provider"),
            _mock_train_dit_module(loss=0.01),
            _mkdir_noop(),
            patch("gpu_pool.push_checkpoint_to_hub", return_value=True),
            patch.dict(os.environ, {"HF_TOKEN": "fake"}, clear=True),
        ):
            train_with_fallback(data_dir="/data/cats", steps=100)
            mock_mod = sys.modules["train_dit"]
            assert mock_mod.train_dit_local.call_args.kwargs["resume"] is None

    def test_checkpoint_sync_with_custom_hub_token(self):
        with (
            patch("gpu_pool.detect_provider_and_log", return_value=Provider.MODAL),
            patch("gpu_pool.pull_checkpoint_from_hub") as mock_pull,
            patch("gpu_pool._setup_data_for_provider"),
            _mock_train_dit_module(loss=0.01),
            _mkdir_noop(),
            patch("gpu_pool.push_checkpoint_to_hub") as mock_push,
            patch.dict(os.environ, {}, clear=True),
        ):
            train_with_fallback(
                data_dir="/data/cats", steps=100, hub_token="hf_custom_token"
            )
            assert mock_pull.call_args.kwargs.get("token") == "hf_custom_token"
            assert mock_push.call_args.kwargs["token"] == "hf_custom_token"


# ── provider fallback chain integration ────────────────────


class TestProviderFallbackIntegration:
    def test_all_fallback_providers_have_configs(self):
        for provider in FALLBACK_CHAIN:
            assert provider in PROVIDER_CONFIGS, f"Missing config for {provider}"

    def test_chain_starts_with_modal(self):
        assert FALLBACK_CHAIN[0] == Provider.MODAL

    def test_local_is_last_resort(self):
        assert FALLBACK_CHAIN[-1] == Provider.LOCAL

    def test_chain_ordering_matches_capability(self):
        im, il, ix = map(
            FALLBACK_CHAIN.index, [Provider.MODAL, Provider.LIGHTNING, Provider.LOCAL]
        )
        assert im < il < ix

    def test_iterate_fallback_from_modal_is_full(self):
        assert iterate_fallback_chain(Provider.MODAL) == list(FALLBACK_CHAIN)

    def test_iterate_fallback_from_local_is_last(self):
        assert iterate_fallback_chain(Provider.LOCAL) == [Provider.LOCAL]


# ── PoolTrainingResult states ──────────────────────────────


class TestPoolTrainingResultStates:
    def test_success_result_fields(self):
        r = PoolTrainingResult(
            provider=Provider.MODAL,
            success=True,
            final_loss=0.00123,
            checkpoint_path="/tmp/model.pt",
            steps_completed=100_000,
            total_time_seconds=3600.0,
        )
        assert r.success is True
        assert r.final_loss == 0.00123
        assert r.steps_completed == 100_000
        assert r.error is None

    def test_failure_result_fields(self):
        r = PoolTrainingResult(
            provider=Provider.LIGHTNING,
            success=False,
            error="Session timeout",
            steps_completed=15_000,
            total_time_seconds=7200.0,
        )
        assert r.success is False
        assert r.error == "Session timeout"
        assert r.final_loss is None

    def test_partial_result_preserves_progress(self):
        r = PoolTrainingResult(
            provider=Provider.KAGGLE,
            success=False,
            error="Preempted",
            steps_completed=45_000,
            total_time_seconds=18000.0,
        )
        assert r.steps_completed == 45_000
        assert r.total_time_seconds == 18000.0
        assert not r.success

    def test_default_result_values(self):
        r = PoolTrainingResult(provider=Provider.LOCAL, success=True)
        assert r.final_loss is None
        assert r.checkpoint_path is None
        assert r.error is None
        assert r.steps_completed == 0
        assert r.total_time_seconds == 0.0
