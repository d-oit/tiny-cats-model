"""tests/test_gpu_pool.py

Unit tests for the GPU pool abstraction module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpu_pool import (
    FALLBACK_CHAIN,
    PROVIDER_CONFIGS,
    PoolTrainingResult,
    Provider,
    ProviderConfig,
    detect_provider,
    estimate_cost,
    estimate_gpu_hours,
    get_provider_config,
    iterate_fallback_chain,
)


class TestProviderEnum:
    """Tests for Provider enum."""

    def test_all_providers_have_string_values(self):
        """All providers have unique string values."""
        values = [p.value for p in Provider]
        assert len(values) == len(set(values))

    def test_provider_values_match_keys(self):
        """Provider values match their names in lowercase."""
        for p in Provider:
            assert p.value == p.name.lower()

    def test_known_providers(self):
        """Verify expected providers exist."""
        expected = {
            "modal",
            "lightning",
            "colab",
            "kaggle",
            "hf_spaces",
            "local",
            "unknown",
        }
        actual = {p.value for p in Provider}
        assert actual == expected


class TestProviderConfig:
    """Tests for ProviderConfig and PROVIDER_CONFIGS."""

    def test_all_fallback_providers_have_configs(self):
        """Every provider in FALLBACK_CHAIN has a config entry."""
        for provider in FALLBACK_CHAIN:
            assert provider in PROVIDER_CONFIGS, f"Missing config for {provider}"

    def test_modal_config(self):
        """Modal config has expected values."""
        cfg = PROVIDER_CONFIGS[Provider.MODAL]
        assert cfg.display_name == "Modal"
        assert "T4" in cfg.gpu_types
        assert cfg.free_credits_per_month == 30.0
        assert cfg.preemptible is True
        assert cfg.timeout_minutes == 1440

    def test_lightning_config(self):
        """Lightning config has expected values."""
        cfg = PROVIDER_CONFIGS[Provider.LIGHTNING]
        assert cfg.display_name == "Lightning AI"
        assert "T4" in cfg.gpu_types
        assert cfg.free_hours_per_day == 22

    def test_colab_config(self):
        """Colab config has expected values."""
        cfg = PROVIDER_CONFIGS[Provider.COLAB]
        assert cfg.display_name == "Google Colab"
        assert "drive" in cfg.data_dir.lower()
        assert "drive" in cfg.checkpoint_dir.lower()

    def test_kaggle_config(self):
        """Kaggle config has expected values."""
        cfg = PROVIDER_CONFIGS[Provider.KAGGLE]
        assert cfg.display_name == "Kaggle"
        assert "P100" in cfg.gpu_types
        assert cfg.timeout_minutes <= 540  # session limit

    def test_hf_spaces_config(self):
        """HF Spaces config has expected values."""
        cfg = PROVIDER_CONFIGS[Provider.HF_SPACES]
        assert cfg.display_name == "HuggingFace Spaces"
        assert cfg.free_hours_per_day == 16

    def test_local_config(self):
        """Local config has expected values."""
        cfg = PROVIDER_CONFIGS[Provider.LOCAL]
        assert cfg.preemptible is False
        assert cfg.timeout_minutes == 0  # no limit
        assert cfg.gpu_types == []

    def test_get_provider_config_defaults_to_local(self):
        """Unknown providers default to LOCAL config."""
        cfg = get_provider_config(Provider.UNKNOWN)
        assert cfg.provider == Provider.LOCAL
        assert cfg.preemptible is False

    def test_provider_config_dataclass_defaults(self):
        """Default ProviderConfig has sensible values."""
        cfg = ProviderConfig(provider=Provider.LOCAL, display_name="Test")
        assert cfg.gpu_types == ["T4"]
        assert cfg.free_hours_per_day is None
        assert cfg.free_credits_per_month == 0.0
        assert cfg.cost_per_hour == 0.0
        assert cfg.preemptible is True
        assert cfg.env_vars == {}


class TestDetectProvider:
    """Tests for detect_provider()."""

    @patch.dict(os.environ, {"MODAL_APP_ID": "ap-12345"}, clear=True)
    def test_detect_modal_by_app_id(self):
        """Detect Modal via MODAL_APP_ID."""
        assert detect_provider() == Provider.MODAL

    @patch.dict(os.environ, {"MODAL_FUNCTION_NAME": "train_dit"}, clear=True)
    def test_detect_modal_by_function_name(self):
        """Detect Modal via MODAL_FUNCTION_NAME."""
        assert detect_provider() == Provider.MODAL

    @patch.dict(os.environ, {"LIGHTNING_APP_STATE_DIR": "/tmp/state"}, clear=True)
    def test_detect_lightning_by_env(self):
        """Detect Lightning via LIGHTNING_APP_STATE_DIR."""
        assert detect_provider() == Provider.LIGHTNING

    @patch.dict(os.environ, {"LIGHTNING_STUDIO_ID": "studio-1"}, clear=True)
    def test_detect_lightning_by_studio_id(self):
        """Detect Lightning via LIGHTNING_STUDIO_ID."""
        assert detect_provider() == Provider.LIGHTNING

    @patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "notebook"}, clear=True)
    def test_detect_kaggle(self):
        """Detect Kaggle via KAGGLE_KERNEL_RUN_TYPE."""
        assert detect_provider() == Provider.KAGGLE

    @patch.dict(os.environ, {"SPACE_ID": "my-space"}, clear=True)
    def test_detect_hf_spaces(self):
        """Detect HF Spaces via SPACE_ID."""
        assert detect_provider() == Provider.HF_SPACES

    @patch.dict(os.environ, {}, clear=True)
    def test_detect_local_with_no_env_vars(self):
        """Detect local when no env vars are set."""
        # On CI/non-GPU machines this returns LOCAL
        result = detect_provider()
        assert result in (Provider.LOCAL, Provider.UNKNOWN)

    @patch.dict(os.environ, {"COLAB_GPU": "1"}, clear=True)
    def test_detect_colab_by_env(self):
        """Detect Colab via COLAB_GPU env var."""
        assert detect_provider() == Provider.COLAB

    def test_detect_colab_by_sys_modules(self):
        """Detect Colab via google.colab in sys.modules."""
        with (
            patch.dict(sys.modules, {"google.colab": MagicMock()}),
            patch.dict(os.environ, {}, clear=True),
        ):
            assert detect_provider() == Provider.COLAB


class TestEstimateGpuHours:
    """Tests for estimate_gpu_hours()."""

    def test_default_estimate(self):
        """Default estimate returns positive hours."""
        hours = estimate_gpu_hours(100_000)
        assert hours > 0

    def test_zero_steps_returns_zero(self):
        """Zero steps returns zero hours."""
        assert estimate_gpu_hours(0) == 0.0

    def test_more_steps_more_hours(self):
        """More steps means more estimated hours."""
        h1 = estimate_gpu_hours(10_000)
        h2 = estimate_gpu_hours(100_000)
        assert h2 > h1

    def test_256_image_size_slower(self):
        """256x256 images take longer than 128x128."""
        h128 = estimate_gpu_hours(10_000, image_size=128)
        h256 = estimate_gpu_hours(10_000, image_size=256)
        assert h256 > h128

    def test_larger_batch_slower_per_step(self):
        """Larger batch size means fewer steps per second (more data/step)."""
        h_small = estimate_gpu_hours(10_000, batch_size=256)
        h_large = estimate_gpu_hours(10_000, batch_size=1024)
        assert h_large > h_small

    def test_returns_float(self):
        """Returns a float value."""
        result = estimate_gpu_hours(100_000)
        assert isinstance(result, float)


class TestEstimateCost:
    """Tests for estimate_cost()."""

    def test_all_providers_returned(self):
        """Estimate cost for all providers."""
        results = estimate_cost(100_000)
        # Should have entries for all config providers
        for p in PROVIDER_CONFIGS:
            assert p.value in results

    def test_specific_provider(self):
        """Estimate cost for a specific provider."""
        results = estimate_cost(100_000, provider=Provider.MODAL)
        assert "modal" in results
        assert "lightning" not in results

    def test_modal_within_free_tier(self):
        """Modal is within free tier for reasonable step counts."""
        results = estimate_cost(10_000, provider=Provider.MODAL)
        assert results["modal"]["within_free_tier"] is True

    def test_results_have_required_keys(self):
        """Each result has the expected keys."""
        results = estimate_cost(1_000, provider=Provider.MODAL)
        result = results["modal"]
        for key in (
            "provider",
            "gpu_hours",
            "estimated_cost",
            "within_free_tier",
            "gpu_type",
            "recommended",
        ):
            assert key in result

    def test_local_has_no_gpu_type(self):
        """Local provider reports CPU when no GPU."""
        results = estimate_cost(1_000, provider=Provider.LOCAL)
        assert results["local"]["gpu_type"] == "CPU"


class TestFallbackChain:
    """Tests for iterate_fallback_chain()."""

    def test_full_chain_with_none(self):
        """None returns the full fallback chain."""
        chain = iterate_fallback_chain(None)
        assert chain == list(FALLBACK_CHAIN)

    def test_chain_from_modal(self):
        """Chain from Modal includes all providers."""
        chain = iterate_fallback_chain(Provider.MODAL)
        assert chain[0] == Provider.MODAL
        assert len(chain) == len(FALLBACK_CHAIN)

    def test_chain_from_colab(self):
        """Chain from Colab starts at Colab, excludes earlier providers."""
        chain = iterate_fallback_chain(Provider.COLAB)
        assert chain[0] == Provider.COLAB
        assert Provider.MODAL not in chain
        assert Provider.LIGHTNING not in chain

    def test_chain_from_last_provider(self):
        """Chain from LOCAL has only LOCAL."""
        chain = iterate_fallback_chain(Provider.LOCAL)
        assert chain == [Provider.LOCAL]

    def test_unknown_returns_full_chain(self):
        """Unknown provider returns full fallback chain."""
        chain = iterate_fallback_chain(Provider.UNKNOWN)
        assert chain == list(FALLBACK_CHAIN)

    def test_chain_is_not_mutated(self):
        """Chain list is a copy, not a reference."""
        chain = iterate_fallback_chain()
        chain.append(Provider.UNKNOWN)
        assert Provider.UNKNOWN not in FALLBACK_CHAIN


class TestPoolTrainingResult:
    """Tests for PoolTrainingResult dataclass."""

    def test_default_result(self):
        """Default result values."""
        result = PoolTrainingResult(provider=Provider.LOCAL, success=True)
        assert result.provider == Provider.LOCAL
        assert result.success is True
        assert result.final_loss is None
        assert result.checkpoint_path is None
        assert result.steps_completed == 0
        assert result.total_time_seconds == 0.0

    def test_failed_result(self):
        """Failed result with error."""
        result = PoolTrainingResult(
            provider=Provider.MODAL,
            success=False,
            error="Preempted after 500 steps",
            steps_completed=500,
            total_time_seconds=3600.0,
        )
        assert result.success is False
        assert result.error == "Preempted after 500 steps"
        assert result.steps_completed == 500
        assert result.total_time_seconds == 3600.0

    def test_successful_result(self):
        """Successful result with metrics."""
        result = PoolTrainingResult(
            provider=Provider.MODAL,
            success=True,
            final_loss=0.00123,
            checkpoint_path="/outputs/checkpoints/dit_model.pt",
            steps_completed=100_000,
            total_time_seconds=7200.0,
        )
        assert result.success is True
        assert result.final_loss == 0.00123
        assert result.checkpoint_path is not None
        assert result.steps_completed == 100_000
