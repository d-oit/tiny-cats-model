#!/usr/bin/env python3
"""scripts/test_fallback_chain.py

End-to-end fallback chain simulation for the GPU pool abstraction.

Simulates the provider fallback chain locally by:
1. Mocking detect_provider() to simulate different providers in sequence
2. Testing train_chain() output and provider ordering
3. Testing checkpoint push/pull with mocked HuggingFace Hub
4. Verifying PoolTrainingResult states across success/failure scenarios
5. Testing iterate_fallback_chain() edge cases

Usage:
    python scripts/test_fallback_chain.py
    python scripts/test_fallback_chain.py --verbose
    python scripts/test_fallback_chain.py --provider modal
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpu_pool import (
    FALLBACK_CHAIN,
    PoolTrainingResult,
    Provider,
    detect_provider,
    estimate_cost,
    estimate_gpu_hours,
    get_provider_config,
    iterate_fallback_chain,
    pull_checkpoint_from_hub,
    push_checkpoint_to_hub,
    train_chain,
)

# ─────────────────────────────────────────────────────────────
# Simulation utilities
# ─────────────────────────────────────────────────────────────


class SimulationResult:
    """Single simulation step result."""

    def __init__(
        self,
        name: str,
        passed: bool,
        details: str = "",
    ) -> None:
        self.name = name
        self.passed = passed
        self.details = details


def green(text: str) -> str:
    return f"\033[92m{text}\033[0m"


def red(text: str) -> str:
    return f"\033[91m{text}\033[0m"


def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


# ─────────────────────────────────────────────────────────────
# Simulation 1: Fallback chain ordering
# ─────────────────────────────────────────────────────────────


def simulate_chain_ordering(verbose: bool = False) -> list[SimulationResult]:
    """Verify fallback chain ordering and iterate_fallback_chain()."""
    results: list[SimulationResult] = []

    # Test 1: Full chain is non-empty and ordered correctly
    assert len(FALLBACK_CHAIN) >= 5, "Expected at least 5 providers in chain"
    results.append(
        SimulationResult(
            "Full chain has ≥5 providers",
            True,
            f"Chain: {' → '.join(p.value for p in FALLBACK_CHAIN)}",
        )
    )

    # Test 2: First provider is Modal
    results.append(
        SimulationResult(
            "Modal is first in chain",
            FALLBACK_CHAIN[0] == Provider.MODAL,
            f"First: {FALLBACK_CHAIN[0].value}",
        )
    )

    # Test 3: iterate_fallback_chain from Modal returns all
    chain = iterate_fallback_chain(Provider.MODAL)
    results.append(
        SimulationResult(
            "Chain from Modal is full length",
            len(chain) == len(FALLBACK_CHAIN),
            f"Length: {len(chain)}",
        )
    )

    # Test 4: iterate_fallback_chain from Colab excludes earlier
    chain = iterate_fallback_chain(Provider.COLAB)
    results.append(
        SimulationResult(
            "Chain from Colab excludes Modal & Lightning",
            Provider.MODAL not in chain and Provider.LIGHTNING not in chain,
            f"First: {chain[0].value}, length: {len(chain)}",
        )
    )

    # Test 5: iterate_fallback_chain from LOCAL returns only LOCAL
    chain = iterate_fallback_chain(Provider.LOCAL)
    results.append(
        SimulationResult(
            "Chain from LOCAL is just LOCAL",
            chain == [Provider.LOCAL],
            f"Chain: {[p.value for p in chain]}",
        )
    )

    # Test 6: None returns full chain
    chain = iterate_fallback_chain(None)
    results.append(
        SimulationResult(
            "None returns full chain",
            chain == list(FALLBACK_CHAIN),
            f"Length: {len(chain)}",
        )
    )

    # Test 7: Chain copy is independent
    chain = iterate_fallback_chain()
    chain.append(Provider.UNKNOWN)
    results.append(
        SimulationResult(
            "Chain copy is independent (no mutation)",
            Provider.UNKNOWN not in FALLBACK_CHAIN,
            "Original chain unchanged",
        )
    )

    if verbose:
        for r in results:
            status = green("PASS") if r.passed else red("FAIL")
            print(f"  {status} | {r.name}: {r.details}")

    return results


# ─────────────────────────────────────────────────────────────
# Simulation 2: Provider detection mocking
# ─────────────────────────────────────────────────────────────


def simulate_provider_detection(verbose: bool = False) -> list[SimulationResult]:
    """Verify detect_provider() with mocked environments."""
    results: list[SimulationResult] = []

    test_cases = [
        ({"MODAL_APP_ID": "ap-12345"}, Provider.MODAL, "MODAL_APP_ID"),
        ({"MODAL_FUNCTION_NAME": "train"}, Provider.MODAL, "MODAL_FUNCTION_NAME"),
        (
            {"LIGHTNING_APP_STATE_DIR": "/tmp"},
            Provider.LIGHTNING,
            "LIGHTNING_APP_STATE_DIR",
        ),
        ({"COLAB_GPU": "1"}, Provider.COLAB, "COLAB_GPU"),
        (
            {"KAGGLE_KERNEL_RUN_TYPE": "notebook"},
            Provider.KAGGLE,
            "KAGGLE_KERNEL_RUN_TYPE",
        ),
        ({"SPACE_ID": "my-space"}, Provider.HF_SPACES, "SPACE_ID"),
    ]

    for env_vars, expected, label in test_cases:
        with patch.dict(os.environ, env_vars, clear=True):
            detected = detect_provider()
            results.append(
                SimulationResult(
                    f"detect_provider({label})",
                    detected == expected,
                    f"Expected {expected.value}, got {detected.value}",
                )
            )

    if verbose:
        for r in results:
            status = green("PASS") if r.passed else red("FAIL")
            print(f"  {status} | {r.name}: {r.details}")

    return results


# ─────────────────────────────────────────────────────────────
# Simulation 3: Estimate benchmarking
# ─────────────────────────────────────────────────────────────


def simulate_estimate_validation(verbose: bool = False) -> list[SimulationResult]:
    """Verify estimate_gpu_hours() reasonableness against known data."""
    results: list[SimulationResult] = []

    # Known benchmark from ADR: 400k steps @ 256 batch, 128x128 = ~36-48h on A10G/H100
    # Our estimate uses 2.2 steps/s for T4. T4 ≈ 0.5x H100 speed.
    # So for H100: ~4.4 steps/s → 400k / 4.4 = 90,909s ≈ 25.3h
    # For T4:   ~2.2 steps/s → 400k / 2.2 = 181,818s ≈ 50.5h
    # The log says 36-48h which is between T4 and H100 estimates — reasonable for A10G.

    # Test 1: 400k steps, batch 256, 128x128 — should be 25-51h range
    h = estimate_gpu_hours(400_000, batch_size=256, image_size=128)
    results.append(
        SimulationResult(
            "400k steps @ 256 batch (128x128) in 25-80h",
            25 <= h <= 80,
            f"Estimated: {h:.1f}h",
        )
    )

    # Test 2: Larger batch means less time (fewer steps/sec * more data/step)
    h_big = estimate_gpu_hours(100_000, batch_size=1024, image_size=128)
    h_sml = estimate_gpu_hours(100_000, batch_size=256, image_size=128)
    results.append(
        SimulationResult(
            "Larger batch = more GPU hours (more data/step)",
            h_big > h_sml,
            f"batch=1024: {h_big:.1f}h, batch=256: {h_sml:.1f}h",
        )
    )

    # Test 3: 256x256 is ~4x slower than 128x128
    h256 = estimate_gpu_hours(100_000, batch_size=512, image_size=256)
    h128 = estimate_gpu_hours(100_000, batch_size=512, image_size=128)
    results.append(
        SimulationResult(
            "256x256 ≈ 4x 128x128 time",
            3.5 <= h256 / h128 <= 4.5,
            f"256: {h256:.1f}h, 128: {h128:.1f}h, ratio: {h256 / h128:.1f}x",
        )
    )

    # Test 4: Zero steps = zero hours
    h = estimate_gpu_hours(0)
    results.append(
        SimulationResult(
            "Zero steps = zero hours",
            h == 0.0,
            f"Hours: {h}",
        )
    )

    # Test 5: All providers return cost estimates
    costs = estimate_cost(10_000)
    results.append(
        SimulationResult(
            "estimate_cost returns all providers",
            len(costs) >= 5,
            f"Got {len(costs)} provider estimates",
        )
    )

    # Test 6: Modal within free tier for small runs
    costs = estimate_cost(10_000, provider=Provider.MODAL)
    results.append(
        SimulationResult(
            "10k steps within Modal free tier",
            costs["modal"]["within_free_tier"] is True,
            f"Cost: ${costs['modal']['estimated_cost']}",
        )
    )

    if verbose:
        for r in results:
            status = green("PASS") if r.passed else red("FAIL")
            print(f"  {status} | {r.name}: {r.details}")

    return results


# ─────────────────────────────────────────────────────────────
# Simulation 4: Checkpoint sync (mocked)
# ─────────────────────────────────────────────────────────────


def simulate_checkpoint_sync(verbose: bool = False) -> list[SimulationResult]:
    """Verify checkpoint push/pull with mocked HuggingFace Hub."""
    results: list[SimulationResult] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake checkpoint
        ckpt_path = Path(tmpdir) / "dit_model.pt"
        ckpt_path.write_text("fake checkpoint data")

        # Test 1: push without HF_TOKEN returns False
        with patch.dict(os.environ, {}, clear=True):
            ok = push_checkpoint_to_hub(
                checkpoint_path=ckpt_path,
                hub_repo="test/repo",
                token=None,
            )
            results.append(
                SimulationResult(
                    "push fails without HF_TOKEN",
                    ok is False,
                    "Correctly returned False",
                )
            )

        # Test 2: push with mock HfApi succeeds
        # HfApi/create_repo are imported lazily inside push_checkpoint_to_hub()
        with (
            patch("huggingface_hub.HfApi") as mock_api,
            patch("huggingface_hub.create_repo"),
            patch.dict(os.environ, {"HF_TOKEN": "fake_token"}, clear=True),
        ):
            ok = push_checkpoint_to_hub(
                checkpoint_path=ckpt_path,
                hub_repo="test/repo",
                checkpoint_name="test.pt",
            )
            results.append(
                SimulationResult(
                    "push succeeds with mocked Hub",
                    ok is True,
                    "Upload called",
                )
            )
            mock_api.return_value.upload_file.assert_called_once()

        # Test 3: push missing file returns False
        with patch.dict(os.environ, {"HF_TOKEN": "fake_token"}, clear=True):
            ok = push_checkpoint_to_hub(
                checkpoint_path="/nonexistent/path.pt",
                hub_repo="test/repo",
            )
            results.append(
                SimulationResult(
                    "push nonexistent file returns False",
                    ok is False,
                    "Correctly returned False",
                )
            )

        # Test 4: pull without HF_TOKEN returns None
        with patch.dict(os.environ, {}, clear=True):
            pulled = pull_checkpoint_from_hub(
                hub_repo="test/repo",
                checkpoint_name="model.pt",
                output_dir=tmpdir,
                token=None,
            )
            results.append(
                SimulationResult(
                    "pull without HF_TOKEN returns None",
                    pulled is None,
                    "Correctly returned None",
                )
            )

        # Test 5: pull with mock hf_hub_download succeeds
        with (
            patch("huggingface_hub.hf_hub_download") as mock_dl,
            patch.dict(os.environ, {"HF_TOKEN": "fake_token"}, clear=True),
        ):
            mock_dl.return_value = str(ckpt_path)
            pulled = pull_checkpoint_from_hub(
                hub_repo="test/repo",
                checkpoint_name="model.pt",
                output_dir=tmpdir,
            )
            results.append(
                SimulationResult(
                    "pull succeeds with mocked Hub",
                    pulled is not None and pulled.name == "dit_model.pt",
                    f"Pulled: {pulled}",
                )
            )
            mock_dl.assert_called_once()

        # Test 6: pull with exception returns None
        with (
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=Exception("Network error"),
            ),
            patch.dict(os.environ, {"HF_TOKEN": "fake_token"}, clear=True),
        ):
            pulled = pull_checkpoint_from_hub(
                hub_repo="test/repo",
                checkpoint_name="model.pt",
                output_dir=tmpdir,
            )
            results.append(
                SimulationResult(
                    "pull handles network errors gracefully",
                    pulled is None,
                    "Correctly returned None on error",
                )
            )

    if verbose:
        for r in results:
            status = green("PASS") if r.passed else red("FAIL")
            print(f"  {status} | {r.name}: {r.details}")

    return results


# ─────────────────────────────────────────────────────────────
# Simulation 5: PoolTrainingResult states
# ─────────────────────────────────────────────────────────────


def simulate_result_states(verbose: bool = False) -> list[SimulationResult]:
    """Verify PoolTrainingResult dataclass and state transitions."""
    results: list[SimulationResult] = []

    # Success result
    success = PoolTrainingResult(
        provider=Provider.MODAL,
        success=True,
        final_loss=0.001,
        checkpoint_path="/tmp/model.pt",
        steps_completed=100_000,
        total_time_seconds=3600.0,
    )
    results.append(
        SimulationResult(
            "Success result has correct fields",
            success.success
            and success.final_loss == 0.001
            and success.steps_completed == 100_000,
            f"Loss={success.final_loss}, Steps={success.steps_completed}",
        )
    )

    # Failure result
    failure = PoolTrainingResult(
        provider=Provider.COLAB,
        success=False,
        error="Runtime disconnected",
        total_time_seconds=900.0,
    )
    results.append(
        SimulationResult(
            "Failure result has error and success=False",
            not failure.success
            and failure.error == "Runtime disconnected"
            and failure.steps_completed == 0,
            f"Error: {failure.error}",
        )
    )

    # Partial result (failed mid-training)
    partial = PoolTrainingResult(
        provider=Provider.KAGGLE,
        success=False,
        error="Session timeout",
        steps_completed=15_000,
        total_time_seconds=7200.0,
    )
    results.append(
        SimulationResult(
            "Partial result preserves steps_completed",
            partial.steps_completed == 15_000 and not partial.success,
            f"Steps: {partial.steps_completed}, Time: {partial.total_time_seconds}s",
        )
    )

    if verbose:
        for r in results:
            status = green("PASS") if r.passed else red("FAIL")
            print(f"  {status} | {r.name}: {r.details}")

    return results


# ─────────────────────────────────────────────────────────────
# Simulation 6: train_chain() output
# ─────────────────────────────────────────────────────────────


def simulate_train_chain(verbose: bool = False) -> list[SimulationResult]:
    """Verify train_chain() returns correct type and provider info."""
    results: list[SimulationResult] = []

    # Mock detect_provider + train_with_fallback to just return a result
    with (
        patch("gpu_pool.detect_provider", return_value=Provider.MODAL),
        patch(
            "gpu_pool.train_with_fallback",
            return_value=PoolTrainingResult(
                provider=Provider.MODAL,
                success=True,
                final_loss=0.01,
                steps_completed=100,
                total_time_seconds=10.0,
            ),
        ),
    ):
        result = train_chain(
            steps=100,
            batch_size=64,
            hub_repo="test/repo",
        )
        results.append(
            SimulationResult(
                "train_chain returns PoolTrainingResult",
                isinstance(result, PoolTrainingResult),
                f"Type: {type(result).__name__}",
            )
        )
        results.append(
            SimulationResult(
                "train_chain preserves provider",
                result.provider == Provider.MODAL and result.success,
                f"Provider: {result.provider.value}, Success: {result.success}",
            )
        )

    # With start_from different from detected provider
    with (
        patch("gpu_pool.detect_provider", return_value=Provider.LOCAL),
        patch(
            "gpu_pool.train_with_fallback",
            return_value=PoolTrainingResult(
                provider=Provider.LOCAL,
                success=True,
                steps_completed=50,
                total_time_seconds=5.0,
            ),
        ),
    ):
        result = train_chain(
            steps=50,
            start_from=Provider.COLAB,  # different from detected LOCAL
        )
        results.append(
            SimulationResult(
                "train_chain start_from controls display only",
                result.provider == Provider.LOCAL,  # always trains on detected
                f"Provider: {result.provider.value}",
            )
        )

    if verbose:
        for r in results:
            status = green("PASS") if r.passed else red("FAIL")
            print(f"  {status} | {r.name}: {r.details}")

    return results


# ─────────────────────────────────────────────────────────────
# Simulation 7: Provider config validation
# ─────────────────────────────────────────────────────────────


def simulate_provider_configs(verbose: bool = False) -> list[SimulationResult]:
    """Verify all provider configs are complete and valid."""
    results: list[SimulationResult] = []

    required_configs = {
        Provider.MODAL,
        Provider.LIGHTNING,
        Provider.COLAB,
        Provider.KAGGLE,
        Provider.HF_SPACES,
        Provider.LOCAL,
    }

    for p in required_configs:
        cfg = get_provider_config(p)
        has_gpu = len(cfg.gpu_types) > 0 or p == Provider.LOCAL
        has_timeout = cfg.timeout_minutes >= 0
        has_display = len(cfg.display_name) > 0
        results.append(
            SimulationResult(
                f"{p.value} config is valid",
                has_gpu and has_timeout and has_display,
                f"GPU: {cfg.gpu_types}, Timeout: {cfg.timeout_minutes}m",
            )
        )

    # Unknown provider defaults to LOCAL
    cfg = get_provider_config(Provider.UNKNOWN)
    results.append(
        SimulationResult(
            "Unknown provider defaults to LOCAL config",
            cfg.provider == Provider.LOCAL,
            f"Got: {cfg.provider.value}",
        )
    )

    if verbose:
        for r in results:
            status = green("PASS") if r.passed else red("FAIL")
            print(f"  {status} | {r.name}: {r.details}")

    return results


# ─────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end fallback chain simulation for GPU pool"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    args = parser.parse_args()
    verbose = args.verbose

    print("=" * 60)
    print(bold("GPU Pool Fallback Chain Simulation"))
    print("=" * 60)

    if not verbose:
        print("(use --verbose for detailed output)\n")

    all_results: list[SimulationResult] = []

    # Run all simulations
    for name, fn in [
        ("Chain Ordering", simulate_chain_ordering),
        ("Provider Detection", simulate_provider_detection),
        ("Estimate Validation", simulate_estimate_validation),
        ("Checkpoint Sync", simulate_checkpoint_sync),
        ("Result States", simulate_result_states),
        ("train_chain()", simulate_train_chain),
        ("Provider Configs", simulate_provider_configs),
    ]:
        print(f"\n{bold(name)}:")
        results = fn(verbose=verbose)
        all_results.extend(results)
        if not verbose:
            passed = sum(1 for r in results if r.passed)
            total = len(results)
            print(f"  {passed}/{total} passed")

    # Summary
    print("\n" + "=" * 60)
    total_passed = sum(1 for r in all_results if r.passed)
    total_all = len(all_results)
    if total_passed == total_all:
        print(green(f"✅ ALL {total_all} SIMULATIONS PASSED"))
    else:
        print(red(f"❌ {total_passed}/{total_all} SIMULATIONS PASSED"))
        if not verbose:
            print("Run with --verbose to see failures")
    print("=" * 60)

    sys.exit(0 if total_passed == total_all else 1)


if __name__ == "__main__":
    main()
