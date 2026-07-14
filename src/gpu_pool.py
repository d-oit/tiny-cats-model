"""src/gpu_pool.py

Unified GPU pool abstraction for training across free GPU providers.

Providers:
- Modal ($30/mo free credits, T4/L4 GPU)
- Lightning AI (free T4 GPU hours/day, Lightning Studio)
- Google Colab (free T4 GPU, daily limit)
- Kaggle (free T4/P100 GPU, 30h/week)
- HuggingFace Spaces (free T4 GPU, limited)
- Tinker (managed LoRA API — LLM fine-tuning only, excluded for custom models)

Architecture:
    ┌──────────────────────────────────────────────────────┐
    │                    GPU Pool                           │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
    │  │  Modal   │→ │Lightning │→ │  Colab   │→ ...      │
    │  │ (T4/L4)  │  │  (T4)    │  │  (T4)    │           │
    │  └──────────┘  └──────────┘  └──────────┘           │
    │        ↓               ↓              ↓              │
    │  ┌──────────────────────────────────────────┐       │
    │  │       HuggingFace Hub (checkpoint sync)   │       │
    │  └──────────────────────────────────────────┘       │
    └──────────────────────────────────────────────────────┘

Usage:
    from gpu_pool import detect_provider, get_provider_config, train_with_fallback

    # Detect current environment
    provider = detect_provider()
    config = get_provider_config(provider)

    # Train with automatic fallback chain
    train_with_fallback(
        data_dir="/data/cats",
        steps=100_000,
        batch_size=512,
        hub_repo="d4oit/tiny-cats-model",
    )
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Provider definitions
# ──────────────────────────────────────────────────────────────────────


class Provider(str, Enum):
    """Supported free GPU providers."""

    MODAL = "modal"
    LIGHTNING = "lightning"
    COLAB = "colab"
    KAGGLE = "kaggle"
    HF_SPACES = "hf_spaces"
    LOCAL = "local"
    UNKNOWN = "unknown"


@dataclass
class ProviderConfig:
    """Configuration for a GPU provider."""

    provider: Provider
    display_name: str
    gpu_types: list[str] = field(default_factory=lambda: ["T4"])
    free_hours_per_day: float | None = None  # None = unlimited (credits-based)
    free_credits_per_month: float = 0.0  # USD
    cost_per_hour: float = 0.0  # USD (if exceeding free tier)
    timeout_minutes: int = 720  # default 12h
    data_dir: str = "/data/cats"
    checkpoint_dir: str = "checkpoints"
    env_vars: dict[str, str] = field(default_factory=dict)
    preemptible: bool = True  # can be revoked at any time


# Provider registry
PROVIDER_CONFIGS: dict[Provider, ProviderConfig] = {
    Provider.MODAL: ProviderConfig(
        provider=Provider.MODAL,
        display_name="Modal",
        gpu_types=["T4", "L4"],
        free_credits_per_month=30.0,
        cost_per_hour=0.59,  # T4
        timeout_minutes=1440,  # 24h max
        data_dir="/data/cats",
        checkpoint_dir="/outputs/checkpoints",
        env_vars={
            "MODAL_GPU_TYPE": "T4",
        },
        preemptible=True,
    ),
    Provider.LIGHTNING: ProviderConfig(
        provider=Provider.LIGHTNING,
        display_name="Lightning AI",
        gpu_types=["T4"],
        free_hours_per_day=22,  # varies
        timeout_minutes=720,
        data_dir="/teamspace/datasets/cats",
        checkpoint_dir="/teamspace/checkpoints",
        env_vars={
            "LIGHTNING_GPU_TYPE": "T4",
        },
        preemptible=True,
    ),
    Provider.COLAB: ProviderConfig(
        provider=Provider.COLAB,
        display_name="Google Colab",
        gpu_types=["T4"],
        free_hours_per_day=12,
        timeout_minutes=720,
        data_dir="/content/drive/MyDrive/tiny-cats-data",
        checkpoint_dir="/content/drive/MyDrive/tiny-cats-checkpoints",
        env_vars={},
        preemptible=True,
    ),
    Provider.KAGGLE: ProviderConfig(
        provider=Provider.KAGGLE,
        display_name="Kaggle",
        gpu_types=["T4", "P100"],
        free_hours_per_day=4.3,  # ~30h/week
        timeout_minutes=540,  # 9h per session
        data_dir="/kaggle/input/cats-dataset",
        checkpoint_dir="/kaggle/working/checkpoints",
        env_vars={},
        preemptible=True,
    ),
    Provider.HF_SPACES: ProviderConfig(
        provider=Provider.HF_SPACES,
        display_name="HuggingFace Spaces",
        gpu_types=["T4"],
        free_hours_per_day=16,
        timeout_minutes=480,
        data_dir="/data/cats",
        checkpoint_dir="/data/checkpoints",
        env_vars={},
        preemptible=True,
    ),
    Provider.LOCAL: ProviderConfig(
        provider=Provider.LOCAL,
        display_name="Local CPU/GPU",
        gpu_types=[],
        timeout_minutes=0,  # no limit
        data_dir="data/cats",
        checkpoint_dir="checkpoints",
        preemptible=False,
    ),
}


# Fallback chain priority (most capable/cost-effective first).
# Iterated by train_with_fallback() and available for callers who need
# multi-provider orchestration via iterate_fallback_chain().
FALLBACK_CHAIN: list[Provider] = [
    Provider.MODAL,
    Provider.LIGHTNING,
    Provider.COLAB,
    Provider.KAGGLE,
    Provider.HF_SPACES,
    Provider.LOCAL,
]


def iterate_fallback_chain(
    start_from: Provider | None = None,
) -> list[Provider]:
    """Return the fallback chain starting from (and including) a provider.

    Args:
        start_from: Provider to start from. If None or not in FALLBACK_CHAIN,
            returns the full chain.

    Returns:
        Ordered list of providers to try.

    Example:
        # Try Lightning, then Colab, then Kaggle, etc.
        for p in iterate_fallback_chain(Provider.LIGHTNING):
            result = try_training_on(p)
            if result.success:
                break
    """
    if start_from is None or start_from not in FALLBACK_CHAIN:
        return list(FALLBACK_CHAIN)

    start_idx = FALLBACK_CHAIN.index(start_from)
    return list(FALLBACK_CHAIN[start_idx:])


def train_chain(
    steps: int = 100_000,
    batch_size: int = 512,
    lr: float = 5e-5,
    image_size: int = 128,
    hub_repo: str = "d4oit/tiny-cats-model",
    hub_token: str | None = None,
    start_from: Provider | None = None,
    **train_kwargs: Any,
) -> PoolTrainingResult:
    """Train on the current provider with planned fallback chain.

    Prints the FALLBACK_CHAIN order for the operator. Training runs on the
    current (detected) provider only — cross-provider orchestration happens
    by running this same script separately on each provider, with HuggingFace
    Hub checkpoint sync bridging the runs.

    To skip the current provider (e.g., after a failure): run on a different
    machine or pass ``start_from`` to begin later in the chain.

    Args:
        steps: Total training steps.
        batch_size: Batch size.
        lr: Learning rate.
        image_size: Image size.
        hub_repo: HF Hub repo id.
        hub_token: HF token.
        start_from: Provider to start from (skips earlier providers).
            If None, starts from the auto-detected current provider.
        **train_kwargs: Additional args for train_dit_local.

    Returns:
        PoolTrainingResult with final status.
    """
    current = detect_provider()
    # start_from controls the fallback-chain display only —
    # actual training always runs on the detected current provider
    chain_start = start_from if start_from is not None else current
    providers = iterate_fallback_chain(chain_start)

    logger.info(f"Fallback chain: {' → '.join(p.value for p in providers)}")
    logger.info(
        f"Currently on: {current.value}. "
        f"Run this script on each provider sequentially; "
        f"checkpoints sync via HuggingFace Hub."
    )

    return train_with_fallback(
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        image_size=image_size,
        hub_repo=hub_repo,
        hub_token=hub_token,
        **train_kwargs,
    )


# ──────────────────────────────────────────────────────────────────────
# Environment detection
# ──────────────────────────────────────────────────────────────────────


def detect_provider() -> Provider:
    """Detect which GPU provider environment we're running in.

    Detection order:
    1. Modal: MODAL_APP_ID env var
    2. Lightning: LIGHTNING_APP_STATE_DIR or /teamspace/ path
    3. Colab: google.colab in sys.modules
    4. Kaggle: KAGGLE_KERNEL_RUN_TYPE env var
    5. HF Spaces: SPACE_ID or HF_HOME env var
    6. Local: none of the above

    Returns:
        Detected provider, or Provider.UNKNOWN if ambiguous.
    """
    # Modal
    if os.environ.get("MODAL_APP_ID") or os.environ.get("MODAL_FUNCTION_NAME"):
        return Provider.MODAL
    if os.environ.get("MODAL_ENVIRONMENT"):
        return Provider.MODAL

    # Lightning AI Studio
    if os.environ.get("LIGHTNING_APP_STATE_DIR"):
        return Provider.LIGHTNING
    if os.environ.get("LIGHTNING_STUDIO_ID"):
        return Provider.LIGHTNING
    if Path("/teamspace").exists():
        return Provider.LIGHTNING

    # Google Colab (check sys.modules — avoids import in non-Colab envs)
    if "google.colab" in sys.modules:
        return Provider.COLAB
    if os.environ.get("COLAB_GPU"):
        return Provider.COLAB

    # Kaggle
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return Provider.KAGGLE
    if os.environ.get("KAGGLE_URL_BASE"):
        return Provider.KAGGLE

    # HuggingFace Spaces
    if os.environ.get("SPACE_ID"):
        return Provider.HF_SPACES
    if os.environ.get("HF_HOME") and Path("/data").exists():
        return Provider.HF_SPACES

    # Check for local GPU availability
    try:
        import torch

        if torch.cuda.is_available():
            return Provider.LOCAL
    except ImportError:
        pass

    return Provider.UNKNOWN


def detect_provider_and_log() -> Provider:
    """Detect provider and log the result with relevant details."""
    provider = detect_provider()
    config = get_provider_config(provider)

    logger.info("=" * 60)
    logger.info(f"DETECTED PROVIDER: {config.display_name} ({provider.value})")
    logger.info(f"  GPU types: {config.gpu_types}")
    logger.info(f"  Data dir: {config.data_dir}")
    logger.info(f"  Checkpoint dir: {config.checkpoint_dir}")
    logger.info(f"  Timeout: {config.timeout_minutes} min")
    logger.info(f"  Preemptible: {config.preemptible}")

    if config.free_credits_per_month:
        logger.info(f"  Free credits: ${config.free_credits_per_month}/mo")
    if config.free_hours_per_day:
        logger.info(f"  Free hours: {config.free_hours_per_day}h/day")
    logger.info("=" * 60)

    return provider


def get_provider_config(provider: Provider) -> ProviderConfig:
    """Get configuration for a provider.

    Args:
        provider: The provider to get config for.

    Returns:
        ProviderConfig, defaulting to LOCAL for unknown providers.
    """
    return PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS[Provider.LOCAL])


# ──────────────────────────────────────────────────────────────────────
# Checkpoint sync via HuggingFace Hub
# ──────────────────────────────────────────────────────────────────────


_HF_AVAILABLE = None


def _check_hf_available() -> bool:
    """Check if huggingface_hub is available (lazy, cached)."""
    global _HF_AVAILABLE
    if _HF_AVAILABLE is None:
        try:
            import huggingface_hub  # noqa: F401

            _HF_AVAILABLE = True
        except ImportError:
            _HF_AVAILABLE = False
    return _HF_AVAILABLE


def push_checkpoint_to_hub(
    checkpoint_path: str | Path,
    hub_repo: str = "d4oit/tiny-cats-model",
    checkpoint_name: str | None = None,
    token: str | None = None,
) -> bool:
    """Push a checkpoint to HuggingFace Hub for cross-provider sync.

    Args:
        checkpoint_path: Local path to checkpoint file or directory.
        hub_repo: HF Hub repo ID (username/repo-name).
        checkpoint_name: Optional name for the checkpoint on the hub.
        token: HF token (defaults to HF_TOKEN env var).

    Returns:
        True if upload succeeded, False otherwise.
    """
    if not _check_hf_available():
        logger.warning(
            "huggingface_hub not installed. Install with: pip install huggingface_hub"
        )
        return False

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        logger.warning("No HF_TOKEN set — cannot push checkpoint to Hub")
        return False

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return False

    try:
        from huggingface_hub import HfApi, create_repo

        api = HfApi(token=token)

        # Ensure repo exists
        try:
            create_repo(hub_repo, exist_ok=True, repo_type="model", token=token)
        except Exception as e:
            # create_repo failure will cascade into upload failure with a
            # confusing error message — fail fast and surface the root cause.
            logger.error(f"Could not create/verify repo {hub_repo}: {e}")
            return False

        # Determine remote path
        if checkpoint_name is None:
            checkpoint_name = checkpoint_path.name

        remote_path = f"checkpoints/pool/{checkpoint_name}"

        if checkpoint_path.is_file():
            api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=remote_path,
                repo_id=hub_repo,
                repo_type="model",
                token=token,
                commit_message=f"pool: push checkpoint {checkpoint_name}",
            )
        elif checkpoint_path.is_dir():
            api.upload_folder(
                folder_path=str(checkpoint_path),
                path_in_repo=remote_path,
                repo_id=hub_repo,
                repo_type="model",
                token=token,
                commit_message=f"pool: push checkpoint dir {checkpoint_name}",
            )

        logger.info(f"✅ Pushed checkpoint to: {hub_repo}/{remote_path}")
        return True

    except Exception as e:
        logger.warning(f"Checkpoint push failed: {e}")
        return False


def pull_checkpoint_from_hub(
    hub_repo: str = "d4oit/tiny-cats-model",
    checkpoint_name: str = "dit_model.pt",
    output_dir: str | Path = "checkpoints",
    token: str | None = None,
) -> Path | None:
    """Pull a checkpoint from HuggingFace Hub for cross-provider resume.

    Args:
        hub_repo: HF Hub repo ID (username/repo-name).
        checkpoint_name: Name of the checkpoint file on the hub.
        output_dir: Local directory to save the checkpoint.
        token: HF token (defaults to HF_TOKEN env var).

    Returns:
        Path to downloaded checkpoint, or None if not found.
    """
    if not _check_hf_available():
        logger.warning("huggingface_hub not installed")
        return None

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    try:
        from huggingface_hub import hf_hub_download

        remote_path = f"checkpoints/pool/{checkpoint_name}"

        # Ensure output_dir exists — hf_hub_download() requires it.
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        local_path = hf_hub_download(
            repo_id=hub_repo,
            filename=remote_path,
            repo_type="model",
            token=token,
            local_dir=str(output_dir),
        )

        logger.info(f"✅ Pulled checkpoint from: {hub_repo}/{remote_path}")
        logger.info(f"   Saved to: {local_path}")
        return Path(local_path)

    except Exception as e:
        logger.info(f"No checkpoint found on Hub ({e}) — starting fresh")
        return None


# ──────────────────────────────────────────────────────────────────────
# Training with fallback chain
# ──────────────────────────────────────────────────────────────────────


@dataclass
class PoolTrainingResult:
    """Result of a pooled training attempt."""

    provider: Provider
    success: bool
    final_loss: float | None = None
    checkpoint_path: str | None = None
    error: str | None = None
    steps_completed: int = 0
    total_time_seconds: float = 0.0


def train_with_fallback(
    data_dir: str | None = None,
    steps: int = 100_000,
    batch_size: int = 512,
    lr: float = 5e-5,
    image_size: int = 128,
    hub_repo: str = "d4oit/tiny-cats-model",
    hub_token: str | None = None,
    **train_kwargs: Any,
) -> PoolTrainingResult:
    """Train TinyDiT on the current provider with Hub-based checkpoint resume.

    NOTE: The function name is historical. ``train_with_fallback`` runs the
    training ONCE on ``detect_provider()``'s reported provider. Cross-provider
    FALLBACK involves running this script separately on each provider (see
    ``train_chain()`` for the orchestration order); checkpoints are bridged
    via HuggingFace Hub. ``iterate_fallback_chain()`` is exposed for callers
    that want to programmatically walk the chain.

    Within a single run, the function:
      1. Pulls the latest checkpoint from Hub (resume if present).
      2. Runs ``train_dit_local()`` end-to-end on the current provider.
      3. Pushes the resulting EMA checkpoint back to Hub.

    If a different provider is needed after a failure, re-run this script
    on that machine — Hub will resume from the latest checkpoint automatically.

    Args:
        data_dir: Dataset directory (auto-detected per provider).
        steps: Total training steps.
        batch_size: Batch size.
        lr: Learning rate.
        image_size: Image size (128 or 256).
        hub_repo: HF Hub repo for checkpoint sync.
        hub_token: HF token for checkpoint sync.
        **train_kwargs: Additional args passed to train_dit_local.

    Returns:
        PoolTrainingResult with final status.
    """
    provider = detect_provider_and_log()
    config = get_provider_config(provider)

    # Use provider defaults
    data_dir = data_dir or config.data_dir

    logger.info(f"Starting pooled training on: {config.display_name}")
    logger.info(
        f"Config: steps={steps:,}, batch_size={batch_size}, "
        f"lr={lr}, image_size={image_size}"
    )

    # Try to pull existing checkpoint from Hub
    checkpoint_name = "dit_model_ema.pt"
    pulled = pull_checkpoint_from_hub(
        hub_repo=hub_repo,
        checkpoint_name=checkpoint_name,
        output_dir=config.checkpoint_dir,
        token=hub_token,
    )
    resume_from = str(pulled) if pulled else None

    if resume_from:
        logger.info(f"Resuming from Hub checkpoint: {resume_from}")
    else:
        logger.info("No Hub checkpoint found — starting fresh")

    # Setup data directory
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.info(f"Data dir not found: {data_dir}, attempting download...")
        _setup_data_for_provider(provider, data_dir)

    # Import and run training
    start_time = time.time()

    try:
        from train_dit import train_dit_local

        # Set checkpoint dir
        checkpoint_dir = config.checkpoint_dir
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        output = train_kwargs.pop(
            "output",
            str(Path(checkpoint_dir) / "dit_model.pt"),
        )
        ema_output = train_kwargs.pop(
            "ema_output",
            str(Path(checkpoint_dir) / "dit_model_ema.pt"),
        )

        final_loss = train_dit_local(
            data_dir=data_dir,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            image_size=image_size,
            output=output,
            ema_output=ema_output,
            resume=resume_from,
            **train_kwargs,
        )

        elapsed = time.time() - start_time

        # Push checkpoint back to Hub for next provider
        hub_token = hub_token or os.environ.get("HF_TOKEN")
        if hub_token:
            push_checkpoint_to_hub(
                checkpoint_path=ema_output,
                hub_repo=hub_repo,
                checkpoint_name=checkpoint_name,
                token=hub_token,
            )
            logger.info("✅ Checkpoint synced to Hub for cross-provider resume")

        logger.info(f"Training completed in {elapsed:.0f}s on {config.display_name}")

        return PoolTrainingResult(
            provider=provider,
            success=True,
            final_loss=final_loss,
            checkpoint_path=output,
            steps_completed=steps,
            total_time_seconds=elapsed,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Training failed on {config.display_name}: {e}")

        # Resolve token from env if not explicitly passed (match success path)
        hub_token = hub_token or os.environ.get("HF_TOKEN")

        # Try to push any partial checkpoint
        partial_path = Path(config.checkpoint_dir) / "dit_model.pt"
        if partial_path.exists() and hub_token:
            push_checkpoint_to_hub(
                checkpoint_path=partial_path,
                hub_repo=hub_repo,
                checkpoint_name=f"partial_{checkpoint_name}",
                token=hub_token,
            )

        return PoolTrainingResult(
            provider=provider,
            success=False,
            error=str(e),
            total_time_seconds=elapsed,
        )


def _setup_data_for_provider(provider: Provider, data_dir: str) -> None:
    """Set up dataset for the current provider.

    Args:
        provider: Current provider.
        data_dir: Target data directory.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    if provider == Provider.KAGGLE:
        # Kaggle datasets should be in /kaggle/input/
        kaggle_input = Path("/kaggle/input/cats-dataset")
        if kaggle_input.exists():
            logger.info(f"Using Kaggle dataset at {kaggle_input}")
            return

    # Try downloading from Oxford IIIT Pet
    try:
        import subprocess

        script_dir = Path(__file__).parent.parent
        download_script = script_dir / "data" / "download.py"

        if download_script.exists():
            result = subprocess.run(
                ["python", str(download_script)],
                env={**os.environ, "DATA_DIR": str(data_path), "CATS_DIR": data_dir},
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                logger.info(f"Dataset downloaded to {data_dir}")
                return
            else:
                logger.error(f"Dataset download failed: {result.stderr}")
        else:
            logger.error(f"Download script not found: {download_script}")

    except Exception as e:
        logger.error(f"Data setup failed: {e}")


# ──────────────────────────────────────────────────────────────────────
# Provider-specific setup hooks
# ──────────────────────────────────────────────────────────────────────


def setup_colab_environment() -> bool:
    """Set up Google Colab environment (mount Drive, install deps).

    Returns:
        True if setup succeeded.
    """
    try:
        from google.colab import drive

        drive.mount("/content/drive")
        logger.info("Google Drive mounted at /content/drive")
    except ImportError:
        logger.warning("Not in Colab — skipping Drive mount")
        return False
    except Exception as e:
        logger.warning(f"Drive mount failed: {e}")

    # Create directories
    for d in [
        "/content/drive/MyDrive/tiny-cats-data",
        "/content/drive/MyDrive/tiny-cats-checkpoints",
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)

    return True


def setup_kaggle_environment() -> bool:
    """Set up Kaggle environment.

    Returns:
        True if setup succeeded.
    """
    # Kaggle provides GPU automatically
    try:
        import torch

        if torch.cuda.is_available():
            logger.info(
                f"Kaggle GPU: {torch.cuda.get_device_name(0)} "
                f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
            )
            return True
        else:
            logger.warning("Kaggle: GPU not available, falling back to CPU")
            return False
    except ImportError:
        return False


def setup_lightning_environment() -> bool:
    """Set up Lightning AI Studio environment.

    Returns:
        True if setup succeeded.
    """
    # Lightning Studio provides GPU automatically
    try:
        import torch

        if torch.cuda.is_available():
            logger.info(
                f"Lightning GPU: {torch.cuda.get_device_name(0)} "
                f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)"
            )
        else:
            logger.warning("Lightning: GPU not available, falling back to CPU")
    except ImportError:
        pass

    # Create directories
    for d in ["/teamspace/datasets/cats", "/teamspace/checkpoints"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    return True


# ──────────────────────────────────────────────────────────────────────
# Time/cost estimation
# ──────────────────────────────────────────────────────────────────────

# Tunable baseline: steps/second on T4 GPU at batch=512, 128x128.
# Sourced from a real T4 run (see AGENTS.md: "Speed: 2.2 steps/s per loop
# iteration"). The benchmark_estimates.py --tune output should NOT be
# auto-applied — the dataset is too sparse (one real entry, plus a
# derived 2.5x-scaled twin). Calibration requires >=3 independent real
# T4 wall-clock measurements before a tune recommendation is trustworthy.
T4_STEPS_PER_SECOND: float = 2.2


def estimate_gpu_hours(
    steps: int,
    batch_size: int = 512,
    image_size: int = 128,
    steps_per_second: float | None = None,
) -> float:
    """Estimate GPU hours needed for training.

    Based on benchmarks: ~2.2 steps/s on T4 for 128x128, batch 512.
    Set T4_STEPS_PER_SECOND or pass steps_per_second to override.

    Args:
        steps: Total training steps.
        batch_size: Batch size.
        image_size: Image size.
        steps_per_second: Override baseline speed (defaults to T4_STEPS_PER_SECOND).

    Returns:
        Estimated GPU hours.
    """
    sps = steps_per_second if steps_per_second is not None else T4_STEPS_PER_SECOND

    # Adjust for image size
    if image_size == 256:
        sps *= 0.25  # ~4x slower

    # Adjust for batch size: larger batches process more data per step,
    # so steps/second decreases (e.g. batch=256 → 1/2 data → 2x faster;
    # batch=1024 → 2x data → 2x slower). The 512 base is the benchmark.
    sps *= 512 / batch_size

    seconds = steps / max(sps, 0.01)
    return seconds / 3600.0


def estimate_cost(
    steps: int,
    provider: Provider | None = None,
    batch_size: int = 512,
    image_size: int = 128,
) -> dict[str, Any]:
    """Estimate cost and time for training on each provider.

    Args:
        steps: Total training steps.
        provider: Specific provider (or None for all).
        batch_size: Batch size.
        image_size: Image size.

    Returns:
        Dictionary with cost/time estimates.
    """
    gpu_hours = estimate_gpu_hours(steps, batch_size, image_size)

    providers_to_check = [provider] if provider else list(PROVIDER_CONFIGS.keys())

    results = {}
    for p in providers_to_check:
        config = PROVIDER_CONFIGS[p]
        cost = config.cost_per_hour * gpu_hours
        within_free = True

        if config.free_credits_per_month:
            within_free = cost <= config.free_credits_per_month

        results[p.value] = {
            "provider": config.display_name,
            "gpu_hours": round(gpu_hours, 1),
            "estimated_cost": round(cost, 2),
            "within_free_tier": within_free,
            "gpu_type": config.gpu_types[0] if config.gpu_types else "CPU",
            "recommended": within_free,
        }

    return results
