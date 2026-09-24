"""src/train_dit.py

Training script for TinyDiT (Diffusion Transformer) with flow matching.

Features:
- Flow matching training objective
- EMA (Exponential Moving Average) weight averaging
- Mixed precision training (AMP)
- Learning rate warmup with cosine annealing
- Gradient clipping
- Checkpoint/resume support
- Modal GPU training
- Progress tracking with wandb-style logging

Usage:
    # Local training
    python src/train_dit.py data/cats --steps 200000 --batch-size 256

    # Modal GPU training
    modal run src/train_dit.py

    # Resume from checkpoint
    python src/train_dit.py data/cats --resume checkpoints/dit_epoch_50.pt

Modal GPU training:
    modal run src/train_dit.py
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import pickle
import signal
import sys
import time
import zipfile
from collections import Counter
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import modal
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Optional auth utilities import (for enhanced error handling)
try:
    from auth_utils import AuthenticationError, require_modal_auth, setup_auth_logging

    AUTH_UTILS_AVAILABLE = True
except ImportError:
    AUTH_UTILS_AVAILABLE = False

    # Fallback for Modal container
    class AuthenticationError(Exception):  # type: ignore
        def __init__(self, message: str, token_type: str | None = None):
            self.message = message
            self.token_type = token_type
            super().__init__(self.message)

    def require_modal_auth():  # type: ignore
        pass

    def setup_auth_logging(level=None):  # type: ignore
        import logging

        return logging.getLogger("tiny_dit")


# Optional experiment tracker import
try:
    from experiment_tracker import ExperimentTracker
except ImportError:
    # Fallback simple tracker (ADR-042)
    class ExperimentTracker:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

        def start_run(self, *args, **kwargs):
            return None

        def log_params(self, *args, **kwargs):
            pass

        def log_metrics(self, *args, **kwargs):
            pass

        def log_model(self, *args, **kwargs):
            pass

        def log_artifact(self, *args, **kwargs):
            pass

        def log_image(self, *args, **kwargs):
            pass

        def end_run(self, *args, **kwargs):
            pass

        def log(self, *args, **kwargs):
            pass

        def close(self):
            pass


# Add project root to path (for local development)
sys.path.insert(0, str(Path(__file__).parent))

# Note: Modal imports (DiT modules) are done inside train_dit_local and
# DiTTrainer.train() — container init is handled by @modal.enter() (ADR-025).
# This avoids ModuleNotFoundError when running on Modal (ADR-030, ADR-042).

# Type hints only (not imported at runtime) - ADR-042
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flow_matching import EMA


# Configure logging
def setup_logging(log_file: str | None = None) -> logging.Logger:
    """Setup logging with console and optional file handlers.

    Args:
        log_file: Optional path to log file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("tiny_dit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# YAML config keys whose spelling differs from the argparse destination.
_CONFIG_KEY_ALIASES = {
    "patience": "early_stopping_patience",
    "min_delta": "early_stopping_min_delta",
    "beta": "ema_beta",
    "level": "augmentation_level",
}


def load_yaml_defaults(parser: argparse.ArgumentParser, config_path: str) -> list[str]:
    """Apply a YAML config file as parser defaults.

    The previous implementation applied the config *after* parse_args() and only
    when the attribute was None — but argparse always populates defaults, so no
    key ever qualified and ``--config`` was silently a no-op. Setting defaults
    before parsing keeps the documented "CLI flags override the config" order.

    Args:
        parser: Argument parser to set defaults on.
        config_path: Path to the YAML config file.

    Returns:
        Config keys that were ignored because no matching argument exists.
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    known_dests = {action.dest for action in parser._actions}
    defaults: dict[str, Any] = {}
    ignored: list[str] = []

    for section in config.values():
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            dest = _CONFIG_KEY_ALIASES.get(key, key.replace("-", "_"))
            if dest in known_dests:
                defaults[dest] = value
            else:
                ignored.append(key)

    parser.set_defaults(**defaults)
    return ignored


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments, with any ``--config`` YAML applied as defaults.
    """
    parser = argparse.ArgumentParser(
        description="Train TinyDiT for cat image generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file (optional YAML)
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Data & output
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to dataset root"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/dit_model.pt",
        help="Output checkpoint path",
    )
    parser.add_argument(
        "--ema-output",
        type=str,
        default="checkpoints/dit_model_ema.pt",
        help="Output EMA checkpoint path",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    # Training
    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
        help=(
            "Global target steps (issue #163): a resume performs "
            "max(0, steps - completed), never steps additional steps"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--warmup-steps", type=int, default=2_000, help="LR warmup steps"
    )
    parser.add_argument(
        "--min-lr", type=float, default=1e-6, help="Minimum LR for cosine decay"
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (0 to disable)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable automatic mixed precision training",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Experiment identity (issue #163 WP4)
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="dit-breed-conditioned-v4",
        help="Experiment identifier recorded in the manifest; resumes must match",
    )
    parser.add_argument(
        "--allow-experiment-mismatch",
        action="store_true",
        help=(
            "Resume even when the checkpoint manifest differs from this run "
            "(explicit override; incompatible resumes are rejected by default)"
        ),
    )

    # Model architecture
    parser.add_argument(
        "--image-size", type=int, default=128, help="Image size (128 or 256)"
    )
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size")
    parser.add_argument(
        "--embed-dim", type=int, default=384, help="Embedding dimension"
    )
    parser.add_argument(
        "--depth", type=int, default=12, help="Number of transformer blocks"
    )
    parser.add_argument(
        "--num-heads", type=int, default=6, help="Number of attention heads"
    )

    # Logging & checkpointing
    parser.add_argument("--log-file", type=str, default=None, help="Path to log file")
    parser.add_argument(
        "--log-interval", type=int, default=100, help="Logging interval in steps"
    )
    parser.add_argument(
        "--save-interval", type=int, default=10_000, help="Checkpoint save interval"
    )
    parser.add_argument(
        "--sample-interval", type=int, default=5_000, help="Sample generation interval"
    )
    parser.add_argument(
        "--num-sample-images",
        type=int,
        default=8,
        help="Number of images to generate during sampling",
    )

    # Early stopping
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Stop if loss doesn't improve for N evaluations (0=disabled)",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.001,
        help="Minimum loss improvement to count as progress",
    )

    # EMA & sampling
    parser.add_argument("--ema-beta", type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument(
        "--cfg-scale", type=float, default=1.5, help="Classifier-free guidance scale"
    )

    # Data augmentation
    parser.add_argument(
        "--augmentation-level",
        type=str,
        default="full",
        choices=["basic", "medium", "full"],
        help="Level of data augmentation",
    )

    # Performance
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    # Validation
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.05,
        help="Fraction of the dataset held out for validation (0 disables)",
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=8,
        help="Number of validation batches per evaluation (0 = all)",
    )

    # Timestep sampling
    parser.add_argument(
        "--timestep-sampling",
        type=str,
        default="uniform",
        choices=["uniform", "logit_normal"],
        help="Timestep distribution (logit_normal concentrates on mid-trajectory)",
    )
    parser.add_argument(
        "--logit-normal-mean",
        type=float,
        default=0.0,
        help="Mean of the logit-normal timestep sampler",
    )
    parser.add_argument(
        "--logit-normal-std",
        type=float,
        default=1.0,
        help="Std of the logit-normal timestep sampler",
    )

    # Read --config before parsing so the YAML can be applied as defaults and
    # explicit CLI flags still win.
    config_preparser = argparse.ArgumentParser(add_help=False)
    config_preparser.add_argument("--config", type=str, default=None)
    config_path = config_preparser.parse_known_args(argv)[0].config

    if config_path:
        ignored = load_yaml_defaults(parser, config_path)
        if ignored:
            logging.getLogger("tiny_dit").warning(
                "Ignoring unknown config keys: %s", ", ".join(sorted(set(ignored)))
            )

    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_memory() -> None:
    """Clean up GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_gpu_memory(logger: logging.Logger, prefix: str = "") -> None:
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        logger.info(
            f"{prefix}GPU Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved"
        )


def build_lr_lambda(
    warmup_steps: int,
    steps: int,
    min_lr_ratio: float = 0.0,
) -> Any:
    """Build the linear-warmup + cosine-decay LR multiplier.

    Args:
        warmup_steps: Number of warmup steps (0.01x -> 1.0x LR).
        steps: Total steps for the cosine horizon.
        min_lr_ratio: ``min_lr / lr``, the floor the cosine decays to. Without
            it the schedule decayed to a literal 0.0 LR (--min-lr was parsed
            and then never read).

    Returns:
        Callable mapping a step index to an LR multiplier.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup: 0.01 -> 1.0
            return 0.01 + 0.99 * float(current_step) / float(max(1, warmup_steps))
        # Cosine annealing: 1.0 -> min_lr_ratio. progress is clamped to [0, 1]
        # because resuming a sliced run with a smaller --steps used to push it
        # past 1, where the cosine turns back up and *raises* the LR.
        progress = float(current_step - warmup_steps) / float(
            max(1, steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(decay, min_lr_ratio)

    return lr_lambda


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    step: int,
    loss: float,
    path: str | Path,
    logger: logging.Logger,
    is_best: bool = False,
    best_loss: float | None = None,
    patience_counter: int = 0,
    val_loss: float | None = None,
    val_loss_ema: float | None = None,
    steps: int | None = None,
    warmup_steps: int | None = None,
    target_steps: int | None = None,
    manifest: dict[str, Any] | None = None,
    seed: int | None = None,
) -> None:
    """Save training checkpoint with EMA weights.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        ema: EMA tracker.
        step: Current training step.
        loss: Current loss value (the selection metric).
        path: Checkpoint path.
        logger: Logger instance.
        is_best: Whether this is the best model.
        best_loss: Best selection loss so far, persisted so early stopping
            survives a resume (it used to reset to inf on every slice).
        patience_counter: Consecutive non-improving evaluations.
        val_loss: Held-out validation loss at this step, when measured.
        val_loss_ema: EMA-weight validation loss, when measured.
        steps: LR-schedule horizon this run is using.
        warmup_steps: LR-schedule warmup this run is using.
        target_steps: Global target step count for this run (issue #163).
        manifest: Immutable experiment manifest; embedded in the checkpoint
            and mirrored into ``training_state.json`` beside it.
        seed: Training seed recorded for reproducibility.
    """
    from training_state import (
        TRAINING_STATE_FILENAME,
        capture_rng_state,
        write_training_state,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ema_shadow_params": ema.shadow_params,
        "ema_step": ema.step,
        "loss": loss,
        "val_loss": val_loss,
        "val_loss_ema": val_loss_ema,
        "best_loss": best_loss,
        "patience_counter": patience_counter,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "target_steps": target_steps,
        "manifest": manifest,
        "seed": seed,
        # RNG snapshot so a provider handoff resumes the exact stream
        # (torch/python/numpy/cuda; issue #163 WP1).
        "rng_state": capture_rng_state(),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "image_size": model.image_size,
            "patch_size": model.patch_size,
            "embed_dim": model.embed_dim,
            "depth": len(model.blocks),
            "num_heads": model.blocks[0].attn.num_heads,
            # Persisted so validation/eval can rebuild the generative model
            # instead of guessing from a hardcoded default (validate_model.py).
            "num_classes": getattr(model, "num_classes", None),
        },
    }

    # Atomic write: write to temp file then rename (prevents corruption on crash)
    tmp_path = str(path) + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, str(path))
    logger.info(f"Saved checkpoint at step {step:,} (loss={loss:.6e}) to {path}")

    # Mirror manifest + progress beside the checkpoint (issue #163 WP1/WP4).
    if manifest is not None:
        write_training_state(
            path.parent / TRAINING_STATE_FILENAME,
            manifest=manifest,
            completed_steps=step,
        )

    if is_best:
        best_path = path.parent / f"best_{path.name}"
        tmp_best_path = str(best_path) + ".tmp"
        torch.save(checkpoint, tmp_best_path)
        os.replace(tmp_best_path, str(best_path))
        logger.info(f"Saved best model to {best_path}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    ema: EMA | None = None,
    logger: logging.Logger | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[nn.Module, torch.optim.Optimizer | None, EMA | None, int]:
    """Load checkpoint for resume training.

    Args:
        path: Checkpoint path.
        model: Model to load weights into.
        optimizer: Optional optimizer to load state.
        ema: Optional EMA to load shadow params.
        logger: Optional logger.
        state: Optional dict populated with the persisted training state
            (``best_loss``, ``patience_counter``, ``val_loss``, ``steps``,
            ``warmup_steps``) so a resumed slice keeps early-stopping and
            LR-schedule continuity instead of restarting them from scratch.

    Returns:
        Tuple of (model, optimizer, ema, completed_steps) where
        ``completed_steps`` is the number of *completed global steps* stored in
        the checkpoint (issue #163 — the caller performs
        ``max(0, target - completed)`` more). When ``path`` exists but is
        unreadable (truncated zip from a preempted run, EOFError on partial
        write, etc.) the file is renamed to ``<path>.corrupt`` for forensics
        and ``completed_steps=0`` is returned so training restarts from
        scratch.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        IncompatibleExperimentError: If the checkpoint tensors do not match
            the current model architecture. Incompatible checkpoints are
            rejected, never silently restarted (issue #163).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except (
        zipfile.BadZipFile,
        RuntimeError,
        EOFError,
        OSError,
        pickle.UnpicklingError,
    ) as exc:
        # Stale or partial checkpoint from a previously preempted run. Quarantine
        # the bad file (so it stays available for post-mortem) and return
        # start_step=0 so train_dit_local restarts from scratch with fresh
        # weights. Without this guard, a left-over `dit_model.pt` at the
        # volume root can poison every subsequent auto-resume attempt until
        # the operator manually deletes it (see ADR-058).
        # Use `with_name(name + ".corrupt")` instead of `with_suffix(...)` so that
        # multi-dot filenames (e.g. `.tar.gz`, `.pt.bak`) still quarantine cleanly.
        quarantine = path.with_name(path.name + ".corrupt")
        try:
            path.rename(quarantine)
        except OSError as rename_exc:  # pragma: no cover - defensive
            if logger:
                logger.warning(
                    f"Could not quarantine corrupt checkpoint {path} -> "
                    f"{quarantine}: {rename_exc}"
                )
        if logger:
            logger.warning(
                f"Checkpoint at {path} is unreadable ({type(exc).__name__}: "
                f"{exc}); moved to {quarantine} and restarting from step 0."
            )
        return model, optimizer, ema, 0
    from dit import load_state_dict_checked
    from training_state import IncompatibleExperimentError

    try:
        load_state_dict_checked(model, checkpoint["model_state_dict"])
    except (ValueError, RuntimeError) as exc:
        # Architecture changed (e.g. a new parameter was added) or the
        # checkpoint is from another model size. The helper validates keys and
        # shapes before copying, so the model is left untouched.
        #
        # Reject instead of silently restarting from step 0 (issue #163):
        # a provider handoff that lands on the wrong architecture must fail
        # clearly so the operator moves the checkpoint aside deliberately.
        raise IncompatibleExperimentError(
            f"Checkpoint at {path} does not match the current model "
            f"architecture: {exc} Refusing to resume. Move the checkpoint "
            "aside (or restore the matching configuration) to continue."
        ) from exc

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if ema and "ema_shadow_params" in checkpoint:
        ema.shadow_params = checkpoint["ema_shadow_params"]
        ema.step = checkpoint.get("ema_step", 0)
        if logger:
            logger.info(f"Loaded EMA state (step {ema.step:,})")

    if state is not None:
        state.update(
            {
                "best_loss": checkpoint.get("best_loss"),
                "patience_counter": checkpoint.get("patience_counter", 0),
                "val_loss": checkpoint.get("val_loss"),
                "val_loss_ema": checkpoint.get("val_loss_ema"),
                "steps": checkpoint.get("steps"),
                "warmup_steps": checkpoint.get("warmup_steps"),
                # Issue #163: experiment identity, progress and RNG stream for
                # exact global-step / cross-provider resume. "loaded" marks a
                # successful load so a stale training_state.json sidecar is
                # never validated after a quarantined (corrupt) checkpoint.
                "loaded": True,
                "manifest": checkpoint.get("manifest"),
                "rng_state": checkpoint.get("rng_state"),
                "seed": checkpoint.get("seed"),
                "target_steps": checkpoint.get("target_steps"),
            }
        )

    # Exact global-step semantics (issue #163): checkpoint["step"] counts
    # *completed* optimizer steps, and a resume must perform exactly
    # max(0, target - completed) more — the old "+ 1" silently skipped one
    # step (45k -> 60k ran 14,999 instead of the required 15,000).
    start_step = int(checkpoint.get("step", 0))
    if logger:
        logger.info(
            f"Loaded checkpoint from {path} ({start_step:,} global steps completed)"
        )

    return model, optimizer, ema, start_step


class TrainingError(Exception):
    """Custom exception for training errors."""

    pass


# Modal setup (ADR-022, ADR-023, ADR-024, ADR-025)
app = modal.App("tiny-dit-training")

# Volume definitions (ADR-024: organized storage with explicit commits)
volume_outputs = modal.Volume.from_name("dit-outputs", create_if_missing=True)
volume_data = modal.Volume.from_name("dit-dataset", create_if_missing=True)

# Optimized container image (ADR-022: fast builds with uv_pip_install)
# Download scripts added for dataset download fallback (ADR-031)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "tar", "curl", "git")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",  # Faster HuggingFace downloads
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",  # Memory optimization
        }
    )
    .uv_pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "pillow==11.0.0",
        "tqdm==4.67.1",
        "onnx==1.17.0",
        "onnxruntime==1.20.0",
        "huggingface_hub",
    )
    .add_local_file("src/train_dit.py", "/app/train_dit.py")
    .add_local_file("src/training_state.py", "/app/training_state.py")
    .add_local_file("src/artifacts.py", "/app/artifacts.py")
    .add_local_file("src/dit.py", "/app/dit.py")
    .add_local_file("src/flow_matching.py", "/app/flow_matching.py")
    .add_local_file("src/dit_validation.py", "/app/dit_validation.py")
    .add_local_file("src/dataset.py", "/app/dataset.py")
    .add_local_file("src/gpu_pool.py", "/app/gpu_pool.py")
    .add_local_file("src/export_dit_onnx.py", "/app/export_dit_onnx.py")
    .add_local_file("src/optimize_onnx.py", "/app/optimize_onnx.py")
    .add_local_file("src/volume_utils.py", "/app/volume_utils.py")
    .add_local_file("src/auth_utils.py", "/app/auth_utils.py")
    .add_local_file("src/retry_utils.py", "/app/retry_utils.py")
    .add_local_file("src/experiment_tracker.py", "/app/experiment_tracker.py")
    .add_local_file("data/download.py", "/app/data/download.py")
    .add_local_file("data/download.sh", "/app/data/download.sh")
)


@app.cls(
    image=image,
    volumes={
        "/outputs": volume_outputs,
        "/data": volume_data,
    },
    gpu=["T4", "L4"],  # T4 ($0.59/hr) for $7 budget; L4 ($0.80/hr) fallback
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=86400,  # 24 hours max for long training runs
    retries=modal.Retries(
        max_retries=10,
        initial_delay=0.0,  # Immediate retry on preemption
    ),
    scaledown_window=300,  # ADR-057: keep container warm 5 min for retries
)
class DiTTrainer:
    """Modal container class for DiT GPU training (ADR-025, ADR-057).

    Uses @modal.enter() for one-time container initialization instead of
    calling _initialize_dit_container() inside the function body. This
    means CUDA warm-up and heavy imports run once per container, not
    per function invocation — cutting cold start latency.
    """

    @modal.enter()
    def enter(self):
        """One-time container init: paths, heavy imports, CUDA warm-up.

        Runs when the Modal container starts (not on each function call),
        eliminating the per-invocation cold-start penalty (ADR-025).
        """
        # Setup paths - files are at /app/ via add_local_file (ADR-022, ADR-030)
        sys.path.insert(0, "/app")
        os.chdir("/app")

        # Pre-import heavy modules
        import torch
        import torchvision  # noqa: F401

        # Warm up CUDA

        if torch.cuda.is_available():
            _ = torch.zeros(1).cuda()
            dummy_input = torch.randn(1, 3, 32, 32).cuda()
            dummy_conv = torch.nn.Conv2d(3, 16, 3).cuda()
            _ = dummy_conv(dummy_input)
            del dummy_input, dummy_conv
            torch.cuda.empty_cache()

    def _push_to_hub(
        self,
        hub_repo: str,
        output: str,
        ema_output: str,
        logger: logging.Logger,
        experiment_id: str | None = None,
        completed_steps: int | None = None,
        training_state_path: str | None = None,
    ) -> None:
        """Push current best + EMA checkpoints to HF Hub (graceful).

        Used both mid-run (hub_push_interval) and at run completion so a
        cancelled GHA slice still syncs its latest checkpoint for
        cross-provider resume (train-pool.yml --no-hub-push disables).

        With ``experiment_id`` + ``completed_steps`` the push uses the
        canonical immutable pool layout (issue #163 WP3); without them it
        falls back to the legacy flat path.
        """
        try:
            from gpu_pool import push_checkpoint_to_hub

            for ckpt_name, ckpt_path in [
                ("dit_model.pt", output),
                ("dit_model_ema.pt", ema_output),
            ]:
                if Path(ckpt_path).exists():
                    push_checkpoint_to_hub(
                        checkpoint_path=ckpt_path,
                        hub_repo=hub_repo,
                        checkpoint_name=ckpt_name,
                        experiment_id=(
                            experiment_id if completed_steps is not None else None
                        ),
                        completed_steps=completed_steps,
                        training_state_path=training_state_path,
                    )
        except Exception as e:
            logger.warning(f"GPU pool: hub push skipped ({e})")

    @modal.method()
    def train(
        self,
        data_dir: str = "/data/cats",
        steps: int = 100_000,
        batch_size: int = 128,
        lr: float = 5e-5,
        image_size: int = 128,
        output: str | None = None,
        ema_output: str | None = None,
        num_workers: int = 0,
        mixed_precision: bool = True,
        gradient_clip: float = 1.0,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 2_000,
        min_lr: float = 1e-6,
        val_split: float = 0.05,
        val_batches: int = 8,
        timestep_sampling: str = "uniform",
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
        log_interval: int = 100,
        save_interval: int = 500,
        early_stopping_patience: int = 15,
        early_stopping_min_delta: float = 0.001,
        sample_interval: int = 2_000,
        log_file: str | None = None,
        ema_beta: float = 0.9999,
        seed: int = 42,
        augmentation_level: str = "full",
        resume_checkpoint: str | None = None,
        hub_resume: bool = False,
        no_hub_push: bool = False,
        hub_push_interval: int = 0,
        experiment_id: str = "dit-breed-conditioned-v4",
        allow_experiment_mismatch: bool = False,
    ) -> dict[str, Any]:
        """Run DiT training (was train_dit_on_gpu, now DiTTrainer.train).

        Container is already initialized by @modal.enter() — no
        explicit _initialize_dit_container() call needed.

        Returns:
            Training status dict.

        Raises:
            AuthenticationError: If Modal authentication fails
        """
        # Setup logging first
        logger = setup_auth_logging(level=logging.INFO)

        # Validate Modal authentication before starting training
        logger.info("=" * 60)
        logger.info("MODAL TRAINING - PRE-FLIGHT CHECKS")
        logger.info("=" * 60)

        try:
            require_modal_auth()
            logger.info("✅ Modal authentication validated")
        except AuthenticationError as e:
            logger.error(f"❌ {e.message}")
            logger.error("")
            logger.error("To fix this:")
            logger.error("  1. Run 'modal token new' to authenticate (Modal 1.0+)")
            logger.error("  2. Verify with: modal token info")
            logger.error(
                "  3. For GitHub Actions, ensure MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are set"
            )
            logger.error("")
            logger.error("See: https://modal.com/docs/reference/cli/token")
            logger.error(
                "See AGENTS.md or agents-docs/auth-troubleshooting.md for help"
            )
            raise

        # Container is already initialized by @modal.enter()

        # Canonical live-checkpoint directory (issue #163 WP2): stable
        # (non-dated) so Modal retries and manually re-triggered runs can
        # find and resume prior progress. A dated directory meant every
        # retry silently restarted from step 0.
        checkpoint_dir = "/outputs/checkpoints/pool"
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        samples_dir = f"{checkpoint_dir}/samples"
        Path(samples_dir).mkdir(parents=True, exist_ok=True)

        # Stable output paths enable resume across retries
        output = output or f"{checkpoint_dir}/dit_model.pt"
        ema_output = ema_output or f"{checkpoint_dir}/dit_model_ema.pt"
        log_file = log_file or f"{checkpoint_dir}/training.log"

        # Auto-resume: if a checkpoint already exists (e.g. from a prior Modal retry),
        # pass it through to train_dit_local so training continues from that step.
        # resume_checkpoint overrides auto-detection when set explicitly.
        #
        # Only resume when the file *looks* like a valid torch checkpoint zip -
        # otherwise a stale, partially-written `dit_model.pt` from a previous
        # preempt would silently restart training at step 0 instead of raising
        # (or worse, raise and abort the run). load_checkpoint() itself also has
        # a defensive try/except as belt-and-braces (ADR-058).
        resume: str | None = resume_checkpoint
        if resume is None and Path(output).exists() and zipfile.is_zipfile(output):
            resume = output
            logger.info(f"Found existing checkpoint; will resume from: {output}")
        elif resume is None and Path(output).exists():
            logger.warning(
                f"Ignoring non-zip file at {output} (likely a stale partial "
                "checkpoint from a previously preempted run); starting fresh. "
                "load_checkpoint() will quarantine the file if it is unreadable."
            )
        elif resume is not None:
            logger.info(f"Using explicit resume checkpoint: {resume}")

        # Layout migration (issue #163 WP2): when the canonical directory has
        # no valid checkpoint, pick up a valid legacy live checkpoint instead
        # of silently restarting from step 0.
        if resume is None:
            from artifacts import find_live_checkpoint

            legacy = find_live_checkpoint("/outputs")
            if legacy is not None:
                resume = str(legacy)
                logger.info(f"Resuming from pre-migration live checkpoint: {legacy}")

        # GPU pool cross-provider resume (train-pool.yml --hub-resume / ADR-055).
        # Pull the last EMA checkpoint from HuggingFace Hub so a prior provider's
        # progress carries over. Degrades gracefully (logs + continues fresh) if
        # huggingface_hub is unavailable or no HF_TOKEN is set in the container.
        hub_repo = "d4oit/tiny-cats-model"
        if hub_resume and resume is None:
            logger.info(f"GPU pool: pulling checkpoint from Hub ({hub_repo})...")
            try:
                from gpu_pool import pull_checkpoint_from_hub

                pulled = pull_checkpoint_from_hub(
                    hub_repo=hub_repo,
                    checkpoint_name="dit_model_ema.pt",
                    output_dir=checkpoint_dir,
                    experiment_id=experiment_id,
                )
                if pulled:
                    resume = str(pulled)
                    logger.info(f"GPU pool: resuming from Hub checkpoint: {resume}")
                else:
                    logger.info("GPU pool: no Hub checkpoint found — starting fresh")
            except Exception as e:  # graceful degradation
                logger.warning(f"GPU pool: hub pull skipped ({e})")

        # Setup training-specific logging (after auth validation)
        logger = setup_logging(log_file)
        logger.info("Starting TinyDiT Modal GPU training")
        logger.info(
            f"Configuration: steps={steps:,}, batch_size={batch_size}, "
            f"image_size={image_size}, lr={lr}"
        )

        try:
            # Check dataset cache (ADR-024: dataset caching in volume)
            if not Path(data_dir).exists() or not list(Path(data_dir).iterdir()):
                logger.info("Dataset not found, downloading...")
                import subprocess

                result = subprocess.run(
                    ["python", "data/download.py"],
                    cwd="/app",
                    env={**os.environ, "DATA_DIR": "/data", "CATS_DIR": "/data/cats"},
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if result.returncode != 0:
                    logger.warning(f"Download failed: {result.stderr}")
                logger.info("Dataset ready")

            # Train
            final_loss = train_dit_local(
                data_dir=data_dir,
                steps=steps,
                batch_size=batch_size,
                lr=lr,
                image_size=image_size,
                output=output,
                ema_output=ema_output,
                num_workers=num_workers,
                mixed_precision=mixed_precision,
                gradient_clip=gradient_clip,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                min_lr=min_lr,
                val_split=val_split,
                val_batches=val_batches,
                timestep_sampling=timestep_sampling,
                logit_normal_mean=logit_normal_mean,
                logit_normal_std=logit_normal_std,
                log_interval=log_interval,
                save_interval=save_interval,
                sample_interval=sample_interval,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                log_file=log_file,
                ema_beta=ema_beta,
                seed=seed,
                logger=logger,
                resume=resume,
                augmentation_level=augmentation_level,
                no_hub_push=no_hub_push,
                hub_push_interval=hub_push_interval,
                hub_repo=hub_repo,
                experiment_id=experiment_id,
                allow_experiment_mismatch=allow_experiment_mismatch,
            )

            # Canonical final artifact package (issue #163 WP2): export ONNX
            # and build artifacts/ only when this run's global target is
            # actually reached — partial slices never publish "final"
            # artifacts, and a broken final package fails the job instead of
            # passing silently (issue #163: never publish a final model
            # unless the target checkpoint exists and passes validation).
            from training_state import TRAINING_STATE_FILENAME, read_training_state

            pool_state = read_training_state(
                Path(output).parent / TRAINING_STATE_FILENAME
            )
            target_steps: int | None = None
            if (
                pool_state is not None
                and "completed_steps" in pool_state
                and "target_steps" in pool_state
                and int(pool_state["completed_steps"])
                >= int(pool_state["target_steps"])
            ):
                target_steps = int(pool_state["target_steps"])
            if target_steps is None:
                logger.info(
                    "Global target not reached (or training state missing) — "
                    "skipping final artifact package for this slice."
                )
            else:
                logger.info("Global target reached — building final artifacts...")
                try:
                    from artifacts import export_paths, package_final_artifacts
                    from export_dit_onnx import export_generator_onnx, load_model
                    from optimize_onnx import optimize_onnx

                    # Export stages in <root>/export/, *never* directly into
                    # artifacts/generator/ — packaging copies sources into the
                    # package (a same-directory source/destination crashed the
                    # live smoke run with SameFileError).
                    paths = export_paths("/outputs")
                    onnx_path = paths["onnx"]

                    model_to_export = load_model(output, image_size=image_size)
                    export_generator_onnx(model_to_export, output_path=onnx_path)
                    logger.info(f"✅ Exported to {onnx_path}")

                    logger.info("Quantizing ONNX model...")
                    optimize_onnx(
                        model_path=onnx_path,
                        output_dir=paths["export_dir"],
                        method="dynamic",
                        model_type="generator",
                    )
                    # optimize_onnx names generator output
                    # "generator_quantized.onnx" (not "model_quantized.onnx").
                    quantized_path = paths["quantized"]
                    if not quantized_path.exists():
                        raise FileNotFoundError(
                            f"Quantized ONNX missing after optimize_onnx: "
                            f"{quantized_path}"
                        )
                    logger.info(f"✅ Quantized model saved to {quantized_path}")

                    package_final_artifacts(
                        "/outputs",
                        raw_checkpoint=output,
                        ema_checkpoint=ema_output,
                        training_log=log_file,
                        samples_dir=str(Path(output).parent / "samples"),
                        onnx=onnx_path,
                        quantized=quantized_path,
                        require_completed=target_steps,
                    )
                    logger.info(
                        "✅ Final artifact package written to /outputs/artifacts/"
                    )
                except Exception as e:
                    # Training + checkpoints are already safe; surface the
                    # packaging failure loudly (the except below commits the
                    # volume before re-raising as TrainingError).
                    logger.error(f"Final artifact packaging failed: {e}")
                    raise TrainingError(f"Final artifact packaging failed: {e}") from e

            # Commit volume after successful training (ADR-024: explicit commits)
            volume_outputs.commit()
            logger.info("Checkpoint committed to volume")

            # GPU pool: push checkpoints to Hub for cross-provider resume
            # (train-pool.yml --no-hub-push disables; degrades gracefully).
            if not no_hub_push:
                logger.info("GPU pool: pushing checkpoints to HuggingFace Hub...")
                self._push_to_hub(
                    hub_repo=hub_repo,
                    output=output,
                    ema_output=ema_output,
                    logger=logger,
                    experiment_id=experiment_id,
                    completed_steps=(
                        int(pool_state["completed_steps"]) if pool_state else None
                    ),
                    training_state_path=(
                        str(Path(output).parent / TRAINING_STATE_FILENAME)
                        if pool_state
                        else None
                    ),
                )

            logger.info("Training completed successfully")
            return {"status": "completed", "output": output, "final_loss": final_loss}

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            # Commit partial state on error
            volume_outputs.commit()
            raise TrainingError(f"Training failed: {e}") from e

        finally:
            cleanup_memory()


def create_dataloader(
    data_dir: str,
    batch_size: int,
    image_size: int,
    num_workers: int = 4,
    augmentation_level: str = "full",
) -> torch.utils.data.DataLoader:
    """Create dataloader for training.

    Args:
        data_dir: Dataset directory.
        batch_size: Batch size.
        image_size: Target image size.
        num_workers: DataLoader workers.
        augmentation_level: Level of data augmentation ("basic", "medium", "full").

    Returns:
        DataLoader yielding (images, breed_indices).
    """
    from dataset import CatBreedGenerationDataset, build_enhanced_transforms

    transform = build_enhanced_transforms(
        train=True,
        image_size=image_size,
        augmentation_level=augmentation_level,  # type: ignore[arg-type]
    )
    dataset = CatBreedGenerationDataset(data_dir, transform=transform)
    class_counts = Counter(label for _, label in dataset.samples)
    sample_weights = [1.0 / class_counts[label] for _, label in dataset.samples]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def train_dit_local(
    data_dir: str,
    steps: int = 100_000,
    batch_size: int = 512,
    lr: float = 5e-5,
    image_size: int = 128,
    output: str = "checkpoints/dit_model.pt",
    ema_output: str = "checkpoints/dit_model_ema.pt",
    num_workers: int = 4,
    mixed_precision: bool = True,
    gradient_clip: float = 1.0,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 2_000,
    min_lr: float = 1e-6,
    val_split: float = 0.05,
    val_batches: int = 8,
    timestep_sampling: str = "uniform",
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
    log_interval: int = 100,
    save_interval: int = 10_000,
    sample_interval: int = 5_000,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.001,
    log_file: str | None = None,
    ema_beta: float = 0.9999,
    seed: int = 42,
    logger: logging.Logger | None = None,
    resume: str | None = None,
    augmentation_level: str = "full",
    no_hub_push: bool = True,
    hub_push_interval: int = 0,
    hub_repo: str = "d4oit/tiny-cats-model",
    experiment_id: str = "dit-breed-conditioned-v4",
    allow_experiment_mismatch: bool = False,
    model_fn: Callable[[], nn.Module] | None = None,
) -> float:
    """Full TinyDiT training loop with flow matching and EMA.

    Args:
        data_dir: Dataset directory.
        steps: Total training steps.
        batch_size: Batch size.
        lr: Learning rate.
        image_size: Image size.
        output: Model checkpoint path.
        ema_output: EMA checkpoint path.
        num_workers: DataLoader workers.
        mixed_precision: Enable AMP.
        gradient_clip: Gradient clipping.
        gradient_accumulation_steps: Number of steps for gradient accumulation.
        warmup_steps: LR warmup steps.
        min_lr: LR floor for the cosine decay.
        val_split: Fraction of the dataset held out for validation (0 disables).
        val_batches: Validation batches per evaluation (0 = all).
        timestep_sampling: "uniform" or "logit_normal".
        logit_normal_mean: Mean of the logit-normal timestep sampler.
        logit_normal_std: Std of the logit-normal timestep sampler.
        log_interval: Logging frequency.
        save_interval: Checkpoint frequency.
        sample_interval: Sampling frequency.
        log_file: Optional log file.
        ema_beta: EMA decay factor.
        seed: Random seed.
        logger: Optional logger instance.
        resume: Optional checkpoint to resume from.
        early_stopping_min_delta: Minimum loss improvement to count as progress.
        augmentation_level: Level of data augmentation.
        experiment_id: Experiment identifier for the immutable manifest
            (issue #163); resumes must carry a matching one.
        allow_experiment_mismatch: Accept a resume whose manifest differs
            from this run (explicit override; rejected by default).
        model_fn: Optional factory replacing the built-in TinyDiT builder —
            a verification seam (WP8) so exact-resume behaviour runs on CPU
            in tests with a tiny model. Production paths leave it None.

    Returns:
        Final training loss.
    """
    # Import DiT modules (works for both local and Modal after path setup)
    from dataset import create_train_val_dataloaders
    from dit import count_parameters, tinydit_128, tinydit_256
    from dit_validation import evaluate_model_and_ema
    from flow_matching import (
        EMA,
        FlowMatchingLoss,
        flow_matching_step,
        sample,
        sample_timesteps,
    )
    from training_state import (
        TRAINING_STATE_FILENAME,
        IncompatibleExperimentError,
        build_manifest,
        manifest_mismatches,
        manifest_of,
        read_training_state,
        restore_rng_state,
        steps_to_run,
        write_training_state,
    )

    # Setup logging
    if logger is None:
        logger = setup_logging(log_file)

    logger.info("=" * 60)
    logger.info("Starting TinyDiT training with flow matching")
    logger.info(f"Configuration: {locals()}")

    set_seed(seed)
    logger.info(f"Random seed set to {seed}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log_gpu_memory(logger, "Initial | ")

    # Create model
    num_classes = 13  # 12 cat breeds + other
    if model_fn is not None:
        # Verification seam (issue #163 WP8): tests inject a tiny model so
        # exact-resume behaviour can be exercised on CPU without building the
        # 33M-param 128x128 generator. Production callers leave this None.
        model = model_fn().to(device)
    elif image_size == 128:
        model = tinydit_128(num_classes=num_classes).to(device)
    elif image_size == 256:
        model = tinydit_256(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unsupported image_size: {image_size}. Use 128 or 256.")

    logger.info(
        f"Model: TinyDiT | Image size: {image_size} | "
        f"Parameters: {count_parameters(model):,} | "
        f"Timesteps: {timestep_sampling}"
    )

    # Immutable experiment manifest (issue #163 WP4). Built from the *model*
    # (source of truth for architecture) and the *dataset on disk*, then
    # embedded in every checkpoint and mirrored to training_state.json.
    blocks = getattr(model, "blocks", [])
    manifest = build_manifest(
        experiment_id=experiment_id,
        data_dir=data_dir,
        image_size=int(getattr(model, "image_size", image_size)),
        patch_size=int(getattr(model, "patch_size", 0)),
        embed_dim=int(getattr(model, "embed_dim", 0)),
        depth=len(blocks),
        num_heads=int(blocks[0].attn.num_heads) if blocks else 0,
        num_classes=int(getattr(model, "num_classes", num_classes) or num_classes),
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        augmentation_level=augmentation_level,
        seed=seed,
        target_steps=steps,
    )

    # Create dataloaders (train + held-out validation, ADR-059 follow-up)
    train_loader, val_loader = create_train_val_dataloaders(
        root=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        augmentation_level=augmentation_level,  # type: ignore[arg-type]
        val_split=val_split,
        seed=seed,
    )
    effective_batch_size = batch_size * gradient_accumulation_steps
    logger.info(
        f"DataLoader created: {len(train_loader)} batches per epoch | "
        f"Effective batch size: {effective_batch_size} "
        f"(batch_size={batch_size} x accumulation_steps={gradient_accumulation_steps})"
    )
    if val_loader is None:
        logger.warning(
            "No validation split: checkpoint selection and early stopping will "
            "fall back to augmented training-batch loss (val_split=0)."
        )
    else:
        logger.info(
            f"Validation split: {val_split:.1%} held out | "
            f"{len(val_loader)} batches | evaluating {val_batches} per checkpoint"
        )

    # Optimizer and loss
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.95))
    loss_fn = FlowMatchingLoss()

    # LR scheduler with warmup and cosine annealing using LambdaLR (ADR-032)
    # The horizon is persisted in checkpoints so a resumed slice keeps decaying
    # along the original curve instead of restarting it.
    schedule_steps = steps
    schedule_warmup_steps = warmup_steps

    def make_scheduler(horizon_steps: int, horizon_warmup: int) -> LambdaLR:
        """Build the LR schedule for a given horizon.

        The base LR is the optimizer's ``initial_lr`` — set by the scheduler,
        and on resume restored from the checkpoint — so ``--min-lr`` stays
        proportional to the LR actually being decayed.
        """
        base_lr = optimizer.param_groups[0].get("initial_lr") or lr
        min_lr_ratio = min(min_lr / base_lr, 1.0) if base_lr > 0 else 0.0
        return LambdaLR(
            optimizer, build_lr_lambda(horizon_warmup, horizon_steps, min_lr_ratio)
        )

    scheduler = make_scheduler(schedule_steps, schedule_warmup_steps)

    # Mixed precision
    scaler = (
        torch.amp.GradScaler("cuda")
        if mixed_precision and torch.cuda.is_available()
        else None
    )
    if scaler:
        logger.info("Mixed precision training enabled (AMP)")

    # Null token for CFG dropout (dedicated index, matches forward_with_cfg)
    null_token = torch.tensor(num_classes, device=device)

    # EMA
    ema = EMA(beta=ema_beta)
    ema.init(model)
    logger.info(f"EMA initialized with beta={ema_beta}")

    tracker = ExperimentTracker("tiny-dit-cats")
    params = {
        "model": "TinyDiT",
        "image_size": image_size,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "learning_rate": lr,
        "mixed_precision": mixed_precision,
        "gradient_clip": gradient_clip,
        "warmup_steps": warmup_steps,
        "steps": steps,
        "ema_beta": ema_beta,
        "timestep_sampling": timestep_sampling,
        "val_split": val_split,
    }
    tracker.start_run(
        params, run_name=f"dit_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    # Resume from checkpoint
    start_step = 0
    resume_state: dict[str, Any] = {}
    if resume:
        logger.info(f"Resuming from checkpoint: {resume}")
        # Keep the local `optimizer` binding: load_checkpoint mutates and
        # returns the same instance, but its return type is Optional —
        # rebinding would poison later optimizer.param_groups access for mypy.
        model, _loaded_optimizer, _ema, start_step = load_checkpoint(
            resume, model, optimizer, ema, logger, state=resume_state
        )
        # Use loaded EMA if available
        ema = _ema if _ema is not None else ema

        # Exact global-step accounting (issue #163): `start_step` is the
        # number of *completed* global steps, so this invocation runs
        # max(0, target - completed) more — a 45k checkpoint with
        # --steps 60000 performs exactly 15,000, and an already-complete
        # checkpoint performs 0 (successful no-op at the end of this function).
        additional_steps = steps_to_run(steps, start_step)
        logger.info(
            f"Global-step resume: checkpoint={start_step:,} completed | "
            f"target={steps:,} | additional_steps={additional_steps:,}"
        )
        if additional_steps == 0:
            logger.info(
                "Checkpoint already at/past the global target; "
                "no training steps will run."
            )

        # Manifest gate (issue #163 WP4): only meaningful when the load
        # actually succeeded — a quarantined (corrupt) checkpoint leaves
        # resume_state empty and restarts fresh instead of consulting a
        # stale training_state.json sidecar.
        if resume_state.get("loaded"):
            saved_manifest = resume_state.get("manifest")
            if not saved_manifest:
                sidecar = read_training_state(
                    Path(resume).parent / TRAINING_STATE_FILENAME
                )
                saved_manifest = manifest_of(sidecar) if sidecar else None
            if saved_manifest:
                mismatches = manifest_mismatches(saved_manifest, manifest)
                if mismatches:
                    detail = "; ".join(mismatches)
                    if allow_experiment_mismatch:
                        logger.warning(
                            "Resuming despite experiment manifest mismatch "
                            f"(--allow-experiment-mismatch): {detail}"
                        )
                    else:
                        raise IncompatibleExperimentError(
                            f"Checkpoint {resume} belongs to a different "
                            f"experiment ({detail}). Refusing to resume; pass "
                            "--allow-experiment-mismatch to override "
                            "deliberately."
                        )
                else:
                    logger.info("Experiment manifest matches; resume accepted")
            else:
                logger.warning(
                    "Checkpoint has no experiment manifest (saved before "
                    "issue #163); architecture was validated but config "
                    "continuity cannot be verified."
                )

            if resume_state.get("rng_state"):
                restore_rng_state(resume_state["rng_state"])
                logger.info(
                    "Restored RNG state (torch/python/numpy/cuda) from checkpoint"
                )

        # Continue the recorded LR horizon. Resuming with a *smaller* --steps
        # than the run was scheduled for used to push the cosine past its end,
        # where the decay term turns back up and raises the LR mid-run.
        recorded_steps = resume_state.get("steps")
        if recorded_steps:
            schedule_steps = max(int(recorded_steps), steps)
            schedule_warmup_steps = int(
                resume_state.get("warmup_steps") or warmup_steps
            )
            if schedule_steps != steps or schedule_warmup_steps != warmup_steps:
                logger.info(
                    f"LR schedule continued from checkpoint: steps={schedule_steps:,}, "
                    f"warmup={schedule_warmup_steps:,} (requested steps={steps:,}, "
                    f"warmup={warmup_steps:,})"
                )
            scheduler = make_scheduler(schedule_steps, schedule_warmup_steps)

        # Position the schedule exactly at the completed step count. After N
        # completed optimizer steps the invariant is last_epoch == N and the
        # param-group LR == lambda(N). Assigning last_epoch alone does *not*
        # recompute the LR, so the first optimizer step after a resume used to
        # run at the init LR (lambda(0)) — materialise it explicitly instead
        # of calling scheduler.step(), which would warn about stepping before
        # optimizer.step().
        scheduler.last_epoch = start_step
        base_lr_resume = optimizer.param_groups[0].get("initial_lr") or lr
        min_lr_ratio_resume = (
            min(min_lr / base_lr_resume, 1.0) if base_lr_resume > 0 else 0.0
        )
        resumed_lr_fn = build_lr_lambda(
            schedule_warmup_steps, schedule_steps, min_lr_ratio_resume
        )
        resumed_lr = base_lr_resume * resumed_lr_fn(start_step)
        for group in optimizer.param_groups:
            group["lr"] = resumed_lr
        scheduler._last_lr = [resumed_lr for _ in optimizer.param_groups]

    # Training state (best_loss/patience are restored so early stopping works
    # across hub-resumed slices instead of restarting on every resume)
    best_loss = float("inf")
    restored_best_loss = resume_state.get("best_loss")
    if restored_best_loss is not None and math.isfinite(float(restored_best_loss)):
        best_loss = float(restored_best_loss)
    patience_counter = int(resume_state.get("patience_counter") or 0)
    if resume and math.isfinite(best_loss):
        logger.info(
            f"Restored early-stopping state: best_loss={best_loss:.6e}, "
            f"patience={patience_counter}"
        )
    shutdown_requested = False

    def signal_handler(signum: int, frame: Any) -> None:
        nonlocal shutdown_requested
        logger.warning(f"Signal {signum} received, finishing current step...")
        shutdown_requested = True

    old_handler = signal.signal(signal.SIGINT, signal_handler)
    old_handler_term = signal.signal(signal.SIGTERM, signal_handler)
    old_handler_hup = signal.signal(signal.SIGHUP, signal_handler)

    def persist(
        path: str,
        loss_value: float,
        is_best: bool = False,
        val_loss: float | None = None,
        val_loss_ema: float | None = None,
    ) -> None:
        """Save the current step together with the early-stopping state."""
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            ema=ema,
            step=step,
            loss=loss_value,
            path=path,
            logger=logger,
            is_best=is_best,
            best_loss=best_loss,
            patience_counter=patience_counter,
            val_loss=val_loss,
            val_loss_ema=val_loss_ema,
            steps=schedule_steps,
            warmup_steps=schedule_warmup_steps,
            target_steps=steps,
            manifest=manifest,
            seed=seed,
        )

    try:
        model.train()
        step = start_step
        accum_step = 0  # Accumulation step counter
        epoch = 0
        avg_loss = 0.0  # Default value if training exits early (ADR-042)
        interval_loss = 0.0
        interval_steps = 0
        interval_start = time.time()
        evaluation_loss = 0.0
        evaluation_steps = 0
        last_val_loss: float | None = None
        last_val_loss_ema: float | None = None
        stop_training = False
        saved_on_shutdown = False

        while step < steps and not stop_training:
            epoch += 1

            for images, breeds in train_loader:
                if step >= steps:
                    break

                images = images.to(device, non_blocking=True)
                breeds = breeds.to(device, non_blocking=True)

                # Sample timesteps
                t = sample_timesteps(
                    batch_size,
                    device,
                    sampling=timestep_sampling,  # type: ignore[arg-type]
                    logit_normal_mean=logit_normal_mean,
                    logit_normal_std=logit_normal_std,
                )

                # Mixed precision context
                context = torch.amp.autocast("cuda") if scaler else nullcontext()

                with context:
                    # Classifier-free guidance: drop breed conditioning 10% of time
                    # Uses dedicated null token (num_classes) matching forward_with_cfg
                    dropout_prob = 0.1
                    drop_mask = (
                        torch.rand(breeds.shape[0], device=device) < dropout_prob
                    )
                    train_breeds = torch.where(drop_mask, null_token, breeds)

                    # Flow matching: x0 is noise, x1 is target image
                    x0 = torch.randn_like(images)
                    pred, target = flow_matching_step(
                        model, x0, images, t, train_breeds
                    )
                    # Normalize loss by accumulation steps for correct gradient scaling
                    loss = loss_fn(pred, target) / gradient_accumulation_steps

                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_step += 1

                # Perform optimizer step after accumulation steps
                if accum_step % gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        if gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), gradient_clip
                            )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), gradient_clip
                            )
                        optimizer.step()

                    scheduler.step()
                    ema.update(model)
                    optimizer.zero_grad()

                    # Track loss
                    interval_loss += loss.item() * gradient_accumulation_steps
                    interval_steps += 1
                    evaluation_loss += loss.item() * gradient_accumulation_steps
                    evaluation_steps += 1
                    step += 1

                    # Logging
                    if step % log_interval == 0:
                        avg_loss = interval_loss / interval_steps
                        # Non-finite avg_loss can occur at the start of a
                        # warmup window (small batch + early-step AMP scaler
                        # has not yet calibrated its dynamic scale) or on an
                        # outlier batch. Emit a one-line warning instead of
                        # the confusing "Loss: inf" line and skip the metric
                        # log; the model + EMA continue to update, and the
                        # next log interval starts clean once interval_loss is
                        # reset below.
                        if not math.isfinite(avg_loss):
                            logger.warning(
                                f"Step {step:,}/{steps:,} | "
                                f"Non-finite avg loss ({avg_loss}); "
                                "skipping this log entry."
                            )
                        else:
                            current_lr = scheduler.get_last_lr()[0]
                            elapsed = time.time() - interval_start
                            steps_per_sec = interval_steps / max(elapsed, 0.001)

                            logger.info(
                                f"Step {step:,}/{steps:,} | "
                                f"Loss: {avg_loss:.6e} | "
                                f"LR: {current_lr:.2e} | "
                                f"Speed: {steps_per_sec:.1f} steps/s | "
                                f"Effective batch: {effective_batch_size}"
                            )
                            log_gpu_memory(logger, "  ")

                            tracker.log_metrics(
                                {
                                    "loss": avg_loss,
                                    "learning_rate": current_lr,
                                },
                                step=step,
                            )

                        interval_loss = 0.0
                        interval_steps = 0
                        interval_start = time.time()

                    # Save checkpoint
                    if step % save_interval == 0:
                        avg_loss = (
                            evaluation_loss / evaluation_steps
                            if evaluation_steps
                            else float("nan")
                        )

                        # Held-out validation drives selection and early
                        # stopping; the augmented training average is only a
                        # fallback when the split is disabled.
                        val_loss: float | None = None
                        val_loss_ema: float | None = None
                        if val_loader is not None:
                            raw_val, ema_val = evaluate_model_and_ema(
                                model,
                                val_loader,
                                device,
                                num_batches=val_batches,
                                seed=seed,
                                ema=ema,
                                timestep_sampling=timestep_sampling,  # type: ignore[arg-type]
                                logit_normal_mean=logit_normal_mean,
                                logit_normal_std=logit_normal_std,
                            )
                            val_loss, val_loss_ema = raw_val, ema_val

                        selection_loss = (
                            val_loss_ema if val_loss_ema is not None else val_loss
                        )
                        if selection_loss is None or not math.isfinite(selection_loss):
                            selection_loss = avg_loss

                        if val_loss is not None:
                            ema_text = (
                                f", ema={val_loss_ema:.6e}"
                                if val_loss_ema is not None
                                else ""
                            )
                            logger.info(
                                f"Validation loss at step {step:,}: "
                                f"raw={val_loss:.6e}{ema_text} | "
                                f"selected={selection_loss:.6e}"
                            )
                            val_metrics = {
                                "val_loss": val_loss,
                                "selection_loss": selection_loss,
                            }
                            if val_loss_ema is not None:
                                val_metrics["val_loss_ema"] = val_loss_ema
                            tracker.log_metrics(val_metrics, step=step)

                        improved = (
                            math.isfinite(selection_loss)
                            and selection_loss < best_loss - early_stopping_min_delta
                        )
                        last_val_loss = val_loss
                        last_val_loss_ema = val_loss_ema
                        persist(
                            output,
                            selection_loss,
                            is_best=improved,
                            val_loss=val_loss,
                            val_loss_ema=val_loss_ema,
                        )
                        # Mid-run Hub push (cadence-capped) so a cancelled
                        # slice still syncs its latest checkpoint for
                        # cross-provider resume (train-pool.yml).
                        if (
                            not no_hub_push
                            and hub_push_interval > 0
                            and step % hub_push_interval == 0
                        ):
                            logger.info(f"GPU pool: mid-run Hub push at step {step}...")
                            try:
                                from gpu_pool import push_checkpoint_to_hub

                                push_checkpoint_to_hub(
                                    checkpoint_path=output,
                                    hub_repo=hub_repo,
                                    checkpoint_name="dit_model.pt",
                                    experiment_id=manifest["experiment_id"],
                                    completed_steps=step,
                                    training_state_path=str(
                                        Path(output).parent / TRAINING_STATE_FILENAME
                                    ),
                                )
                                push_checkpoint_to_hub(
                                    checkpoint_path=ema_output,
                                    hub_repo=hub_repo,
                                    checkpoint_name="dit_model_ema.pt",
                                    experiment_id=manifest["experiment_id"],
                                    completed_steps=step,
                                    training_state_path=str(
                                        Path(output).parent / TRAINING_STATE_FILENAME
                                    ),
                                )
                            except Exception as e:
                                logger.warning(
                                    f"GPU pool: mid-run hub push skipped ({e})"
                                )
                        if improved:
                            best_loss = selection_loss
                            patience_counter = 0
                            logger.info(f"New best loss: {best_loss:.6e}")
                        elif early_stopping_patience > 0:
                            patience_counter += 1
                            logger.info(
                                f"Loss plateau detected ({patience_counter}/{early_stopping_patience} evaluations)"
                            )

                        # Adaptive early stopping check (self-learning)
                        evaluation_loss = 0.0
                        evaluation_steps = 0

                        if (
                            early_stopping_patience > 0
                            and patience_counter >= early_stopping_patience
                        ):
                            logger.info(
                                f"Early stopping triggered at step {step:,}. "
                                f"Loss hasn't improved for {early_stopping_patience} evaluations."
                            )
                            logger.info(
                                f"Final best loss: {best_loss:.6e} at step {step:,}"
                            )
                            persist(
                                output,
                                best_loss,
                                val_loss=last_val_loss,
                                val_loss_ema=last_val_loss_ema,
                            )
                            persist(
                                ema_output,
                                best_loss,
                                val_loss=last_val_loss,
                                val_loss_ema=last_val_loss_ema,
                            )
                            step = steps  # Break outer loop
                            break

                    # Generate samples
                    if step % sample_interval == 0:
                        logger.info(f"Generating samples at step {step:,}...")
                        model.eval()
                        sample_breeds = torch.arange(min(8, num_classes), device=device)
                        generated = sample(
                            model,
                            sample_breeds,
                            num_steps=50,
                            device=device,
                            image_size=image_size,
                            cfg_scale=1.5,
                            progress=False,
                        )
                        model.train()
                        # Save samples (optional, requires PIL)
                        try:
                            from PIL import Image

                            samples_dir = Path(output).parent / "samples"
                            samples_dir.mkdir(parents=True, exist_ok=True)

                            for i in range(len(generated)):
                                img = (
                                    (
                                        generated[i].permute(1, 2, 0).cpu().numpy()
                                        * 127.5
                                        + 127.5
                                    )
                                    .clip(0, 255)
                                    .astype("uint8")
                                )
                                Image.fromarray(img).save(
                                    samples_dir / f"step_{step:,}_breed_{i}.png"
                                )
                            logger.info(f"Saved samples to {samples_dir}")
                        except ImportError:
                            logger.info("PIL not available, skipping sample save")

                # Check for shutdown
                if shutdown_requested:
                    logger.info("Shutdown requested, saving checkpoint...")
                    persist(
                        output,
                        avg_loss,
                        val_loss=last_val_loss,
                        val_loss_ema=last_val_loss_ema,
                    )
                    persist(
                        ema_output,
                        avg_loss,
                        val_loss=last_val_loss,
                        val_loss_ema=last_val_loss_ema,
                    )
                    # Exit the outer loop too, otherwise every remaining batch
                    # re-runs both 500MB saves until the container is killed.
                    stop_training = True
                    saved_on_shutdown = True
                    break

            # Epoch cleanup
            cleanup_memory()

        # Final save. Skip when this invocation trained no steps (e.g. a resume
        # already at/past the target) — otherwise best_loss stays inf and gets
        # written as 0.0, clobbering a good checkpoint's metadata.
        if step > start_step:
            if evaluation_steps:
                final_eval_loss = evaluation_loss / evaluation_steps
            else:
                final_eval_loss = float("nan")

            if last_val_loss is not None:
                # Held-out validation already drove selection for this run, so
                # keep it instead of replacing it with the augmented training
                # mean (the two are not the same metric).
                final_loss = best_loss
            else:
                final_loss = min(best_loss, final_eval_loss)
            if not math.isfinite(final_loss):
                final_loss = avg_loss
            best_loss = final_loss
            logger.info("=" * 60)
            logger.info(f"Training complete. Final loss: {best_loss:.6e}")

            # The shutdown branch already wrote both checkpoints; skip the
            # duplicate 500MB pair at the timeout boundary.
            if not saved_on_shutdown:
                persist(
                    output,
                    best_loss,
                    val_loss=last_val_loss,
                    val_loss_ema=last_val_loss_ema,
                )
                persist(
                    ema_output,
                    best_loss,
                    val_loss=last_val_loss,
                    val_loss_ema=last_val_loss_ema,
                )

            log_gpu_memory(logger, "Final | ")

            tracker.log_metrics({"final_loss": best_loss, "total_steps": step})
            tracker.log_artifact(output)
            tracker.log_artifact(ema_output)
        else:
            logger.info(
                "No steps trained (checkpoint already at/past target steps); "
                "keeping the existing checkpoint unchanged."
            )
            # A resume may point at a different file than the output path
            # (e.g. hub-resume pulls dit_model_ema.pt). Preserve the loaded
            # model at the requested output so callers can rely on it.
            if resume and not Path(output).exists() and Path(resume).exists():
                import shutil

                Path(output).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(resume, output)
                logger.info(f"Copied resumed checkpoint to {output}")
            # Never clobber a valid completed checkpoint with a no-op one
            # (issue #163) — but make sure the output directory at least has
            # a training state describing the checkpoint that lives there.
            state_path = Path(output).parent / TRAINING_STATE_FILENAME
            if not state_path.exists():
                write_training_state(
                    state_path, manifest=manifest, completed_steps=step
                )
            best_loss = float("nan")

        tracker.end_run()

        return best_loss

    finally:
        if shutdown_requested:
            logger.info(f"Training ended at step {step}/{steps} due to signal shutdown")
        else:
            logger.info(f"Training ended at step {step}/{steps}")
        signal.signal(signal.SIGINT, old_handler)
        signal.signal(signal.SIGTERM, old_handler_term)
        signal.signal(signal.SIGHUP, old_handler_hup)


@app.local_entrypoint()
def main(
    data_dir: str = "/data/cats",
    steps: int = 100_000,
    batch_size: int = 128,
    lr: float = 5e-5,
    image_size: int = 128,
    output: str = "/outputs/dit_model.pt",
    ema_output: str = "/outputs/dit_model_ema.pt",
    num_workers: int = 0,
    mixed_precision: bool = True,
    gradient_clip: float = 1.0,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 2_000,
    min_lr: float = 1e-6,
    val_split: float = 0.05,
    val_batches: int = 8,
    timestep_sampling: str = "uniform",
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
    save_interval: int = 500,
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 0.001,
    augmentation_level: str = "full",
    resume: str | None = None,
    hub_resume: bool = False,
    no_hub_push: bool = False,
    hub_push_interval: int = 0,
    experiment_id: str = "dit-breed-conditioned-v4",
    allow_experiment_mismatch: bool = False,
):
    """Local entrypoint for Modal CLI (ADR-025: @modal.enter() class pattern).

    Usage:
        modal run src/train_dit.py --steps 100000
        modal run src/train_dit.py --steps 100000 --batch-size 512 --lr 5e-5
        modal run src/train_dit.py --save-interval 1000
        modal run src/train_dit.py --resume /outputs/checkpoints/dit/current/dit_model.pt
        modal run src/train_dit.py --steps 20000 --hub-resume --no-hub-push
    """
    trainer = DiTTrainer()
    result = trainer.train.remote(
        data_dir=data_dir,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        image_size=image_size,
        output=output,
        ema_output=ema_output,
        num_workers=num_workers,
        mixed_precision=mixed_precision,
        gradient_clip=gradient_clip,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
        val_split=val_split,
        val_batches=val_batches,
        timestep_sampling=timestep_sampling,
        logit_normal_mean=logit_normal_mean,
        logit_normal_std=logit_normal_std,
        save_interval=save_interval,
        early_stopping_patience=early_stopping_patience,
        augmentation_level=augmentation_level,
        resume_checkpoint=resume,
        hub_resume=hub_resume,
        no_hub_push=no_hub_push,
        hub_push_interval=hub_push_interval,
        experiment_id=experiment_id,
        allow_experiment_mismatch=allow_experiment_mismatch,
    )
    print(f"Training completed: {result}")


if __name__ == "__main__":
    args = parse_args()
    try:
        train_dit_local(
            data_dir=args.data_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            image_size=args.image_size,
            output=args.output,
            ema_output=args.ema_output,
            num_workers=args.num_workers,
            mixed_precision=args.mixed_precision,
            gradient_clip=args.gradient_clip,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            min_lr=args.min_lr,
            val_split=args.val_split,
            val_batches=args.val_batches,
            timestep_sampling=args.timestep_sampling,
            logit_normal_mean=args.logit_normal_mean,
            logit_normal_std=args.logit_normal_std,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            sample_interval=args.sample_interval,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            log_file=args.log_file,
            ema_beta=args.ema_beta,
            resume=args.resume,
            augmentation_level=args.augmentation_level,
            experiment_id=args.experiment_id,
            allow_experiment_mismatch=args.allow_experiment_mismatch,
        )
    except (TrainingError, Exception) as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        sys.exit(130)
