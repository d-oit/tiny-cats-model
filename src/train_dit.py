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
import signal
import sys
import time
import zipfile
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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
        "--steps", type=int, default=100_000, help="Total training steps"
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

    args = parser.parse_args()

    # Load config from YAML if provided
    if args.config:
        import yaml

        with open(args.config) as f:
            config = yaml.safe_load(f)

        # Apply config values as defaults (CLI args still override)
        for section in config.values():
            if isinstance(section, dict):
                for key, value in section.items():
                    if getattr(args, key.replace("-", "_"), None) is None:
                        setattr(args, key.replace("-", "_"), value)

    return args


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


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA,
    step: int,
    loss: float,
    path: str | Path,
    logger: logging.Logger,
    is_best: bool = False,
) -> None:
    """Save training checkpoint with EMA weights.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        ema: EMA tracker.
        step: Current training step.
        loss: Current loss value.
        path: Checkpoint path.
        logger: Logger instance.
        is_best: Whether this is the best model.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ema_shadow_params": ema.shadow_params,
        "ema_step": ema.step,
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "image_size": model.image_size,
            "patch_size": model.patch_size,
            "embed_dim": model.embed_dim,
            "depth": len(model.blocks),
            "num_heads": model.blocks[0].attn.num_heads,
        },
    }

    # Atomic write: write to temp file then rename (prevents corruption on crash)
    tmp_path = str(path) + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, str(path))
    logger.info(f"Saved checkpoint at step {step:,} (loss={loss:.6e}) to {path}")

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
) -> tuple[nn.Module, torch.optim.Optimizer | None, EMA | None, int]:
    """Load checkpoint for resume training.

    Args:
        path: Checkpoint path.
        model: Model to load weights into.
        optimizer: Optional optimizer to load state.
        ema: Optional EMA to load shadow params.
        logger: Optional logger.

    Returns:
        Tuple of (model, optimizer, ema, start_step). When ``path`` exists but
        is unreadable (truncated zip from a preempted run, EOFError on partial
        write, etc.) the file is renamed to ``<path>.corrupt`` for forensics
        and ``start_step=0`` is returned so training restarts from scratch.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except (zipfile.BadZipFile, RuntimeError, EOFError, OSError) as exc:
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
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if ema and "ema_shadow_params" in checkpoint:
        ema.shadow_params = checkpoint["ema_shadow_params"]
        ema.step = checkpoint.get("ema_step", 0)
        if logger:
            logger.info(f"Loaded EMA state (step {ema.step:,})")

    start_step = checkpoint.get("step", 0) + 1
    if logger:
        logger.info(f"Loaded checkpoint from {path} (resuming at step {start_step:,})")

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
    .add_local_file("src/dit.py", "/app/dit.py")
    .add_local_file("src/flow_matching.py", "/app/flow_matching.py")
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

        # Use a stable (non-dated) checkpoint directory so Modal retries and
        # manually re-triggered runs can find and resume prior progress.
        # A dated directory meant every retry silently restarted from step 0.
        checkpoint_dir = "/outputs/checkpoints/dit/current"
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        samples_dir = f"{checkpoint_dir}/samples"
        Path(samples_dir).mkdir(parents=True, exist_ok=True)

        # Stable output paths enable resume across retries
        output = output or f"{checkpoint_dir}/dit_model.pt"
        ema_output = ema_output or f"{checkpoint_dir}/dit_model_ema.pt"
        log_file = log_file or f"{checkpoint_dir}/dit_training.log"

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
            )

            # Export to ONNX and Quantize (Issue #63)
            logger.info("Exporting to ONNX...")
            try:
                from export_dit_onnx import export_generator_onnx, load_model
                from optimize_onnx import optimize_onnx

                onnx_path = "/outputs/generator.onnx"
                quant_dir = "/outputs"

                # Load best model for export
                model_to_export = load_model(output, image_size=image_size)
                export_generator_onnx(model_to_export, output_path=onnx_path)
                logger.info(f"✅ Exported to {onnx_path}")

                logger.info("Quantizing ONNX model...")
                optimize_onnx(
                    model_path=onnx_path,
                    output_dir=quant_dir,
                    method="dynamic",
                    model_type="generator",
                )
                logger.info(
                    f"✅ Quantized model saved to {quant_dir}/generator_quantized.onnx"
                )

                # Copy best .pt to root for easier CI download
                import shutil

                shutil.copy2(output, "/outputs/tinydit_final.pt")
                logger.info("✅ Copied best model to /outputs/tinydit_final.pt")

            except Exception as e:
                logger.warning(f"ONNX export/quantization failed: {e}")

            # Commit volume after successful training (ADR-024: explicit commits)
            volume_outputs.commit()
            logger.info("Checkpoint committed to volume")

            # GPU pool: push checkpoints to Hub for cross-provider resume
            # (train-pool.yml --no-hub-push disables; degrades gracefully).
            if not no_hub_push:
                logger.info("GPU pool: pushing checkpoints to HuggingFace Hub...")
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
                            )
                except Exception as e:  # graceful degradation
                    logger.warning(f"GPU pool: hub push skipped ({e})")

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
    # Use ImageFolder directly
    from torchvision.datasets import ImageFolder

    from dataset import build_enhanced_transforms

    transform = build_enhanced_transforms(
        train=True,
        image_size=image_size,
        augmentation_level=augmentation_level,  # type: ignore[arg-type]
    )
    dataset = ImageFolder(data_dir, transform=transform)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
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

    Returns:
        Final training loss.
    """
    # Import DiT modules (works for both local and Modal after path setup)
    from dit import count_parameters, tinydit_128, tinydit_256
    from flow_matching import (
        EMA,
        FlowMatchingLoss,
        flow_matching_step,
        sample,
        sample_t,
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
    if image_size == 128:
        model = tinydit_128(num_classes=num_classes).to(device)
    elif image_size == 256:
        model = tinydit_256(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unsupported image_size: {image_size}. Use 128 or 256.")

    logger.info(
        f"Model: TinyDiT | Image size: {image_size} | "
        f"Parameters: {count_parameters(model):,}"
    )

    # Create dataloader
    train_loader = create_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        augmentation_level=augmentation_level,
    )
    effective_batch_size = batch_size * gradient_accumulation_steps
    logger.info(
        f"DataLoader created: {len(train_loader)} batches per epoch | "
        f"Effective batch size: {effective_batch_size} "
        f"(batch_size={batch_size} x accumulation_steps={gradient_accumulation_steps})"
    )

    # Optimizer and loss
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.95))
    loss_fn = FlowMatchingLoss()

    # LR scheduler with warmup and cosine annealing using LambdaLR (ADR-032)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup: 0.01 -> 1.0
            return 0.01 + 0.99 * float(current_step) / float(max(1, warmup_steps))
        # Cosine annealing: 1.0 -> 0.0
        progress = float(current_step - warmup_steps) / float(
            max(1, steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

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
    }
    tracker.start_run(
        params, run_name=f"dit_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    # Resume from checkpoint
    start_step = 0
    if resume:
        logger.info(f"Resuming from checkpoint: {resume}")
        model, optimizer, _ema, start_step = load_checkpoint(
            resume, model, optimizer, ema, logger
        )
        # Use loaded EMA if available
        ema = _ema if _ema is not None else ema
        # Adjust scheduler to current step without triggering the
        # "scheduler.step() before optimizer.step()" warning
        scheduler.last_epoch = start_step - 1

    # Training state
    best_loss = float("inf")
    shutdown_requested = False
    patience_counter = 0
    last_eval_step = 0
    loss_history: list[float] = []  # Track loss trajectory for adaptive early stopping

    def signal_handler(signum: int, frame: Any) -> None:
        nonlocal shutdown_requested
        logger.warning(f"Signal {signum} received, finishing current step...")
        shutdown_requested = True

    old_handler = signal.signal(signal.SIGINT, signal_handler)
    old_handler_term = signal.signal(signal.SIGTERM, signal_handler)
    old_handler_hup = signal.signal(signal.SIGHUP, signal_handler)

    try:
        model.train()
        step = start_step
        accum_step = 0  # Accumulation step counter
        epoch = 0
        avg_loss = 0.0  # Default value if training exits early (ADR-042)

        while step < steps:
            epoch += 1
            epoch_loss = 0.0
            epoch_start = time.time()

            for images, breeds in train_loader:
                if step >= steps:
                    break

                images = images.to(device, non_blocking=True)
                breeds = breeds.to(device, non_blocking=True)

                # Sample timesteps
                t = sample_t(batch_size, device)

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
                    epoch_loss += loss.item() * gradient_accumulation_steps
                    step += 1

                    # Logging
                    if step % log_interval == 0:
                        avg_loss = epoch_loss / log_interval
                        # Non-finite avg_loss can occur at the start of a
                        # warmup window (small batch + early-step AMP scaler
                        # has not yet calibrated its dynamic scale) or on an
                        # outlier batch. Emit a one-line warning instead of
                        # the confusing "Loss: inf" line and skip the metric
                        # log; the model + EMA continue to update, and the
                        # next log interval starts clean once epoch_loss is
                        # reset below.
                        if not math.isfinite(avg_loss):
                            logger.warning(
                                f"Step {step:,}/{steps:,} | "
                                f"Non-finite avg loss ({avg_loss}); "
                                "skipping this log entry."
                            )
                        else:
                            current_lr = scheduler.get_last_lr()[0]
                            elapsed = time.time() - epoch_start
                            steps_per_sec = log_interval / max(elapsed, 0.001)

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

                        epoch_loss = 0.0
                        epoch_start = time.time()

                    # Save checkpoint
                    if step % save_interval == 0:
                        save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            ema=ema,
                            step=step,
                            loss=avg_loss,
                            path=output,
                            logger=logger,
                            is_best=(avg_loss < best_loss),
                        )
                        if avg_loss < best_loss - early_stopping_min_delta:
                            best_loss = avg_loss
                            patience_counter = 0
                            logger.info(f"New best loss: {best_loss:.6e}")
                        else:
                            patience_counter += 1
                            logger.info(
                                f"Loss plateau detected ({patience_counter}/{early_stopping_patience} evaluations)"
                            )

                        # Adaptive early stopping check (self-learning)
                        if patience_counter >= early_stopping_patience:
                            logger.info(
                                f"Early stopping triggered at step {step:,}. "
                                f"Loss hasn't improved for {early_stopping_patience} evaluations."
                            )
                            logger.info(
                                f"Final best loss: {best_loss:.6e} at step {step:,}"
                            )
                            save_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                ema=ema,
                                step=step,
                                loss=best_loss,
                                path=output,
                                logger=logger,
                            )
                            save_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                ema=ema,
                                step=step,
                                loss=best_loss,
                                path=ema_output,
                                logger=logger,
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
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        ema=ema,
                        step=step,
                        loss=avg_loss,
                        path=output,
                        logger=logger,
                    )
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        ema=ema,
                        step=step,
                        loss=avg_loss,
                        path=ema_output,
                        logger=logger,
                    )
                    break

            # Epoch cleanup
            cleanup_memory()

        # Final save
        logger.info("=" * 60)
        logger.info(f"Training complete. Final loss: {best_loss:.6e}")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            ema=ema,
            step=step,
            loss=best_loss,
            path=output,
            logger=logger,
        )
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            ema=ema,
            step=step,
            loss=best_loss,
            path=ema_output,
            logger=logger,
        )

        log_gpu_memory(logger, "Final | ")

        tracker.log_metrics({"final_loss": best_loss, "total_steps": step})
        tracker.log_artifact(output)
        tracker.log_artifact(ema_output)
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
    save_interval: int = 500,
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 0.001,
    augmentation_level: str = "full",
    resume: str | None = None,
    hub_resume: bool = False,
    no_hub_push: bool = False,
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
        save_interval=save_interval,
        early_stopping_patience=early_stopping_patience,
        augmentation_level=augmentation_level,
        resume_checkpoint=resume,
        hub_resume=hub_resume,
        no_hub_push=no_hub_push,
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
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            sample_interval=args.sample_interval,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            log_file=args.log_file,
            ema_beta=args.ema_beta,
            resume=args.resume,
            augmentation_level=args.augmentation_level,
        )
    except (TrainingError, Exception) as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        sys.exit(130)
