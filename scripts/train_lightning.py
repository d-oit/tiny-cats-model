#!/usr/bin/env python3
"""Training script for Lightning AI Studio.

Trains TinyDiT on Lightning AI's free T4 GPU, with checkpoint sync
via HuggingFace Hub for cross-provider resume.

Lightning AI Studio provides free T4 GPU hours per day. This script:
1. Detects the Lightning Studio environment
2. Pulls latest checkpoint from HuggingFace Hub (if available)
3. Sets up data in /teamspace/datasets/cats
4. Runs training with periodic checkpoint saves
5. Pushes checkpoints back to HuggingFace Hub for cross-provider resume

Usage (in Lightning Studio terminal):
    python scripts/train_lightning.py --steps 50000 --batch-size 256

    # Resume from Hub checkpoint
    python scripts/train_lightning.py --steps 50000 --hub-resume

    # Push checkpoint without training (sync only)
    python scripts/train_lightning.py --sync-only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────
# Path setup
# ──────────────────────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

# ──────────────────────────────────────────────────────────────────────
# Lightning-specific defaults
# ──────────────────────────────────────────────────────────────────────
LIGHTNING_DATA_DIR = os.environ.get("LIGHTNING_DATA_DIR", "/teamspace/datasets/cats")
LIGHTNING_CHECKPOINT_DIR = os.environ.get(
    "LIGHTNING_CHECKPOINT_DIR", "/teamspace/checkpoints"
)
LIGHTNING_HUB_REPO = os.environ.get("LIGHTNING_HUB_REPO", "d4oit/tiny-cats-model")

logger = logging.getLogger("lightning_train")


def setup_logging(log_file: str | None = None) -> logging.Logger:
    """Configure logging for Lightning training."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | [Lightning] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def detect_lightning_environment() -> dict[str, Any]:
    """Detect and report the Lightning Studio environment.

    Returns:
        Dict with environment info.
    """
    info: dict[str, Any] = {
        "is_lightning": False,
        "studio_id": os.environ.get("LIGHTNING_STUDIO_ID", "unknown"),
        "has_gpu": False,
        "gpu_name": "N/A",
        "gpu_memory_gb": 0.0,
    }

    # Check for Lightning Studio markers
    if os.environ.get("LIGHTNING_APP_STATE_DIR"):
        info["is_lightning"] = True
    if os.environ.get("LIGHTNING_STUDIO_ID"):
        info["is_lightning"] = True
    if Path("/teamspace").exists():
        info["is_lightning"] = True

    # Check GPU
    try:
        import torch

        if torch.cuda.is_available():
            info["has_gpu"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = (
                torch.cuda.get_device_properties(0).total_memory / 1e9
            )
    except ImportError:
        pass

    return info


def setup_lightning_dirs() -> None:
    """Create required directories in Lightning Studio."""
    for d in [LIGHTNING_DATA_DIR, LIGHTNING_CHECKPOINT_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.info(f"  Directory OK: {d}")


def download_data(data_dir: str) -> bool:
    """Download cat dataset to Lightning Studio.

    Args:
        data_dir: Target data directory.

    Returns:
        True if data is ready.
    """
    data_path = Path(data_dir)
    if data_path.exists() and list(data_path.iterdir()):
        logger.info(f"Dataset already at {data_dir}")
        return True

    logger.info("Downloading dataset...")
    data_path.mkdir(parents=True, exist_ok=True)

    download_script = _project_root / "data" / "download.py"

    if not download_script.exists():
        logger.error(f"Download script not found: {download_script}")
        return False

    try:
        import subprocess

        result = subprocess.run(
            ["python", str(download_script)],
            env={
                **os.environ,
                "DATA_DIR": str(data_path),
                "CATS_DIR": str(data_dir),
            },
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            logger.info("Dataset downloaded successfully")
            return True
        else:
            logger.error(f"Download failed: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("Download timed out")
        return False
    except Exception as e:
        logger.error(f"Download error: {e}")
        return False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train TinyDiT on Lightning AI Studio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core training
    parser.add_argument("--steps", type=int, default=50_000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=128, help="Image size")

    # Paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default=LIGHTNING_DATA_DIR,
        help="Dataset directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=LIGHTNING_CHECKPOINT_DIR,
        help="Checkpoint output directory",
    )

    # Hub sync
    parser.add_argument(
        "--hub-repo",
        type=str,
        default=LIGHTNING_HUB_REPO,
        help="HuggingFace repo for checkpoint sync",
    )
    parser.add_argument(
        "--hub-token",
        type=str,
        default=None,
        help="HF token (default: HF_TOKEN env var)",
    )
    parser.add_argument(
        "--hub-resume",
        action="store_true",
        help="Pull checkpoint from Hub before training",
    )
    parser.add_argument(
        "--no-hub-push",
        action="store_true",
        help="Skip pushing checkpoint to Hub after training",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only sync checkpoints (pull + push), no training",
    )

    # Training options
    # mixed_precision is always enabled for GPU training — no CLI flag needed
    parser.add_argument(
        "--gradient-clip", type=float, default=1.0, help="Gradient clipping"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=2_000, help="LR warmup steps"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1_000,
        help="Checkpoint save interval in steps",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=5_000,
        help="Sample generation interval",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--augmentation-level",
        type=str,
        default="full",
        choices=["basic", "medium", "full"],
        help="Data augmentation level",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")

    return parser.parse_args()


def main() -> int:
    """Main entry point for Lightning AI Studio training."""
    args = parse_args()
    setup_logging()

    # Detect environment
    env_info = detect_lightning_environment()
    logger.info("=" * 60)
    logger.info("LIGHTNING AI STUDIO — TINYDIT TRAINING")
    logger.info("=" * 60)
    logger.info(f"Studio ID: {env_info['studio_id']}")
    logger.info(f"GPU: {env_info['gpu_name']} ({env_info['gpu_memory_gb']:.1f} GB)")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Checkpoint dir: {args.checkpoint_dir}")

    setup_lightning_dirs()

    # Sync-only mode: just pull/push checkpoints
    if args.sync_only:
        logger.info("Sync-only mode — pulling/pushing checkpoints")
        from gpu_pool import pull_checkpoint_from_hub, push_checkpoint_to_hub

        pulled = pull_checkpoint_from_hub(
            hub_repo=args.hub_repo,
            checkpoint_name="dit_model_ema.pt",
            output_dir=args.checkpoint_dir,
            token=args.hub_token,
        )
        if pulled:
            logger.info(f"Pulled checkpoint: {pulled}")

        ema_path = Path(args.checkpoint_dir) / "dit_model_ema.pt"
        if ema_path.exists():
            push_checkpoint_to_hub(
                checkpoint_path=ema_path,
                hub_repo=args.hub_repo,
                checkpoint_name="dit_model_ema.pt",
                token=args.hub_token,
            )
            logger.info("Pushed checkpoint to Hub")
        return 0

    # Download dataset
    if not download_data(args.data_dir):
        logger.error("Cannot proceed without dataset")
        return 1

    # Pull checkpoint from Hub for resume
    resume_from: str | None = None
    if args.hub_resume:
        from gpu_pool import pull_checkpoint_from_hub

        pulled = pull_checkpoint_from_hub(
            hub_repo=args.hub_repo,
            checkpoint_name="dit_model_ema.pt",
            output_dir=args.checkpoint_dir,
            token=args.hub_token,
        )
        if pulled:
            resume_from = str(pulled)
            logger.info(f"Resuming from Hub checkpoint: {resume_from}")

    # Also check for local checkpoint
    local_ema = Path(args.checkpoint_dir) / "dit_model_ema.pt"
    if resume_from is None and local_ema.exists():
        import zipfile

        if zipfile.is_zipfile(local_ema):
            resume_from = str(local_ema)
            logger.info(f"Resuming from local checkpoint: {resume_from}")

    if resume_from:
        logger.info("Resuming from checkpoint")
    else:
        logger.info("Starting fresh training")

    # Note: train_dit_local handles its own signal handling for SIGINT,
    # SIGTERM, and SIGHUP internally (graceful checkpoint on shutdown).

    # Run training
    start_time = time.time()

    try:
        from train_dit import train_dit_local

        output = str(Path(args.checkpoint_dir) / "dit_model.pt")
        ema_output = str(Path(args.checkpoint_dir) / "dit_model_ema.pt")

        final_loss = train_dit_local(
            data_dir=args.data_dir,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            image_size=args.image_size,
            output=output,
            ema_output=ema_output,
            num_workers=args.num_workers,
            mixed_precision=True,
            gradient_clip=args.gradient_clip,
            warmup_steps=args.warmup_steps,
            save_interval=args.save_interval,
            sample_interval=args.sample_interval,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=0.001,
            augmentation_level=args.augmentation_level,
            seed=args.seed,
            resume=resume_from,
        )

        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.0f}s")
        logger.info(f"Final loss: {final_loss:.6e}")

        # Push checkpoint to Hub
        if not args.no_hub_push:
            from gpu_pool import push_checkpoint_to_hub

            for ckpt_name, ckpt_path in [
                ("dit_model.pt", output),
                ("dit_model_ema.pt", ema_output),
            ]:
                if Path(ckpt_path).exists():
                    push_checkpoint_to_hub(
                        checkpoint_path=ckpt_path,
                        hub_repo=args.hub_repo,
                        checkpoint_name=ckpt_name,
                        token=args.hub_token,
                    )

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1
    finally:
        # Push partial checkpoint on failure
        partial_ema = Path(args.checkpoint_dir) / "dit_model_ema.pt"
        if partial_ema.exists() and not args.no_hub_push:
            try:
                from gpu_pool import push_checkpoint_to_hub

                push_checkpoint_to_hub(
                    checkpoint_path=partial_ema,
                    hub_repo=args.hub_repo,
                    checkpoint_name="partial_dit_model_ema.pt",
                    token=args.hub_token,
                )
                logger.info("Pushed partial checkpoint to Hub")
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
