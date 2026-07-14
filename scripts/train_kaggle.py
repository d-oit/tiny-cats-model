#!/usr/bin/env python3
"""Training script for Kaggle.

Trains TinyDiT on Kaggle's free GPU (T4/P100, 30h/week), with
checkpoint persistence and HuggingFace Hub sync.

Kaggle provides free GPU access with 30 hours/week. This script:
1. Detects the Kaggle environment and GPU
2. Uses Kaggle dataset if available, or downloads from source
3. Pulls latest checkpoint from HuggingFace Hub (if available)
4. Runs training with periodic checkpoint saves
5. Pushes checkpoints to Hub for cross-provider resume

Kaggle specifics:
- Sessions limited to ~9 hours
- GPU: T4 (16GB) or P100 (16GB)
- Output directory: /kaggle/working/ (persists for session)
- Dataset input: /kaggle/input/ (read-only, can be added via Kaggle UI)

Usage (in Kaggle notebook/script):
    # Install deps (first time)
    !pip install torch torchvision pillow tqdm huggingface_hub

    # Run training
    !python scripts/train_kaggle.py --steps 30000 --batch-size 256

    # Resume from Hub
    !python scripts/train_kaggle.py --steps 30000 --hub-resume
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────
# Path setup
# ──────────────────────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

# Kaggle-specific paths
KAGGLE_WORKING = "/kaggle/working"
KAGGLE_INPUT = "/kaggle/input"
KAGGLE_DATA_DIR = os.environ.get("KAGGLE_DATA_DIR", f"{KAGGLE_WORKING}/cats-data")
KAGGLE_CHECKPOINT_DIR = os.environ.get(
    "KAGGLE_CHECKPOINT_DIR", f"{KAGGLE_WORKING}/checkpoints"
)
KAGGLE_HUB_REPO = os.environ.get("KAGGLE_HUB_REPO", "d4oit/tiny-cats-model")
KAGGLE_MAX_SESSION_HOURS = 9  # Kaggle session limit

logger = logging.getLogger("kaggle_train")


def setup_logging(log_file: str | None = None) -> logging.Logger:
    """Configure logging for Kaggle training."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | [Kaggle] %(message)s",
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


def detect_kaggle_environment() -> dict[str, Any]:
    """Detect Kaggle environment and GPU.

    Returns:
        Dict with environment info.
    """
    info: dict[str, Any] = {
        "is_kaggle": False,
        "has_gpu": False,
        "gpu_name": "N/A",
        "gpu_memory_gb": 0.0,
        "session_limit_hours": KAGGLE_MAX_SESSION_HOURS,
    }

    # Detect Kaggle
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        info["is_kaggle"] = True
    if os.environ.get("KAGGLE_URL_BASE"):
        info["is_kaggle"] = True

    # Detect GPU
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

    # Check for internet (Kaggle sometimes restricts internet)
    info["internet_enabled"] = (
        os.environ.get("KAGGLE_IS_COMPETITION") is None
        and os.environ.get("KAGGLE_KERNEL_RUN_TYPE") != "batch"
    )

    return info


def setup_kaggle_dirs() -> tuple[Path, Path]:
    """Create Kaggle working directories.

    Returns:
        Tuple of (data_dir, checkpoint_dir).
    """
    data_dir = Path(KAGGLE_DATA_DIR)
    checkpoint_dir = Path(KAGGLE_CHECKPOINT_DIR)

    for d in [data_dir, checkpoint_dir]:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Directory OK: {d}")

    return data_dir, checkpoint_dir


def prepare_kaggle_data(data_dir: str | Path) -> bool:
    """Prepare dataset for Kaggle training.

    Tries in order:
    1. Use /kaggle/input/cats-dataset if available (from Kaggle dataset)
    2. Copy from /kaggle/input/oxford-pet-dataset if available
    3. Download fresh

    Args:
        data_dir: Target data directory.

    Returns:
        True if data is ready.
    """
    data_path = Path(data_dir)

    # Already has data
    if data_path.exists() and list(data_path.iterdir()):
        logger.info(f"Dataset ready at {data_dir}")
        return True

    # Try Kaggle dataset inputs
    kaggle_inputs = [
        Path("/kaggle/input/cats-dataset"),
        Path("/kaggle/input/oxford-pet-dataset"),
        Path("/kaggle/input/oxford-iiit-pet"),
    ]

    for kaggle_ds in kaggle_inputs:
        if kaggle_ds.exists() and list(kaggle_ds.iterdir()):
            logger.info(f"Using Kaggle dataset: {kaggle_ds}")
            # Symlink or copy
            for item in kaggle_ds.iterdir():
                dest = data_path / item.name
                if item.is_dir() and not item.name.startswith("."):
                    if not dest.exists():
                        shutil.copytree(str(item), str(dest))
                elif item.is_file():
                    shutil.copy2(str(item), str(dest))
            logger.info(f"Copied dataset to {data_dir}")
            return True

    # Download fresh
    logger.info("No Kaggle dataset found — downloading fresh...")
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
            timeout=900,
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
        description="Train TinyDiT on Kaggle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core training
    parser.add_argument("--steps", type=int, default=30_000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=128, help="Image size")

    # Paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default=KAGGLE_DATA_DIR,
        help="Dataset directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=KAGGLE_CHECKPOINT_DIR,
        help="Checkpoint directory",
    )

    # Resume / Hub
    parser.add_argument(
        "--hub-resume",
        action="store_true",
        help="Pull latest checkpoint from Hub before training",
    )
    parser.add_argument(
        "--hub-repo",
        type=str,
        default=KAGGLE_HUB_REPO,
        help="HF Hub repo for checkpoint sync",
    )
    parser.add_argument(
        "--hub-token",
        type=str,
        default=None,
        help="HF token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--no-hub-push",
        action="store_true",
        help="Skip pushing checkpoint to Hub after training",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only sync checkpoints from Hub, no training",
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
        default=500,
        help="Checkpoint save interval (frequent for Kaggle preemption)",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=3_000,
        help="Sample generation interval",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
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
    """Main entry point for Kaggle training."""
    args = parse_args()
    setup_logging()

    # Detect environment
    env_info = detect_kaggle_environment()

    logger.info("=" * 60)
    logger.info("KAGGLE — TINYDIT TRAINING")
    logger.info("=" * 60)
    logger.info(f"Kaggle environment: {env_info['is_kaggle']}")
    logger.info(f"GPU: {env_info['gpu_name']} ({env_info['gpu_memory_gb']:.1f} GB)")
    logger.info(f"Internet enabled: {env_info['internet_enabled']}")
    logger.info(f"Session limit: {env_info['session_limit_hours']}h")
    logger.info("Weekly GPU quota: 30h")

    if not env_info["has_gpu"]:
        logger.warning(
            "No GPU detected! Enable GPU in Kaggle: "
            "Notebook → Settings → Accelerator → GPU T4 x2"
        )

    data_dir, checkpoint_dir = setup_kaggle_dirs()
    logger.info(f"Data: {data_dir}")
    logger.info(f"Checkpoints: {checkpoint_dir}")

    # Sync-only mode
    if args.sync_only:
        logger.info("Sync-only mode — pulling checkpoint from Hub")
        from gpu_pool import pull_checkpoint_from_hub

        pulled = pull_checkpoint_from_hub(
            hub_repo=args.hub_repo,
            checkpoint_name="dit_model_ema.pt",
            output_dir=str(checkpoint_dir),
            token=args.hub_token,
        )
        if pulled:
            logger.info(f"Pulled checkpoint: {pulled}")
        else:
            logger.info("No checkpoint on Hub")
        return 0

    # Prepare dataset
    if not prepare_kaggle_data(data_dir):
        logger.error("Cannot proceed without dataset")
        return 1

    # Pull checkpoint from Hub for resume
    resume_from: str | None = None
    if args.hub_resume:
        from gpu_pool import pull_checkpoint_from_hub

        pulled = pull_checkpoint_from_hub(
            hub_repo=args.hub_repo,
            checkpoint_name="dit_model_ema.pt",
            output_dir=str(checkpoint_dir),
            token=args.hub_token,
        )
        if pulled:
            resume_from = str(pulled)
            logger.info(f"Resuming from Hub: {resume_from}")

    # Also check local checkpoint
    local_ema = checkpoint_dir / "dit_model_ema.pt"
    if resume_from is None and local_ema.exists():
        import zipfile

        if zipfile.is_zipfile(local_ema):
            resume_from = str(local_ema)
            logger.info(f"Resuming from local: {resume_from}")

    if resume_from:
        logger.info("✅ Resuming from checkpoint")
    else:
        logger.info("Starting fresh training")

    # Adaptive batch size for Kaggle GPU memory
    actual_batch = args.batch_size
    if env_info["has_gpu"]:
        gpu_mem = env_info["gpu_memory_gb"]
        if gpu_mem <= 15 and args.batch_size > 128:
            actual_batch = 128
            logger.info(
                f"Reducing batch size {args.batch_size} → {actual_batch} "
                f"(limited GPU memory: {gpu_mem:.1f} GB)"
            )

    # Run training
    start_time = time.time()

    try:
        from train_dit import train_dit_local

        output = str(checkpoint_dir / "dit_model.pt")
        ema_output = str(checkpoint_dir / "dit_model_ema.pt")

        final_loss = train_dit_local(
            data_dir=str(data_dir),
            steps=args.steps,
            batch_size=actual_batch,
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
        logger.info(f"Training completed in {elapsed:.0f}s ({elapsed / 3600:.1f}h)")
        logger.info(f"Final loss: {final_loss:.6e}")

        # Push to Hub
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
            logger.info("Checkpoints synced to Hub from Kaggle")

        # Save to Kaggle output
        kaggle_output = Path("/kaggle/working")
        logger.info(f"Checkpoints in: {checkpoint_dir}")
        logger.info(f"Kaggle output: {kaggle_output}")
        logger.info(
            "Download checkpoints before session ends: Kaggle → Output → checkpoints/"
        )

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted — checkpoint saved")
        return 130
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1
    finally:
        # Push partial checkpoint
        partial_ema = checkpoint_dir / "dit_model_ema.pt"
        if partial_ema.exists() and not args.no_hub_push:
            try:
                from gpu_pool import push_checkpoint_to_hub

                push_checkpoint_to_hub(
                    checkpoint_path=partial_ema,
                    hub_repo=args.hub_repo,
                    checkpoint_name="partial_dit_model_ema.pt",
                    token=args.hub_token,
                )
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
