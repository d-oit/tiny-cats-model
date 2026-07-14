#!/usr/bin/env python3
"""Training script for Google Colab.

Trains TinyDiT on Colab's free T4 GPU, with Google Drive checkpoint
persistence and HuggingFace Hub sync for cross-provider resume.

Colab provides free T4 GPU access (with daily limits). This script:
1. Mounts Google Drive for persistent storage
2. Detects GPU availability (T4, V100, or CPU fallback)
3. Pulls latest checkpoint from HuggingFace Hub (if available)
4. Downloads dataset if not in Drive
5. Runs training with periodic checkpoint saves to Drive
6. Pushes checkpoints to HuggingFace Hub for cross-provider resume

Usage (in Colab cell):
    !pip install torch torchvision pillow tqdm huggingface_hub
    !python scripts/train_colab.py --steps 10000 --batch-size 256

    # Resume from Drive checkpoint
    !python scripts/train_colab.py --steps 10000 --resume

    # Sync checkpoints between Drive and Hub
    !python scripts/train_colab.py --sync-only
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

# Colab-specific paths (Drive-mounted)
COLAB_DRIVE_BASE = "/content/drive/MyDrive"
COLAB_DATA_DIR = os.environ.get(
    "COLAB_DATA_DIR",
    f"{COLAB_DRIVE_BASE}/tiny-cats-data",
)
COLAB_CHECKPOINT_DIR = os.environ.get(
    "COLAB_CHECKPOINT_DIR",
    f"{COLAB_DRIVE_BASE}/tiny-cats-checkpoints",
)
COLAB_HUB_REPO = os.environ.get("COLAB_HUB_REPO", "d4oit/tiny-cats-model")

logger = logging.getLogger("colab_train")


def setup_logging(log_file: str | None = None) -> logging.Logger:
    """Configure logging for Colab training."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | [Colab] %(message)s",
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


def mount_google_drive() -> bool:
    """Mount Google Drive in Colab.

    Returns:
        True if Drive was mounted successfully.
    """
    try:
        from google.colab import drive

        logger.info("Mounting Google Drive...")
        drive.mount("/content/drive")
        logger.info("Drive mounted at /content/drive")

        # Verify mount
        if Path("/content/drive/MyDrive").exists():
            logger.info("Drive mount verified")
            return True
        else:
            logger.warning("Drive mount may have failed — MyDrive not found")
            return False

    except ImportError:
        logger.warning("Not in Colab environment — google.colab not available")
        return False
    except Exception as e:
        logger.error(f"Drive mount failed: {e}")
        return False


def detect_colab_gpu() -> dict[str, Any]:
    """Detect Colab GPU information.

    Returns:
        Dict with GPU info.
    """
    info: dict[str, Any] = {
        "has_gpu": False,
        "gpu_name": "N/A",
        "gpu_memory_gb": 0.0,
        "is_colab": False,
    }

    # Check if in Colab
    if "google.colab" in sys.modules:
        info["is_colab"] = True

    # Check GPU
    try:
        import torch

        if torch.cuda.is_available():
            info["has_gpu"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = (
                torch.cuda.get_device_properties(0).total_memory / 1e9
            )

            # Assign GPU to Colab:0 for TF compat
            if info["is_colab"]:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    except ImportError:
        pass

    return info


def setup_colab_dirs() -> tuple[Path, Path]:
    """Create Colab directories.

    Returns:
        Tuple of (data_dir, checkpoint_dir).
    """
    data_dir = Path(COLAB_DATA_DIR)
    checkpoint_dir = Path(COLAB_CHECKPOINT_DIR)

    for d in [data_dir, checkpoint_dir]:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Directory OK: {d}")

    return data_dir, checkpoint_dir


def download_data_if_needed(data_dir: str | Path) -> bool:
    """Download dataset if not present in Drive.

    Args:
        data_dir: Target data directory.

    Returns:
        True if data is ready.
    """
    data_path = Path(data_dir)
    if data_path.exists() and list(data_path.iterdir()):
        logger.info(f"Dataset ready at {data_dir}")
        return True

    logger.info("Downloading dataset (one-time, cached in Drive)...")
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
            logger.info("Dataset downloaded and cached in Drive")
            return True
        else:
            logger.error(f"Download failed: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("Download timed out (>15 min)")
        return False
    except Exception as e:
        logger.error(f"Download error: {e}")
        return False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train TinyDiT on Google Colab",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core training
    parser.add_argument("--steps", type=int, default=20_000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=128, help="Image size")

    # Paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default=COLAB_DATA_DIR,
        help="Dataset directory (in Drive)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=COLAB_CHECKPOINT_DIR,
        help="Checkpoint directory (in Drive)",
    )

    # Resume
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in Drive or Hub",
    )

    # Hub sync
    parser.add_argument(
        "--hub-repo",
        type=str,
        default=COLAB_HUB_REPO,
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
        help="Only sync checkpoints between Drive and Hub",
    )
    parser.add_argument(
        "--no-drive-mount",
        action="store_true",
        help="Skip Drive mount (use local paths)",
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
        help="Checkpoint save interval (lower for Colab preemption)",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=2_000,
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
    """Main entry point for Colab training."""
    args = parse_args()
    setup_logging()

    # Detect environment
    gpu_info = detect_colab_gpu()

    logger.info("=" * 60)
    logger.info("GOOGLE COLAB — TINYDIT TRAINING")
    logger.info("=" * 60)
    logger.info(f"Colab environment: {gpu_info['is_colab']}")
    logger.info(f"GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_memory_gb']:.1f} GB)")

    if not gpu_info["has_gpu"]:
        logger.warning(
            "No GPU detected! Training will be very slow on CPU. "
            "Enable GPU: Runtime → Change runtime type → T4 GPU"
        )

    # Mount Drive
    if gpu_info["is_colab"] and not args.no_drive_mount:
        if not mount_google_drive():
            logger.error("Drive mount failed — cannot persist checkpoints")
            return 1

    data_dir, checkpoint_dir = setup_colab_dirs()
    logger.info(f"Data: {data_dir}")
    logger.info(f"Checkpoints: {checkpoint_dir}")

    # Sync-only mode
    if args.sync_only:
        logger.info("Sync-only mode — syncing checkpoints Drive ↔ Hub")
        from gpu_pool import pull_checkpoint_from_hub, push_checkpoint_to_hub

        pulled = pull_checkpoint_from_hub(
            hub_repo=args.hub_repo,
            checkpoint_name="dit_model_ema.pt",
            output_dir=str(checkpoint_dir),
            token=args.hub_token,
        )
        if pulled:
            logger.info(f"Pulled from Hub: {pulled}")

        ema_path = checkpoint_dir / "dit_model_ema.pt"
        if ema_path.exists():
            push_checkpoint_to_hub(
                checkpoint_path=ema_path,
                hub_repo=args.hub_repo,
                checkpoint_name="dit_model_ema.pt",
                token=args.hub_token,
            )
            logger.info("Pushed to Hub")
        return 0

    # Download dataset
    if not download_data_if_needed(data_dir):
        logger.error("Cannot proceed without dataset")
        return 1

    # Determine resume checkpoint
    resume_from: str | None = None

    if args.resume:
        # Try Hub first, then Drive local
        from gpu_pool import pull_checkpoint_from_hub

        pulled = pull_checkpoint_from_hub(
            hub_repo=args.hub_repo,
            checkpoint_name="dit_model_ema.pt",
            output_dir=str(checkpoint_dir),
            token=args.hub_token,
        )
        if pulled:
            resume_from = str(pulled)
            logger.info(f"Resuming from Hub checkpoint: {resume_from}")

    # Fall back to local Drive checkpoint
    local_ema = checkpoint_dir / "dit_model_ema.pt"
    if resume_from is None and local_ema.exists():
        import zipfile

        if zipfile.is_zipfile(local_ema):
            resume_from = str(local_ema)
            logger.info(f"Resuming from Drive checkpoint: {resume_from}")
        else:
            logger.warning(f"Corrupt checkpoint at {local_ema} — starting fresh")

    if resume_from:
        logger.info("✅ Will resume from checkpoint")
    else:
        logger.info("Starting fresh training")

    # Run training
    start_time = time.time()

    try:
        from train_dit import train_dit_local

        output = str(checkpoint_dir / "dit_model.pt")
        ema_output = str(checkpoint_dir / "dit_model_ema.pt")

        final_loss = train_dit_local(
            data_dir=str(data_dir),
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
            logger.info("Checkpoints synced to Hub from Colab")

        # Show Drive location for reference
        logger.info(f"Checkpoints saved to Drive: {checkpoint_dir}")
        logger.info(
            "To continue later, run: "
            "!python scripts/train_colab.py --steps 20000 --resume"
        )

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted — checkpoints saved to Drive")
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
