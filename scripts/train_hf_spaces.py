#!/usr/bin/env python3
"""Training script for HuggingFace Spaces.

Trains TinyDiT on HuggingFace Spaces' free T4 GPU (Docker-based space).
Designed to be triggered manually or via scheduled Space rebuilds.

HuggingFace Spaces provides free T4 GPU access (~16 hours/day) in a
Docker container. This script:
1. Detects the HF Spaces environment
2. Pulls latest checkpoint from Hub
3. Downloads dataset if needed
4. Runs training with periodic saves
5. Pushes checkpoint back to Hub

Usage (in Spaces Docker container or local):
    # Run from Spaces app.py as a background task
    python scripts/train_hf_spaces.py --steps 20000 --batch-size 256

    # Resume from Hub checkpoint
    python scripts/train_hf_spaces.py --steps 20000 --hub-resume

    # Check env and exit
    python scripts/train_hf_spaces.py --check

Space Configuration (README.md in Space repo):
    sdk: docker
    app_port: 7860
    gpu: t4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

SPACES_DATA_DIR = os.environ.get("SPACES_DATA_DIR", "/data/cats")
SPACES_CHECKPOINT_DIR = os.environ.get("SPACES_CHECKPOINT_DIR", "/data/checkpoints")
SPACES_HUB_REPO = os.environ.get("SPACES_HUB_REPO", "d4oit/tiny-cats-model")

logger = logging.getLogger("hf_spaces_train")


def setup_logging(log_file: str | None = None) -> logging.Logger:
    """Configure logging for Spaces training."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | [Spaces] %(message)s",
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


def detect_spaces_environment() -> dict:
    """Detect HF Spaces environment and GPU."""
    info = {
        "is_spaces": bool(os.environ.get("SPACE_ID")),
        "space_id": os.environ.get("SPACE_ID", "unknown"),
        "has_gpu": False,
        "gpu_name": "N/A",
        "gpu_memory_gb": 0.0,
    }
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


def download_data(data_dir: str) -> bool:
    """Download dataset if needed."""
    data_path = Path(data_dir)
    if data_path.exists() and list(data_path.iterdir()):
        logger.info(f"Dataset ready at {data_dir}")
        return True

    logger.info("Downloading dataset...")
    data_path.mkdir(parents=True, exist_ok=True)
    download_script = _project_root / "data" / "download.py"
    if not download_script.exists():
        logger.error("Download script not found")
        return False

    try:
        import subprocess

        result = subprocess.run(
            ["python", str(download_script)],
            env={**os.environ, "DATA_DIR": str(data_path), "CATS_DIR": data_dir},
            capture_output=True,
            text=True,
            timeout=900,
        )
        if result.returncode == 0:
            logger.info("Dataset downloaded successfully")
            return True
        logger.error(f"Download failed: {result.stderr[:500]}")
        return False
    except Exception as e:
        logger.error(f"Download error: {e}")
        return False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Train TinyDiT on HF Spaces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--steps", type=int, default=20_000, help="Training steps")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    p.add_argument("--image-size", type=int, default=128, help="Image size")
    p.add_argument(
        "--data-dir", type=str, default=SPACES_DATA_DIR, help="Dataset directory"
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=SPACES_CHECKPOINT_DIR,
        help="Checkpoint directory",
    )
    p.add_argument("--hub-resume", action="store_true", help="Pull checkpoint from Hub")
    p.add_argument("--hub-repo", type=str, default=SPACES_HUB_REPO, help="HF Hub repo")
    p.add_argument("--hub-token", type=str, default=None, help="HF token")
    p.add_argument("--no-hub-push", action="store_true", help="Skip pushing to Hub")
    p.add_argument("--check", action="store_true", help="Check environment and exit")
    # mixed_precision is always enabled for GPU training — no CLI flag needed
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--sample-interval", type=int, default=2_000)
    p.add_argument("--early-stopping-patience", type=int, default=10)
    p.add_argument(
        "--augmentation-level",
        type=str,
        default="full",
        choices=["basic", "medium", "full"],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging()

    env_info = detect_spaces_environment()
    logger.info("=" * 60)
    logger.info(f"HF SPACES — TINYDIT TRAINING | Space: {env_info['space_id']}")
    logger.info(f"GPU: {env_info['gpu_name']} ({env_info['gpu_memory_gb']:.1f} GB)")
    logger.info("=" * 60)

    if args.check:
        return 0

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if not download_data(args.data_dir):
        return 1

    # Pull checkpoint from Hub
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

    local_ema = Path(args.checkpoint_dir) / "dit_model_ema.pt"
    if resume_from is None and local_ema.exists():
        import zipfile

        if zipfile.is_zipfile(local_ema):
            resume_from = str(local_ema)

    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
    else:
        logger.info("Starting fresh training")

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
        logger.info(f"Training completed in {elapsed:.0f}s ({elapsed / 3600:.1f}h)")
        logger.info(f"Final loss: {final_loss:.6e}")

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
            logger.info("Checkpoints synced to Hub")
        return 0
    except KeyboardInterrupt:
        logger.info("Training interrupted — checkpoint saved")
        return 130
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1
    finally:
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
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
