#!/usr/bin/env python3
"""Lightning pool and Studio training entrypoint.

Pool slices use canonical `/tmp` data and fail-closed Hub resume. Campaign flags
match Kaggle. Training needs CUDA; sync-only remains CPU-capable.

Usage:
    python scripts/train_lightning.py --steps 50000 --batch-size 256
    python scripts/train_lightning.py --steps 15000 --hub-resume --force-download
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

# Path setup.
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from slice_runtime import (  # noqa: E402
    pull_resume_checkpoint,
    push_progress_backstop,
)

# Lightning-specific defaults.
# Pool slices run on ephemeral containers with no teamspace mounts, so the
# fresh materialisation targets live under /tmp (never the Studio-mounted
# /teamspace paths): data arriving from a mount would carry a different
# dataset identity than the canonical download.
LIGHTNING_DATA_DIR = os.environ.get("LIGHTNING_DATA_DIR", "/tmp/tiny-cats-data")
LIGHTNING_CHECKPOINT_DIR = os.environ.get(
    "LIGHTNING_CHECKPOINT_DIR", "/tmp/tiny-cats-checkpoints"
)
LIGHTNING_HUB_REPO = os.environ.get("LIGHTNING_HUB_REPO", "d4oit/tiny-cats-model")
# Forced canonical downloads get Kaggle's full grace period (the Oxford host
# is unreliable; the Zenodo mirror is ~789 MB).
LIGHTNING_DATA_PREP_TIMEOUT_SECONDS = 2_700

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
    """Detect Studio markers and CUDA availability."""
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


def setup_lightning_dirs(data_dir: str, checkpoint_dir: str) -> None:
    """Create required directories in Lightning Studio."""
    for d in [data_dir, checkpoint_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.info(f"  Directory OK: {d}")


def download_data(data_dir: str, *, force_download: bool = False) -> bool:
    """Download cat dataset for the Lightning pool slice.

    Args:
        data_dir: Target data directory.
        force_download: Always materialise the dataset through the canonical
            ``data/download.py`` even when the directory already has content.
            Pool slices must never silently reuse mounted or previously
            repacked data: a differing ``relpath:size`` fingerprint fails the
            resume manifest gate (ADR-063/ADR-067), and the slice would train
            a non-canonical dataset identity. A failed download propagates as
            ``False`` so the caller cannot proceed on stale data.

    Returns:
        True if data is ready.
    """
    data_path = Path(data_dir)
    if not force_download and data_path.exists() and list(data_path.iterdir()):
        logger.info(f"Dataset already at {data_dir}")
        return True

    if force_download and data_path.exists() and any(data_path.iterdir()):
        logger.info(
            f"Discarding pre-existing {data_dir} so the dataset fingerprint "
            "matches the canonical download"
        )
        import shutil

        try:
            shutil.rmtree(data_path)
        except OSError as exc:
            logger.error(f"Cannot clear {data_dir} for fingerprint parity: {exc}")
            return False

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
            # The forced canonical download (Oxford can be slow; Zenodo mirror
            # ~789 MB) gets the full grace period; a normal fresh download
            # keeps the historical cap.
            timeout=(LIGHTNING_DATA_PREP_TIMEOUT_SECONDS if force_download else 600),
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train TinyDiT on Lightning AI Studio",
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
        default=500,
        help="Checkpoint save interval in steps",
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

    # Campaign contract (ADR-063/ADR-067): control-plane slices must stay
    # manifest-identical to the Modal/Kaggle runs — same experiment identity,
    # LR floor and horizon, seed — or the resume manifest gate rejects the
    # checkpoint. Same flags and meanings as scripts/train_kaggle.py.
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="dit-breed-conditioned-v4",
        help="Manifest identity; resumes must match",
    )
    parser.add_argument(
        "--min-lr", type=float, default=1e-6, help="LR floor for the cosine decay"
    )
    parser.add_argument(
        "--schedule-steps",
        type=int,
        default=None,
        help="Campaign-wide LR horizon (global target) so slices cannot shorten the cosine",
    )
    parser.add_argument(
        "--hub-push-interval",
        type=int,
        default=5_000,
        help="Canonical mid-run Hub push cadence in steps (0 disables)",
    )
    parser.add_argument(
        "--allow-experiment-mismatch",
        action="store_true",
        help="Resume despite a manifest mismatch (explicit migration only)",
    )
    parser.add_argument(
        "--allow-legacy-dataset-hash",
        action="store_true",
        help=(
            "Accept a pre-v2 dataset identity when the saved hash equals the "
            "legacy fingerprint of the current data"
        ),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help=(
            "Materialise the dataset with data/download.py even if data is "
            "already present (fingerprint parity)"
        ),
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for Lightning AI Studio training."""
    args = parse_args(argv)
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

    setup_lightning_dirs(args.data_dir, args.checkpoint_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    # Sync-only mode: canonical experiment-aware pull only (no legacy flat
    # publication) — CPU is fine here because no training runs.
    if args.sync_only:
        logger.info("Sync-only mode — pulling the canonical experiment snapshot")
        try:
            pulled = pull_resume_checkpoint(
                hub_repo=args.hub_repo,
                checkpoint_dir=checkpoint_dir,
                experiment_id=args.experiment_id,
                token=args.hub_token,
            )
            if pulled:
                logger.info(f"Pulled checkpoint: {pulled}")
            else:
                logger.info("No checkpoint on Hub")
            return 0
        finally:
            if not args.no_hub_push:
                push_progress_backstop(
                    checkpoint_dir=checkpoint_dir,
                    hub_repo=args.hub_repo,
                    experiment_id=args.experiment_id,
                    hub_push_interval=args.hub_push_interval,
                    token=args.hub_token,
                )

    # A pool slice must train on a GPU; there is no CPU fallback. Refuse before
    # downloading ~789 MB of data on a CPU-only runner.
    if not env_info["has_gpu"]:
        logger.error(
            "No CUDA GPU detected — Lightning pool slices require CUDA; "
            "refusing to train on CPU"
        )
        return 1

    # Download dataset
    if not download_data(args.data_dir, force_download=args.force_download):
        logger.error("Cannot proceed without dataset")
        return 1

    # Pull checkpoint from Hub for resume
    resume_from: str | None = None
    if args.hub_resume:
        pulled = pull_resume_checkpoint(
            hub_repo=args.hub_repo,
            checkpoint_dir=checkpoint_dir,
            experiment_id=args.experiment_id,
            token=args.hub_token,
        )
        if pulled is None:
            logger.error(
                f"Canonical resume failed: experiment '{args.experiment_id}' has "
                "no committed snapshot on the Hub. Refusing to substitute a "
                "stale local checkpoint or start a campaign slice from scratch."
            )
            return 1
        resume_from = str(pulled)
        logger.info(f"Resuming from Hub checkpoint: {resume_from}")
    else:
        # Without --hub-resume an explicit local checkpoint keeps working the
        # way it always has for ad-hoc (non-pool) studio runs.
        local_ema = checkpoint_dir / "dit_model_ema.pt"
        if local_ema.exists():
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

        output = str(checkpoint_dir / "dit_model.pt")
        ema_output = str(checkpoint_dir / "dit_model_ema.pt")

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
            min_lr=args.min_lr,
            schedule_steps=args.schedule_steps,
            save_interval=args.save_interval,
            sample_interval=args.sample_interval,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=0.001,
            augmentation_level=args.augmentation_level,
            hub_push_interval=args.hub_push_interval,
            no_hub_push=args.no_hub_push,
            hub_repo=args.hub_repo,
            experiment_id=args.experiment_id,
            allow_experiment_mismatch=args.allow_experiment_mismatch,
            allow_legacy_dataset_hash=args.allow_legacy_dataset_hash,
            seed=args.seed,
            resume=resume_from,
        )

        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.0f}s")
        logger.info(f"Final loss: {final_loss:.6e}")

        # Hub sync is canonical (ADR-064): train_dit_local publishes immutable
        # step-XXXXXX snapshots + the latest/manifest.json pointer every
        # --hub-push-interval steps. The legacy flat final pushes that used to
        # live here wrote a parallel layout nothing consumes.
        if args.no_hub_push or args.hub_push_interval <= 0:
            logger.info("Hub sync disabled (--no-hub-push / --hub-push-interval 0)")
        else:
            logger.info(
                f"Canonical Hub sync performed by the trainer every "
                f"{args.hub_push_interval} steps"
            )

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1
    finally:
        # Canonical safety net for a run that ended between the trainer's
        # interval-boundary pushes (covers training exits and early stops).
        if not args.no_hub_push:
            push_progress_backstop(
                checkpoint_dir=checkpoint_dir,
                hub_repo=args.hub_repo,
                experiment_id=args.experiment_id,
                hub_push_interval=args.hub_push_interval,
                token=args.hub_token,
            )


if __name__ == "__main__":
    sys.exit(main())
