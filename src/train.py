"""src/train.py

Training entrypoint for the cats classifier.

Features:
- Error handling with retry logic
- Structured logging (console + file)
- Mixed precision training (AMP)
- Gradient clipping
- Memory management (GC, CUDA cache)
- Learning rate warmup
- Checkpoint recovery
- Modal GPU training with optimizations (ADR-022 to ADR-025)

Usage:
    python src/train.py data/cats
    python src/train.py data/cats --epochs 20 --batch-size 64 --backbone resnet34

Modal GPU training:
    modal run src/train.py data/cats
    modal run src/train.py -- --epochs 20 --batch-size 64 --backbone resnet34
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import modal
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


# Configure logging
def setup_logging(log_file: str | None = None) -> logging.Logger:
    """Setup logging with console and optional file handlers.

    Args:
        log_file: Optional path to log file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("cats_classifier")
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
        description="Train cats classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_dir", type=str, help="Path to dataset root (ImageFolder format)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        help="Model backbone",
        choices=["resnet18", "resnet34", "resnet50", "mobilenet_v3_small"],
    )
    parser.add_argument(
        "--output", type=str, default="cats_model.pt", help="Output checkpoint path"
    )
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument(
        "--no-pretrained", action="store_true", help="Disable pretrained weights"
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable automatic mixed precision training",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (0 to disable)",
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=2, help="Number of LR warmup epochs"
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--log-file", type=str, default=None, help="Path to log file (optional)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Checkpoint save interval in epochs",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_gpu_memory(logger: logging.Logger, prefix: str = "") -> None:
    """Log GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        logger.info(
            f"{prefix}GPU Memory: {allocated:.1f}MB allocated, "
            f"{reserved:.1f}MB reserved"
        )


def cleanup_memory() -> None:
    """Clean up GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TrainingError(Exception):
    """Custom exception for training errors."""

    pass


class DataLoadError(Exception):
    """Custom exception for data loading errors."""

    pass


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    scaler: torch.cuda.amp.GradScaler | None = None,
    gradient_clip: float = 1.0,
    grad_accum_steps: int = 1,
) -> tuple[float, float]:
    """Run one training epoch with mixed precision and gradient clipping."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for batch_idx, (xb, yb) in enumerate(loader):
        try:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            context = torch.cuda.amp.autocast() if scaler else nullcontext()
            with context:
                pred = model(xb)
                loss = loss_fn(pred, yb) / grad_accum_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                if scaler:
                    if gradient_clip > 0:
                        scaler.unscale_(optimizer)
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
                optimizer.zero_grad()

            with torch.no_grad():
                total_loss += loss.item() * xb.size(0) * grad_accum_steps
                correct += (pred.argmax(dim=1) == yb).sum().item()
                total += xb.size(0)

            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(loader)} | "
                    f"Loss: {loss.item() * grad_accum_steps:.4f}"
                )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM at batch {batch_idx}, clearing cache...")
                cleanup_memory()
                continue
            raise

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[float, float]:
    """Run validation with memory cleanup."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    try:
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = (
                    xb.to(device, non_blocking=True),
                    yb.to(device, non_blocking=True),
                )
                pred = model(xb)
                loss = loss_fn(pred, yb)
                total_loss += loss.item() * xb.size(0)
                correct += (pred.argmax(dim=1) == yb).sum().item()
                total += xb.size(0)

        return total_loss / max(total, 1), correct / max(total, 1)

    finally:
        cleanup_memory()
        log_gpu_memory(logger, prefix="After validation | ")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    path: str | Path,
    logger: logging.Logger,
    is_best: bool = False,
) -> None:
    """Save training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "timestamp": datetime.now().isoformat(),
    }

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path} (val_acc={val_acc:.4f})")

    if is_best:
        best_path = path.parent / f"best_{path.name}"
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model to {best_path}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    logger: logging.Logger | None = None,
) -> tuple[nn.Module, torch.optim.Optimizer | None, int]:
    """Load checkpoint for resume training."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    if logger:
        logger.info(f"Loaded checkpoint from {path} (epoch {start_epoch - 1})")

    return model, optimizer, start_epoch


# =============================================================================
# Modal Configuration (ADR-022, ADR-023, ADR-024, ADR-025)
# =============================================================================

app = modal.App("tiny-cats-model")

# Volume definitions (ADR-024: organized storage with explicit commits)
volume_outputs = modal.Volume.from_name("cats-model-outputs", create_if_missing=True)
volume_data = modal.Volume.from_name("cats-dataset", create_if_missing=True)

# Optimized container image (ADR-022: fast builds with uv_pip_install)
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
    )
    .add_local_dir("data/", "/app/data/", copy=True)
    .add_local_file("src/train.py", "/app/train.py")
    .add_local_file("src/dataset.py", "/app/dataset.py")
    .add_local_file("src/model.py", "/app/model.py")
)


def _initialize_container():
    """Initialize container environment for faster training start (ADR-025)."""
    import torch

    # Setup paths
    sys.path.insert(0, "/app/src")
    os.chdir("/app")

    # Pre-import heavy modules
    import torchvision  # noqa: F401

    # Warm up CUDA
    if torch.cuda.is_available():
        _ = torch.zeros(1).cuda()
        dummy_input = torch.randn(1, 3, 32, 32).cuda()
        dummy_conv = torch.nn.Conv2d(3, 16, 3).cuda()
        _ = dummy_conv(dummy_input)
        del dummy_input, dummy_conv
        torch.cuda.empty_cache()


@app.function(
    image=image,
    volumes={
        "/outputs": volume_outputs,
        "/data": volume_data,
    },
    gpu="T4",
    timeout=3600,
    # Retry configuration (ADR-023: automatic recovery from transient failures)
    retries=modal.Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=10.0,
        max_delay=60.0,  # Max allowed: 60 seconds
    ),
)
def train_on_gpu(
    data_dir: str = "/data/cats",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    backbone: str = "resnet18",
    output: str | None = None,
    num_workers: int = 0,
    pretrained: bool = True,
    mixed_precision: bool = True,
    gradient_clip: float = 1.0,
    warmup_epochs: int = 2,
) -> dict[str, Any]:
    """Modal function for GPU training with error handling.

    Args:
        data_dir: Path to dataset directory.
        epochs: Number of epochs.
        batch_size: Batch size.
        lr: Learning rate.
        backbone: Model backbone.
        output: Output checkpoint path (auto-generated if None).
        num_workers: DataLoader workers.
        pretrained: Use pretrained weights.
        mixed_precision: Enable AMP training.
        gradient_clip: Gradient clipping value.
        warmup_epochs: LR warmup epochs.

    Returns:
        Dictionary with training status and output path.
    """
    # Initialize container (ADR-025: cold start optimization)
    _initialize_container()

    from datetime import datetime

    # Create dated checkpoint directory (ADR-024: organized storage)
    run_date = datetime.now().strftime("%Y-%m-%d")
    checkpoint_dir = f"/outputs/checkpoints/classifier/{run_date}"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Use dated directory for output and logs
    output = output or f"{checkpoint_dir}/cats_model.pt"
    log_file = f"{checkpoint_dir}/training.log"

    # Setup logging
    logger = setup_logging(log_file)
    logger.info("Starting Modal GPU training")
    logger.info(
        f"Configuration: data_dir={data_dir}, epochs={epochs}, batch_size={batch_size}"
    )

    try:
        # Check dataset cache (ADR-024: dataset caching in volume)
        if not Path(data_dir).exists() or not list(Path(data_dir).iterdir()):
            logger.info("Dataset not found, downloading...")
            Path("/data").mkdir(parents=True, exist_ok=True)

            result = subprocess.run(
                ["python", "data/download.py"],
                cwd="/app",
                env={**os.environ, "DATA_DIR": "/data", "CATS_DIR": "/data/cats"},
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                logger.warning(f"Python download failed: {result.stderr}")
                logger.info("Trying bash download script...")
                subprocess.run(
                    ["bash", "data/download.sh"],
                    cwd="/app",
                    env={**os.environ, "DATA_DIR": "/data", "CATS_DIR": "/data/cats"},
                    check=True,
                    capture_output=True,
                    text=True,
                )

            logger.info("Dataset downloaded successfully")

        # Train with error handling
        val_acc = train(
            data_dir=data_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            backbone=backbone,
            output=output,
            num_workers=num_workers,
            pretrained=pretrained,
            mixed_precision=mixed_precision,
            gradient_clip=gradient_clip,
            warmup_epochs=warmup_epochs,
            log_file=log_file,
            logger=logger,
        )

        # Commit volume after successful training (ADR-024: explicit commits)
        volume_outputs.commit()
        logger.info("Checkpoint committed to volume")

        # Cleanup old checkpoints (keep last 5)
        try:
            from volume_utils import cleanup_old_checkpoints

            cleanup_old_checkpoints(
                volume_outputs, "/outputs/checkpoints/classifier", keep_last_n=5
            )
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

        logger.info("Training completed successfully")
        return {"status": "completed", "output": output, "val_acc": val_acc}

    except subprocess.CalledProcessError as e:
        logger.error(f"Dataset download failed: {e}")
        logger.error(f"stdout: {e.stdout}, stderr: {e.stderr}")
        volume_outputs.commit()
        raise TrainingError(f"Dataset download failed: {e}") from e

    except RuntimeError as e:
        logger.error(f"Runtime error during training: {e}")
        log_gpu_memory(logger, "Before cleanup | ")
        cleanup_memory()
        volume_outputs.commit()
        raise TrainingError(f"Training failed: {e}") from e

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        volume_outputs.commit()
        raise TrainingError(f"Training failed with unexpected error: {e}") from e

    finally:
        cleanup_memory()
        logger.info("Final memory cleanup completed")


@app.local_entrypoint()
def main(
    data_dir: str = "/data/cats",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    backbone: str = "resnet18",
    output: str | None = None,
    num_workers: int = 0,
    pretrained: bool = True,
    mixed_precision: bool = True,
    gradient_clip: float = 1.0,
    warmup_epochs: int = 2,
):
    """Local entrypoint for Modal CLI.

    Usage:
        modal run src/train.py data/cats --epochs 20 --batch-size 64
        modal run src/train.py -- --epochs 20 --batch-size 64 --backbone resnet34
    """
    result = train_on_gpu.remote(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        backbone=backbone,
        output=output,
        num_workers=num_workers,
        pretrained=pretrained,
        mixed_precision=mixed_precision,
        gradient_clip=gradient_clip,
        warmup_epochs=warmup_epochs,
    )
    print(f"Training completed: {result}")


# =============================================================================
# Local Training Functions (unchanged from original)
# =============================================================================


def train(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    backbone: str = "resnet18",
    output: str = "cats_model.pt",
    num_workers: int = 2,
    pretrained: bool = True,
    mixed_precision: bool = False,
    gradient_clip: float = 1.0,
    warmup_epochs: int = 2,
    log_file: str | None = None,
    logger: logging.Logger | None = None,
    seed: int = 42,
) -> float:
    """Full training loop with validation, checkpointing, and memory management."""
    if logger is None:
        logger = setup_logging(log_file)

    logger.info("=" * 60)
    logger.info("Starting cats classifier training")
    logger.info(f"Configuration: {locals()}")

    set_seed(seed)
    logger.info(f"Random seed set to {seed}")

    data_path = Path(data_dir)
    if not data_path.exists():
        raise DataLoadError(f"Dataset directory not found: {data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log_gpu_memory(logger, "Initial | ")

    from dataset import cats_dataloader
    from model import cats_model, count_parameters

    try:
        train_loader, val_loader = cats_dataloader(
            root=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        logger.info(
            f"Data loaders created: {len(train_loader)} train batches, "
            f"{len(val_loader)} val batches"
        )
    except Exception as e:
        raise DataLoadError(f"Failed to create data loaders: {e}") from e

    num_classes = len(train_loader.dataset.dataset.classes)  # type: ignore
    model = cats_model(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
    ).to(device)

    logger.info(
        f"Model: {backbone} | Classes: {num_classes} | "
        f"Parameters: {count_parameters(model):,}"
    )

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    main_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )

    scaler = (
        torch.cuda.amp.GradScaler()
        if mixed_precision and torch.cuda.is_available()
        else None
    )
    if scaler:
        logger.info("Mixed precision training enabled (AMP)")

    if batch_size < 32:
        logger.warning(f"Small batch size ({batch_size}) may affect training stability")

    best_val_acc = 0.0
    start_epoch = 1
    shutdown_requested = False

    def signal_handler(signum: int, frame: Any) -> None:
        nonlocal shutdown_requested
        logger.warning(f"Received signal {signum}, will finish current epoch and stop")
        shutdown_requested = True

    old_handler = signal.signal(signal.SIGINT, signal_handler)
    old_handler_term = signal.signal(signal.SIGTERM, signal_handler)

    try:
        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.time()

            try:
                train_loss, train_acc = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=device,
                    epoch=epoch,
                    logger=logger,
                    scaler=scaler,
                    gradient_clip=gradient_clip,
                )

                val_loss, val_acc = validate(
                    model=model,
                    loader=val_loader,
                    loss_fn=loss_fn,
                    device=device,
                    logger=logger,
                )

                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]

                elapsed = time.time() - epoch_start

                logger.info(
                    f"Epoch {epoch:>3}/{epochs} | "
                    f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
                    f"Val loss: {val_loss:.4f} acc: {val_acc:.3f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )

                is_best = val_acc > best_val_acc
                if is_best:
                    best_val_acc = val_acc

                if epoch % 1 == 0 or is_best:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        val_acc=val_acc,
                        path=output,
                        logger=logger,
                        is_best=is_best,
                    )

                cleanup_memory()

                if shutdown_requested:
                    logger.info("Shutdown requested, saving checkpoint and stopping...")
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        val_acc=val_acc,
                        path=output,
                        logger=logger,
                        is_best=False,
                    )
                    break

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"OOM at epoch {epoch}, trying to recover...")
                    cleanup_memory()
                    continue
                raise

            except Exception as e:
                logger.error(f"Error in epoch {epoch}: {e}", exc_info=True)
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_acc=best_val_acc,
                    path=output.replace(".pt", "_error.pt"),
                    logger=logger,
                    is_best=False,
                )
                raise TrainingError(f"Training failed at epoch {epoch}: {e}") from e

        logger.info("=" * 60)
        logger.info(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
        logger.info(f"Model saved to: {output}")
        log_gpu_memory(logger, "Final | ")

        return best_val_acc

    finally:
        signal.signal(signal.SIGINT, old_handler)
        signal.signal(signal.SIGTERM, old_handler_term)


if __name__ == "__main__":
    args = parse_args()
    try:
        train(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            backbone=args.backbone,
            output=args.output,
            num_workers=args.num_workers,
            pretrained=not args.no_pretrained,
            mixed_precision=args.mixed_precision,
            gradient_clip=args.gradient_clip,
            warmup_epochs=args.warmup_epochs,
            log_file=args.log_file,
        )
    except (TrainingError, DataLoadError) as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        sys.exit(130)
