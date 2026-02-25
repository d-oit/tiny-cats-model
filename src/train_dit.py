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
import os
import signal
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import modal
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dit import count_parameters, tinydit_128
from flow_matching import EMA, FlowMatchingLoss, flow_matching_step, sample, sample_t


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
    parser.add_argument("data_dir", type=str, help="Path to dataset root")
    parser.add_argument(
        "--steps", type=int, default=200_000, help="Total training steps"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
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
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
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
        "--warmup-steps", type=int, default=10_000, help="LR warmup steps"
    )
    parser.add_argument(
        "--log-interval", type=int, default=100, help="Logging interval in steps"
    )
    parser.add_argument(
        "--save-interval", type=int, default=10_000, help="Checkpoint save interval"
    )
    parser.add_argument(
        "--sample-interval", type=int, default=5_000, help="Sample generation interval"
    )
    parser.add_argument("--log-file", type=str, default=None, help="Path to log file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--ema-beta", type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument(
        "--num-sample-images",
        type=int,
        default=8,
        help="Number of images to generate during sampling",
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=1.5, help="Classifier-free guidance scale"
    )
    return parser.parse_args()


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

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint at step {step:,} (loss={loss:.4f}) to {path}")

    if is_best:
        best_path = path.parent / f"best_{path.name}"
        torch.save(checkpoint, best_path)
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
        Tuple of (model, optimizer, ema, start_step).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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
    .add_local_file("src/train_dit.py", "/app/train_dit.py")
    .add_local_file("src/dit.py", "/app/dit.py")
    .add_local_file("src/flow_matching.py", "/app/flow_matching.py")
    .add_local_file("src/dataset.py", "/app/dataset.py")
)


def _initialize_dit_container():
    """Initialize container environment for faster DiT training start (ADR-025)."""
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
    gpu="A10G",  # Better for transformer training (ADR-023)
    timeout=86400,  # 24 hours max for long training runs
    # Retry configuration (ADR-023: automatic recovery from transient failures)
    retries=modal.Retries(
        max_retries=2,  # Fewer retries for long jobs
        backoff_coefficient=2.0,
        initial_delay=30.0,  # Longer initial delay
        max_delay=60.0,  # Max allowed: 60 seconds
    ),
)
def train_dit_on_gpu(
    data_dir: str = "/data/cats",
    steps: int = 200_000,
    batch_size: int = 256,
    lr: float = 1e-4,
    image_size: int = 128,
    output: str | None = None,
    ema_output: str | None = None,
    num_workers: int = 0,
    mixed_precision: bool = True,
    gradient_clip: float = 1.0,
    warmup_steps: int = 10_000,
    log_interval: int = 100,
    save_interval: int = 10_000,
    sample_interval: int = 5_000,
    log_file: str | None = None,
    ema_beta: float = 0.9999,
    seed: int = 42,
) -> dict[str, Any]:
    """Modal function for DiT GPU training.

    Args:
        data_dir: Dataset directory.
        steps: Total training steps.
        batch_size: Batch size.
        lr: Learning rate.
        image_size: Image size.
        output: Output checkpoint path (auto-generated if None).
        ema_output: EMA checkpoint path (auto-generated if None).
        num_workers: DataLoader workers.
        mixed_precision: Enable AMP.
        gradient_clip: Gradient clipping.
        warmup_steps: LR warmup.
        log_interval: Logging frequency.
        save_interval: Checkpoint frequency.
        sample_interval: Sampling frequency.
        log_file: Log file path (auto-generated if None).
        ema_beta: EMA decay.
        seed: Random seed.

    Returns:
        Training status dict.
    """
    # Initialize container (ADR-025)
    _initialize_dit_container()

    from datetime import datetime

    # Create dated checkpoint directory (ADR-024: organized storage)
    run_date = datetime.now().strftime("%Y-%m-%d")
    checkpoint_dir = f"/outputs/checkpoints/dit/{run_date}"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    samples_dir = f"{checkpoint_dir}/samples"
    Path(samples_dir).mkdir(parents=True, exist_ok=True)

    # Use dated directory for outputs and logs
    output = output or f"{checkpoint_dir}/dit_model.pt"
    ema_output = ema_output or f"{checkpoint_dir}/dit_model_ema.pt"
    log_file = log_file or f"{checkpoint_dir}/dit_training.log"

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
            warmup_steps=warmup_steps,
            log_interval=log_interval,
            save_interval=save_interval,
            sample_interval=sample_interval,
            log_file=log_file,
            ema_beta=ema_beta,
            seed=seed,
            logger=logger,
        )

        # Commit volume after successful training (ADR-024: explicit commits)
        volume_outputs.commit()
        logger.info("Checkpoint committed to volume")

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
) -> torch.utils.data.DataLoader:
    """Create dataloader for training.

    Args:
        data_dir: Dataset directory.
        batch_size: Batch size.
        image_size: Target image size.
        num_workers: DataLoader workers.

    Returns:
        DataLoader yielding (images, breed_indices).
    """
    # Use ImageFolder directly
    from torchvision.datasets import ImageFolder

    from dataset import build_transforms

    transform = build_transforms(train=True, image_size=image_size)
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
    steps: int = 200_000,
    batch_size: int = 256,
    lr: float = 1e-4,
    image_size: int = 128,
    output: str = "checkpoints/dit_model.pt",
    ema_output: str = "checkpoints/dit_model_ema.pt",
    num_workers: int = 4,
    mixed_precision: bool = True,
    gradient_clip: float = 1.0,
    warmup_steps: int = 10_000,
    log_interval: int = 100,
    save_interval: int = 10_000,
    sample_interval: int = 5_000,
    log_file: str | None = None,
    ema_beta: float = 0.9999,
    seed: int = 42,
    logger: logging.Logger | None = None,
    resume: str | None = None,
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
        warmup_steps: LR warmup steps.
        log_interval: Logging frequency.
        save_interval: Checkpoint frequency.
        sample_interval: Sampling frequency.
        log_file: Log file path.
        ema_beta: EMA decay rate.
        seed: Random seed.
        logger: Logger instance.
        resume: Checkpoint to resume from.

    Returns:
        Final training loss.
    """
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
    model = tinydit_128(num_classes=num_classes).to(device)

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
    )
    logger.info(f"DataLoader created: {len(train_loader)} batches per epoch")

    # Optimizer and loss
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.95))
    loss_fn = FlowMatchingLoss(prediction_type="velocity")

    # LR scheduler with warmup
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    main_scheduler = CosineAnnealingLR(optimizer, T_max=steps - warmup_steps)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )

    # Mixed precision
    scaler = (
        torch.cuda.amp.GradScaler()
        if mixed_precision and torch.cuda.is_available()
        else None
    )
    if scaler:
        logger.info("Mixed precision training enabled (AMP)")

    # EMA
    ema = EMA(beta=ema_beta)
    ema.init(model)
    logger.info(f"EMA initialized with beta={ema_beta}")

    # Resume from checkpoint
    start_step = 0
    if resume:
        logger.info(f"Resuming from checkpoint: {resume}")
        model, optimizer, _ema, start_step = load_checkpoint(
            resume, model, optimizer, ema, logger
        )
        # Use loaded EMA if available
        ema = _ema if _ema is not None else ema
        # Adjust scheduler to current step
        for _ in range(start_step):
            scheduler.step()

    # Training state
    best_loss = float("inf")
    shutdown_requested = False

    def signal_handler(signum: int, frame: Any) -> None:
        nonlocal shutdown_requested
        logger.warning(f"Signal {signum} received, finishing current step...")
        shutdown_requested = True

    old_handler = signal.signal(signal.SIGINT, signal_handler)
    old_handler_term = signal.signal(signal.SIGTERM, signal_handler)

    try:
        model.train()
        step = start_step
        epoch = 0

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
                context = torch.cuda.amp.autocast() if scaler else nullcontext()

                with context:
                    # Flow matching step
                    pred, target = flow_matching_step(model, images, images, t, breeds)
                    loss = loss_fn(pred, target)

                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), gradient_clip
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), gradient_clip
                        )
                    optimizer.step()

                scheduler.step()
                ema.update(model)
                optimizer.zero_grad()

                # Track loss
                epoch_loss += loss.item()
                step += 1

                # Logging
                if step % log_interval == 0:
                    avg_loss = epoch_loss / log_interval
                    current_lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - epoch_start
                    steps_per_sec = log_interval / max(elapsed, 0.001)

                    logger.info(
                        f"Step {step:,}/{steps:,} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Speed: {steps_per_sec:.1f} steps/s"
                    )
                    log_gpu_memory(logger, "  ")

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
                    if avg_loss < best_loss:
                        best_loss = avg_loss

                # Generate samples
                if step % sample_interval == 0:
                    logger.info(f"Generating samples at step {step:,}...")
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
                    # Save samples (optional, requires PIL)
                    try:
                        from PIL import Image

                        samples_dir = Path(output).parent / "samples"
                        samples_dir.mkdir(parents=True, exist_ok=True)

                        for i in range(len(generated)):
                            img = (
                                (
                                    generated[i].permute(1, 2, 0).cpu().numpy() * 127.5
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
        logger.info(f"Training complete. Final loss: {best_loss:.4f}")

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
        return best_loss

    finally:
        signal.signal(signal.SIGINT, old_handler)
        signal.signal(signal.SIGTERM, old_handler_term)


@app.local_entrypoint()
def main(
    data_dir: str = "/data/cats",
    steps: int = 200_000,
    batch_size: int = 256,
    lr: float = 1e-4,
    image_size: int = 128,
    output: str = "/outputs/dit_model.pt",
    ema_output: str = "/outputs/dit_model_ema.pt",
    num_workers: int = 0,
    mixed_precision: bool = True,
    gradient_clip: float = 1.0,
    warmup_steps: int = 10_000,
):
    """Local entrypoint for Modal CLI.

    Usage:
        modal run src/train_dit.py data/cats --steps 200000
        modal run src/train_dit.py -- --steps 200000 --batch-size 256 --lr 0.0001
    """
    result = train_dit_on_gpu.remote(
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
        warmup_steps=warmup_steps,
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
            warmup_steps=args.warmup_steps,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            sample_interval=args.sample_interval,
            log_file=args.log_file,
            ema_beta=args.ema_beta,
            resume=args.resume,
        )
    except (TrainingError, Exception) as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        sys.exit(130)
