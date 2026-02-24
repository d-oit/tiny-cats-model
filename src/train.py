"""src/train.py

Training entrypoint for the cats classifier.

Usage:
    python src/train.py data/cats
    python src/train.py data/cats --epochs 20 --batch-size 64 --backbone resnet34

Modal GPU training:
    modal run src/train.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import modal
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path for relative imports when run directly
sys.path.insert(0, str(Path(__file__).parent))

from dataset import cats_dataloader
from model import cats_model, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cats classifier")
    parser.add_argument("data_dir", type=str, help="Path to dataset root (ImageFolder format)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Model backbone")
    parser.add_argument("--output", type=str, default="cats_model.pt", help="Output checkpoint path")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weights")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        correct += (pred.argmax(dim=1) == yb).sum().item()
        total += xb.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | " f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run validation. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total_loss += loss.item() * xb.size(0)
            correct += (pred.argmax(dim=1) == yb).sum().item()
            total += xb.size(0)

    return total_loss / total, correct / total


stub = modal.App("tiny-cats-model")

volume_outputs = modal.Volume.from_name("cats-model-outputs", create_if_missing=True)
volume_data = modal.Volume.from_name("cats-model-data", create_if_missing=True)

LOCAL_SRC = Path(__file__).parent.absolute()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.0.0", "torchvision>=0.15.0", "Pillow>=9.0.0", "tqdm>=4.65.0")
    .add_local_dir(str(LOCAL_SRC), "/app/src", copy=True)
)


@stub.function(
    image=image,
    volumes={
        "/outputs": volume_outputs,
        "/data": volume_data,
    },
    gpu="T4",
    timeout=3600,
)
def train_modal(
    data_dir: str = "/data/cats",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    backbone: str = "resnet18",
    output: str = "/outputs/cats_model.pt",
    num_workers: int = 0,
    pretrained: bool = True,
) -> dict:
    """Modal function for GPU training."""
    import os
    import sys

    sys.path.insert(0, "/app/src")
    os.chdir("/app")

    if not Path(data_dir).exists():
        print("Downloading dataset...")
        Path("/data").mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["bash", "data/download.sh"],
            cwd="/app",
            env={**os.environ, "DATA_DIR": "/data"},
            check=True,
        )

    train(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        backbone=backbone,
        output=output,
        num_workers=num_workers,
        pretrained=pretrained,
    )
    return {"status": "completed", "output": output}


def train(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    backbone: str = "resnet18",
    output: str = "cats_model.pt",
    num_workers: int = 2,
    pretrained: bool = True,
) -> None:
    """Full training loop with validation and checkpoint saving."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = cats_dataloader(
        root=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = cats_model(
        num_classes=len(train_loader.dataset.dataset.classes),  # type: ignore
        backbone=backbone,
        pretrained=pretrained,
    ).to(device)

    print(f"Model: {backbone} | Parameters: {count_parameters(model):,}")

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.3f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output)
            print(f"  -> Saved best model to {output} (val_acc={val_acc:.4f})")

    print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output}")


@stub.local_entrypoint()
def main(
    data_dir: str = "/data/cats",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    backbone: str = "resnet18",
    output: str = "/outputs/cats_model.pt",
    num_workers: int = 0,
    pretrained: bool = True,
):
    """Local entrypoint for Modal CLI."""
    train_modal.remote(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        backbone=backbone,
        output=output,
        num_workers=num_workers,
        pretrained=pretrained,
    )


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone=args.backbone,
        output=args.output,
        num_workers=args.num_workers,
        pretrained=not args.no_pretrained,
    )
