"""src/eval.py

Evaluation script for the cats classifier.

Usage:
    python src/eval.py
    python src/eval.py --data-dir data/cats --checkpoint cats_model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from dataset import cats_dataloader
from model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cats classifier")
    parser.add_argument(
        "--data-dir", type=str, default="data/cats", help="Path to dataset root"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="cats_model.pt", help="Path to model checkpoint"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--backbone", type=str, default="resnet18", help="Model backbone used during training"
    )
    return parser.parse_args()


def evaluate(
    data_dir: str = "data/cats",
    checkpoint: str = "cats_model.pt",
    batch_size: int = 32,
    backbone: str = "resnet18",
) -> dict:
    """Evaluate a trained model and print results.

    Returns:
        Dictionary with keys: accuracy, correct, total, class_names.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_loader = cats_dataloader(root=data_dir, batch_size=batch_size)
    class_names = val_loader.dataset.dataset.classes
    num_classes = len(class_names)

    model = load_checkpoint(checkpoint, num_classes=num_classes, backbone=backbone)
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    per_class_correct = {c: 0 for c in class_names}
    per_class_total = {c: 0 for c in class_names}

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            for pred, label in zip(preds, yb):
                class_name = class_names[label.item()]
                per_class_total[class_name] += 1
                if pred == label:
                    correct += 1
                    per_class_correct[class_name] += 1
            total += yb.size(0)

    accuracy = correct / total if total > 0 else 0.0

    print(f"\n=== Evaluation Results ===")
    print(f"Overall accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"\nPer-class accuracy:")
    for cls in class_names:
        cls_total = per_class_total[cls]
        cls_correct = per_class_correct[cls]
        cls_acc = cls_correct / cls_total if cls_total > 0 else 0.0
        print(f"  {cls:20s}: {cls_acc:.4f} ({cls_correct}/{cls_total})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "class_names": class_names,
        "per_class": {c: per_class_correct[c] / max(per_class_total[c], 1) for c in class_names},
    }


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        backbone=args.backbone,
    )
