"""src/eval.py

Evaluation script for the cats classifier.

Usage:
    python src/eval.py
    python src/eval.py --data-dir data/cats --checkpoint cats_model.pt
    python src/eval.py --data-dir data/cats --checkpoint cats_model.pt \
        --report evaluation_report.json --max-failures 15
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from dataset import cats_dataloader
from model import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cats classifier")
    parser.add_argument(
        "--data-dir", type=str, default="data/cats", help="Path to dataset root"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="cats_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        help="Model backbone used during training",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="evaluation_report.json",
        help="Path to write the JSON evaluation report",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=0,
        help="Max failure cases to include in report/print (0 = no cap)",
    )
    return parser.parse_args()


def _confusion_stats(
    y_true: list[int], y_pred: list[int], n_classes: int
) -> list[list[int]]:
    """Return an n_classes x n_classes confusion matrix (rows=true, cols=pred)."""
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for truth, pred in zip(y_true, y_pred):
        matrix[truth][pred] += 1
    return matrix


def _per_class_prf(
    matrix: list[list[int]],
) -> tuple[list[float], list[float], list[float]]:
    """Per-class precision/recall/F1 computed by hand from a confusion matrix."""
    n = len(matrix)
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for c in range(n):
        tp = matrix[c][c]
        fp = sum(matrix[r][c] for r in range(n)) - tp
        fn = sum(matrix[c][k] for k in range(n)) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return precisions, recalls, f1s


def evaluate(
    data_dir: str = "data/cats",
    checkpoint: str = "cats_model.pt",
    batch_size: int = 32,
    backbone: str = "resnet18",
    report: str = "evaluation_report.json",
    max_failures: int = 0,
) -> dict:
    """Evaluate a trained model and print results.

    Returns:
        Dictionary with keys: accuracy, correct, total, class_names,
        per_class; plus precision, recall, f1, macro_f1, weighted_f1,
        confusion_matrix, num_failures, failure_cases.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_loader = cats_dataloader(root=data_dir, batch_size=batch_size)
    class_names = val_loader.dataset.dataset.classes  # type: ignore[attr-defined]
    num_classes = len(class_names)

    model = load_checkpoint(checkpoint, num_classes=num_classes, backbone=backbone)
    model = model.to(device)
    model.eval()

    softmax = torch.nn.Softmax(dim=1)
    all_labels: list[int] = []
    all_preds: list[int] = []
    pred_confidences: list[float] = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probs = softmax(logits)
            preds = logits.argmax(dim=1)
            for pred, label, row in zip(preds, yb, probs):
                all_labels.append(label.item())
                all_preds.append(pred.item())
                pred_confidences.append(row[pred].item())

    correct = sum(int(p == t) for p, t in zip(all_preds, all_labels))
    total = len(all_labels)
    accuracy = correct / total if total > 0 else 0.0

    # Per-class accuracy (unchanged semantics, recomputed from collected results)
    per_class_correct = dict.fromkeys(class_names, 0)
    per_class_total = dict.fromkeys(class_names, 0)
    for pred, label in zip(all_preds, all_labels):
        class_name = class_names[label]
        per_class_total[class_name] += 1
        if pred == label:
            per_class_correct[class_name] += 1
    per_class = {
        c: per_class_correct[c] / max(per_class_total[c], 1) for c in class_names
    }

    # Map each val-sample position k back to its image path.
    # val_loader.dataset is a torch.utils.data.Subset; its .dataset is the
    # ImageFolder (class order alphabetical: 0=cat, 1=other) and .indices
    # holds original positions of the (identically ordered) full dataset.
    subset = val_loader.dataset  # type: ignore[attr-defined]
    full_samples = subset.dataset.samples  # type: ignore[attr-defined]
    paths = [str(full_samples[subset.indices[k]][0]) for k in range(total)]

    confusion_matrix = _confusion_stats(all_labels, all_preds, num_classes)
    precisions, recalls, f1s = _per_class_prf(confusion_matrix)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    weighted_f1 = sum(
        per_class_total[class_names[c]] * f1s[c] for c in range(num_classes)
    ) / max(total, 1)

    full_failures = [
        {
            "image": paths[k],
            "true": class_names[all_labels[k]],
            "predicted": class_names[all_preds[k]],
            "confidence": pred_confidences[k],
        }
        for k in range(total)
        if all_preds[k] != all_labels[k]
    ]
    num_failures = len(full_failures)
    failures = full_failures[:max_failures] if max_failures > 0 else full_failures

    print("\n=== Evaluation Results ===")
    print(f"Overall accuracy: {accuracy:.4f} ({correct}/{total})")
    print("\nPer-class accuracy:")
    for cls in class_names:
        cls_total = per_class_total[cls]
        cls_correct = per_class_correct[cls]
        cls_acc = cls_correct / cls_total if cls_total > 0 else 0.0
        print(f"  {cls:20s}: {cls_acc:.4f} ({cls_correct}/{cls_total})")

    print("\nPer-class precision / recall / F1:")
    for c, cls in enumerate(class_names):
        print(
            f"  {cls:20s}: precision={precisions[c]:.4f} "
            f"recall={recalls[c]:.4f} f1={f1s[c]:.4f} "
            f"(support={per_class_total[cls]})"
        )
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    header = "  " + "".join(f"{cls:>10s}" for cls in class_names)
    print(header)
    for r, cls in enumerate(class_names):
        row = "  " + "".join(
            f"{confusion_matrix[r][c]:>10d}" for c in range(num_classes)
        )
        print(f"{cls:>6s}" + f" {row}")

    if failures:
        print("\nTop failures:")
        for i, fc in enumerate(failures[:15], 1):
            print(
                f"  {i:>2}. {fc['image']}  true={fc['true']:5s} "
                f"pred={fc['predicted']:5s} conf={fc['confidence']:.4f}"
            )

    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "data_dir": data_dir,
        "device": str(device),
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "class_names": class_names,
        "precision": {c: precisions[i] for i, c in enumerate(class_names)},
        "recall": {c: recalls[i] for i, c in enumerate(class_names)},
        "f1": {c: f1s[i] for i, c in enumerate(class_names)},
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": confusion_matrix,
        "num_failures": num_failures,
        "failure_cases": failures,
    }
    with open(report, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nReport written to {report}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "class_names": class_names,
        "per_class": per_class,
        "precision": {c: precisions[i] for i, c in enumerate(class_names)},
        "recall": {c: recalls[i] for i, c in enumerate(class_names)},
        "f1": {c: f1s[i] for i, c in enumerate(class_names)},
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": confusion_matrix,
        "num_failures": num_failures,
        "failure_cases": failures,
    }


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        backbone=args.backbone,
        report=args.report,
        max_failures=args.max_failures,
    )
