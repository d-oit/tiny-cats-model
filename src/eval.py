"""Evaluate a cats classifier against a validation or held-out split."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from dataset import cats_eval_loader
from model import load_checkpoint_with_metadata
from model_contract import (
    CLASSIFIER_IMAGE_SIZE,
    CLASSIFIER_LABELS,
    classifier_labels,
    validate_classifier_metadata,
)
from safe_checkpoint import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cats classifier")
    parser.add_argument("--data-dir", default="data/cats", help="Dataset root")
    parser.add_argument("--checkpoint", default="cats_model.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--absolute-paths", action="store_true")
    parser.add_argument(
        "--report", default="artifacts/evaluation/evaluation_report.json"
    )
    parser.add_argument("--max-failures", type=int, default=0)
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
    """Calculate per-class precision, recall and F1 from a confusion matrix."""
    n = len(matrix)
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for c in range(n):
        tp = matrix[c][c]
        fp = sum(matrix[r][c] for r in range(n)) - tp
        fn = sum(matrix[c][k] for k in range(n)) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return precisions, recalls, f1s


def _aggregate_f1(f1s: list[float], support: list[int]) -> tuple[float, float]:
    """Return macro and support-weighted F1, deriving total from support."""
    if len(f1s) != len(support):
        raise ValueError(
            f"f1s and support lengths differ ({len(f1s)} != {len(support)})"
        )
    total = sum(support)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    weighted_f1 = sum(f * s for f, s in zip(f1s, support)) / total if total else 0.0
    return macro_f1, weighted_f1


def evaluate(
    data_dir: str = "data/cats",
    checkpoint: str = "cats_model.pt",
    batch_size: int = 32,
    backbone: str | None = None,
    report: str = "evaluation_report.json",
    max_failures: int = 0,
    *,
    num_classes: int | None = None,
    split_manifest: str | None = None,
    test_dir: str | None = None,
    split: str = "val",
    absolute_paths: bool = False,
) -> dict:
    """Evaluate a checkpoint and write metrics plus split/checkpoint provenance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, metadata = load_checkpoint_with_metadata(checkpoint)
    validate_classifier_metadata(
        metadata,
        expected_backbone=backbone,
        expected_num_classes=num_classes,
    )
    labels = list(classifier_labels(metadata))
    architecture = metadata.get("architecture")
    input_metadata = metadata.get("input")
    if not isinstance(architecture, dict) or not isinstance(input_metadata, dict):
        raise ValueError("Checkpoint architecture and input metadata must be objects")
    num_classes_value = architecture.get("num_classes")
    backbone_value = architecture.get("name")
    input_shape = input_metadata.get("shape")
    if isinstance(num_classes_value, bool) or not isinstance(
        num_classes_value, (int, str)
    ):
        raise ValueError("Checkpoint num_classes must be an integer")
    if not isinstance(backbone_value, str):
        raise ValueError("Checkpoint backbone name must be a string")
    if (
        not isinstance(input_shape, list)
        or len(input_shape) < 3
        or not isinstance(input_shape[2], int)
    ):
        raise ValueError("Checkpoint input shape must contain integer image dimensions")
    metadata_classes = int(num_classes_value)
    model_backbone = backbone_value
    image_size = input_shape[2]
    if image_size != CLASSIFIER_IMAGE_SIZE:
        raise ValueError(f"Unsupported checkpoint image size: {image_size}")

    loader, samples, _dataset_classes, provenance = cats_eval_loader(
        data_dir,
        split=split,
        split_manifest=split_manifest,
        test_dir=test_dir,
        batch_size=batch_size,
        image_size=image_size,
    )
    if set(labels) != set(CLASSIFIER_LABELS):
        raise ValueError(
            f"Checkpoint labels must map to {CLASSIFIER_LABELS}, got {labels}"
        )
    label_remap = torch.tensor(
        [labels.index(label) for label in CLASSIFIER_LABELS],
        dtype=torch.long,
        device=device,
    )
    model = model.to(device)
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    pred_confidences: list[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            yb = label_remap[yb]
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            for pred, label, row in zip(preds, yb, probs):
                all_labels.append(label.item())
                all_preds.append(pred.item())
                pred_confidences.append(row[pred].item())

    if len(labels) != metadata_classes or any(
        label < 0 or label >= metadata_classes for label in all_labels + all_preds
    ):
        raise ValueError("Dataset labels do not match checkpoint output labels")
    total = len(all_labels)
    correct = sum(int(p == t) for p, t in zip(all_preds, all_labels))
    accuracy = correct / total if total else 0.0
    per_class_correct = dict.fromkeys(labels, 0)
    per_class_total = dict.fromkeys(labels, 0)
    for pred, truth in zip(all_preds, all_labels):
        per_class_total[labels[truth]] += 1
        if pred == truth:
            per_class_correct[labels[truth]] += 1
    per_class = {
        name: per_class_correct[name] / max(per_class_total[name], 1) for name in labels
    }
    confusion_matrix = _confusion_stats(all_labels, all_preds, metadata_classes)
    precisions, recalls, f1s = _per_class_prf(confusion_matrix)
    support = [per_class_total[name] for name in labels]
    macro_f1, weighted_f1 = _aggregate_f1(f1s, support)

    data_root = Path(test_dir) if test_dir is not None else Path(data_dir)
    paths: list[str] = []
    for sample_path, _ in samples:
        path = Path(sample_path)
        paths.append(
            str(path.resolve())
            if absolute_paths
            else path.relative_to(data_root).as_posix()
        )
    failure_cases = [
        {
            "image": paths[i],
            "true": labels[all_labels[i]],
            "predicted": labels[all_preds[i]],
            "confidence": pred_confidences[i],
        }
        for i in range(total)
        if all_preds[i] != all_labels[i]
    ]
    num_failures = len(failure_cases)
    failures = failure_cases[:max_failures] if max_failures > 0 else failure_cases
    model_hash = sha256_file(checkpoint)
    per_class_support = {name: per_class_total[name] for name in labels}

    print(f"Overall accuracy: {accuracy:.4f} ({correct}/{total})")
    for name in labels:
        print(f"  {name}: {per_class[name]:.4f} (support={per_class_total[name]})")
    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "data_dir": data_dir,
        "device": str(device),
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "evaluated_samples": total,
        "class_names": labels,
        "labels": labels,
        "per_class_support": per_class_support,
        "dataset_hash": provenance["dataset_hash"],
        "split_hash": provenance["split_hash"],
        "model_sha256": model_hash,
        "split_source": provenance["source"],
        "backbone": model_backbone,
        "precision": {name: precisions[i] for i, name in enumerate(labels)},
        "recall": {name: recalls[i] for i, name in enumerate(labels)},
        "f1": {name: f1s[i] for i, name in enumerate(labels)},
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": confusion_matrix,
        "num_failures": num_failures,
        "image_paths": paths,
        "failure_cases": failures,
    }
    report_path = Path(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_data, indent=2) + "\n")
    print(f"Report written to {report_path}")
    return {
        **report_data,
        "per_class": per_class,
        "failure_cases": failures,
    }


def main() -> None:
    args = parse_args()
    evaluate(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        backbone=args.backbone,
        num_classes=args.num_classes,
        report=args.report,
        max_failures=args.max_failures,
        split_manifest=args.split_manifest,
        test_dir=args.test_dir,
        split=args.split,
        absolute_paths=args.absolute_paths,
    )


if __name__ == "__main__":
    main()
