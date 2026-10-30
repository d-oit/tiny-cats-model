"""tests/test_eval.py

Unit tests for the metric math in src/eval.py: confusion matrix,
per-class precision/recall/F1, and macro/support-weighted F1 aggregation.

NOTE: Bare `from eval import ...` works because tests/conftest.py inserts
`src/` on sys.path (same pattern as `from model import cats_model`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from eval import _aggregate_f1, _confusion_stats, _per_class_prf, evaluate


def test_confusion_stats_counts_correctly() -> None:
    """Rows=true, cols=pred; each (truth, pred) pair increments one cell."""
    matrix = _confusion_stats(
        y_true=[0, 0, 1, 1, 1], y_pred=[0, 1, 1, 1, 0], n_classes=2
    )
    assert matrix == [[1, 1], [1, 2]]


def test_confusion_stats_absent_class() -> None:
    """Unused classes still produce an all-zero row and column."""
    matrix = _confusion_stats(y_true=[0, 1], y_pred=[0, 0], n_classes=3)
    assert matrix == [[1, 0, 0], [1, 0, 0], [0, 0, 0]]


def test_per_class_prf_known_matrix() -> None:
    """Hand-computed P/R/F1 for a small matrix."""
    matrix = [[4, 1], [2, 3]]
    precisions, recalls, f1s = _per_class_prf(matrix)
    assert precisions == pytest.approx([4 / 6, 3 / 4])
    assert recalls == pytest.approx([4 / 5, 3 / 5])
    # F1 = 2PR/(P+R) = 8/11 and 2/3
    assert f1s == pytest.approx([8 / 11, 2 / 3])


def test_per_class_prf_zero_support() -> None:
    """Classes with no TP/FP/FN must not divide by zero."""
    precisions, recalls, f1s = _per_class_prf([[3, 0], [0, 0]])
    assert precisions == [3 / 3, 0.0]
    assert recalls == [3 / 3, 0.0]
    assert f1s == [1.0, 0.0]


def test_aggregate_f1_macro_and_weighted() -> None:
    """Macro is a plain mean; weighted uses per-class support proportions."""
    macro_f1, weighted_f1 = _aggregate_f1(f1s=[0.9, 0.5], support=[90, 10])
    assert macro_f1 == pytest.approx(0.7)
    assert weighted_f1 == pytest.approx(0.86)


def test_aggregate_f1_empty_and_zero_total() -> None:
    """Empty f1 list and zero support both guard against div-by-zero."""
    assert _aggregate_f1([], []) == (0.0, 0.0)
    # Zero total arises from empty support; macro F1 is still returned.
    assert _aggregate_f1([0.8], [0]) == (0.8, 0.0)


def test_aggregate_f1_length_mismatch_raises() -> None:
    """Unequal f1s/support lengths fail loudly instead of silently dropping."""
    with pytest.raises(ValueError, match="lengths differ"):
        _aggregate_f1([0.9, 0.5], [90])


def test_published_report_numbers_regression() -> None:
    """Aggregate math reproduces the committed evaluation_report.json.

    The version-controlled report is the golden source, so the test never goes
    stale on retrain: _per_class_prf(confusion_matrix) plus _aggregate_f1 over
    per-class support (row sums) must reproduce the report's per-class F1 and
    macro/weighted F1. A bug in either helper or a broken report breaks this.
    """
    report = json.loads(
        (Path(__file__).parent.parent / "evaluation_report.json").read_text()
    )
    matrix = report["confusion_matrix"]
    _, _, f1s = _per_class_prf(matrix)
    assert dict(zip(report["class_names"], f1s)) == pytest.approx(report["f1"])

    support = [sum(row) for row in matrix]
    macro_f1, weighted_f1 = _aggregate_f1(f1s, support)
    assert macro_f1 == pytest.approx(report["macro_f1"])
    assert weighted_f1 == pytest.approx(report["weighted_f1"])


def test_evaluate_checkpoint_contract_and_explicit_test_directory(
    tmp_path: Path,
) -> None:
    from model import cats_model
    from model_contract import CLASSIFIER_LABELS, classifier_metadata

    data_root = tmp_path / "data"
    test_dir = tmp_path / "held-out"
    for root in (data_root, test_dir):
        for folder in ("cat", "other"):
            directory = root / folder
            directory.mkdir(parents=True)
            Image.new("RGB", (260, 260), color=(30, 70, 120)).save(
                directory / "sample.jpg"
            )
    model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
    metadata = classifier_metadata(backbone="resnet18", num_classes=2)
    checkpoint = tmp_path / "classifier.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "metadata": metadata}, checkpoint
    )

    report_path = tmp_path / "report.json"
    result = evaluate(
        data_dir=str(data_root),
        checkpoint=str(checkpoint),
        test_dir=str(test_dir),
        split="test",
        batch_size=2,
        report=str(report_path),
    )
    saved = json.loads(report_path.read_text())
    assert all(not Path(path).is_absolute() for path in saved["image_paths"])
    assert result["evaluated_samples"] == 2
    assert result["labels"] == list(CLASSIFIER_LABELS)
    assert result["split_source"] == "test_dir"
    assert result["per_class_support"] == {"cat": 1, "not_cat": 1}
    assert saved["dataset_hash"] is None
    assert saved["split_hash"] is None
    assert saved["model_sha256"]
    assert all(
        not Path(item["image"]).is_absolute() for item in result["failure_cases"]
    )
    with pytest.raises(ValueError, match="backbone"):
        evaluate(
            data_dir=str(data_root),
            checkpoint=str(checkpoint),
            test_dir=str(test_dir),
            backbone="resnet34",
            report=str(tmp_path / "bad.json"),
        )
    with pytest.raises(ValueError, match="num_classes"):
        evaluate(
            data_dir=str(data_root),
            checkpoint=str(checkpoint),
            test_dir=str(test_dir),
            num_classes=1,
            report=str(tmp_path / "bad-classes.json"),
        )
    with pytest.raises(FileNotFoundError, match="Test directory"):
        evaluate(
            data_dir=str(data_root),
            checkpoint=str(checkpoint),
            test_dir=str(tmp_path / "missing"),
            report=str(tmp_path / "missing.json"),
        )


def test_evaluate_test_split_never_falls_back(tmp_path: Path) -> None:
    from model import cats_model
    from model_contract import classifier_metadata

    data_root = tmp_path / "data"
    for folder in ("cat", "other"):
        directory = data_root / folder
        directory.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(directory / "image.jpg")
    model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": classifier_metadata(backbone="resnet18", num_classes=2),
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="requires"):
        evaluate(
            data_dir=str(data_root),
            checkpoint=str(checkpoint),
            split="test",
            report=str(tmp_path / "report.json"),
        )


@pytest.mark.slow
def test_evaluate_on_real_data_matches_committed_report(tmp_path: Path) -> None:
    """End-to-end evaluate() on real data/checkpoint matches the report.

    Codifies the one-off CPU verification as a repeatable (slow) test:
    evaluate() against the real validation split and checkpoint must reproduce
    the committed evaluation_report.json metrics. CI does not ship data/cats or
    checkpoints, so the test self-skips there; run it locally with
    `python -m pytest tests/test_eval.py -m slow`.
    """
    root = Path(__file__).parent.parent
    data_dir = root / "data/cats"
    checkpoint = root / "checkpoints/best_cats_model_v2.pt"
    if not data_dir.is_dir() or not checkpoint.is_file():
        pytest.skip("data/cats or checkpoint not present")

    report_path = tmp_path / "eval_real.json"
    result = evaluate(
        data_dir=str(data_dir),
        checkpoint=str(checkpoint),
        batch_size=32,
        report=str(report_path),
    )
    expected = json.loads(
        (Path(__file__).parent.parent / "evaluation_report.json").read_text()
    )
    assert result["accuracy"] == pytest.approx(expected["accuracy"])
    assert result["macro_f1"] == pytest.approx(expected["macro_f1"])
    assert result["weighted_f1"] == pytest.approx(expected["weighted_f1"])
    assert result["confusion_matrix"] == expected["confusion_matrix"]
