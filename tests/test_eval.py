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
    # F1 via 2PR/(P+R): 8/11 and 2/3
    assert f1s == pytest.approx([0.7272727272727273, 0.6666666666666666])


def test_per_class_prf_zero_support() -> None:
    """Classes with no TP/FP/FN must not divide by zero."""
    precisions, recalls, f1s = _per_class_prf([[3, 0], [0, 0]])
    assert precisions == [3 / 3, 0.0]
    assert recalls == [3 / 3, 0.0]
    assert f1s == [1.0, 0.0]


def test_aggregate_f1_macro_and_weighted() -> None:
    """Macro is a plain mean; weighted uses per-class support proportions."""
    macro_f1, weighted_f1 = _aggregate_f1(f1s=[0.9, 0.5], support=[90, 10], total=100)
    assert macro_f1 == pytest.approx(0.7)
    assert weighted_f1 == pytest.approx(0.86)


def test_aggregate_f1_empty_and_zero_total() -> None:
    """Empty f1 list and zero total both guard against div-by-zero."""
    assert _aggregate_f1([], [], 0) == (0.0, 0.0)
    assert _aggregate_f1([0.8], [5], 0) == (0.8, 0.0)


def test_published_report_numbers_regression() -> None:
    """Pin the macro/weighted F1 from the committed evaluation_report.json.

    If a future retrain changes the report, update these constants.
    """
    f1s = [0.9947753396029259, 0.9974987493746873]
    support = [479, 999]
    total = 1478
    macro_f1, weighted_f1 = _aggregate_f1(f1s, support, total)
    assert macro_f1 == pytest.approx(0.9961370444888066)
    assert weighted_f1 == pytest.approx(0.9966161287517687)

    # Same F1s are produced by _per_class_prf on the same confusion matrix.
    _, _, matrix_f1s = _per_class_prf([[476, 3], [2, 997]])
    assert matrix_f1s == pytest.approx(f1s)


def test_evaluate_wiring_reports_correct_aggregates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """evaluate() end-to-end metric wiring on synthetic deterministic data.

    Deterministic 7-sample 2-class run: 5 cats (3 correct, 2 mispredicted as
    other) and 2 other (1 correct, 1 mispredicted as cat) -> confusion matrix
    [[3, 2], [1, 1]]. The dataloader and checkpoint seams are patched so the
    real metric plumbing (support built from class names, aggregate F1s,
    failure paths, report write) runs against known inputs. Imbalanced support
    (5:2) and distinct class F1s (2/3 vs 2/5) make weighted != macro, so a
    swapped or miskeyed support list would break weighted_f1.
    """
    samples = [(f"/img/{i}.jpg", 0 if i < 5 else 1) for i in range(7)]

    class FakeImageFolder:
        def __init__(self, samples):
            self.classes = ["cat", "other"]
            self.samples = samples

    class FakeSubset:
        def __init__(self, samples):
            self.dataset = FakeImageFolder(samples)
            self.indices = list(range(7))

    class FakeValLoader:
        def __init__(self):
            self.dataset = FakeSubset(samples)
            self.xb = torch.zeros(7, 1, 1, 1)
            self.yb = torch.tensor([0, 0, 0, 0, 0, 1, 1])

        def __iter__(self):
            yield self.xb, self.yb

    # logits.argmax(dim=1) -> [0, 0, 0, 1, 1, 1, 0] (3 correct cats, 2
    # cat->other, 1 correct other, 1 other->cat).
    logits = torch.tensor(
        [
            [2.0, 0.0],
            [2.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [0.0, 2.0],
            [0.0, 2.0],
            [2.0, 0.0],
        ]
    )

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, xb):
            return logits

    fake_loader = FakeValLoader()
    monkeypatch.setattr("eval.cats_dataloader", lambda **kw: (None, fake_loader))
    monkeypatch.setattr("eval.load_checkpoint", lambda *a, **k: FakeModel())

    report = tmp_path / "report.json"
    result = evaluate(
        data_dir="unused",
        checkpoint="unused.pt",
        batch_size=8,
        report=str(report),
    )

    assert result["accuracy"] == pytest.approx(4 / 7)
    assert result["confusion_matrix"] == [[3, 2], [1, 1]]
    assert result["f1"] == {"cat": pytest.approx(2 / 3), "other": pytest.approx(2 / 5)}
    assert result["precision"] == {
        "cat": pytest.approx(3 / 4),
        "other": pytest.approx(1 / 3),
    }
    assert result["recall"] == {
        "cat": pytest.approx(3 / 5),
        "other": pytest.approx(1 / 2),
    }
    # Imbalanced support and distinct class F1s keep weighted != macro.
    assert result["macro_f1"] == pytest.approx(8 / 15)
    assert result["weighted_f1"] == pytest.approx(62 / 105)
    assert result["macro_f1"] != result["weighted_f1"]
    # Failure wiring: mispredictions at val positions 3, 4, 6 map to images.
    assert result["num_failures"] == 3
    assert result["failure_cases"][0] == {
        "image": "/img/3.jpg",
        "true": "cat",
        "predicted": "other",
        "confidence": pytest.approx(0.8808, abs=1e-3),
    }

    # Report file persists the same aggregates.
    saved = json.loads(report.read_text())
    assert saved["accuracy"] == result["accuracy"]
    assert saved["macro_f1"] == result["macro_f1"]
    assert saved["weighted_f1"] == result["weighted_f1"]
