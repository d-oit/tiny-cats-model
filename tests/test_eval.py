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
