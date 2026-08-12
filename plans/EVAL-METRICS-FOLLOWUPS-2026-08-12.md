# Eval Metrics Unit Tests — Followups

**Date:** 2026-08-12
**Status:** Most implemented 2026-08-12; FU-4/FU-6 deliberately deferred
**Related:** `src/eval.py`, `tests/test_eval.py`, `evaluation_report.json`

## Context (what is done, so this doc stands alone)

`src/eval.py` metric math is unit-tested and the aggregate-F1 helper is
hardened:

- **`_aggregate_f1(f1s, support)`** (next to `_per_class_prf`); `evaluate()`
  delegates to it. Total is derived from `support`; unequal `f1s`/`support`
  lengths raise `ValueError` instead of silently dropping entries.
- **`tests/test_eval.py`** (10 tests): `_confusion_stats`, `_per_class_prf`,
  `_aggregate_f1` (incl. a length-mismatch guard), a report-driven regression
  against the committed `evaluation_report.json`, an end-to-end `evaluate()`
  wiring test on deterministic synthetic data, and a `@pytest.mark.slow`
  real-data eval test.
- Verification: 10/10 eval tests pass; full suite green; ruff + mypy clean;
  prior CPU rerun of `eval.py` bit-identical to the committed report.

---

## FU-1 — `_aggregate_f1` accepts contradictory input silently

**Status:** Implemented 2026-08-12.

**What/Why (was):** The helper range-indexed over `len(f1s)` and trusted a
separate `total` param — length mismatches silently dropped entries, and
`total != sum(support)` produced garbage (weighted > 1.0).

**Applied:** derivation + fail-loud:

```python
def _aggregate_f1(f1s: list[float], support: list[int]) -> tuple[float, float]:
    if len(f1s) != len(support):
        raise ValueError(f"f1s and support lengths differ ({len(f1s)} != {len(support)})")
    total = sum(support)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    weighted_f1 = sum(f * s for f, s in zip(f1s, support)) / total if total > 0 else 0.0
    return macro_f1, weighted_f1
```

`evaluate()` now calls `_aggregate_f1(f1s, support)`; `total` is derived
identically (it always equals `sum(support)` in the only caller).
Covered by `test_aggregate_f1_length_mismatch_raises` + the wiring test.

## FU-2 — Test `test_aggregate_f1_empty_and_zero_total` enshrined an impossible state

**Status:** Implemented 2026-08-12.

**Applied:** the contradictory 3-arg case is gone (signature no longer takes
`total`). Zero-total branch is now exercised with empty support / zero
per-class support: `_aggregate_f1([], []) == (0.0, 0.0)` and
`_aggregate_f1([0.8], [0]) == (0.8, 0.0)`.

## FU-3 — Regression pin broke on the next retrain

**Status:** Implemented 2026-08-12 — now auto-refreshing.

**Applied:** `test_published_report_numbers_regression` no longer hardcodes
floats. It loads the version-controlled `evaluation_report.json` as the golden
source and asserts `_per_class_prf(confusion_matrix)` reproduces the report's
per-class F1 and `_aggregate_f1(f1s, support=row-sums)` reproduces its
macro/weighted F1. A retrain regenerating the report updates the golden
automatically — no manual constant edits; a broken helper or broken report
still fails. AGENTS.md notes this.

## FU-4 — Metrics helpers testable only via a heavy import chain

**Status:** Deferred — needs plan-level approval; not in this batch.

**What/Why:** `from eval import ...` pulls `torch`, `dataset.py`, and `model.py`
at import time (~9s collection for the fast metric tests).

**Resolution (when pursued):** extract pure metric functions into a standalone
module (e.g. `src/metrics.py`), re-export from `eval.py`, migrate the tests to
import from `metrics`. Changes module layout, so it requires an approved change
to the eval code layout. Revisit only if `eval.py`'s imports deepen or the
collection cost matters.

## FU-5 — Cosmetic: magic float literals in `test_per_class_prf_known_matrix`

**Status:** Implemented 2026-08-12.

**Applied:** `pytest.approx([8 / 11, 2 / 3])` replaces the decimal literals.

## FU-6 — Mutate-and-catch is manual; consider automating for `src/eval.py`

**Status:** Deferred (optional tooling).

**What/Why:** The wiring test's value was proven by a manual mutation
(swapped `support` order → `weighted_f1` 0.47619 vs 0.59048 → test failed).
That guard is not enforced on future edits.

**Resolution (when pursued):** a lightweight mutation spot-check in the quality
gate (`scripts/quality-gate.sh`) over `src/eval.py` — guarded mutations:
support ordering, support key lookup, aggregate guards. Not added to avoid
churning the bash quality gate for marginal coverage. The FU-1 length-guard and
report-driven regression already reduce the drift surface.

## FU-7 — `evaluate()` real-data paths ununit-tested (integration)

**Status:** Implemented 2026-08-12.

**Applied:** `test_evaluate_on_real_data_matches_committed_report`
(`@pytest.mark.slow`) runs `evaluate()` on the real `data/cats` +
`checkpoints/best_cats_model_v2.pt` and asserts accuracy/macro/weighted F1 and
confusion matrix equal the committed report. Self-skips when data or the
checkpoint are absent — CI ships neither, so the default `Test` job stays fast
and green; run locally via `pytest tests/test_eval.py -m slow`.

## FU-8 — AGENTS.md test documentation stale w.r.t. `tests/test_eval.py`

**Status:** Implemented 2026-08-12.

**Applied:** Testing section now lists `tests/test_eval.py` (metric math +
report regression) and the `-m slow` real-data eval, with a note that
`evaluation_report.json` is the auto-refreshing golden source (no constant
edits on retrain).

---

## Non-followups (recorded for completeness)

- **Roast findings 1/2 (helper invariants)** → resolved as FU-1/FU-2.
- **Plan contingency (CPU rerun > 10 min)** — not triggered: the rerun took
  239s. No action.
- **The wiring gap identified in review** — closed by
  `test_evaluate_wiring_reports_correct_aggregates`; no open work.
