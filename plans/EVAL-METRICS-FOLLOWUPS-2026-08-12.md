# Eval Metrics Unit Tests — Followups

**Date:** 2026-08-12
**Status:** Open (accepted debt + actionable items)
**Related:** `src/eval.py`, `tests/test_eval.py`, `evaluation_report.json`

## Context (what is done, so this doc stands alone)

`src/eval.py` metric math is now unit-tested. Delivered and verified this session:

- **`_aggregate_f1(f1s, support, total)`** extracted (next to `_per_class_prf`); `evaluate()` delegates to it.
- **`tests/test_eval.py`** (8 tests): `_confusion_stats`, `_per_class_prf`, `_aggregate_f1`, a published-report regression pin, and `test_evaluate_wiring_reports_correct_aggregates` — an end-to-end `evaluate()` test with dataloader/checkpoint seams patched, on deterministic synthetic data.
- Verification: 8/8 eval tests pass; full suite 332 passed, 1 skipped, 1 xpassed; ruff check + format clean; CPU rerun of `eval.py` bit-identical to committed `evaluation_report.json` (temp file removed).

The followups below are the residual items surfaced by the review; none block current functionality.

---

## FU-1 — `_aggregate_f1` accepts contradictory input silently (actionable, needs sign-off)

**Status:** Open — latent only, unreachable in production today.

**What/Why:** The helper range-indexes over `len(f1s)` and trusts `total`:

- `len(f1s) != len(support)` → silently ignores trailing support (or IndexErrors) instead of failing loud.
- `total != sum(support)` → produces garbage, e.g. weighted F1 > 1.0, with no guard.
- `total` is redundant: `evaluate()` is the only caller and always passes `total == sum(support)`.

**Evidence (demonstrated against the real helper):**

```
_aggregate_f1([0.5], [100, 200], 300)  -> (0.5, 0.1666667)   # drops support[2]
_aggregate_f1([0.9, 0.5], [90, 10], 50) -> (0.7, 1.72)       # weighted > 1.0
```

**Resolution path (pick one):**
- Derive `total = sum(support)` internally and drop the param → `_aggregate_f1(f1s, support)`. Deletes the whole inconsistency class. Requires changing the approved signature.
- Or keep signature but assert `len(f1s) == len(support)` and `sum(support) == total` (defensive, fail-loud).

**Trigger:** next approved change touching `src/eval.py` metric code, or any new caller of the helper. Re-verify with the same battery (tests + behavior proof) afterwards.

## FU-2 — Test `test_aggregate_f1_empty_and_zero_total` enshrines an impossible state

**Status:** Open — blocked on FU-1.

**What/Why:** The case `([0.8], [5], 0)` has nonempty support with `total == 0`, which `evaluate()` can never produce. It only exercises the `total > 0` guard branch in isolation.

**Resolution path:** rework the test once FU-1 lands (e.g. assert empty-support behavior for the zero-total branch instead of a contradictory mixed state).

## FU-3 — Regression pin breaks on the next retrain

**Status:** Accepted debt (documented in test docstring).

**What/Why:** `test_published_report_numbers_regression` hardcodes floats from the committed `evaluation_report.json` (`f1s=[0.9947753396029259, 0.9974987493746873]`, `support=[479,999]`, `total=1478` → macro/weighted pins). A future retrain regenerating the report will fail this test until the constants are updated by hand.

**Resolution path (pick one):**
- Read goldens from `evaluation_report.json` at test time (no manual edits, but weaker — pins the report file instead of the numbers).
- Or keep hardcoded and add a one-line note to the quality-gate / AGENTS.md "regenerate constants" reminder.

**Trigger:** next `evaluation_report.json` regeneration.

## FU-4 — Metrics helpers are testable only via a heavy import chain

**Status:** Open — low priority, do not do casually.

**What/Why:** `from eval import ...` pulls `torch`, `dataset.py`, and `model.py` at module-import time; the 8 fast metric tests cost ~9s of collection runtime for that import. Acceptable now; grows if `eval.py`'s imports deepen.

**Resolution path:** extract pure metric functions (`_confusion_stats`, `_per_class_prf`, `_aggregate_f1`) into a standalone module (e.g. `src/metrics.py`) and have `eval.py` re-export/import them. Only if/when import cost matters — this changes module layout, so it needs plan-level approval and a caller migration.

## FU-5 — Cosmetic: magic float literals in `test_per_class_prf_known_matrix`

**Status:** Open — trivial.

**What/Why:** Asserts `0.7272727272727273` / `0.6666666666666666` with an explanatory comment; `pytest.approx([8 / 11, 2 / 3])` is self-documenting and equivalent.

**Resolution:** `8 / 11` and `2 / 3` literals; no behavior change. Safe to fold into any future `test_eval.py` edit.

## FU-6 — Mutate-and-catch is manual; consider automating for `src/eval.py`

**Status:** Optional.

**What/Why:** The wiring test's value was proven by a manual mutation (temporarily swapped `support` order in `evaluate()` → test failed with `weighted_f1` 0.47619 vs expected 0.59048; restored → green). That guard is not enforced on future edits.

**Resolution path (optional):** add `src/eval.py` to a lightweight mutation spot-check in the quality gate (`scripts/quality-gate.sh`) or a tool if one is already adopted. Guarded mutations: support ordering, support key lookup, aggregate guards.

## FU-7 — `evaluate()` real-data paths remain ununit-tested (integration candidate)

**Status:** Open — optional.

**What/Why:** `test_evaluate_wiring_reports_correct_aggregates` covers the metric wiring on a synthetic 2-class run with seams patched. Real-data behavior (actual `data/cats` + `checkpoints/best_cats_model_v2.pt` forward pass, `--max-failures` cap rendering, alternate backbones) is only proven by the one-time CPU verification run, not codified in CI.

**Resolution path:** add a `@pytest.mark.slow` integration test that runs `evaluate()` on the real checkpoint and asserts equality with the committed report (why the plan's Verification 2 exists). Excluded from the default suite by the `-m 'not slow'` filter; run in CI's slow job.

**Trigger:** when eval output/report format next changes, or when a slow test lane is wanted.

## FU-8 — AGENTS.md test documentation is stale w.r.t. `tests/test_eval.py`

**Status:** Open — docs hygiene.

**What/Why:** AGENTS.md "Testing" lists test modules (test_gpu_pool, test_train_chain, ...) but not `tests/test_eval.py` or the regression-pin refresh rule from FU-3.

**Resolution:** add a line to the Testing section noting `tests/test_eval.py` pins `evaluation_report.json` numbers and that constants require updating on retrain.

---

## Non-followups (recorded for completeness)

- **Roast findings 1/2 (helper invariants)** → FU-1/FU-2. Not fixed in the original delivery because the signature is plan-approved; behavior is provably identical and production-unreachable.
- **Plan contingency (CPU rerun > 10 min)** — not triggered: the rerun took 239s. No action.
- **The wiring gap identified in review** — closed by `test_evaluate_wiring_reports_correct_aggregates`; no open work.
