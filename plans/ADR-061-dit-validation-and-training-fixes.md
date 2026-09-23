# ADR-061: Trustworthy DiT validation — held-out loss, EMA selection, working gates

- **Status:** Accepted (2026-09-20)
- **Date:** 2026-09-20
- **Deciders:** tiny-cats-model maintainers
- **Related:** `src/train_dit.py`, `src/dataset.py`, `src/dit_validation.py`,
  `src/validate_model.py`, `src/flow_matching.py`, `AGENTS.md`,
  `agents-docs/training.md`, `scripts/train_dit_high_accuracy.sh`,
  ADR-028 (validation gates), ADR-059 (loss accounting), ADR-060 (architecture fixes)

## Context

A validation pass on 2026-09-20 (test suite, a 30-step local CPU training run on the
real 7,390-image `data/cats`, and `src/validate_model.py` against both the production
classifier and a generator checkpoint) found that the training loop works but that
almost every signal used to *judge* it was unsound.

**What the run confirmed**

- The loop is healthy: flow-matching loss started at **1.390** and fell to **1.212**
  in 30 steps at batch 8 — i.e. from the theoretical noise floor
  `E[|x1 − x0|²] ≈ 1.35`, matching ADR-060's measured 1.38 zero-predictor baseline.
- Consequently the ~**6e-2** losses recorded in the July `dit_training.log` are not a
  velocity-MSE baseline at all and must not be used as one (consistent with ADR-060
  declaring every pre-fix checkpoint worthless).

**Defects found**

1. **No unbiased selection signal.** `create_dataloader()` built one loader; the
   `evaluation_loss` that drove `is_best` and early stopping was averaged over the
   *same augmented training batches* as the logged loss, so overfitting (7.4k images,
   13 classes, 33M params) was unobservable and patience was noise-driven.
2. **EMA was never measured.** Samples and checkpoint selection used the raw model;
   `ema_output` was written and nothing read it.
3. **Resume discarded early-stopping state.** `best_loss`/`patience_counter` reset to
   `inf`/`0` on every resume, so early stopping could never fire across hub-resumed
   pool slices, and `is_best` could fire on a worse model. `lr_lambda` was also
   re-derived from the current `--steps`, and `progress > 1` made the cosine term turn
   back **up** — resuming a slice with a smaller `--steps` *raised* the LR.
4. **`--config` was a no-op.** The YAML was applied with
   `if getattr(args, key) is None`, but argparse always populates defaults, so no key
   ever qualified. `configs/dit_train_config.yaml` had zero effect.
5. **`--min-lr` was parsed and never read**, so the cosine decayed to a literal 0.0.
6. **`max_final_loss` was never enforced.** `check_training_metrics` formatted the
   comparison into the message string and then returned `passed=True`
   unconditionally: a generator at 1.2776 against a 0.5 threshold reported
   "Training Metrics: PASS".
7. **The generative gate never ran.** `generate_sample_and_check_quality` required
   `num_classes` in the checkpoint `config`, which `save_checkpoint()` never wrote, so
   every generator reported "Not a generative model". It also only understood the
   `ema_shadow_params` layout and used plain `load_state_dict` (partial load before
   raising).
8. **Optional dependencies failed the report.** With onnxruntime absent (or no `.onnx`
   artifact), the gate returned `passed=False`, so it reported **FAILED 5/7** for the
   99.66%-accuracy classifier and a green local gate was unreachable.
9. **Docs/CLI drift.** `AGENTS.md` and `agents-docs/training.md` documented positional
   Modal commands that ADR-048/050 established as broken, and
   `scripts/train_dit_high_accuracy.sh` passed `--data-dir data/cats`, which does not
   exist inside the container (`/data/cats`).

## Decision

1. **Hold out a validation split.** New `dataset.create_train_val_dataloaders()`
   partitions with a seeded generator and gives the validation half eval transforms
   (resize + normalize, no augmentation). Default `--val-split 0.05`.
2. **Select and early-stop on held-out loss, preferring EMA.** New
   `src/dit_validation.py` provides `evaluate_flow_loss()` (fixed batch window,
   seeded timestep/noise so the metric is step-comparable — `sample_t()` gained an
   optional `generator`) and `evaluate_model_and_ema()`, which applies the EMA
   weights in place, evaluates, and restores the training parameters exactly.
   `is_best`/patience use the EMA held-out loss when available, else the raw held-out
   loss, else the training average only when the split is disabled.
3. **Persist training state in checkpoints.** `save_checkpoint()` now stores
   `best_loss`, `patience_counter`, `val_loss`, `val_loss_ema`, `steps`,
   `warmup_steps`, and `config["num_classes"]`; `load_checkpoint()` populates an
   optional `state` dict. On resume the LR horizon becomes
   `max(recorded_steps, steps)`, `progress` is clamped to `[0, 1]`, and the
   `--min-lr` floor is computed from the optimizer's `initial_lr` so it stays
   proportional to the LR actually being decayed.
4. **Make `--config` real.** The YAML is now applied via `parser.set_defaults()`
   *before* parsing (CLI flags still win), with an alias map for keys whose spelling
   differs from the destination (`patience`, `min_delta`, `beta`, `level`) and a
   warning for unknown keys. Unknown keys no longer create junk attributes.
5. **Enforce every gate threshold.** `max_final_loss` returns a failing
   `ValidationResult` like the accuracy check already did.
6. **Add `ValidationResult.skipped`.** Missing optional dependencies/artifacts are
   skipped, not failed — and are excluded from the pass/fail counts and reported
   separately (`skipped_checks`).
7. **Rebuild models from the checkpoint's own config** (`build_model_from_checkpoint`,
   `is_generative_checkpoint`, `extract_state_dict`) instead of hardcoding
   `tinydit_128`/128 and requiring `num_classes`; load generators through
   `load_state_dict_checked` so an incompatible checkpoint cannot half-load.
8. **Correct the documented commands** to `--data-dir` + `/data/cats` for Modal.

## Consequences

### Positive

- Checkpoint selection, early stopping, and the reported final loss now come from an
  augmentation-free held-out set, and the EMA checkpoint (the one that ships) is the
  one being selected.
- Early stopping survives resumes; the LR curve is continuous across sliced runs and
  floors at `--min-lr` instead of 0.
- The gate is meaningful again: the classifier now passes with two checks *skipped*,
  and a generator at loss > threshold now **fails** instead of passing silently.
- YAML configs and `--min-lr` do what they say.

### Negative / neutral

- Validation adds forward passes at every `save_interval` (≈4 batches by default) and
  clones the trainable parameters once per evaluation for the EMA swap.
- `--val-split 0` restores the old behaviour exactly, including its blind spots.
- A checkpoint's `steps` now pins the LR horizon on resume; changing the target
  horizon means the schedule stretches rather than restarts.
- The generator still has no FID; sample quality remains a degenerate-output sanity
  check, not a perceptual metric.

## Implementation

| File | Change |
|---|---|
| `src/dit_validation.py` | New: `evaluate_flow_loss`, `evaluate_model_and_ema` |
| `src/dataset.py` | `create_train_val_dataloaders`, `_weighted_generator_loader` |
| `src/train_dit.py` | Val split + EMA selection, `build_lr_lambda`, `load_yaml_defaults` + config pre-parse, `--val-split`/`--val-batches`, checkpoint state, `persist()` helper |
| `src/flow_matching.py` | `sample_t(..., generator=...)` for reproducible evaluation |
| `src/validate_model.py` | Enforce `max_final_loss`, `skipped` results, generative detection, config-driven model rebuild, checked loader |
| `AGENTS.md`, `agents-docs/training.md` | Corrected Modal commands, `--min-lr`/`--val-split`/`--config` documented |
| `scripts/train_dit_high_accuracy.sh` | `/data/cats` for Modal runs, `${1:-}` guards |
| `tests/test_validate_model.py`, `tests/test_train_dit.py` | 32 regression tests for the above |

## References

- **ADR-028** — model validation gates.
- **ADR-059** — interval/evaluation loss split, T4 baseline.
- **ADR-060** — missing positional embeddings and broken unpatchify.
- **ADR-048/051/054** — Modal CLI `--data-dir` and absolute container paths.
