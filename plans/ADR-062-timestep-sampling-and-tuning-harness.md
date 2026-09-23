# ADR-062: Timestep-sampling option and a DiT configuration comparison harness

- **Status:** Accepted (2026-09-20)
- **Date:** 2026-09-20
- **Deciders:** tiny-cats-model maintainers
- **Related:** `src/flow_matching.py`, `src/train_dit.py`, `src/dit_validation.py`,
  `scripts/tune_dit_configs.py`, `configs/dit_train_config.yaml`, ADR-059, ADR-060, ADR-061

## Context

ADR-060 left two open items: re-evaluate the learning-rate/warmup recipe and settle
training-recipe details now that the generator can finally represent spatial
structure. Evaluating any of them required two things the repo did not have:

1. **A selectable timestep distribution.** `sample_t()` was uniform-only, so the
   standard flow-matching alternative (SD3-style logit-normal, which concentrates
   capacity on the middle of the trajectory instead of the t≈0/t≈1 tails) could not
   even be tried.
2. **A comparable metric across configurations.** `train_dit_local()` scores each run
   with the validation loss of *that run's* timestep distribution, so two arms
   differing in `--timestep-sampling` could not be ranked by it.

## Decision

1. **Add the option, keep the measured default.** `flow_matching.sample_timesteps()`
   dispatches on `sampling` to either the existing `sample_t()` or the new
   `sample_t_logit_normal()` (`t = sigmoid(mean + std * N(0, 1))`, mean 0 / std 1
   matching SD3). Exposed as `--timestep-sampling {uniform,logit_normal}` plus
   `--logit-normal-mean`/`--logit-normal-std`. The held-out evaluation takes the same
   arguments so the reported loss measures the objective actually optimised.
2. **Add `scripts/tune_dit_configs.py`.** It trains each named arm with
   `train_dit_local()`, rebuilds the arm's checkpoint, and re-scores **every** arm on
   the same held-out split under a fixed reference distribution (uniform). Arms are
   declared on the CLI (`--arms label=…,lr=…,warmup_steps=…`), results print as a
   table and can be written to JSON.
3. **Record the measurement.** A 250-step, batch-8, lr-1e-3 run on the real
   7,390-image dataset (16 held-out batches, seed 42) gives:

   | Arm | Held-out loss | Δ vs best |
   |---|---|---|
   | uniform, lr 1e-3, warmup 20 | **0.728487** | — |
   | logit-normal, lr 1e-3, warmup 20 | 0.732695 | +0.004208 |
   | uniform, lr 5e-4, warmup 100 | 0.817498 | +0.089011 |
   | uniform, lr 5e-5 (production), warmup 20 | 1.096956 | +0.368469 |

## Consequences

### Positive

- The timestep distribution is now a one-flag experiment, and held-out loss is
  comparable across arms instead of being distribution-dependent.
- There is a reusable, documented way to answer "which config is worth GPU time?"
  without hand-running shells.
- The LR question now has a number behind it: at 250 steps the production
  `lr 5e-5` lands at **1.097** against **0.728** for `lr 1e-3`, i.e. the shipping
  recipe is still an order of magnitude away from fast early learning. This is
  consistent with ADR-060's observation and is the strongest evidence yet that the
  production LR/warmup needs revisiting.

### Negative / neutral

- **The logit-normal switch is not justified by this measurement.** 0.7285 vs
  0.7327 is a 0.6% gap at a budget where a single arm's seed-to-seed noise is of the
  same order; `uniform` therefore remains both the code default and the value in
  `configs/dit_train_config.yaml`. The option ships for a GPU-scale test.
- The LR ranking at short budgets favours aggressive LRs (fast early descent), so
  these numbers say nothing about final image quality. Warmup was varied together
  with LR in the third arm, so its individual effect is not isolated.
- The harness loads each arm's ~500 MB checkpoint to rebuild the model, which is the
  dominant per-arm overhead. Scores are deleted after use unless
  `--keep-checkpoints` is passed.

### Verified separately

ADR-060's open item about the Modal `dit-dataset` volume holding only ~2,900 images
is **stale**: `modal volume ls dit-dataset /cats/{cat,other}` reports 2,400 cats and
4,990 others (7,390 images), matching `data/download.py`. No refresh is needed.

## Implementation

| File | Change |
|---|---|
| `src/flow_matching.py` | `TimestepSampling`, `sample_t_logit_normal`, `sample_timesteps` |
| `src/dit_validation.py` | Timestep-sampling kwargs threaded into the held-out loss |
| `src/train_dit.py` | `--timestep-sampling`/`--logit-normal-mean`/`--logit-normal-std`, tracker params, 128-step model log line |
| `configs/dit_train_config.yaml` | `timestep_sampling` + `validation` sections (uniform default) |
| `scripts/tune_dit_configs.py` | New comparison harness |
| `tests/test_flow_matching.py` | `TestTimestepSampling` (range, concentration, determinism, dispatch errors) |
| `AGENTS.md`, `agents-docs/training.md` | Option documented + harness usage |

## References

- **Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image
  Synthesis" (2024)** — logit-normal timestep sampling.
- **ADR-059** — evaluation-window loss accounting and the T4 baseline.
- **ADR-060** — positional embeddings, unpatchify, and the LR/warmup open item.
- **ADR-061** — held-out validation, EMA selection, and working validation gates.
