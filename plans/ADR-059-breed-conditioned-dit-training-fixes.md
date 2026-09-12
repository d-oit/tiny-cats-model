# ADR-059: Breed-conditioned DiT training fixes and T4 baseline recalibration

- **Status:** Accepted (2026-09-12)
- **Date:** 2026-09-12
- **Deciders:** tiny-cats-model maintainers
- **Related:** `src/dataset.py`, `src/train_dit.py`, `src/dit.py`, `src/gpu_pool.py`,
  `.github/workflows/train-pool.yml`, `scripts/benchmark_estimates.py`

## Context

Four independent defects were found while preparing the third breed-conditioned
generator run (`breed-conditioned-v3`):

1. **Conditioning collapse.** TinyDiT conditions on 13 classes (12 cat breeds +
   `other`), but `data/cats` is a two-folder dataset (`cat/`, `other/`) and
   `create_dataloader()` used `ImageFolder`, which only produced binary labels.
   Embeddings 2-12 therefore never received a gradient — the generator could not
   learn breed conditioning at all.

2. **Loss accounting.** `avg_loss` was averaged over the *logging* interval
   (`log_interval`, default 100) and reused for checkpoint `is_best` decisions and
   early stopping (every `save_interval`, default 500). A smoke run on 2026-09-12
   even reported `Final loss: 0.000000e+00` because `best_loss` was never updated
   when no checkpoint interval completed.

3. **Silent CI training failures.** In `train-pool.yml`, `${HUBSYNC:+"$HUBSYNC"}`
   expanded `--no-hub-push` into a single argument (`" --no-hub-push"`), so the
   Modal CLI exited with `Got unexpected extra argument`. Because the step piped
   through `tee` without `set -o pipefail`, the job still reported success. Every
   scheduled run since the workflow was introduced (2026-08-10) trained nothing —
   the artifacts contain 149 bytes of usage error.

4. **Wrong planning baseline.** `T4_STEPS_PER_SECOND = 2.2` came from the printed
   per-interval speed of an older run, not wall-clock. Direct Modal measurements
   on 2026-09-12 (batch 32/64/128, 128x128, AMP) show ~37 images/s on a T4, i.e.
   ~0.06 batch-512-equivalent steps/s — the old figure overstated throughput by
   roughly 37x.

## Decision

1. **Label the generator dataset by filename.** New
   `CatBreedGenerationDataset` in `src/dataset.py` maps Oxford Pets filenames
   (`Abyssinian_*.jpg`, ...) to indices 0-11 and every `other/` image to index 12.
   Unknown `cat/` filenames raise instead of silently training the wrong class.
   `create_dataloader()` uses a `WeightedRandomSampler` so the 2.4k cats and 5k
   others contribute equally per epoch.

2. **Evaluate loss over the checkpoint window.** Loss is accumulated separately
   for logging (reset every `log_interval`) and evaluation (reset every
   `save_interval`). Checkpoint `is_best`, patience, and the final reported loss
   all use the evaluation window; `--early-stopping-patience 0` disables early
   stopping, and the final loss can no longer be `0.0` when no checkpoint fired.

3. **Use `F.scaled_dot_product_attention`** in `Attention` (same scaling,
   kernel-selected implementation) instead of an explicit softmax.

4. **Recalibrate the T4 baseline to `0.06` batch-512-equivalent steps/s** and
   record the three real 2026-09-12 measurements in
   `scripts/benchmark_estimates.py`. The 400k A10G/H100 entry is marked
   unmeasured (`actual_hours=None`): `docs/TRAINING_400K_LOG.md` only contains a
   36-48h *estimate*, so it no longer pollutes `--tune`. `--tune` now reports
   3 real benchmarks, mean error -4.9%, recommended baseline 0.06.

5. **Fix the workflow flag passing.** `train-pool.yml` builds flag arrays and
   adds `set -o pipefail`, so a failing `modal run` fails the job. Modal defaults
   move to 60k steps / batch 32 (most optimizer steps per wall-clock second at
   this throughput) and output to
   `/outputs/checkpoints/dit/breed-conditioned-v3/`. The CPU pool-runner job is
   opt-in for explicit non-modal providers (not on schedule or `provider=all`).

## Consequences

### Positive

- All 13 conditioning embeddings are trained and class-balanced; samples can be
  conditioned on specific breeds.
- Checkpoint selection and early stopping use a meaningful loss average.
- Scheduled Modal runs actually train, and failures surface as job failures.
- Cost/GPU-hour estimates are grounded in measured T4 throughput.

### Negative / neutral

- The corrected baseline is much slower than the old figure: 100k steps at
  batch 512 is ~463h on one T4, so long runs need Hub-resumed slices or a
  faster GPU.
- Measurements are startup-inclusive and were taken with `num_workers=0`
  (CPU dataloading); enabling workers could raise throughput, in which case the
  baseline must be re-measured.
- `provider=all` now runs the Modal job only; the pool runner remains a
  simulation of the non-modal providers.

## Implementation

| File | Change |
|---|---|
| `src/dataset.py` | `CAT_BREEDS`, `OTHER_CLASS_INDEX`, `CatBreedGenerationDataset` |
| `src/train_dit.py` | Filename-labeled dataset + `WeightedRandomSampler`; interval/evaluation loss split; patience=0 support |
| `src/dit.py` | `F.scaled_dot_product_attention` |
| `src/gpu_pool.py` | `T4_STEPS_PER_SECOND: 2.2 → 0.06` with measurement provenance |
| `.github/workflows/train-pool.yml` | Flag arrays, `pipefail`, 60k/32 defaults, `breed-conditioned-v3` output, opt-in pool runner |
| `scripts/train_lightning.py` | `setup_lightning_dirs()` honors `--data-dir` / `--checkpoint-dir` |
| `tests/test_dataset.py` | Breed-mapping, unknown-breed, and sampler-balance tests |
| `scripts/benchmark_estimates.py` | 3 real T4 benchmarks; 400k entry excluded from tune |
| `scripts/test_fallback_chain.py` | Estimate range updated to the measured T4 baseline (600-1200h for 400k @ 256) |
| `scripts/quality-gate.sh` | Runs the fallback-chain simulation so baseline changes can't slip past locally |

## References

- **ADR-036** — high-accuracy training configuration.
- **ADR-057** — Modal CLI verification & best-practice audit.
- **ADR-058** — T4/L4 GPU selection and save interval.
- `docs/TRAINING_400K_LOG.md` — plan-only 400k log (no measured wall-clock).
