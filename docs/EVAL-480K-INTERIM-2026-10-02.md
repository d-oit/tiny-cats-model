# 480k Interim Checkpoint Evaluation — 2026-10-02

Interim data point on the `dit-breed-conditioned-v4` pool, measured with the
same protocol as `docs/EVAL-436K-CHECKPOINT-2026-10-02.md` so the numbers are
directly comparable (988 samples, 30 steps, CFG 1.5, seed 42, EMA weights vs
all `data/cats/cat` photos). Produced by `bash scripts/eval_checkpoint.sh`.

## Checkpoint

| Field | Value |
|-------|-------|
| Source | Modal volume `dit-outputs` → `checkpoints/pool/dit_model_ema.pt` |
| Step | 480,000 (clean slice completion; `exit_reason: completed`) |

## Results vs the 436k baseline

| Metric | 436k | 480k | Δ |
|--------|------|------|---|
| FID | 263.31 | **261.72** | −1.59 |
| Inception Score | 1.60 ± 0.07 | **1.59 ± 0.07** | −0.01 |
| Precision | 0.50 | **0.50** | 0.00 |
| Recall | 0.48 | **0.47** | −0.01 |

## Sampler / CFG sweep

CPU-only, on the 480k EMA weights: 338 samples per arm (26 per breed) against
the same 240-image real subset, so arms are directly comparable to each other.
Absolute FID is offset by sample size — the (30, 1.5) arm is the sweep's
control, not the 436k number above.

| Sampling steps | CFG | FID | Inception Score |
|----------------|-----|-----|-----------------|
| 30 | 1.5 (control) | 282.13 | 1.54 |
| 50 | 1.5 | 280.41 | 1.64 |
| 50 | 2.0 | 280.25 | 1.64 |
| 50 | 3.0 | 279.37 | 1.70 |
| 100 | 1.5 | 279.65 | 1.72 |
| **100** | **3.0** | **278.73** | **1.77** |

- Sampling steps help monotonically; most of the gain arrives by 50 steps, and
  30 → 100 buys FID −3.4 / IS +0.23 against the control.
- CFG 1.5 → 2.0 is within noise; 3.0 adds ~0.9 FID / 0.06 IS at 50 steps and
  stacks with 100 steps.
- **Artifacts are identical in every arm** (sharpness 0.47–0.48, saturation
  0.65–0.66 vs 0.055 / 0.304 for real photos): tuning improves the metrics
  without fixing the noise and over-saturation — evidence the plateau is
  training-side, not sampler-side.

**Applied:** demo defaults 50 → 100 sampling steps (`src/app_gradio.py`,
`frontend/src/constants.ts`). CFG stays at 1.5 by default — its benefit was
measured on the PyTorch checkpoint while both demos run the quantized ONNX
model, and higher CFG amplifies the existing over-saturation. The CFG slider
goes to 3.0 for anyone who wants it.

## Interpretation — quality has plateaued

44,000 additional steps (436k → 480k) moved FID by 0.6% and left IS, precision
and recall flat. At this rate the remaining 120k steps to 600k are unlikely to
reach the ADR-036 targets (FID ~30–40, IS ~5–6); they would cost roughly 25
further T4-hours for single-digit FID movement.

Before spending the rest of the budget on steps alone, the cheaper levers are
worth testing on the existing 480k checkpoint:

- ~~more sampling steps at inference and a CFG sweep~~ — done above; worth
  ~0.23 IS and ~1% FID, and it does not touch the artifacts,
- checking whether the residual sharpness/saturation gap (noisy,
  over-saturated samples) is a sampler or an EMA/learning-rate issue,
- data-side: the 13th "Other" class and ~2.4k training images may cap fidelity.

This mirrors the low IS across both measurements: the model is not
mode-diverse yet, which more steps at the same recipe have not fixed.