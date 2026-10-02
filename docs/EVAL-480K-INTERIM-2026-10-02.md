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

## Interpretation — quality has plateaued

44,000 additional steps (436k → 480k) moved FID by 0.6% and left IS, precision
and recall flat. At this rate the remaining 120k steps to 600k are unlikely to
reach the ADR-036 targets (FID ~30–40, IS ~5–6); they would cost roughly 25
further T4-hours for single-digit FID movement.

Before spending the rest of the budget on steps alone, the cheaper levers are
worth testing on the existing 480k checkpoint:

- more sampling steps at inference (30 → 50/100) and CFG sweep (1.5 → 2–3),
- checking whether the residual sharpness/saturation gap (noisy,
  over-saturated samples) is a sampler or an EMA/learning-rate issue,
- data-side: the 13th "Other" class and ~2.4k training images may cap fidelity.

This mirrors the low IS across both measurements: the model is not
mode-diverse yet, which more steps at the same recipe have not fixed.