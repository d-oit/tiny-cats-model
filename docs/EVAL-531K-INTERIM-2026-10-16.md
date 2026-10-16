# 531k Interim Checkpoint Evaluation — 2026-10-16

Pre-lift control point on the `dit-breed-conditioned-v4` pool, measured with
the same protocol as `docs/EVAL-436K-CHECKPOINT-2026-10-02.md` and
`docs/EVAL-480K-INTERIM-2026-10-02.md` (988 samples = 13 breeds × 76,
30 steps, CFG 1.5, seed 42, EMA weights vs all `data/cats/cat` photos) so
the numbers stay directly comparable. Produced by
`bash scripts/eval_checkpoint.sh checkpoints/pool/dit_model_ema_531238.pt eval_out/samples-531k`.

## Checkpoint

| Field | Value |
|-------|-------|
| Source | Modal volume `dit-outputs` → `checkpoints/pool/dit_model_ema.pt`, pulled 2026-10-16 |
| Step | 531,238 (the pool's last save before the provider stall) |
| LR horizon recorded in the checkpoint | `steps=540000` (the last slice's target), no `schedule_steps` — pre-fix layout |

## Results

| Metric | 436k | 480k | 531k | Δ vs 480k |
|--------|------|------|------|-----------|
| FID | 263.31 | 261.72 | **263.00** | +1.28 |
| Inception Score | 1.60 ± 0.07 | 1.59 ± 0.07 | **1.59 ± 0.06** | 0.00 |
| Precision | 0.50 | 0.50 | **0.50** | 0.00 |
| Recall | 0.48 | 0.47 | **0.47** | 0.00 |

## Interpretation — the floor-bound steps trained nothing measurable

The 51,238 steps between 480k and 531,238 ran with the cosine decay pinned at
the `--min-lr` floor (≈2% of nominal LR; see
`docs/LR-PLATEAU-DIAGNOSIS-2026-10-02.md`), and every metric is unchanged
within noise — the +1.28 FID move is the same magnitude as the −1.59
"improvement" measured from 436k to 480k. This is the baseline that makes the
remediation testable: the next slice resumes at `--min-lr 1e-5` (≈20% of
nominal) with a campaign-wide horizon of 600k, so movement at the next
measurement is attributable to the lifted floor, not to additional steps at
the old one.

The checkpoint also predates the 13-day provider stall (2026-10-03 →
2026-10-16), so no training happened between this save and this measurement.

> **Artifacts note:** the eval was run against the volume copy pulled to
> `checkpoints/pool/dit_model_ema_531238.pt` (the pre-existing local
> `dit_model_ema.pt` is the 480k checkpoint). Report:
> `eval_out/samples-531k/report.json`.
