# 436k Checkpoint Evaluation — 2026-10-02

Generation-quality snapshot of the `dit-breed-conditioned-v4` pool checkpoint
mid-training (target 600k), measured with the fixed FID implementation
(PR #174 — the previous Newton-Schulz matrix square root returned NaN for
rank-deficient covariances).

## Checkpoint

| Field | Value |
|-------|-------|
| Experiment | `dit-breed-conditioned-v4` |
| Source | Modal volume `dit-outputs` → `checkpoints/pool/dit_model_ema.pt` |
| Step (checkpoint header) | 436,000 |
| `training_state.json` | 434,000 completed (written 2026-10-01 18:54 UTC) |
| Weights | EMA |
| Architecture | TinyDiT 12L/384d, patch 16, 128×128, 13 breeds |

## Method

- 26 generated samples (13 breeds × 2), 30 sampling steps, CFG 1.5, seed 42.
- 240 real Oxford-IIIT-Pet cat images as the reference set.
- Metrics via `src/evaluate_full.py` (torchvision InceptionV3 features).

## Results

| Metric | 436k (measured) | ADR-036 target (400k) | Reading |
|--------|-----------------|-----------------------|---------|
| FID | **316.6** | ~30–40 | very poor |
| Inception Score | **1.25 ± 0.14** | ~5–6 | very poor |
| Precision | **0.50** | ~0.75 | half the samples off-manifold |
| Recall | **0.36** | ~0.55 | modes not covered |

Image-statistics sanity check vs real photos (26 vs 26):

| Statistic | Generated | Real |
|-----------|-----------|------|
| Sharpness (high-freq energy) | 0.48 | 0.06 |
| Luminance | 0.39 | 0.40 |
| Saturation | 0.65 | 0.35 |

## Interpretation

The generator still produces noisy, over-saturated images with limited
diversity at 436k steps — consistent with a model mid-training after the
recent root-cause fixes (#159 conditioning, #160 positional embeddings and
unpatchify). Not releaseable; the 600k target is justified.

## Caveats

- FID on 26 generated samples is high-variance; treat it as an order of
  magnitude, not a precise score. Re-measure with ≥1000 samples at 600k.
- The reference set is a 240-image subset (every 10th image) for CPU speed.
- ADR-036's expected numbers came from a different configuration (batch 512,
  longer warmup) and are directional only for this pool experiment.
