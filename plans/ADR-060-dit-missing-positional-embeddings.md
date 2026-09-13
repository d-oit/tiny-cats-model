# ADR-060: TinyDiT spatial-structure bugs — missing positional embeddings and broken unpatchify

- **Status:** Accepted (2026-09-13)
- **Date:** 2026-09-13
- **Deciders:** tiny-cats-model maintainers
- **Related:** `src/dit.py`, `src/train_dit.py`, `tests/test_flow_matching.py`, ADR-017, ADR-019

## Context

The DiT generator has never produced meaningful images. A diagnostic on the
`breed-conditioned-v3` checkpoint (and the earlier Sept 1 / Feb checkpoints)
established:

- Every checkpoint generates pure RGB noise at `cfg` 1.0 and 1.5. The Feb
  samples in `samples/generated/step_0/` that ADR-019 described as "coherent
  structure" are noise.
- On a controlled flow-matching loss evaluation the model scores ~1.17 against
  a zero-predictor baseline of 1.38 and a **single-scalar linear predictor** of
  1.15 — the transformer is worse than `a * x_t`.
- The model can overfit one batch (1.21 → 0.18 in 120 steps at lr 1e-3), so the
  optimizer and loss are fine.
- Layer-wise probing showed `patch_embed`, every block, and `final_layer` are
  **exactly permutation-equivariant**: `f(permute_patches(x)) ==
  permute_patches(f(x))`.

Two architecture bugs explain all of it:

1. **No positional embeddings.** `TinyDiT` never added any position information
   after `PatchEmbed`, so the model is a bag-of-patches transformer. It cannot
   represent spatial layout, which is exactly what image generation requires.
2. **Broken unpatchify.** `forward()` did
   `x.transpose(1, 2).reshape(B, C, H, W)`, which interleaves the token
   dimension with pixels: every output patch received values from *every*
   token. A per-token-constant probe confirmed each 16x16 output patch had
   std 1.12 instead of 0.

## Decision

1. **Add a learnable positional embedding** to `TinyDiT`
   (`nn.Parameter` of shape `(1, num_patches, embed_dim)`, trunc-normal
   `std=0.02` like the reference DiT), added to the patch embeddings in
   `forward()`.
2. **Fix the unpatchify layout** to the standard patch reconstruction:
   `(B, N, P*P*C) -> (B, n, n, P, P, C) -> permute -> (B, C, H, W)`, so each
   token reconstructs its own patch.
3. **Keep the timestep-embedding scaling** (`t * 1000` before the DDPM
   sinusoids): with raw `t in [0, 1]` the embedding changes by only ~0.19 in
   relative L2 from t=0.1 to t=0.9, versus ~1.27 when scaled. It is the
   standard convention and removes a needless handicap, though on its own it
   did not change the loss curve.
4. **Regression tests** in `tests/test_flow_matching.py`:
   `test_has_positional_embedding`, `test_output_is_position_aware` (permuting
   patches must change the output), and
   `test_unpatchify_places_each_token_in_its_patch`.
5. **Graceful incompatible-checkpoint handling**: `load_checkpoint()` now
   catches `RuntimeError` from `load_state_dict` (e.g. a checkpoint without
   `pos_embed`) and restarts from step 0 instead of crashing the training job.
6. **Retrain on a fresh output path** (`breed-conditioned-v4`): every existing
   DiT checkpoint predates the architecture fix and is worthless.

## Consequences

### Positive

- The generator can finally represent spatial structure. The validation run
  (2,000 steps, lr 5e-4, warmup 100, on the 2,900-image volume) drove the
  flow-matching loss from ~1.39 to **0.63** by step 1,900, versus ~1.26 for an
  identical run without the fix and a ~1.17 plateau after 47,000 steps before.
- Samples at step 2,000 show coherent spatial structure (per-patch content,
  light/dark layout) instead of uniform noise. Recognisable images still
  require a full-length run.
- Regression tests pin both failure modes, so neither can silently return.

### Negative / neutral

- All pre-existing DiT checkpoints are incompatible (handled by the restart
  logic); no result from the previous generator effort is salvageable.
- ADR-019's sample-quality assessment was wrong and is superseded by this ADR.
- The training experiments also found two training-loop bugs, fixed alongside:
  the shutdown handler re-saved 500MB x2 on every remaining batch until
  SIGKILL, and a resume past the target steps overwrote a good checkpoint with
  `loss=0.0`.

### Open items

- The Modal `dit-dataset` volume holds 2,900 images (2,400 cats + 500 dogs)
  while `data/download.py` now prepares ~7,400 — refresh it before the full
  retrain.
- Learning-rate/warmup experiments (lr 5e-4 + warmup 100 learns faster than
  the production lr 5e-5 + 2,000-step warmup) should be re-evaluated once the
  architecture fix is in.

## References

- **DiT** — Peebles & Xie, "Scalable Diffusion Models with Transformers" (2022):
  learnable positional embeddings and zero-initialised final layer.
- **ADR-017** — TinyDiT training infrastructure.
- **ADR-019** — TinyDiT sample evaluation results (superseded assessment).
- **ADR-059** — breed-conditioned training fixes and T4 baseline.
