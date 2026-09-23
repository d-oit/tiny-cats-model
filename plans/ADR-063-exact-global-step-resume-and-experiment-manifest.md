# ADR-063: Exact global-step resume and the immutable experiment manifest

- **Status:** Accepted (2026-09-23)
- **Date:** 2026-09-23
- **Deciders:** tiny-cats-model maintainers
- **Related:** `src/train_dit.py`, `src/training_state.py`,
  `tests/test_training_state.py`, `tests/test_resume_exactness.py`,
  `agents-docs/training.md`, AGENTS.md, issue #163 (WP1 + WP4),
  ADR-058 (checkpoint quarantine), ADR-059 (pool hardening),
  ADR-061 (held-out validation), ADR-062 (timestep sampling)

## Context

Issue #163 tracks what is required for a **complete, resumable 400k-step
DiT run** across Modal and future GPU-pool providers. Work packages 1 (exact
global-step semantics) and 4 (immutable experiment manifest) are the contract
everything else — Hub transport, bounded slices, workflow hardening — builds
on, and both were unsound in the code:

1. **Off-by-one in resume.** `checkpoint["step"]` counts *completed* optimizer
   steps (`persist()` runs after `step += 1`), but `load_checkpoint()` returned
   `checkpoint["step"] + 1`. A checkpoint at 45,000 with `--steps 60000`
   therefore performed **14,999** steps instead of the required 15,000.
2. **First resumed optimizer step ran at the init LR.** The resume path set
   `scheduler.last_epoch = start_step - 1` but assigning `last_epoch` does not
   recompute the param-group LR — the groups still held `lambda(0)` (0.01× for
   a warmup schedule), so every resumed slice started with one step at 1% LR.
3. **Silent restarts on architecture mismatch.** `load_checkpoint()` caught
   the `ValueError` from `load_state_dict_checked()` and restarted from step 0
   with fresh weights — a provider handoff landing on the wrong architecture
   would quietly destroy continuity instead of failing clearly.
4. **Non-zip corrupt checkpoints crashed.** Garbage bytes raise
   `pickle.UnpicklingError`, which was *not* in the quarantine `except` tuple
   (only truncated *zips* were handled).
5. **No experiment identity.** Nothing recorded which dataset, breed mapping,
   architecture, optimizer settings or seed a checkpoint belonged to, so any
   resume — compatible or not — was accepted.

Research grounding: the PyTorch saving/loading guidance (state dicts for
model/optimizer, RNG state carried in the checkpoint, restore last and never
reseed afterwards) and the HF Hub upload guide (uploads are atomic per commit
and resumable by re-running; the `.pt` is the transport unit, so identity must
travel *inside* it).

## Decision

1. **`--steps` is the global target.** `load_checkpoint()` now returns the
   number of *completed* global steps, and `training_state.steps_to_run()`
   computes `max(0, target − completed)` (the `+1` is gone). Already-complete
   checkpoints exit successfully without a training step and without writing a
   checkpoint — the existing `step > start_step` guard is preserved, and the
   no-op path may only create a missing `training_state.json`, never overwrite
   a checkpoint.
2. **New `src/training_state.py`** owns:
   - the **manifest** (`build_manifest`) — experiment_id, repository, git_sha,
     dataset id/version/hash, breed-mapping hash, image/patch size, embed dim,
     depth, heads, num classes, optimizer, lr, warmup, scheduler, batch size,
     grad-accum, augmentation level, seed, target steps, provider;
   - **`training_state.json`** — manifest flattened top-level + atomic
     `completed_steps`/`written_at`/`schema_version` (tmp file, fsync,
     `os.replace`) written beside every checkpoint;
   - **RNG capture/restore** — `torch`/`python`/`numpy`/CUDA states saved in
     every checkpoint and restored after load, before the loop;
   - `steps_to_run()` — the single source of resume arithmetic.
3. **Manifest gate on resume.** The embedded manifest is authoritative (it
   survives Hub transport); the sidecar is a fallback for legacy checkpoints.
   Fields in `IMMUTABLE_FIELDS` (architecture, dataset hash, breed mapping,
   optimizer, lr, warmup, scheduler, batch size, grad-accum, augmentation,
   seed, experiment id) must match or the resume raises
   `IncompatibleExperimentError` listing every difference — unless
   `--allow-experiment-mismatch` is passed. `target_steps`, `git_sha`,
   `provider` and `dataset_version` are recorded but **not** compared: slices
   raise the target (60k → 400k), code advances between slices, and providers
   are expected to differ. Validation only runs after a *successful* load, so
   a stale sidecar can never poison a corrupt-checkpoint restart.
4. **Rejection, not restart.** Architecture mismatches now raise
   `IncompatibleExperimentError` (from `load_state_dict_checked`'s clean
   failure) instead of returning `start_step=0`. Corrupt/unreadable files keep
   the ADR-058 quarantine, extended with `pickle.UnpicklingError`.
5. **Scheduler positioned exactly.** On resume: `last_epoch = completed` plus
   an explicit `group["lr"] = initial_lr * λ(completed)` materialisation
   (no `scheduler.step()` call, so no "step before optimizer.step" warning),
   restoring the exact invariant an uninterrupted run has at that point.
6. **`model_fn` seam on `train_dit_local`** so tests inject a tiny DiT and the
   *real* loop's exactness runs on CPU in CI (issue WP8's resume test:
   target 20, slice 1 = 10, slice 2 = exactly 10 more).

## Consequences

### Positive

- Acceptance criterion met and pinned by tests: 45k → 60k is exactly 15,000;
  60k → 400k is 340,000; completed target is a zero-step success with the
  checkpoint byte-identical.
- A provider handoff now carries model + optimizer + scheduler horizon + EMA +
  early-stopping state + RNG stream + experiment identity in one file, and the
  next provider either validates it or fails with a field-level diff.
- Resume no longer emits a bogus 0.01×-LR step; the LR curve is continuous
  across slices.
- `training_state.json` makes progress inspectable without unpickling
  500 MB checkpoints (basis for future workflow/slice reporting, WP6/WP7).

### Negative / neutral

- Resumes of *pre-existing* checkpoints (no manifest) log a warning and
  proceed with architecture validation only — they cannot be config-verified.
- Changing lr/batch/seed/augmentation between slices now requires an explicit
  `--allow-experiment-mismatch`; that is deliberate friction (it is exactly
  what the issue asks to reject).
- The manifest's `dataset_hash` fingerprints paths + sizes (not content) —
  cheap on 7,390 images, detects add/replace/truncate, not in-place same-size
  edits.
- Exact RNG continuation covers the main process; DataLoader worker RNGs
  (`num_workers > 0`) are re-seeded per epoch by PyTorch and remain best
  effort.

## Implementation

| File | Change |
|---|---|
| `src/training_state.py` | New: manifest, `training_state.json`, RNG state, `steps_to_run`, `IncompatibleExperimentError` |
| `src/train_dit.py` | Completed-steps resume, manifest gate + override, RNG restore, scheduler positioning, sidecar writes, `pickle.UnpicklingError` quarantine, `--experiment-id`/`--allow-experiment-mismatch`, `model_fn` seam, Modal image ships `training_state.py` |
| `tests/test_training_state.py` | Step arithmetic (45k/60k/400k), manifest fields + mismatch matrix, atomic state round-trip, RNG restore, fingerprints |
| `tests/test_resume_exactness.py` | Real-loop e2e: 0→20, 10→20 exactly 10, no-op (same + different output path), corrupt quarantine, arch/optimizer rejection + override |
| `tests/test_train_dit.py` | Round-trip assertion follows completed-steps semantics (42, not 43) |
| `AGENTS.md`, `agents-docs/training.md` | Global-target semantics, manifest/override rules, sliced examples |

## References

- **Issue #163** — Complete DiT training / GPU-pool hardening plan (WP1, WP4).
- **PyTorch**, *Saving and Loading Models* —
  https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
- **PyTorch forums** — RNG generator state save/resume, exact recovery
  (restore last, never reseed afterwards).
- **Hugging Face Hub**, *Upload files to the Hub* — atomic-per-commit,
  resumable uploads:
  https://huggingface.co/docs/huggingface_hub/en/guides/upload
- **ADR-058** — corrupt-checkpoint quarantine (preserved and extended).
- **ADR-061** — checkpoint state (early stopping, LR horizon) round-trip.
