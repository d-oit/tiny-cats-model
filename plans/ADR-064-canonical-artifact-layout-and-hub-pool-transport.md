# ADR-064: Canonical artifact layout and HF Hub pool transport

- **Status:** Accepted (2026-09-23)
- **Date:** 2026-09-23
- **Deciders:** tiny-cats-model maintainers
- **Related:** `src/artifacts.py`, `src/gpu_pool.py`, `src/train_dit.py`,
  `src/export_dit_onnx.py`, `src/evaluate_full.py`, `src/benchmark_inference.py`,
  `src/eval.py`, `src/validate_model.py`, `src/verify_checkpoint.py`,
  `src/upload_to_huggingface.py`, `src/upload_to_hub.py`,
  `.github/workflows/train.yml`, `.github/workflows/train-pool.yml`,
  `.github/workflows/upload-hub.yml`, `.agents/skills/model-training/SKILL.md`,
  `tests/test_artifacts.py`, `tests/test_hub_transport.py`,
  issue #163 (WP2 + WP3), ADR-063 (exact resume + manifest)

## Context

Issue #163 WP2 and WP3 cover the *storage* contract for a resumable 400k-step
run: where checkpoints and final artifacts live, and how independent GPU
sessions hand state to each other. Both were ambiguous or wrong:

1. **Ambiguous checkpoint locations.** Live checkpoints moved between
   `/outputs/checkpoints/dit/current` (DiTTrainer default) and
   `/outputs/checkpoints/dit/breed-conditioned-v4` (train-pool.yml), while
   the "final" generator was copied to the volume root as `tinydit_final.pt`
   — a name baked into three workflows, six scripts and the training skill.
2. **No artifact package.** ONNX exports, quantized models, reports and
   samples were scattered across the volume root with no manifest and no
   rule about *when* publication is allowed — a partial slice happily
   overwrote `tinydit_final.pt`.
3. **Flat Hub transport with no ordering guarantees.**
   `checkpoints/pool/<name>` was a single mutable path: two providers could
   silently overwrite each other, an interrupted upload could leave a
   half-written checkpoint as the newest state, there was no step identity,
   no stale-result protection and no retry/backoff (the issue explicitly
   calls out retries and "never replace a newer checkpoint").

Research grounding: HF Hub uploads are atomic per commit and resumable by
re-running (huggingface_hub upload guide), which makes a *pointer document*
a cheap, safe commit boundary — upload all snapshot files first, flip the
pointer last.

## Decision

1. **Canonical live layout** (new `src/artifacts.py`):

   ```
   checkpoints/pool/{dit_model.pt, dit_model_ema.pt,
                     training_state.json, training.log}
   ```

   DiTTrainer defaults here; train-pool.yml passes the same explicit paths.
2. **Canonical final artifact package**:

   ```
   artifacts/
   ├── generator/   model.pt, model_ema.pt, model.onnx, model_quantized.onnx
   ├── training/    final_checkpoint.pt, training_state.json, training.log
   ├── evaluation/  evaluation_report.json, benchmark_report.json,
   │                validation_report.json, samples/
   └── manifest.json   (sha256 + size per file, experiment id, git SHA)
   ```

   `package_final_artifacts()` builds it **only when the run's global target
   is reached** (`training_state.completed_steps >= target_steps`), refuses
   to publish a missing/unreadable/short-of-target checkpoint
   (`ArtifactPackageError`), and a packaging failure raises `TrainingError`
   — partial slices never publish "final" artifacts, and a broken package
   fails the job instead of passing silently (WP7's "never publish unless
   validated").
3. **Legacy locations are read-only migration inputs.**
   `find_live_checkpoint()` picks up a valid zip checkpoint from
   `checkpoints/dit/current`, `checkpoints/dit/breed-conditioned-v4` or
   `checkpoints/dit` when the canonical directory is empty, so the layout
   change never silently restarts an in-flight experiment. Nothing writes
   `tinydit_final.pt` anymore; scripts and workflows were moved to canonical
   paths/defaults (`checkpoints/pool/...`, `artifacts/...`).
4. **Report defaults moved to `artifacts/evaluation/`** in
   `evaluate_full.py`, `benchmark_inference.py`, `eval.py` and
   `validate_model.py` (all write sites create parent directories).
5. **Hub transport** (`gpu_pool.py`) gains an opt-in canonical mode used
   when `experiment_id` **and** `completed_steps` are passed:

   ```
   checkpoints/pool/<experiment-id>/
   ├── latest/manifest.json   # pointer: step_dir, files, updated_at — uploaded LAST
   ├── step-0060000/          # immutable snapshot: dit_model.pt,
   │                          # dit_model_ema.pt, training_state.json
   └── step-0120000/
   ```

   - **Stale guard:** if the remote pointer's `completed_steps` exceeds the
     local one, the push is rejected *before any upload*.
   - **Pointer-last commit boundary:** snapshot files upload first; the
     pointer flip is the final commit, so interrupted pushes stay invisible
     and pullers always resolve the last complete snapshot. Same-step pushes
     (model + EMA + state) merge the pointer's `files` list.
   - **Validate before activation:** pulls structurally validate the
     download (torch zip); corrupt files quarantine to `*.corrupt` and
     return `None`. Missing/invalid pointers fall back to the legacy flat
     path (transition support for pre-WP3 writers).
   - **Retries:** every Hub operation runs through `retry_utils`
     (`upload_with_retry`, exponential backoff + jitter). Unreadable pointer
     *errors* abort the push rather than risk clobbering unknown newer
     state; only a genuine 404/missing counts as "no pointer".
   - **Token hygiene:** tokens are only passed to the SDK; no log line emits
     them (covered by a test).
   - Writers: DiTTrainer final push, mid-run `hub_push_interval` push,
     `train_with_fallback` (default experiment id
     `dit-breed-conditioned-v4`, falling back to legacy when no
     `training_state.json` exists yet).
   - The legacy flat layout (`experiment_id=None`) is byte-for-byte the old
     behavior, so `scripts/test_fallback_chain.py` and the provider scripts
     keep working unchanged; migrating those entry points is WP5/WP6 work.

## Consequences

### Positive

- One answer to "where is the checkpoint/artifact": `checkpoints/pool/` and
  `artifacts/`, documented and enforced by code, with `artifacts/manifest.json`
  for bit-level verification.
- Publication is gated: a preempted slice cannot masquerade as a final
  release, and a failed ONNX export at target fails loudly.
- Cross-provider handoff becomes ordered and idempotent: immutable step
  snapshots + pointer flip mean retries/interruptions cannot produce a
  "newest" checkpoint that is half-written, and a slow provider cannot
  regress the experiment's state.
- In-flight Modal runs migrate automatically via `find_live_checkpoint()`.

### Negative / neutral

- ONNX export + artifact packaging now run only at target completion, so
  intermediate slices no longer refresh `model.onnx` (previously every run
  re-exported). Anything needing ONNX mid-experiment must run
  `export_dit_onnx.py` explicitly.
- Two Hub layouts coexist during the transition (nested per-experiment and
  legacy flat) until the provider scripts adopt `experiment_id` (WP5/WP6).
- The pointer design means `latest/` does not duplicate the multi-hundred-MB
  weights; readers must resolve the pointer (one extra small download).
- `validate_model.py` now writes `artifacts/evaluation/validation_report.json`
  by default instead of writing nothing when `--output` was omitted.

## Implementation

| File | Change |
|---|---|
| `src/artifacts.py` | New: `pool_paths`, `find_live_checkpoint`, `package_final_artifacts`, `ArtifactPackageError` |
| `src/gpu_pool.py` | Canonical push/pull with pointer, stale guard, validation, retries; legacy mode preserved |
| `src/train_dit.py` | Canonical `/outputs/checkpoints/pool` + `training.log`, legacy migration, target-gated ONNX + package, experiment-aware hub pushes/pull |
| `src/export_dit_onnx.py`, `src/verify_checkpoint.py` | Defaults → `checkpoints/pool/dit_model.pt` |
| `src/evaluate_full.py`, `src/benchmark_inference.py`, `src/eval.py`, `src/validate_model.py` | Report/sample defaults → `artifacts/evaluation/…` (parent dirs created) |
| `src/upload_to_huggingface.py`, `src/upload_to_hub.py` | Examples/defaults → `artifacts/generator/model.pt` |
| `.github/workflows/train.yml`, `upload-hub.yml`, `train-pool.yml` | `tinydit_final.pt` removed; canonical volume paths |
| `.agents/skills/model-training/SKILL.md`, `agents-docs/training.md` | Canonical paths + layout/transport docs |
| `tests/test_artifacts.py`, `tests/test_hub_transport.py` | Layout, packaging gates, Hub round trip/stale/corrupt/interrupted/retry/token matrix |

## References

- **Issue #163** — WP2 (canonical checkpoint/artifact layout), WP3 (HF Hub
  as cross-provider transport).
- **Hugging Face Hub**, *Upload files to the Hub* — atomic per-commit,
  resumable uploads: https://huggingface.co/docs/huggingface_hub/en/guides/upload
- **ADR-063** — exact global-step resume and the experiment manifest that
  this transport moves.
- **ADR-059** — prior workflow failure modes (`pipefail`, flag arrays) this
  layout complements.
