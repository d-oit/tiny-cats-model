# ADR-066: Production training workflows — verified slices and honest failures

- **Status:** Accepted (2026-09-24)
- **Date:** 2026-09-24
- **Deciders:** tiny-cats-model maintainers
- **Related:** issue #163 (WP7, WP8, WP9), `.github/workflows/train.yml`,
  `.github/workflows/train-pool.yml`, `.github/workflows/ci.yml`,
  `src/providers.py`, `scripts/verify_training_pipeline.py`,
  `scripts/quality-gate.sh`, `agents-docs/training.md`,
  `.agents/skills/model-training/SKILL.md`,
  ADR-057 (Modal CLI verification), ADR-059 (silent train-pool failures),
  ADR-063 (exact resume + manifest), ADR-064 (canonical artifacts + Hub
  transport), ADR-065 (control plane + slices)

## Context

WP1–WP6 made the *training contract* correct: exact global-step resume, an
immutable experiment manifest, a canonical checkpoint/artifact layout, an
ordered HF Hub transport, and a control plane that plans bounded slices and
refuses to fake GPU providers. What was still missing for production was the
**publication contract** in the workflows themselves:

1. **A secret-gated step that never ran.** `train.yml` published with
   `if: env.HF_TOKEN != ''`. GitHub evaluates that condition against the
   *job/workflow* `env` context, and the `secrets` context is not allowed in
   `if`; `HF_TOKEN` was only set as a *step* env, so the condition was always
   false. Both classifier and DiT publication steps silently did nothing —
   the same "green job, no work done" class as ADR-059.
2. **Exit codes papered over by `|| true`.** Evaluation, benchmarking, ONNX
   download and the upload fallback all swallowed failures, so a broken run
   could still be reported as a success.
3. **A timeout that could not exist.** `train-dit` declared
   `timeout-minutes: 1440`, but GitHub-hosted runners hard-cap a job at 6h
   (360 min). The declared horizon was fiction, and a single dispatch job was
   implicitly expected to cover a 100k-step run at batch 128.
4. **Publication without verification.** Nothing re-read `training_state.json`
   or proved the checkpoint was loadable before publishing; a preempted slice
   could still overwrite the published generator.
5. **An unsupported provider could still train.** `train-slice` hard-coded
   `--provider modal` and did not depend on `provider-gate`, so
   `-f provider=kaggle` ran the gate *and* launched a Modal slice.
6. **No single automated end-to-end verification**, and no documentation of the
   canonical 400k procedure.

## Decision

### 1. GitHub Actions never publishes unverified artifacts (WP7)

`src/providers.py` gains a `verify` subcommand — control-plane logic stays in
unit-tested Python, not shell (ADR-065):

```
python src/providers.py verify --state-file training_state.json \
  --checkpoint checkpoints/pool/dit_model.pt --target 60000
```

`verify_checkpoint()` returns a `CheckpointVerification` and exits:

| code | meaning |
|------|---------|
| `0` | checkpoint is a valid torch zip **and** `completed_steps >= target` |
| `3` | valid but short of target (`partial`) |
| `1` | missing/corrupt checkpoint, unreadable state, or nothing to verify |

The workflows branch on that code instead of parsing logs:

- `1` → hard failure (`::error::`), because the next slice would otherwise
  resume from stale state.
- `3` → `::warning::` and publication is skipped; the checkpoint is resumable
  by design.
- `0` → the artifact download/publish steps run.

`train-pool.yml` verifies every slice's checkpoint; `train.yml` verifies the
single slice before packaging.

ONNX verification needed the same treatment: `verify_checkpoint.py` treated a
failing ONNX Runtime check as a warning, so a broken export could still pass the
publish gate. It gains `--require-onnx`, which `train.yml` now passes — a
checkpoint that loads but whose ONNX cannot run is a failure, not a note.

### 2. Secrets are exposed through job `env`, not `if`

Both training jobs set `env: HF_TOKEN: ${{ secrets.HF_TOKEN }}` (plus the Modal
tokens) at **job** level so `if: env.HF_TOKEN != ''` actually evaluates. This is
the only supported way to condition a step on a secret being present.

### 3. `pipefail` + `tee` everywhere, `|| true` only for genuinely optional work

Every step whose output is piped through `tee` starts with `set -o pipefail`
(ADR-059's regression guard). Remaining tolerances are explicit and narrow:

- a missing optional artifact logs `::warning::` and names the artifact;
- a *required* final artifact (`model.pt`, `model.onnx`,
  `model_quantized.onnx`, `manifest.json`) fails the job with `::error::`;
- evaluation and benchmark reports are required once the target is reached —
  the previous `|| true` hid broken runs.

### 4. Timeouts reflect reality: one job = one bounded slice

`train-dit` is now `timeout-minutes: 350` (GitHub's hosted-runner cap is 6h),
defaults to `steps: 25000` / `batch_size: 32` (a slice that fits the window at
the measured T4 throughput), and documents that longer horizons continue via
`train-pool.yml`. `train.yml` also launches through `providers.py launch`
instead of a hand-written `modal run`, so `train.yml` and `train-pool.yml`
share one launch recipe (canonical `checkpoints/pool/` paths, `--experiment-id`,
Hub flags).

`build_launch_command()` gained optional `--warmup-steps`,
`--gradient-accumulation-steps` and `--early-stopping-patience` flags; they are
only emitted when requested, so the pinned default command is unchanged.
Because GitHub allows at most 10 `workflow_dispatch` inputs, the remaining DiT
knobs stay in `providers.py` rather than becoming inputs.

### 5. An unsupported provider cannot reach a slice

`train-slice` now declares `needs: [plan-slices, provider-gate]` and runs only
when the gate was **skipped** — i.e. only when modal was actually requested.
`-f provider=all` inspects providers without training, and
`-f provider=kaggle` fails at the gate with instructions. `pool-summary`
distinguishes `provider gate failed` / `no slice ran` / `slices completed` /
`slices failed` instead of printing a vacuous success, and skips Hub
verification when `push_to_hub` was false.

### 6. One end-to-end verification pipeline (WP8)

`scripts/verify_training_pipeline.py` runs on CPU with no network:

| stage | asserts |
|-------|---------|
| `smoke` | transforms/breed mapping (13 conditioning labels), DiT build, forward shape, backward reaches every parameter, checkpoint round trip |
| `resume` | **real** `train_dit_local` loop: `0 -> 10`, then `10 -> 20` performs exactly 10 more steps |
| `export` | ONNX export, ONNX Runtime inference, dynamic quantization, quantized inference, smaller file |
| `package` | `package_final_artifacts` manifest lists generator + EMA + state files, and a checkpoint below `require_completed` is **refused** |

Provider-boundary simulation (Modal → HF Hub → Lightning → HF Hub → Modal)
stays in the pytest suites (`tests/test_hub_transport.py`,
`tests/test_train_chain.py`, `scripts/test_fallback_chain.py`). The new script
runs in `ci.yml` and as step 9 of `scripts/quality-gate.sh`.

### 7. One documented 400k procedure (WP9)

`README.md`, `agents-docs/training.md` and
`.agents/skills/model-training/SKILL.md` document the same canonical procedure:
local smoke → one bounded Modal slice → continue with **increasing global
targets** → or let `train-pool.yml` slice automatically; which providers are
launchable versus manual; the slice contract; the publication gate; and a
recovery table (preemption, cancellation, Modal timeout, corrupt checkpoint,
missing Hub checkpoint, provider handoff, target already reached,
green-log-but-failed run).

## Consequences

### Positive

- A failed run can no longer present as successful: exit codes propagate, logs
  are preserved, and `exit_reason` is recorded per session.
- Publication is gated on evidence (valid checkpoint + reached target + valid
  ONNX), so a preempted slice cannot overwrite the published generator.
- `train.yml` and `train-pool.yml` share one launch recipe and one verification
  command; fixing a provider or a flag is a one-place change.
- Requesting an unsupported provider is a fast, explicit failure.
- The export/package half of the pipeline is covered automatically in CI.

### Negative / neutral

- `train.yml`'s dispatch defaults changed (`steps` 100k → 25k, `batch_size`
  128 → 32) to fit a real 6h hosted-runner session. Longer targets must be
  reached by repeated slices or by `train-pool.yml`.
- Four low-value `workflow_dispatch` inputs were removed to stay under
  GitHub's 10-input limit; those knobs are now constants in `providers.py`.
- Evaluation/benchmark generation now fails the job when it is required,
  which is noisier than the old `|| true` — deliberately so.
- `optimize_onnx`'s generator validator is pinned to 128px inputs, so the
  verification script quantizes without its built-in accuracy check and
  verifies the quantized model directly instead.

## References

- **Issue #163** — WP7 (hardened workflows), WP8 (end-to-end verification),
  WP9 (documented 400k procedure).
- **ADR-059** — the `tee`-masked failure this ADR guards against.
- **ADR-065** — control plane, bounded slices, provider gate.
- **GitHub Docs**, *Using secrets in GitHub Actions* — secrets are not
  available in `if` conditions; job `env` is the supported pattern:
  https://docs.github.com/actions/security-guides/using-secrets-in-github-actions
- **GitHub Docs**, *Usage limits* — GitHub-hosted jobs are limited to 6 hours:
  https://docs.github.com/actions/reference/limits
