# Training Guides

## Issue #163 delivery status (WP1–WP9)

All nine hardening work packages from
[issue #163](https://github.com/d-oit/tiny-cats-model/issues/163) are merged:

| WPs | PR | What landed |
|-----|----|-------------|
| WP1 + WP4 | #164 | Exact global-step resume, `training_state.json`, immutable experiment manifest |
| WP2 + WP3 | #165 | Canonical checkpoint/artifact layout, HF Hub pool transport |
| WP5 + WP6 | #166 | `providers.py` control plane (plan/gate/launch/verify), bounded resumable slices |
| WP7–WP9 | #167 | Publication gates (`verify_checkpoint.py`), end-to-end pipeline verification, production 400k runbook |

**Acceptance:** 17 of the 20 checklist items in issue #163 are verified by
unit/integration tests and documentation; the quality gate is green on `main`
(format, ruff, actionlint, yamllint, mypy, pytest, fallback-chain simulation,
and `scripts/verify_training_pipeline.py` incl. quantized ONNX).

**Production run (2026-09-25): the global 400k target is reached.** The
scheduled pool slice reported a no-op at the target and `providers.py verify`
returned `completed_steps=400000, reached_target=true, checkpoint_valid=true`
(run `36160359407`); Modal executed the real GPU slices that got it there.
The last two checklist items — generated evaluation/benchmark reports and a
reproducible final artifact manifest — close with the completion chain, which
packages the artifacts from the volume, verifies them (ONNX incl.), generates
`evaluation_report.json` + `benchmark_report.json`, and publishes to the Hub.
It no-ops training because the checkpoint is already at the target:

```bash
gh workflow run train.yml -f steps=400000 -f push_to_hub=true
```

**Completion chain result (run `36176647013`, 2026-09-25): all steps green.**
The package verified (`model.pt` @ step 400,000 + ONNX + quantized ONNX via
`verify_checkpoint.py --require-onnx`), both reports were generated and
published to [`d4oit/tiny-cats-model`](https://huggingface.co/d4oit/tiny-cats-model)
(`evaluation/evaluation_report.json`, `evaluation/benchmark_report.json`), and
the Hub publication was verified. Benchmark (CPU, PyTorch):

| Metric | batch=1 | batch=4 | batch=8 |
|--------|---------|---------|---------|
| Latency p50 / p95 / p99 | 34.67 / 35.60 / 42.95 ms | — | — |
| Throughput | 28.1 img/s | 43.7 img/s | 49.4 img/s |

A further training slice is started with the pool workflow (only needed for a
*higher* target — an already-complete target is a successful no-op):

```bash
gh workflow run train-pool.yml -f steps=400000 -f slice_size=25000
```

## Modal GPU Training (Class-based with @modal.enter())

Training scripts now use the `@app.cls` + `@modal.enter()` pattern (ADR-025, ADR-057):
- Container init (CUDA warm-up, path setup) runs ONCE per container via `@modal.enter()`
- Training runs via `@modal.method()` without re-initializing on each call
- `scaledown_window=300` keeps containers warm for 5 min between runs

### Running Training

### Classifier (train.py)
```bash
# Modal GPU training
modal run src/train.py data/cats --epochs 20 --batch-size 64

# Local CPU testing (debug)
python src/train.py --data-dir data/cats --epochs 1 --batch-size 8
```

### DiT Generator (train_dit.py)
```bash
# Modal GPU training (100k steps, early stopping)
modal run src/train_dit.py --data-dir /data/cats --steps 100000 --batch-size 512

# Long run (explicit horizon, gradient accumulation)
modal run src/train_dit.py --data-dir /data/cats \
  --steps 400000 \
  --batch-size 256 \
  --gradient-accumulation-steps 2 \
  --augmentation-level full

# Local CPU testing
python src/train_dit.py --data-dir data/cats --steps 100 --batch-size 8

# YAML config (applied as defaults; explicit flags still override)
python src/train_dit.py --data-dir data/cats --config configs/dit_train_config.yaml
```

## Training Options

### Classifier (train.py)
| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 10 | Number of epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--backbone` | resnet18 | Model backbone |
| `--output` | cats_model.pt | Output checkpoint |
| `--no-pretrained` | false | Disable pretrained |

### DiT (train_dit.py)
| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | required | Dataset root (`/data/cats` inside Modal) |
| `--steps` | 100,000 | Global target steps (resume = `max(0, steps − completed)`) |
| `--batch-size` | 512 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--gradient-accumulation-steps` | 1 | Effective batch = batch × steps |
| `--augmentation-level` | full | basic/medium/full |
| `--warmup-steps` | 2,000 | LR warmup steps |
| `--min-lr` | 1e-6 | LR floor for the cosine decay |
| `--schedule-steps` | = `--steps` | Campaign-wide LR-cosine horizon; slices never shorten it (`train-pool` passes the global target) |
| `--val-split` | 0.05 | Held-out fraction for selection/early stopping (0 disables) |
| `--val-batches` | 8 | Validation batches per evaluation (0 = all) |
| `--timestep-sampling` | uniform | `uniform` or `logit_normal` |
| `--logit-normal-mean` | 0.0 | Mean of the logit-normal sampler |
| `--logit-normal-std` | 1.0 | Std of the logit-normal sampler |
| `--save-interval` | 10,000 | Checkpoint frequency (Modal default: 500) |
| `--sample-interval` | 5,000 | Sample generation frequency |
| `--config` | - | YAML config applied as defaults |
| `--experiment-id` | dit-breed-conditioned-v4 | Manifest identity; resumes must match |
| `--allow-experiment-mismatch` | false | Override manifest rejection (explicit migration) |

Checkpoint selection and early stopping use the held-out validation loss (EMA
weights when available). The training-batch average is only a fallback when
`--val-split 0` is set.

### Exact global-step resume & experiment manifest (issue #163)

`--steps` is a **global target**, never "train this many additional steps":n
```bash
# Slice 1: trains 0 -> 60,000
modal run src/train_dit.py --data-dir /data/cats --steps 60000 --batch-size 32

# Slice 2 (same experiment, e.g. after preemption): performs exactly
# max(0, 400000 - 60000) = 340,000 more steps — not another 400,000
modal run src/train_dit.py --data-dir /data/cats --steps 400000 --batch-size 32

# Target already reached? Exits 0 without training or touching the checkpoint
modal run src/train_dit.py --data-dir /data/cats --steps 60000
```

State written with every checkpoint:

- **Embedded in `*.pt`**: manifest, completed/target steps, seed, optimizer,
  EMA, and a torch/python/numpy/CUDA RNG snapshot restored on resume.
- **`training_state.json`** beside the checkpoint: the manifest flattened to
  top level plus `completed_steps` / `target_steps`, written atomically so
  tooling can inspect a run without unpickling the checkpoint.

Resume rules:

| Situation | Behaviour |
|-----------|-----------|
| Manifest matches | Resume; performs exactly `max(0, target − completed)` steps |
| Target raised (60k → 400k) | Allowed — slices raise the global target |
| Architecture / dataset / seed / optimizer-critical change | `IncompatibleExperimentError` (pass `--allow-experiment-mismatch` to override) |
| Checkpoint at/past target | Successful no-op; checkpoint bytes untouched |
| Corrupt / truncated checkpoint | Quarantined to `*.corrupt`, restart from 0 (ADR-058) |

See ADR-063 for the full field list and rationale.

### Canonical checkpoint & artifact layout (issue #163 WP2/WP3)

Layout under the outputs root (`/outputs` on Modal, repo root locally):

```
checkpoints/pool/            # live checkpoint (canonical)
├── dit_model.pt
├── dit_model_ema.pt
├── training_state.json
└── training.log
artifacts/
├── generator/               # model.pt, model_ema.pt, model.onnx, model_quantized.onnx
├── training/                # final_checkpoint.pt, training_state.json, training.log
├── evaluation/              # evaluation_report.json, benchmark_report.json, samples/
└── manifest.json            # sha256 manifest of the package
```

- The final `artifacts/` package is built **only when the run's global
  target is reached**; partial slices never publish final artifacts, and a
  broken package fails the job (`ArtifactPackageError` → `TrainingError`)
  instead of passing silently.
- Legacy locations (`checkpoints/dit/current`,
  `checkpoints/dit/breed-conditioned-v4`, `tinydit_final.pt`) are read-only
  migration inputs — resume picks them up when the canonical directory is
  empty, but nothing writes them anymore.
- Report-writing scripts default to `artifacts/evaluation/…`.

HF Hub is the cross-provider checkpoint transport (WP3):

```
checkpoints/pool/<experiment-id>/
├── latest/manifest.json     # pointer: step, files, updated_at (uploaded LAST)
├── step-0060000/            # immutable snapshot: dit_model.pt,
│                            # dit_model_ema.pt, training_state.json
└── step-0120000/
```

- The pointer flip is the commit boundary: interrupted pushes stay
  invisible and pullers always resolve the newest committed snapshot.
- A push whose completed step is below the remote pointer is **rejected**
  (a stale provider result never overwrites a newer checkpoint).
- Pulls validate the download (torch zip) before activation; corrupt files
  quarantine to `*.corrupt`.
- Transient Hub errors retry with exponential backoff (`retry_utils`).
- The legacy flat `checkpoints/pool/<name>` layout stays readable when
  `experiment_id` is not passed (transition support).

### Comparing configurations

`scripts/tune_dit_configs.py` trains several arms and scores them all on one
common held-out metric (uniform timesteps), so arms that differ in
`--timestep-sampling` stay comparable:

```bash
python scripts/tune_dit_configs.py --data-dir data/cats --steps 300 \
  --arms label=uniform,timestep_sampling=uniform \
  --arms label=logit,timestep_sampling=logit_normal --json-out tune.json
```

## Production 400k Runbook (issue #163 WP9)

This section records the completed 400k milestone and the bounded-slice
workflow used for it. The manual Modal examples below are a separate,
potentially paid path; the free-only `train-pool.yml` workflow rejects Modal.

### 0. Local smoke (CPU, seconds)

```bash
python src/train_dit.py \
  --data-dir data/cats \
  --steps 100 \
  --batch-size 8
```

### 1. Legacy Modal slice (separate from the free-only pool)

```bash
modal run src/train_dit.py \
  --data-dir /data/cats \
  --steps 60000 \
  --batch-size 32 \
  --lr 5e-5 \
  --warmup-steps 2000 \
  --save-interval 5000 \
  --hub-push-interval 5000 \
  --hub-resume
```

### 2. Continue the SAME experiment with increasing GLOBAL targets

The examples below are manual Modal commands, not the free-only pool
workflow. `--steps` is a global target, never an additional-step count:

```bash
modal run src/train_dit.py --data-dir /data/cats --steps 120000 --batch-size 32 --hub-resume
modal run src/train_dit.py --data-dir /data/cats --steps 180000 --batch-size 32 --hub-resume
# ... 240000 -> 300000 -> 360000 -> 400000
```

### 3. Or let GitHub Actions orchestrate every slice

```bash
# Free-only bounded pool; each provider session targets at most 15k steps
gh workflow run train-pool.yml \
  -f steps=615000 \
  -f slice_size=15000 \
  -f provider=auto
```

campaign it reports `campaign_complete=true` and `slice_count=0`; the matrix
keeps a nonempty sentinel entry, but both the provider gate and train job are
skipped and no provider session is launched.

`provider=auto` tries Kaggle, then Lightning only when the operator has
explicitly verified the Lightning account's free-only billing protections.
Fallback occurs only on a definite prelaunch refusal. An ambiguous submission
or a job that started and failed stops the campaign; it is never retried on a
second provider. For a plan that requires training, `provider=modal` and
`provider=all` are rejected by this free-only workflow. An already-complete
target is a successful no-op and skips the provider gate.

Lightning requires an explicit teamspace (`LIGHTNING_TEAMSPACE`), the
`LIGHTNING_API_KEY` secret, and the `LIGHTNING_FREE_ONLY_VERIFIED=true`
repository variable. Set the attestation only after confirming that
promotional credits are the sole funding source, auto-reload is off, no
purchased/subscription credits or BYOC billing can fund the job, and exhausted
credits stop the workload without a paid charge. Otherwise leave it unset; the
adapter blocks before loading the SDK or submitting.

Each successful slice must advance the canonical pointer beyond its recorded
pre-slice step. Checkpoint verification accepts the requested target or an
explicitly recorded convergence at the actual completed step.

### Slice contract

Every provider session: identifies the experiment → pulls the newest valid Hub
checkpoint → validates it against the immutable manifest → reads the global step
→ trains only to the requested target → pushes `step-XXXXXX/` snapshots plus a
`latest/` pointer → handles SIGTERM/SIGINT with a usable checkpoint → writes a
machine-readable provider report.

### Recovery

| Situation | What happens / what to do |
|-----------|---------------------------|
| Provider preemption | Checkpoint survives; the slice reports `partial`; re-run only after checking the Hub pointer and the provider report |
| GitHub Actions cancellation | The canonical mid-run push remains available; the next plan reads the Hub pointer's `completed_steps` |
| Kaggle or Lightning timeout | Slice reports partial progress; inspect the provider report and canonical pointer before continuing |
| Corrupt checkpoint | Quarantined to `*.corrupt` locally; a corrupt Hub download is quarantined and validation returns `None` so the previous valid snapshot is used |
| Missing Hub checkpoint | A new campaign may start at step 0; a resume requires an available canonical snapshot |
| Provider handoff | Hub is the only state transport; the next eligible provider resumes from `latest/` regardless of which provider wrote it |
| Final target already reached | The plan reports `campaign_complete=true`, `slice_count=0`, and skips the train job without launching a GPU |
| Required Hub pointer lookup fails | Planning fails closed; no matrix is emitted and no provider can launch |
| Workflow failed but log looks green | Check `provider-report-<target>.json`, the before/after pointer evidence, and `exit_reason`; a failed launch step fails the job |
| `IncompatibleExperimentError` on every slice | Check the canonical Hub checkpoint's experiment manifest and dataset identity; start a new experiment or restore the matching settings instead of forcing a mismatch |

### Publication gate

A "final model" is published only when all of these hold:

1. `training_state.json` records `completed_steps >= target_steps`.
2. The checkpoint is a valid torch (zip) archive — `python src/providers.py verify`.
3. `artifacts/generator/{model.pt,model.onnx,model_quantized.onnx}` + `manifest.json` exist.
4. ONNX Runtime inference succeeds and evaluation/benchmark reports are written.
5. The Hub upload is listed back afterwards.

Anything short of the target reports `partial` and publishes nothing.

## Error Handling & Logging

### Pre-flight Checks
- Auth validation before training starts
- Clear error messages for auth failures

### Structured Logging
- Console + file with timestamps
- Logs in: `/outputs/checkpoints/*/training.log`

### Cleanup
- Volume commit after successful training
- Volume commit on error (partial state saved)
- Old checkpoints auto-cleaned (keep last 5)

## GPU Selection

| GPU | Best For | Cost |
|-----|----------|------|
| T4 | Classifier, DiT (cost-optimized) | Low ($0.59/hr) |
| L4 | DiT fallback | Low ($0.80/hr) |
| A10G | DiT training (if preemption is critical) | Medium ($1.10/hr) |
| L40S | Non-spot DiT training | High ($1.95/hr) |
| A100 | Large models | High ($2.10/hr) |

## Free GPU Pool Training

`train-pool.yml` is the free-only control plane. It never trains on the GitHub
runner and rejects paid-provider selections:

```bash
# Free-only default: Kaggle, then Lightning only after its eligibility gate
gh workflow run train-pool.yml \
  -f steps=615000 \
  -f slice_size=15000 \
  -f provider=auto

# A completed target plans successfully but launches no provider job
gh workflow run train-pool.yml \
  -f steps=600000 \
  -f slice_size=15000 \
  -f provider=auto

# Modal and "all" are rejected; do not use them in this workflow.
```

Auto fallback is limited to definite prelaunch refusals. Once a provider may
have accepted a job, or training has started, failure stops the campaign to
avoid duplicate GPU work. Every successful slice must advance the canonical
Hub pointer; early stopping is complete only when the checkpoint state records
convergence for that slice's target.

The operational workflow runs only on `main` and shares the pool concurrency
group. Inspection and EMA retrieval do not launch a GPU. The `inspect` action
also reads the current Kaggle kernel status without modifying it; if it reports
`QUEUED`, `RUNNING`, or `NEW_SCRIPT`, do not dispatch another slice until the
kernel reaches a terminal status.

```bash
# List membership candidates; confirm the exact owner/teamspace in the account UI
gh workflow run gpu-pool-check.yml --ref main -f action=inspect

# Fetch the canonical EMA plus step/SHA evidence as a seven-day artifact
gh workflow run gpu-pool-check.yml --ref main -f action=fetch-ema
```

`action=lightning-smoke` is not a generic probe. It is permitted only after
the account's free-only protections have been verified and
`LIGHTNING_FREE_ONLY_VERIFIED=true` is deliberately set. Never infer
eligibility from the presence of credits or from a successful teamspace lookup.

Control-plane planning and provider gates (no training launch):

```bash
python src/providers.py plan --steps 615000 --slice-size 15000
python src/providers.py gate --provider auto --strict
```

Each provider session is bounded, resumes from the canonical Hub pointer, and
uploads a machine-readable `provider-report-<target>.json` with provider,
GPU, job id, timestamps, completed step, checkpoint URI, and exit reason.

## Verification

```bash
# Test setup (no import errors)
modal run src/train_dit.py --help

# Verify checkpoint
python src/verify_checkpoint.py --checkpoint checkpoints/pool/dit_model.pt

# Export and test ONNX
python src/export_dit_onnx.py --verify --test

# Control plane: are the provider artifacts good enough to publish?
python src/providers.py verify \
  --state-file checkpoints/pool/training_state.json \
  --checkpoint checkpoints/pool/dit_model.pt \
  --target 60000
# exit 0 = target reached, 3 = valid but partial, 1 = missing/corrupt

# End-to-end pipeline: CPU smoke + exact resume (10 -> 20) + ONNX export +
# ONNX Runtime inference + quantized ONNX + artifact package/refusal gates
python scripts/verify_training_pipeline.py
python scripts/verify_training_pipeline.py --stage export   # one stage only
```

## Common Issues

| Issue | Solution |
|-------|----------|
| AuthError | Run `modal token new` |
| OOM | Reduce batch-size or use gradient accumulation |
| CUDA error | Use `--device cpu` |
| Import errors | Check files in Modal container |

## Testing

```bash
# Train chain and fallback integration tests
pytest tests/test_train_chain.py -v

# Fallback chain end-to-end simulation
python scripts/test_fallback_chain.py

# GPU hour estimate calibration and drift check
python scripts/benchmark_estimates.py
python scripts/benchmark_estimates.py --tune

# CI runs the drift check automatically (warns if error > 75%)
```

## References

- [GPU Pool Abstraction](../src/gpu_pool.py) — multi-provider training with HF Hub checkpoint sync
- [Train Chain Tests](../tests/test_train_chain.py)
- [Model Training Skill](../.agents/skills/model-training/SKILL.md)
- [ADR-057: Modal CLI Verification & Best Practices](../plans/ADR-057-modal-cli-verification-and-best-practices-2026.md)
- [ADR-025: Cold Start Optimization](../plans/ADR-025-modal-cold-start-optimization.md)
- [ADR-058: GPU Selection & Cost Optimization](../plans/ADR-058-dit-l40s-non-spot-and-save-interval.md)
- [ADR-063: Exact Global-Step Resume & Experiment Manifest](../plans/ADR-063-exact-global-step-resume-and-experiment-manifest.md)
- [ADR-064: Canonical Artifact Layout & HF Hub Pool Transport](../plans/ADR-064-canonical-artifact-layout-and-hub-pool-transport.md)
