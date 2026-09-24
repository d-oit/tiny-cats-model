# Training Guides

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

This is the one canonical procedure for a complete 400k DiT run. It is built
from **bounded, resumable slices**: GitHub Actions is the control plane, real GPU
providers are the execution plane, and HuggingFace Hub is the state transport.

### 0. Local smoke (CPU, seconds)

```bash
python src/train_dit.py \
  --data-dir data/cats \
  --steps 100 \
  --batch-size 8
```

### 1. First Modal slice (bounded to fit one runner session)

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

`--steps` is a global target, never an additional-step count. The examples
below are the only correct way to advance a 400k run by hand:

```bash
modal run src/train_dit.py --data-dir /data/cats --steps 120000 --batch-size 32 --hub-resume
modal run src/train_dit.py --data-dir /data/cats --steps 180000 --batch-size 32 --hub-resume
# ... 240000 -> 300000 -> 360000 -> 400000
```

### 3. Or let GitHub Actions orchestrate every slice

```bash
# Full 400k run, 60k per provider session, sequential + fail-fast
train-pool.yml -f steps=400000 -f slice_size=60000

# Default schedule: every 6h, 25k per session (fits the runner window on a T4)
# Single verified production slice through the main workflow
gh workflow run train.yml -f steps=25000 -f batch_size=32
```

The pool workflow plans `[60000, 120000, ..., 400000]`, filters out slices the
Hub pointer already covers, and runs them with `max-parallel: 1` / `fail-fast:
true`. Each session resumes exactly to its target, so the plan is idempotent:
re-running it skips finished work.

### Which providers actually train?

| Provider | Control plane | How to run |
|----------|---------------|------------|
| Modal | ✅ `providers.py launch --provider modal` | `train-pool.yml` / `train.yml` |
| Lightning AI | ❌ unsupported (fails clearly) | `python scripts/train_lightning.py --hub-resume` |
| Kaggle | ❌ unsupported (fails clearly) | `python scripts/train_kaggle.py --hub-resume` |
| Colab | ❌ unsupported (no headless API) | run the notebook with `--resume` |
| HF Spaces | ❌ unsupported (no GPU-job API) | `python scripts/train_hf_spaces.py --hub-resume` |

Requesting an unsupported provider **exits 2 with instructions** — the control
plane never falls back to CPU simulation. Because GitHub-hosted runners
hard-cap a job at 6 hours, any single job is one bounded slice regardless of the
provider.

### Slice contract

Every provider session: identifies the experiment → pulls the newest valid Hub
checkpoint → validates it against the immutable manifest → reads the global step
→ trains only to the requested target → pushes `step-XXXXXX/` snapshots plus a
`latest/` pointer → handles SIGTERM/SIGINT with a usable checkpoint → writes a
machine-readable provider report.

### Recovery

| Situation | What happens / what to do |
|-----------|---------------------------|
| Provider preemption | Checkpoint survives; the slice reports `partial`; re-run the same command or let the next scheduled pool run resume from the Hub pointer |
| GitHub Actions cancellation | `hub_push_interval` has already synced a mid-run checkpoint; `train-pool.yml` re-plans from the pointer's `completed_steps` |
| Modal timeout | Slice reports `partial` with the last completed step; lower `slice_size` and continue |
| Corrupt checkpoint | Quarantined to `*.corrupt` locally; a corrupt Hub download is quarantined and validation returns `None` so the previous valid snapshot is used |
| Missing Hub checkpoint | First run (nothing to resume); training starts at step 0 |
| Provider handoff | Hub is the only state transport; the next provider resumes from `latest/` regardless of which provider wrote it |
| Final target already reached | The run is a successful no-op: 0 steps, checkpoint bytes untouched, `exit_reason: completed` |
| Workflow failed but log looks green | Check the `provider-report-<target>.json` artifact and `exit_reason`; training output is piped through `tee` under `set -o pipefail`, so a failed `modal run` fails the job |
| `IncompatibleExperimentError` on every slice | `--hub-resume` now prefers the Hub checkpoint, so a stale file left on the Modal volume by an unrelated run can no longer shadow the experiment. If it still fails, the *Hub* checkpoint itself belongs to a different experiment: start a new `--experiment-id` (or re-run with the original hyperparameters) rather than forcing `--allow-experiment-mismatch`, which would silently continue with a different LR horizon |

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

Train across multiple free GPU providers with automatic checkpoint sync:

```bash
# Check provider status and cost estimates
python -c "from gpu_pool import estimate_cost; print(estimate_cost(100000))"

# Train on current provider with Hub checkpoint sync
python -c "from gpu_pool import train_with_fallback; train_with_fallback(steps=50000)"

# Print fallback chain
python -c "from gpu_pool import train_chain; train_chain(steps=20000)"
```

Provider scripts with Hub sync:
```bash
python scripts/train_lightning.py --steps 20000 --hub-resume  # Lightning AI
python scripts/train_colab.py --steps 20000 --resume           # Google Colab
python scripts/train_kaggle.py --steps 20000 --hub-resume      # Kaggle
python scripts/train_hf_spaces.py --steps 20000 --hub-resume   # HF Spaces
```

Pool CI workflow (GitHub Actions = control plane, ADR-065):
```bash
# Plan bounded slices toward the global target (default 400k, 25k/session)
gh workflow run train-pool.yml -f steps=400000 -f slice_size=60000

# Unsupported providers FAIL CLEARLY at the gate (never silent CPU training)
gh workflow run train-pool.yml -f provider=kaggle   # exits 2 with instructions

# Control-plane primitives (unit-tested; used by train-pool.yml)
python src/providers.py plan   --steps 400000 --slice-size 60000
python src/providers.py gate   --provider lightning --strict
python src/providers.py launch --provider modal --target 60000
```

Each slice is one bounded provider session that resumes exactly to its
global target (`--steps` is never additive), pushes checkpoints to the Hub
pool, and uploads a machine-readable `provider-report-<target>.json`
(provider, GPU, VRAM, job id, start/end, completed step, checkpoint URI,
exit reason).

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
