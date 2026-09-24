# AGENTS.md

AI agent guidance for tiny-cats-model (cat image classification with DiT).

## Quick Commands

```bash
# Install & setup
pip install -r requirements.txt && bash data/download.sh

# Training (Modal GPU) - Optimized (100k steps with early stopping)
bash scripts/train_dit_high_accuracy.sh

# Local testing
python src/train_dit.py --data-dir data/cats --steps 100 --batch-size 8

# Quality gate
bash scripts/quality-gate.sh
```

## Training

### Modal GPU Training

```bash
# Classifier (resnet18)
modal run src/train.py --data-dir /data/cats --epochs 20 --batch-size 64

# DiT Generator (optimized - 100k with early stopping)
modal run src/train_dit.py --data-dir /data/cats --steps 100000 --batch-size 512

# Custom configuration
modal run src/train_dit.py --data-dir /data/cats --steps 50000 --batch-size 512 --lr 5e-5 --warmup-steps 2000

# YAML config (applied as defaults; explicit flags still win)
python src/train_dit.py --data-dir data/cats --config configs/dit_train_config.yaml
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--steps` | 100,000 | Global target steps — a resume performs `max(0, steps − completed)` (issue #163) |
| `--batch-size` | 512 | Batch size (increased for better gradients) |
| `--lr` | 5e-5 | Learning rate |
| `--warmup-steps` | 2,000 | LR warmup (shorter = faster convergence) |
| `--min-lr` | 1e-6 | LR floor for the cosine decay |
| `--val-split` | 0.05 | Held-out fraction used for checkpoint selection (0 disables) |
| `--timestep-sampling` | uniform | `uniform` or `logit_normal` (mid-trajectory focus) |
| `--augmentation-level` | full | basic/medium/full |
| `--experiment-id` | dit-breed-conditioned-v4 | Manifest identity; resumes must match |
| `--allow-experiment-mismatch` | false | Override manifest rejection (explicit migration) |

> **Resume semantics (issue #163):** `--steps` is a *global* target — resuming
> a checkpoint at 60k with `--steps 400000` performs exactly 340,000 more
> steps; an already-complete target is a successful no-op that never
> overwrites the checkpoint. Every checkpoint embeds an immutable experiment
> manifest (architecture, dataset hash, breed mapping, optimizer-critical
> settings, seed), mirrored to `training_state.json` beside it; a resume whose
> manifest differs fails clearly unless `--allow-experiment-mismatch` is
> passed. Corrupt checkpoints quarantine to `*.corrupt` (ADR-058);
> architecture-incompatible ones are rejected, never silently restarted.

> **Modal paths:** use absolute container paths (`/data/cats`, `/outputs/...`) — relative
> `data/cats` only exists on the local machine, and Modal 1.0+ requires `--data-dir`
> (positional args fail with `Got unexpected extra argument`). See ADR-048/ADR-051/ADR-054.
> Selection and early stopping use the held-out validation loss (EMA weights when available),
> not the augmented training-batch average.

### Early Stopping

Training automatically stops when loss plateaus for 3 consecutive evaluations (every 10k steps). This typically occurs at 50k-80k steps, saving 60-80% cost.

## GitHub Actions

Production training runs as **bounded, resumable slices** (ADR-065/066).
GitHub Actions is the control plane — it plans slices, launches real GPU sessions
and verifies artifacts; it never trains on CPU. `--steps` is always the *global*
target, and GitHub-hosted runners cap a job at 6h, so one job is one slice.

```bash
# Single verified production slice (Modal, <= 6h)
gh workflow run train.yml -f steps=25000 -f batch_size=32

# Full 400k run as bounded slices (sequential, fail-fast, resumable)
gh workflow run train-pool.yml -f steps=400000 -f slice_size=60000

# Unsupported providers (kaggle/lightning/colab/hf_spaces) fail clearly at the gate
gh workflow run train-pool.yml -f provider=kaggle

# Control plane without launching anything
python src/providers.py gate   --provider all
python src/providers.py plan   --steps 400000 --slice-size 60000
python src/providers.py verify --state-file checkpoints/pool/training_state.json \
  --checkpoint checkpoints/pool/dit_model.pt --target 60000

# Monitor runs
gh run list
gh run view <run-id>
gh run watch

# Check secrets
gh secret list
```

> **Note:** The `Speed: X steps/s` log line reports the current logging interval, not
> wall-clock throughput. Measured T4 throughput on 2026-09-12 was 1.16 steps/s at batch 32,
> 0.45 at batch 64, and 0.17 at batch 128 (128x128, AMP) — the planning baseline
> `gpu_pool.T4_STEPS_PER_SECOND = 0.06` (batch-512-equivalent) reflects those and is
> confirmed by `python scripts/benchmark_estimates.py --tune`. Also: scheduled
> `train-pool.yml` runs before 2026-09-12 exited immediately without training (flags were
> passed as a single argument, masked by a missing `pipefail`), so always check
> `modal-training.log` for real training output. See
> `plans/ADR-057-modal-cli-verification-and-best-practices-2026.md` for the container
> overhead diagnosis and mitigation options (`@modal.enter`, `single_use_containers=True`,
> larger `save_interval`, GH-Action container reuse).

## Authentication

### Modal (1.0+)
```bash
modal token new          # Configure (NOT 'token set')
modal token info         # Verify
```

### HuggingFace
```bash
# Local
export HF_TOKEN=hf_xxx

# GitHub Secrets
gh secret set HF_TOKEN --body "hf_xxx"
```

Generate token: https://huggingface.co/settings/tokens (write permission)

## Code Style

- Ruff linting + formatting (88 char line)
- Type hints required
- 500 LOC max per file

## Testing

```bash
# All tests (unit + GPU pool + train chain simulation)
pytest tests/ -v

# GPU pool abstraction tests
pytest tests/test_gpu_pool.py -v

# Eval metric math (confusion matrix, P/R/F1, aggregate F1) + report regression
# evaluation_report.json is the golden source; a retrain regenerating it
# auto-refreshes the report-driven regression test (no constant edits).
pytest tests/test_eval.py -v

# Slow real-data eval (self-skips if data/cats or checkpoint absent)
pytest tests/test_eval.py -m slow

# Train chain & fallback integration tests
pytest tests/test_train_chain.py -v

# Fallback chain simulation (standalone)
python scripts/test_fallback_chain.py

# End-to-end training pipeline: CPU smoke + exact resume (10 -> 20) + ONNX
# export/runtime inference + quantized ONNX + artifact package/refusal gates
python scripts/verify_training_pipeline.py

# GPU hour estimation calibration
python scripts/benchmark_estimates.py
python scripts/benchmark_estimates.py --tune

# E2E (Playwright)
npx playwright test

# Full verification
bash scripts/quality-gate.sh
```

## Security

- Never commit tokens/secrets
- Use environment variables or GitHub Secrets
- Token rotation: Every 90 days

## PR Rules

- Run quality gate before commit
- Never merge if CI fails
- Use specialist agents for CI fixes

## Extended Documentation

- [Training](agents-docs/training.md)
- [CI/CD](agents-docs/ci-cd.md)
- [Security](agents-docs/security.md)
- [Skills](agents-docs/skills.md)
- [Auth Troubleshooting](agents-docs/auth-troubleshooting.md)
- [Learnings](agents-docs/learnings.md)

# Token Optimization Rules

Never run raw testing, linting, or building commands directly in the terminal.

- For build/test: `bash .agents/skills/token_safe_exec.sh "<command>"`
- For lint/format: `python .agents/skills/smart_lint.py "<command>"`
