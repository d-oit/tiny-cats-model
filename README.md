# tiny-cats-model

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/d-oit/tiny-cats-model/actions/workflows/ci.yml/badge.svg)](https://github.com/d-oit/tiny-cats-model/actions/workflows/ci.yml)
[![Code Quality: Ruff](https://img.shields.io/badge/code%20quality-ruff-ff0000)](https://github.com/astral-sh/ruff)
[![Type Check: mypy](https://img.shields.io/badge/type%20check-mypy-blue)](https://github.com/python/mypy)
[![HuggingFace Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/d4oit/tiny-cats-model)

A cats classifier and generator built on PyTorch with ResNet-18 and TinyDiT, following 2026 best practices for AI-agent-friendly repositories.

## Features

- **Classification**: ResNet-18 fine-tuned for cat breed classification (13 breeds)
- **Generation**: TinyDiT diffusion model for conditional cat image generation
- **Interactive Tutorials**: 3 Jupyter notebooks with Google Colab support
- **Automated Deployment**: CI/CD pipeline with automated HuggingFace uploads
- **Comprehensive Testing**: 215+ E2E tests covering all user journeys
- **ONNX Export**: Quantized models for web deployment (11MB classifier, 33MB generator)

## Quick Links

- 📚 [Tutorial Notebooks](notebooks/README.md) - Interactive guides with Colab
- 🤗 [HuggingFace Model](https://huggingface.co/d4oit/tiny-cats-model) - Download models
- 📖 [Documentation](docs/) - Setup guides and ADRs
- 🧪 [E2E Tests](tests/e2e/) - Playwright test suite

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download & prepare dataset
bash data/download.sh

# 3. Train the model
python src/train.py data/cats

# 4. Evaluate
python src/eval.py
```

## Project Structure

```
tiny-cats-model/
├── src/
│   ├── train.py           # Training entrypoint
│   ├── eval.py            # Evaluation script
│   ├── model.py           # Model definition
│   ├── dataset.py         # DataLoader factory
│   └── export_onnx.py     # ONNX export
├── tests/
│   └── test_dataset.py    # Unit tests
├── data/
│   ├── cats/              # Dataset (gitignored)
│   └── download.sh        # Dataset download script
├── .agents/skills/        # Agent automation skills
│   ├── cli-usage/         # Training & evaluation commands
│   ├── testing-workflow/  # CI verification
│   ├── code-quality/      # Linting & formatting
│   ├── gh-actions/       # CI/CD debugging
│   ├── git-workflow/      # Branch & PR management
│   ├── security/          # Secrets handling
│   └── model-training/   # GPU training
├── .github/workflows/    # CI/CD pipelines
├── plans/                # Architecture decision records
├── AGENTS.md             # AI agent guidance
├── CLAUDE.md             # Claude CLI reference
├── modal.yml             # Modal GPU config
└── requirements.txt      # Dependencies
```

## Training Options

```bash
# Default (10 epochs, resnet18)
python src/train.py data/cats

# Custom training
python src/train.py data/cats \
  --epochs 20 \
  --batch-size 64 \
  --lr 0.0001 \
  --backbone resnet34 \
  --output my_model.pt

# Train without pretrained weights
python src/train.py data/cats --no-pretrained
```

## Modal Training (GPU)

```bash
export MODAL_TOKEN_ID=your_token_id
export MODAL_TOKEN_SECRET=your_token_secret

# Classifier training
modal run src/train.py --data-dir /data/cats --epochs 20 --batch-size 64

# DiT generator: --steps is a GLOBAL target, not an additional-step count
modal run src/train_dit.py --data-dir /data/cats --steps 60000 --batch-size 32
```

> **Security**: Never commit secrets. Use environment variables or GitHub Secrets.

## Production 400k DiT Training

A 400k-step run is a sequence of **bounded, resumable slices**, not one job.
`--steps` is always the *global* target, so every slice resumes exactly where
HuggingFace Hub left off (`max(0, target − completed)`); a checkpoint at 45k run
with `--steps 60000` performs exactly 15,000 more steps. GitHub Actions is the
control plane: it plans slices, launches real GPU sessions and verifies the
artifacts — it never trains on CPU.

```bash
# 1. Local smoke test (CPU, seconds)
python src/train_dit.py --data-dir data/cats --steps 100 --batch-size 8

# 2. One bounded Modal slice (<= 6h; continues the same experiment)
modal run src/train_dit.py \
  --data-dir /data/cats \
  --steps 60000 \
  --batch-size 32 \
  --lr 5e-5 \
  --warmup-steps 2000 \
  --save-interval 5000 \
  --hub-push-interval 5000 \
  --hub-resume

# 3. Continue with INCREASING GLOBAL TARGETS (not additional steps)
modal run src/train_dit.py --data-dir /data/cats --steps 120000 --batch-size 32 --hub-resume
modal run src/train_dit.py --data-dir /data/cats --steps 400000 --batch-size 32 --hub-resume
```

Or let GitHub Actions orchestrate the whole run: every 6 hours `train-pool.yml`
plans the remaining slices toward 400k and launches one bounded provider session
per slice, each pushing `step-XXXXXX/` snapshots plus a `latest/` pointer to the
Hub (resume is automatic and idempotent).

```bash
# Full 400k run as bounded slices (default 25k/session, sequential, fail-fast)
gh workflow run train-pool.yml -f steps=400000 -f slice_size=60000

# Single verified production run (one slice) through the main workflow
gh workflow run train.yml -f steps=25000 -f batch_size=32

# Unsupported providers FAIL CLEARLY instead of silently training on CPU
gh workflow run train-pool.yml -f provider=kaggle   # exits 2 with instructions
```

A run only publishes a "final model" when its global target is reached *and* the
checkpoint verifies: the workflow re-reads `training_state.json`, checks the
checkpoint is a valid torch archive, runs ONNX Runtime inference, and verifies the
Hub upload afterwards. A preempted slice reports `partial`, keeps a resumable
checkpoint and publishes nothing.

### Free GPU Pool Providers

| Provider | Free Tier | GPU Types | Max Session | Control plane |
|----------|-----------|-----------|-------------|---------------|
| Modal | $30/mo credits | T4, L4 | 24h | ✅ launchable |
| Lightning AI | 22h/day free | T4, L4, L40S | Unlimited | manual (`scripts/train_lightning.py`) |
| Google Colab | Free GPU runtime | T4, V100 | 12h | manual (no headless API) |
| Kaggle | 30h/week free | P100, T4 | 9h | manual (`scripts/train_kaggle.py`) |
| HF Spaces | 16h/day GPU | T4-small | Unlimited | manual (`scripts/train_hf_spaces.py`) |

```bash
# Inspect the control plane without launching anything
python src/providers.py gate   --provider all
python src/providers.py gate   --provider lightning --strict   # exits 2
python src/providers.py plan   --steps 400000 --slice-size 60000
python src/providers.py launch --provider modal --target 60000
python src/providers.py verify --state-file training_state.json --target 60000
```

See `src/gpu_pool.py` for the provider abstraction, `src/providers.py` for the
control plane, and `agents-docs/training.md` for the full 400k runbook
(including recovery from preemption, cancellation, and corrupt checkpoints).

## Development

```bash
# Run all tests (unit, GPU pool, train chain)
pytest tests/ -v

# Specific test suites
pytest tests/test_gpu_pool.py -v       # GPU pool abstraction
pytest tests/test_train_chain.py -v    # Train chain & fallback

# Fallback chain simulation (38 checks)
python scripts/test_fallback_chain.py

# End-to-end training pipeline (smoke + exact resume + ONNX + artifact package)
python scripts/verify_training_pipeline.py

# GPU hour estimation & calibration
python scripts/benchmark_estimates.py
python scripts/benchmark_estimates.py --steps 50000

# Lint code (auto-fix)
ruff check . --fix

# Format code
ruff format .

# Type check
mypy .

# Full verification
bash scripts/quality-gate.sh
```

## Dataset

Default: Oxford IIIT Pet Dataset (cats subset). The `data/download.sh` script downloads and prepares the dataset. Replace the URL with your own source if needed.

## License

MIT
