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
modal run src/train.py

# DiT generator training (optimized)
modal run src/train_dit.py --steps 100000 --batch-size 512
```

> **Security**: Never commit secrets. Use environment variables or GitHub Secrets.

## Free GPU Pool Training

Train across multiple free GPU providers with automatic checkpoint sync via HuggingFace Hub:

| Provider | Free Tier | GPU Types | Max Session |
|----------|-----------|-----------|-------------|
| Modal | $30/mo credits | T4, L4 | 24h |
| Lightning AI | 22h/day free | T4, L4, L40S | Unlimited |
| Google Colab | Free GPU runtime | T4, V100 | 12h |
| Kaggle | 30h/week free | P100, T4 | 9h |
| HF Spaces | 16h/day GPU | T4-small | Unlimited |

```bash
# Check which provider you're on
python -c "from src.gpu_pool import detect_provider; print(detect_provider())"

# Estimate cost across all providers
python -c "from src.gpu_pool import estimate_cost; print(estimate_cost(50000))"

# Train on current provider with Hub sync
python scripts/train_lightning.py --steps 20000 --hub-resume

# Manual pool run via GitHub Actions
gh workflow run train-pool.yml -f provider=all -f steps=20000
```

See `src/gpu_pool.py` for the full provider abstraction and `agents-docs/training.md` for detailed setup guides.

## Development

```bash
# Run all tests (unit, GPU pool, train chain)
pytest tests/ -v

# Specific test suites
pytest tests/test_gpu_pool.py -v       # GPU pool abstraction
pytest tests/test_train_chain.py -v    # Train chain & fallback

# Fallback chain simulation (38 checks)
python scripts/test_fallback_chain.py

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
