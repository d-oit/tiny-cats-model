# tiny-cats-model

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/d-oit/tiny-cats-model/actions/workflows/ci.yml/badge.svg)](https://github.com/d-oit/tiny-cats-model/actions/workflows/ci.yml)
[![Code Quality: Ruff](https://img.shields.io/badge/code%20quality-ruff-ff0000)](https://github.com/astral-sh/ruff)
[![Type Check: mypy](https://img.shields.io/badge/type%20check-mypy-blue)](https://github.com/python/mypy)

A cats classifier built on PyTorch with ResNet-18, following 2026 best practices for AI-agent-friendly repositories.

## Features

- ResNet-18 fine-tuned for cat classification (cat / not-cat or breed labels)
- Dataset download script (Oxford IIIT Pet compatible)
- Modal-based GPU training support
- Full CI via GitHub Actions (lint → test → type-check)
- Agent Skills for autonomous task execution
- ONNX model export support

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
modal run src/train.py
```

> **Security**: Never commit secrets. Use environment variables or GitHub Secrets.

## Development

```bash
# Run tests
pytest tests/ -v

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
