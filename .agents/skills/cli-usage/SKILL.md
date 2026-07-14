---
name: cli-usage
description: Use when invoking training, evaluation, and dataset preparation via CLI. Provides all standard commands for this project.
triggers:
  - "train model"
  - "run training"
  - "evaluate"
  - "download dataset"
  - "cli commands"
  - "modal"
---

# Skill: cli-usage

This skill covers all CLI commands for operating the tiny-cats-model project.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download and prepare dataset
bash data/download.sh
# Dataset will be at: data/cats/cat/ and data/cats/other/
```

## Training

```bash
# Basic training (10 epochs, resnet18)
python src/train.py data/cats

# Custom training
python src/train.py data/cats \
  --epochs 20 \
  --batch-size 64 \
  --lr 0.0001 \
  --backbone resnet34 \
  --output my_model.pt

# Training without pretrained weights
python src/train.py data/cats --no-pretrained
```

## Evaluation

```bash
# Evaluate with default settings
python src/eval.py

# Evaluate with custom checkpoint
python src/eval.py \
  --data-dir data/cats \
  --checkpoint cats_model.pt \
  --backbone resnet18
```

## Testing

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Linting & Formatting

```bash
# Lint
ruff check .

# Auto-fix lint issues
ruff check . --fix

# Format code
ruff format .

# Check formatting without modifying
ruff format --check .
```

## Modal GPU Training (Modal 1.0+)

```bash
# Set credentials globally (Modal 1.0+ uses 'token new')
modal token new

# Classifier training on GPU
modal run src/train.py --epochs 20 --batch-size 64

# DiT generator training on GPU
modal run src/train_dit.py --steps 100000 --batch-size 512

# Local CPU testing (debug)
python src/train_dit.py --data-dir data/cats --steps 100 --batch-size 8
```

### Modal best practices

- Modal **timeout is 24h** per function call. Our scripts comply (3600 s for classifier,
  86400 s = 24 h for DiT). For runs > 24h, design as reentrant + use `--detach`.
- Always call `volume.commit()` after writing a checkpoint volume. We do this in both
  `train.py` and `train_dit.py`.
- **`modal token new`** is the canonical auth command. The old `modal token set` (Modal
  0.x) no longer works.
- The live per-iteration `Speed: 2.2 steps/s` printed by the trainer reflects GPU
  forward+backward only — wall-clock between step reports includes container cold-start,
  image pull, dataset download, and per-iteration volume/ONNX overhead. See
  `plans/ADR-057-modal-cli-verification-and-best-practices-2026.md` for the full breakdown.
- When triggering from GitHub Actions, prefer **reusing a running run** over
  cancelling/re-triggering — every re-trigger pays container cold-start + image-pull cost.

## GPU Pool Training

```bash
# Check which free GPU provider you're on
python -c "from gpu_pool import detect_provider; print(detect_provider())"

# Estimate cost across all providers
python -c "from gpu_pool import estimate_cost; print(estimate_cost(50000))"

# Train with automatic fallback chain
python -c "from gpu_pool import train_chain; train_chain(steps=20000)"

# Provider-specific scripts with Hub sync
python scripts/train_lightning.py --steps 20000 --hub-resume
python scripts/train_kaggle.py --steps 20000 --hub-resume
```

## Full Verification

```bash
# Run the quality gate (lint + format + typecheck + test + verify)
bash scripts/quality-gate.sh
```

## Command Reference

| Task | Command |
|------|---------|
| Install | `pip install -r requirements.txt` |
| Dataset | `bash data/download.sh` |
| Train (classifier) | `python src/train.py data/cats` |
| Train (DiT) | `python src/train_dit.py --data-dir data/cats --steps 100000` |
| Evaluate | `python src/eval.py` |
| Tests | `pytest tests/ -v` |
| Fallback sim | `python scripts/test_fallback_chain.py` |
| Benchmark | `python scripts/benchmark_estimates.py` |
| Lint | `ruff check .` |
| Format | `ruff format .` |
| CI verify | `bash scripts/quality-gate.sh` |
