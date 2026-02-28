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
flake8 . --max-line-length=88

# Auto-fix lint issues
ruff check . --fix

# Format code
black .

# Check formatting without modifying
black --check .
```

## Modal GPU Training (Modal 1.0+)

```bash
# Set credentials globally (Modal 1.0+ uses 'token new')
modal token new

# Run training on GPU
modal run src/train.py

# Run with custom options
modal run src/train.py -- --epochs 20 --batch-size 64
```

## Full Verification

```bash
# Run complete CI check suite
bash .agents/skills/testing-workflow/verify.sh

# Or use the quality gate
bash .agents/skills/git-workflow/quality-gate.sh

# Or use the new scripts
bash scripts/quality-gate.sh
```

## Command Reference

| Task | Command |
|------|---------|
| Install | `pip install -r requirements.txt` |
| Dataset | `bash data/download.sh` |
| Train | `python src/train.py data/cats` |
| Evaluate | `python src/eval.py` |
| Tests | `pytest tests/ -v` |
| Lint | `ruff check . && flake8 .` |
| Format | `black .` |
| CI verify | `bash .agents/skills/testing-workflow/verify.sh` |
