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
modal run src/train.py data/cats --epochs 20 --batch-size 64

# DiT Generator (optimized - 100k with early stopping)
modal run src/train_dit.py data/cats --steps 100000 --batch-size 512

# Custom configuration
modal run src/train_dit.py data/cats --steps 50000 --batch-size 512 --lr 5e-5 --warmup-steps 2000
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--steps` | 100,000 | Max training steps (early stopping may stop earlier) |
| `--batch-size` | 512 | Batch size (increased for better gradients) |
| `--lr` | 5e-5 | Learning rate |
| `--warmup-steps` | 2,000 | LR warmup (shorter = faster convergence) |
| `--augmentation-level` | full | basic/medium/full |

### Early Stopping

Training automatically stops when loss plateaus for 3 consecutive evaluations (every 10k steps). This typically occurs at 50k-80k steps, saving 60-80% cost.

## GitHub Actions

```bash
# Trigger training (optimized defaults)
gh workflow run train.yml

# Custom configuration
gh workflow run train.yml -f steps=50000 -f batch_size=512

# Monitor runs
gh run list
gh run view <run-id>
gh run watch

# Check secrets
gh secret list
```

## Training

### Modal GPU Training

```bash
# Classifier (resnet18)
modal run src/train.py data/cats --epochs 20 --batch-size 64

# DiT Generator (optimized - 100k with early stopping)
modal run src/train_dit.py data/cats --steps 100000 --batch-size 512

# Custom configuration
modal run src/train_dit.py data/cats --steps 50000 --batch-size 512 --lr 5e-5 --warmup-steps 2000
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--steps` | 100,000 | Max training steps (early stopping may stop earlier) |
| `--batch-size` | 512 | Batch size (increased for better gradients) |
| `--lr` | 5e-5 | Learning rate |
| `--warmup-steps` | 2,000 | LR warmup (shorter = faster convergence) |
| `--augmentation-level` | full | basic/medium/full |

### Early Stopping

Training automatically stops when loss plateaus for 3 consecutive evaluations (every 10k steps). This typically occurs at 50k-80k steps, saving 60-80% cost.

## GitHub Actions

```bash
# Trigger training (optimized defaults)
gh workflow run train.yml

# Custom configuration
gh workflow run train.yml -f steps=50000 -f batch_size=512

# Monitor runs
gh run list
gh run view <run-id>
gh run watch

# Check secrets
gh secret list
```

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
pytest tests/ -v
npx playwright test
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
