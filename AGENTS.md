# AGENTS.md

AI agent guidance for tiny-cats-model (cat image classification with DiT).

## Setup

- Install deps: `pip install -r requirements.txt`
- Download data: `bash data/download.sh`

## Build & Run

- Train classifier (Modal GPU): `modal run src/train.py data/cats`
- Train DiT (Modal GPU): `modal run src/train_dit.py data/cats`
- Train local (CPU debug): `python src/train.py data/cats --epochs 1 --batch-size 8`
- Evaluate: `python src/eval.py`

## Verification

- Verify checkpoint: `python src/verify_checkpoint.py --checkpoint checkpoints/tinydit_final.pt`
- Export and test ONNX: `python src/export_dit_onnx.py --verify --test`
- Run E2E tests: `npx playwright test`

## Code Style

- PEP 8 with Ruff
- Line length: 88 chars
- Type hints required for new code
- Format: `ruff format .`
- Lint: `ruff check . --fix`
- 500 LOC max per file

## Testing

- Run tests: `pytest tests/ -v`
- Quality gate (lint + format + test): `bash scripts/quality-gate.sh`
- Pre-commit hooks: `pre-commit install && pre-commit run --all-files`

## Security

- Never hardcode tokens or secrets
- Never commit `.env` files
- Use environment variables for credentials
- Modal tokens: configured globally via `modal token set`

## PR Instructions

- Run quality gate before committing: `bash scripts/quality-gate.sh`
- Never skip CI checks before merging
- Never merge if CI fails

## Extended Docs

- [Training detailed options](agents-docs/training.md)
- [CI/CD workflows & debugging](agents-docs/ci-cd.md)
- [Security best practices](agents-docs/security.md)
- [Agent skills reference](agents-docs/skills.md)
- [Learnings & patterns](agents-docs/learnings.md)
