# AGENTS.md

AI agent guidance for tiny-cats-model (cat image classification with DiT).

## Setup

- Install deps: `pip install -r requirements.txt`
- Download data: `bash data/download.sh`

## Build & Run

### Training

**Classifier Training:**
```bash
# Modal GPU training (recommended)
modal run src/train.py data/cats --epochs 20 --batch-size 64

# Local CPU testing (debug)
python src/train.py data/cats --epochs 1 --batch-size 8
```

**DiT Training (with enhanced augmentation):**
```bash
# Modal GPU training (300k steps, full augmentation)
modal run src/train_dit.py data/cats --steps 300000 --batch-size 256 --augmentation-level full

# With gradient accumulation (effective batch 512)
modal run src/train_dit.py data/cats --steps 300000 --batch-size 256 --gradient-accumulation-steps 2

# Local CPU testing (debug)
python src/train_dit.py data/cats --steps 100 --batch-size 8
```

### Evaluation & Benchmarks

```bash
# Full evaluation (FID, IS, Precision/Recall)
python src/evaluate_full.py --checkpoint checkpoints/tinydit_final.pt \
    --generate-samples --num-samples 500 \
    --compute-fid --real-dir data/cats/test --fake-dir samples/evaluation \
    --report-path evaluation_report.json

# Inference benchmarks
python src/benchmark_inference.py --model checkpoints/tinydit_final.pt \
    --device cpu --num-warmup 10 --num-runs 100 \
    --benchmark-throughput --batch-sizes 1,4,8,16 \
    --report-path benchmark_report.json
```

### HuggingFace Upload

```bash
# Upload complete model package
python src/upload_to_huggingface.py \
    --classifier checkpoints/classifier.pt \
    --generator checkpoints/tinydit_final.pt \
    --onnx-classifier frontend/public/models/cats_classifier.onnx \
    --onnx-generator frontend/public/models/generator.onnx \
    --evaluation-report evaluation_report.json \
    --benchmark-report benchmark_report.json \
    --samples-dir samples/evaluation \
    --repo-id d4oit/tiny-cats-model

# Requires HF_TOKEN environment variable
export HF_TOKEN=hf_...
```

## Verification

- Verify checkpoint: `python src/verify_checkpoint.py --checkpoint checkpoints/tinydit_final.pt`
- Export and test ONNX: `python src/export_dit_onnx.py --verify --test`
- Run E2E tests: `npx playwright test`
- Run Python tests: `pytest tests/ -v`

## Code Style

- PEP 8 with Ruff
- Line length: 88 chars
- Type hints required for new code
- Format: `ruff format .`
- Lint: `ruff check . --fix`
- 500 LOC max per file

## Testing

### Python Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=term-missing

# Run specific test file
pytest tests/test_dataset.py -v
```

### E2E Tests (Playwright)
```bash
# Run all E2E tests
npx playwright test

# Run with UI
npm run test:e2e:ui

# Run in headed mode (visible browser)
npx playwright test --headed

# Run specific test file
npx playwright test tests/e2e/classification.spec.ts

# Generate test report
npx playwright show-report
```

### Quality Gate
```bash
# Full quality gate (lint + format + test)
bash scripts/quality-gate.sh

# Pre-commit hooks
pre-commit install && pre-commit run --all-files
```

## Security

- Never hardcode tokens or secrets
- Never commit `.env` files
- Use environment variables for credentials
- Modal tokens: configured globally via `modal token set`
- HuggingFace token: `export HF_TOKEN=hf_...`

## PR Instructions

- Run quality gate before committing: `bash scripts/quality-gate.sh`
- Never skip CI checks before merging
- Never merge if CI fails
- Use atomic commits with clear messages

## Extended Docs

- [Training detailed options](agents-docs/training.md)
- [CI/CD workflows & debugging](agents-docs/ci-cd.md)
- [Security best practices](agents-docs/security.md)
- [Agent skills reference](agents-docs/skills.md)
- [Learnings & patterns](agents-docs/learnings.md)
- [Full Training & HuggingFace Upload Plan](plans/ADR-035-full-model-training-huggingface-upload-2026.md)
