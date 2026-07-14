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
python src/train.py data/cats --epochs 1 --batch-size 8
```

### DiT Generator (train_dit.py)
```bash
# Modal GPU training (300k steps)
modal run src/train_dit.py data/cats --steps 300000 --batch-size 256

# High-accuracy (400k steps, gradient accumulation)
modal run src/train_dit.py data/cats \
  --steps 400000 \
  --batch-size 256 \
  --gradient-accumulation-steps 2 \
  --augmentation-level full

# Local CPU testing
python src/train_dit.py data/cats --steps 100 --batch-size 8
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
| `--steps` | 200,000 | Training steps |
| `--batch-size` | 256 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--gradient-accumulation-steps` | 1 | Effective batch = batch × steps |
| `--augmentation-level` | full | basic/medium/full |
| `--warmup-steps` | 10,000 | LR warmup steps |
| `--save-interval` | 10,000 | Checkpoint frequency |
| `--sample-interval` | 5,000 | Sample generation frequency |

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

Pool CI workflow:
```bash
gh workflow run train-pool.yml -f steps=20000 -f batch_size=256
gh workflow run train-pool.yml -f providers="modal,lightning"
```

## Verification

```bash
# Test setup (no import errors)
modal run src/train_dit.py --help

# Verify checkpoint
python src/verify_checkpoint.py --checkpoint checkpoints/tinydit_final.pt

# Export and test ONNX
python src/export_dit_onnx.py --verify --test
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
