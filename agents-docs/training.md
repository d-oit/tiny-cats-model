# Modal GPU Training

## Authentication (Modal 1.0+)

Modal 1.0+ uses `modal token new` not `modal token set`:

```bash
# Configure token (Modal 1.0+)
modal token new

# Verify
modal token info
modal token list
```

## Running Training

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
| T4 | Classifier | Low |
| A10G | DiT training | Medium |
| A100 | Large models | High |

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

## References

- [Model Training Skill](../.agents/skills/model-training/SKILL.md)
- [ADR-042: Modal Training Enhancement](../plans/ADR-042-modal-training-enhancement.md)
- [ADR-041: Authentication Error Handling](../plans/ADR-041-authentication-error-handling-2026.md)
