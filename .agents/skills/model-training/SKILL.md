---
name: model-training
description: Use for model training, hyperparameter tuning, and Modal GPU training.
---

# Skill: model-training

This skill covers model training workflows for tiny-cats-model.

## Authentication (Modal 1.0+)

```bash
# Configure Modal token (Modal 1.0+ uses 'token new' not 'token set')
modal token new

# Verify token status
modal token info

# List available profiles
modal token list

# Validate programmatically
python -c "from auth_utils import AuthValidator; print(AuthValidator().check_modal_auth())"
```

## Local Training

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

# Use specific device
python src/train.py data/cats --device cuda
python src/train.py data/cats --device cpu
```

## DiT Training (Local)

```bash
# Local CPU testing (debug)
python src/train_dit.py --data-dir data/cats --steps 100 --batch-size 8

# Full training (requires GPU)
python src/train_dit.py --data-dir data/cats --steps 200000 --batch-size 256
```

## Modal GPU Training

```bash
# Classifier training on GPU
modal run src/train.py --data-dir data/cats --epochs 20 --batch-size 64

# DiT training on GPU (300k steps)
modal run src/train_dit.py --data-dir data/cats --steps 300000 --batch-size 256

# High-accuracy DiT training (400k steps, gradient accumulation)
modal run src/train_dit.py --data-dir data/cats \
  --steps 400000 \
  --batch-size 256 \
  --gradient-accumulation-steps 2 \
  --lr 5e-5 \
  --warmup-steps 15000 \
  --augmentation-level full

# Or use the training script (recommended)
bash scripts/train_dit_high_accuracy.sh

# Verify training setup (no import errors)
modal run src/train_dit.py --help
```

## Modal Best Practices

### Error Handling & Logging
- **Pre-flight checks**: Auth validation before training starts
- **Structured logging**: Console + file with timestamps
- **Volume commits**: Explicit commits after successful operations
- **Cleanup**: Old checkpoints auto-cleaned (keep last 5)

### Cleanup Pattern
```python
# Cleanup old checkpoints (train.py)
from volume_utils import cleanup_old_checkpoints
cleanup_old_checkpoints(volume_outputs, "/outputs/checkpoints/classifier", keep_last_n=5)

# Memory cleanup (both scripts)
cleanup_memory()  # gc.collect() + torch.cuda.empty_cache()
```

### GPU Selection
| GPU | Best For | Cost |
|-----|----------|------|
| T4 | Classifier training | Low |
| A10G | DiT training | Medium |
| A100 | Large models | High |

### Retry Configuration
```python
retries=modal.Retries(
    max_retries=3,
    backoff_coefficient=2.0,
    initial_delay=10.0,
    max_delay=60.0,
)
```

## Hyperparameters

### Classifier (train.py)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--backbone` | resnet18 | Model architecture |
| `--device` | cuda/cpu | Compute device |

### DiT (train_dit.py)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--steps` | 200,000 | Training steps |
| `--batch-size` | 256 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--gradient-accumulation-steps` | 1 | Effective batch = batch × steps |
| `--augmentation-level` | full | basic/medium/full |

## Model Evaluation

```bash
# Evaluate default model
python src/eval.py

# Evaluate custom checkpoint
python src/eval.py \
  --data-dir data/cats \
  --checkpoint cats_model.pt \
  --backbone resnet18

# Full evaluation (FID, IS, Precision/Recall)
python src/evaluate_full.py --checkpoint checkpoints/tinydit_final.pt \
    --generate-samples --num-samples 500 \
    --compute-fid --real-dir data/cats/test --fake-dir samples/evaluation

# Benchmark inference
python src/benchmark_inference.py --model checkpoints/tinydit_final.pt \
    --device cpu --num-warmup 10 --num-runs 100 \
    --benchmark-throughput --batch-sizes 1,4,8,16
```

## Checkpoint Management

```bash
# Default checkpoints
checkpoints/
├── classifier/          # Classifier model
│   └── 2026-02-25/
│       └── best_cats_model.pt
└── dit/                # DiT model
    └── 2026-02-25/
        ├── dit_model.pt
        └── dit_model_ema.pt  # EMA weights (use for inference)

# List checkpoints
ls -la checkpoints/

# Verify checkpoint
python src/verify_checkpoint.py --checkpoint checkpoints/tinydit_final.pt
```

## Dataset Preparation

```bash
# Download dataset
bash data/download.sh

# Download via Python (for Modal container)
python data/download.py

# Dataset structure
data/cats/
├── cat/        # Cat images (12 breeds)
└── other/      # Non-cat images
```

## Training Tips

1. **Start small** - 100 steps locally, then scale to Modal
2. **Monitor progress** - Check logs in volume: `/outputs/checkpoints/*/training.log`
3. **Use EMA weights** - `dit_model_ema.pt` for better inference
4. **Validate first** - Run `modal run --help` before full training
5. **Cleanup** - Old checkpoints auto-removed (keep last 5)

## Common Issues

| Issue | Solution |
|-------|----------|
| AuthError | Run `modal token new` (Modal 1.0+) |
| OOM errors | Reduce batch-size or use gradient accumulation |
| Slow training | Use A10G GPU instead of T4 |
| CUDA error | Use `--device cpu` for local testing |
| Import errors | Verify `sys.path` in Modal container |
| Download failed | Check `data/download.py` in container |

## Modal Configuration

```python
# GPU selection in train.py/train_dit.py
@app.function(gpu="A10G")  # T4, A10G, A100

# Timeout for long training
@app.function(timeout=86400)  # 24 hours max

# Volumes for persistent storage
volumes={
    "/outputs": modal.Volume.from_name("dit-outputs"),
    "/data": modal.Volume.from_name("dit-dataset"),
}
```
