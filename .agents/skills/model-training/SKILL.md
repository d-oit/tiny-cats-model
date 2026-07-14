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

### Container Pattern: @app.cls + @modal.enter() (ADR-025, ADR-057)

Training scripts now use the class-based pattern with `@modal.enter()` for
one-time container initialization (CUDA warm-up, heavy imports) that runs
once per container instead of once per function invocation:

```python
@app.cls(
    image=image, volumes={...}, gpu=["T4", "L4"],
    timeout=86400,
    retries=modal.Retries(max_retries=10, initial_delay=0.0),
    scaledown_window=300,  # ADR-057: keep container warm 5 min
)
class DiTTrainer:
    @modal.enter()
    def enter(self):
        """One-time init: paths, CUDA warm-up, heavy imports."""
        sys.path.insert(0, "/app")
        os.chdir("/app")
        # warm up CUDA...

    @modal.method()
    def train(self, ...):
        # Training logic — container already initialized
        ...
```

### Error Handling & Logging
- **Pre-flight checks**: Auth validation before training starts
- **Structured logging**: Console + file with timestamps
- **Volume commits**: Explicit commits after successful operations
- **Cleanup**: Old checkpoints auto-cleaned (keep last 5)

### GPU Selection
| GPU | Best For | Cost |
|-----|----------|------|
| T4 | Classifier training, DiT (cost-optimized) | Low ($0.59/hr) |
| L4 | DiT fallback | Low ($0.80/hr) |
| A10G | DiT training (if preemption is critical) | Medium ($1.10/hr) |
| L40S | Non-spot DiT training | High ($1.95/hr) |
| A100 | Large models | High ($2.10/hr) |

### Free GPU Pool Training

Multi-provider training with HF Hub checkpoint sync.
See `src/gpu_pool.py` for the full abstraction.

```bash
# Check provider and cost estimates
python -c "from gpu_pool import detect_provider_and_log, estimate_cost; detect_provider_and_log(); print(estimate_cost(50000))"

# Train with fallback chain
python -c "from gpu_pool import train_with_fallback; train_with_fallback(steps=50000)"

# Print the fallback chain order
python -c "from gpu_pool import train_chain; train_chain(steps=20000)"
```

Provider-specific scripts with Hub checkpoint sync:
```bash
python scripts/train_lightning.py --steps 20000 --hub-resume   # Lightning AI
python scripts/train_colab.py --steps 20000 --resume            # Colab
python scripts/train_kaggle.py --steps 20000 --hub-resume       # Kaggle
python scripts/train_hf_spaces.py --steps 20000 --hub-resume    # HF Spaces
```

Pool CI workflow:
```bash
gh workflow run train-pool.yml -f steps=20000 -f provider=modal
```

**Testing the GPU pool:**
```bash
# Unit tests for gpu_pool.py (provider detection, cost estimation, fallback chain)
pytest tests/test_gpu_pool.py -v

# Integration tests for train_chain() and train_with_fallback()
pytest tests/test_train_chain.py -v

# End-to-end fallback chain simulation (38 checks)
python scripts/test_fallback_chain.py
```

**GPU hour estimation and calibration:**
```bash
# Full calibration report
python scripts/benchmark_estimates.py

# Print tuned T4_STEPS_PER_SECOND constant
python scripts/benchmark_estimates.py --tune

# Estimate cost for specific step count
python scripts/benchmark_estimates.py --steps 50000

# JSON output for CI drift check
python scripts/benchmark_estimates.py --json
```

The tunable `T4_STEPS_PER_SECOND` constant in `src/gpu_pool.py` controls
the baseline speed used by `estimate_gpu_hours()`. Calibrate it against
real training runs with the benchmark script.

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
| Slow training | Use T4/L4 GPU fallback for cost optimization |
| CUDA error | Use `--device cpu` for local testing |
| Import errors | Verify `sys.path` in Modal container |
| Download failed | Check `data/download.py` in container |

## Modal Configuration

```python
# Class-based pattern (ADR-025, ADR-057) — see src/train_dit.py, src/train.py
@app.cls(
    image=image,
    volumes={"/outputs": volume_outputs, "/data": volume_data},
    gpu=["T4", "L4"],  # Cost-optimized: T4 ($0.59/hr), L4 fallback ($0.80/hr)
    timeout=86400,  # 24 hours max
    retries=modal.Retries(max_retries=10, initial_delay=0.0),
    scaledown_window=300,  # Keep container warm 5 min for retries
)
class DiTTrainer:
    @modal.enter()
    def enter(self): ...  # One-time init
    @modal.method()
    def train(self, ...): ...  # Training logic
```
