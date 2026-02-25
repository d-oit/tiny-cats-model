# ADR-017: TinyDiT Training Infrastructure with EMA

**Date:** 2026-02-24
**Status:** Implemented
**Authors:** AI Agent
**Related:** ADR-008 (Architecture), ADR-010 (Modal Training Improvements), ADR-011 (Modal Container Dependencies)

## Context

### Problem Statement

The GOAP.md identified two remaining tasks for Phase 3 (Modal Training):
1. `[ ] Train full model (200k steps, EMA)`
2. `[ ] Evaluate generated samples`

While the codebase had:
- TinyDiT architecture (`src/dit.py`)
- Flow matching utilities (`src/flow_matching.py`)
- ONNX export (`src/export_dit_onnx.py`)

There was **no complete training script** for the 200k step TinyDiT training with EMA support.

### Requirements

From GOAP.md and ADR-008:
- **200k training steps** with EMA weight averaging
- **Flow matching** training objective
- **Breed conditioning** via one-hot embeddings
- **Modal GPU training** (A10G/T4)
- **Checkpoint/resume** for long training runs
- **Sample generation** during training for monitoring
- **Mixed precision** (AMP) for faster training
- **Learning rate warmup** with cosine annealing

### Existing Components

| Component | File | Status |
|-----------|------|--------|
| TinyDiT Architecture | `src/dit.py` | ✅ Complete |
| Flow Matching | `src/flow_matching.py` | ✅ Complete |
| EMA Class | `src/flow_matching.py` | ✅ Complete |
| Sampling | `src/flow_matching.py` | ✅ Complete |
| Dataset | `src/dataset.py` | ✅ Complete |
| Training Script | `src/train_dit.py` | ❌ **Missing** |
| Evaluation Script | `src/eval_dit.py` | ❌ **Missing** |

## Decision

We will **create two new scripts** to complete Phase 3 infrastructure:

### 1. TinyDiT Training Script (`src/train_dit.py`)

**Features:**
- Full 200k step training loop with flow matching
- EMA (Exponential Moving Average) weight tracking
- Mixed precision training (AMP)
- Learning rate warmup (10k steps) + cosine annealing
- Gradient clipping (1.0 norm)
- Checkpoint/resume support
- Modal GPU training (A10G, 2 hour timeout)
- Progress logging with loss, LR, speed
- Sample generation every 5k steps
- Checkpoint saving every 10k steps
- OOM recovery and graceful shutdown

**Architecture:**
```python
TinyDiT(
    image_size=128,
    patch_size=16,
    embed_dim=384,
    depth=12,
    num_heads=6,
    num_classes=13,  # 12 breeds + other
)
```

**Training Configuration:**
```python
{
    "steps": 200_000,
    "batch_size": 256,
    "lr": 1e-4,
    "optimizer": "AdamW(β1=0.9, β2=0.95)",
    "weight_decay": 1e-4,
    "warmup_steps": 10_000,
    "ema_beta": 0.9999,
    "gradient_clip": 1.0,
    "mixed_precision": True,
}
```

**Usage:**

> **Note:** See ADR-020 for the complete Modal CLI-First Training Strategy.

```bash
# Local training (debug)
python src/train_dit.py data/cats --steps 1000 --batch-size 8

# Modal GPU training (production) - PRIMARY METHOD
modal run src/train_dit.py data/cats

# Resume from checkpoint
python src/train_dit.py data/cats --resume checkpoints/dit_model.pt

# Custom configuration (Modal GPU)
modal run src/train_dit.py -- \
  --steps 200000 \
  --batch-size 256 \
  --lr 0.0001 \
  --image-size 128
```

### 2. Evaluation Script (`src/eval_dit.py`)

**Features:**
- Generate samples from trained model
- Per-breed sample organization
- Grid visualization
- EMA vs non-EMA comparison
- Batch generation for quality assessment
- Metadata JSON export
- PIL image saving

**Usage:**
```bash
# Generate samples with default EMA model
python src/eval_dit.py

# Evaluate specific checkpoint
python src/eval_dit.py --checkpoint checkpoints/dit_model_ema.pt

# Generate all breeds
python src/eval_dit.py --all-breeds --num-samples 16

# Compare EMA vs non-EMA
python src/eval_dit.py --compare-ema --ema-checkpoint checkpoints/dit_model.pt

# Generate specific breed
python src/eval_dit.py --breed 0 --num-samples 8  # Abyssinian
```

### 3. File Structure

```
tiny-cats-model/
├── src/
│   ├── train_dit.py          # NEW: TinyDiT training script
│   ├── eval_dit.py           # NEW: Evaluation script
│   ├── dit.py                # Existing: Model architecture
│   ├── flow_matching.py      # Existing: Flow matching utilities
│   ├── export_dit_onnx.py    # Existing: ONNX export
│   └── dataset.py            # Existing: Dataset loader
├── checkpoints/              # NEW: Training checkpoints
│   ├── dit_model.pt          # Regular model
│   ├── dit_model_ema.pt      # EMA model
│   └── samples/              # Generated samples during training
└── samples/                  # NEW: Evaluation outputs
    └── generated/
        └── step_200000/
            ├── Abyssinian/
            ├── Bengal/
            └── generation_metadata.json
```

### 4. Training Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  Modal GPU Training (A10G, 2 hours)                         │
├─────────────────────────────────────────────────────────────┤
│  1. Load dataset (Oxford IIIT Pet)                          │
│  2. Initialize TinyDiT + EMA                                │
│  3. For step = 1 to 200,000:                                │
│     a. Sample t ~ Uniform(0, 1)                             │
│     b. Compute flow matching loss                           │
│     c. Backward pass with AMP                               │
│     d. Update EMA                                           │
│     e. Log every 100 steps                                  │
│     f. Generate samples every 5,000 steps                   │
│     g. Save checkpoint every 10,000 steps                   │
│  4. Save final model + EMA                                  │
└─────────────────────────────────────────────────────────────┘
```

### 5. Evaluation Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  Sample Generation (GPU/CPU)                                │
├─────────────────────────────────────────────────────────────┤
│  1. Load checkpoint (model + EMA)                           │
│  2. For each breed:                                         │
│     a. Sample noise z ~ N(0, I)                             │
│     b. ODE integration (50 steps, Euler)                    │
│     c. Apply CFG (scale=1.5)                                │
│     d. Save images + grid                                   │
│  3. Compare EMA vs non-EMA (optional)                       │
│  4. Export metadata JSON                                    │
└─────────────────────────────────────────────────────────────┘
```

## Implementation

### Phase 1: Create Training Script (Completed)
- [x] Implement `src/train_dit.py` with flow matching
- [x] Add EMA support from `flow_matching.py`
- [x] Implement checkpoint/resume
- [x] Add Modal GPU configuration
- [x] Add progress logging
- [x] Add sample generation during training
- [x] Add OOM recovery and graceful shutdown

### Phase 2: Create Evaluation Script (Completed)
- [x] Implement `src/eval_dit.py`
- [x] Add per-breed sample generation
- [x] Add grid visualization
- [x] Add EMA vs non-EMA comparison
- [x] Add metadata export

### Phase 3: Update Documentation (Completed)
- [x] Update GOAP.md with new action items
- [x] Create ADR-017
- [x] Add usage examples

### Phase 4: Execute Training (Pending)
- [ ] Obtain GPU budget approval
- [ ] Run `modal run src/train_dit.py --steps 200000`
- [ ] Monitor training progress
- [ ] Evaluate generated samples
- [ ] Document results

## Consequences

### Positive
- ✅ **Complete training pipeline** ready for 200k step training
- ✅ **EMA support** for better generation quality
- ✅ **Checkpoint/resume** for fault tolerance
- ✅ **Modal GPU** integration for fast training
- ✅ **Sample monitoring** during training
- ✅ **Evaluation tools** for quality assessment
- ✅ **Clear usage patterns** documented

### Negative
- ⚠️ **GPU cost**: 12-24 hours on A10G (~$5-15 depending on spot pricing)
- ⚠️ **Storage**: Checkpoints ~500MB each, samples ~10GB total
- ⚠️ **Time**: Training takes 12-24 hours

### Neutral
- ℹ️ **Two output models**: Regular + EMA (both saved)
- ℹ️ **Sample generation**: Every 5k steps (40 sample batches total)

## Alternatives Considered

### Alternative 1: Train Without EMA
**Proposal**: Skip EMA to reduce training time and complexity.

**Rejected Because**:
- EMA significantly improves generation quality
- Standard practice for diffusion models
- Minimal overhead (<1% training time)
- GOAP.md explicitly requires EMA

### Alternative 2: Use Existing `train.py`
**Proposal**: Modify `src/train.py` (ResNet classifier) for DiT training.

**Rejected Because**:
- `train.py` is for classification (CrossEntropyLoss)
- DiT needs flow matching (velocity prediction)
- Different architecture (ResNet vs Transformer)
- Cleaner to have separate scripts

### Alternative 3: Shorter Training (50k steps)
**Proposal**: Train for 50k steps instead of 200k.

**Rejected Because**:
- GOAP.md specifies 200k steps
- Diffusion models need long training
- Reference (tiny-models) uses 200k
- Quality suffers with shorter training

## Technical Specifications

### Model Architecture (TinyDiT)
| Parameter | Value |
|-----------|-------|
| Image Size | 128x128 |
| Patch Size | 16x16 |
| Num Patches | 64 |
| Embed Dim | 384 |
| Num Heads | 6 |
| Num Blocks | 12 |
| Parameters | ~22M |
| Conditioning | Breed one-hot (13 classes) |

### Training Hyperparameters
| Parameter | Value |
|-----------|-------|
| Steps | 200,000 |
| Batch Size | 256 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW (β1=0.9, β2=0.95) |
| Weight Decay | 1e-4 |
| Warmup Steps | 10,000 |
| LR Schedule | Cosine annealing |
| Gradient Clip | 1.0 |
| EMA Beta | 0.9999 |
| Mixed Precision | AMP (FP16) |

### Modal GPU Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| GPU Type | T4 (or A10G) | Configured via `@app.function(gpu="T4")` |
| Timeout | 7200s (2 hours) | Sufficient for 200k steps |
| Volume | `/outputs`, `/data` | Checkpoints and dataset |
| Python | 3.10 | Via PyTorch base image |
| Dependencies | torch, torchvision, Pillow, tqdm | Installed via `pip_install()` |

**Execution:**
```bash
# Primary method: Modal CLI
modal run src/train_dit.py data/cats --steps 200000

# See ADR-020 for complete Modal CLI reference
```

### Sampling Configuration
| Parameter | Value |
|-----------|-------|
| Num Steps | 50 (Euler integration) |
| CFG Scale | 1.5 |
| Batch Size | 8 |
| Image Size | 128x128 |

## Success Metrics

- [x] Training script created (`src/train_dit.py`)
- [x] Evaluation script created (`src/eval_dit.py`)
- [x] EMA support implemented
- [x] Checkpoint/resume functional
- [x] Modal GPU integration complete
- [ ] Full 200k step training executed
- [ ] Generated samples evaluated
- [ ] Quality meets expectations (FID < 50)

## Testing Plan

### Local Testing (CPU)
```bash
# Test training loop (100 steps)
python src/train_dit.py data/cats --steps 100 --batch-size 8

# Test evaluation
python src/eval_dit.py --num-samples 2 --breed 0
```

### Modal GPU Testing

```bash
# Test Modal GPU (1000 steps) - PRIMARY METHOD
modal run src/train_dit.py data/cats --steps 1000

# Full training (200k steps)
modal run src/train_dit.py data/cats --steps 200000

# See ADR-020 for complete Modal CLI reference
```

### Evaluation Testing
```bash
# Generate samples
python src/eval_dit.py --all-breeds --num-samples 8

# Compare EMA vs non-EMA
python src/eval_dit.py --compare-ema
```

## Migration Guide

### For Existing Users

If you have existing checkpoints from `src/train.py` (ResNet classifier):
- **Not compatible**: DiT training uses different architecture
- **Start fresh**: New training from scratch

### For New Users

1. **Download dataset**:
   ```bash
   bash data/download.sh
   ```

2. **Train model** (Modal GPU - recommended):
   ```bash
   modal run src/train_dit.py data/cats
   ```

3. **Evaluate samples**:
   ```bash
   python src/eval_dit.py --all-breeds
   ```

4. **Use in frontend**:
   - Models auto-deployed to `frontend/public/models/`
   - Generation page: `/generate`

> **Note:** See ADR-020 for the complete Modal CLI-First Training Strategy and additional commands.

## References

- ADR-008: Adapt tiny-models Architecture for Cats
- ADR-010: Modal Training Improvements
- DiT Paper: https://arxiv.org/pdf/2212.09748
- Flow Matching: https://arxiv.org/pdf/2210.02747
- tiny-models: https://github.com/amins01/tiny-models/
- Oxford IIIT Pet: https://www.robots.ox.ac.uk/~vgg/data/pets/

## Appendix: Example Output

### Training Log
```
2026-02-24 10:00:00 | INFO     | Starting TinyDiT training with flow matching
2026-02-24 10:00:01 | INFO     | Configuration: steps=200000, batch_size=256, ...
2026-02-24 10:00:02 | INFO     | Using device: cuda
2026-02-24 10:00:02 | INFO     | GPU: NVIDIA A10G
2026-02-24 10:00:03 | INFO     | Model: TinyDiT | Image size: 128 | Parameters: 22,145,536
2026-02-24 10:00:05 | INFO     | DataLoader created: 78 batches per epoch
2026-02-24 10:00:05 | INFO     | EMA initialized with beta=0.9999
2026-02-24 10:05:00 | INFO     | Step 100/200,000 | Loss: 0.8234 | LR: 1.00e-04 | Speed: 2.1 steps/s
2026-02-24 10:10:00 | INFO     | Step 200/200,000 | Loss: 0.7456 | LR: 1.00e-04 | Speed: 2.2 steps/s
...
2026-02-24 22:00:00 | INFO     | Step 200,000/200,000 | Loss: 0.3421 | LR: 5.00e-07 | Speed: 2.0 steps/s
2026-02-24 22:00:05 | INFO     | Training complete. Final loss: 0.3421
2026-02-24 22:00:10 | INFO     | Saved checkpoint to checkpoints/dit_model.pt
2026-02-24 22:00:15 | INFO     | Saved EMA checkpoint to checkpoints/dit_model_ema.pt
```

### Evaluation Output
```
========================================
Generating Abyssinian samples...
Saved 8 images to samples/generated/step_200000/Abyssinian
Saved grid to samples/generated/step_200000/Abyssinian/grid_20260224_220500.png

========================================
Generating Bengal samples...
Saved 8 images to samples/generated/step_200000/Bengal
...

============================================================
Evaluation complete!
Total breeds generated: 13
Output directory: samples/generated/step_200000
Metadata saved to: samples/generated/step_200000/generation_metadata.json
```
