# ADR-036: High-Accuracy Training Configuration for TinyDiT

**Date:** 2026-02-27
**Status:** Proposed
**Authors:** AI Agent (GOAP System)
**Related:** ADR-017 (TinyDiT Training), ADR-035 (Full Model Training), GOAP.md Phase 17

## Context

### Current Training State

The current TinyDiT model has been trained with:
- **Steps:** 200,000
- **Batch Size:** 256
- **Learning Rate:** 1e-4
- **Dataset:** Oxford IIIT Pet (2,400 cat images, 12 breeds + other)
- **Checkpoint:** `checkpoints/tinydit_final.pt` (129MB)
- **HuggingFace:** Uploaded to d4oit/tiny-cats-model

### Problem Statement

The current model, while functional, can be significantly improved:

1. **Limited training steps:** 200k steps may not be sufficient for full convergence
2. **Small dataset:** Only 2,400 cat images across 12 breeds
3. **Basic hyperparameters:** Standard LR and batch size
4. **No gradient accumulation:** Limited effective batch size
5. **No extended training plan:** No clear path to higher accuracy

### Motivation

Higher accuracy models provide:
- **Better sample quality:** More realistic cat images
- **Improved breed discrimination:** Clearer breed-specific features
- **Lower FID scores:** Better generative performance
- **Production readiness:** More reliable for real-world use
- **Research value:** Better baseline for future work

## Decision

We will implement a **high-accuracy training configuration** with extended training, larger effective batch sizes, and optimized hyperparameters.

### Training Configuration Comparison

| Parameter | Current | High-Accuracy | Rationale |
|-----------|---------|---------------|-----------|
| **Steps** | 200,000 | 400,000 | Double training for better convergence |
| **Batch Size** | 256 | 256 | Keep GPU memory manageable |
| **Gradient Accumulation** | 1 | 2 | Effective batch = 512 |
| **Learning Rate** | 1e-4 | 5e-5 | Lower LR for finer convergence |
| **Warmup Steps** | 10,000 | 15,000 | Longer warmup for stability |
| **Augmentation** | Basic | Full | More data diversity |
| **EMA Beta** | 0.9999 | 0.9999 | Same (proven effective) |

### Recommended Training Command

```bash
# High-accuracy configuration (400k steps, effective batch 512)
modal run src/train_dit.py data/cats \
  --steps 400000 \
  --batch-size 256 \
  --gradient-accumulation-steps 2 \
  --lr 5e-5 \
  --warmup-steps 15000 \
  --augmentation-level full
```

### Alternative Configurations

#### Option A: Conservative Improvement (300k steps)
```bash
modal run src/train_dit.py data/cats \
  --steps 300000 \
  --batch-size 256 \
  --lr 1e-4 \
  --augmentation-level full
```
- **Cost:** ~$10-15 on Modal (H100/A10G)
- **Time:** ~18-24 hours
- **Expected improvement:** Moderate (10-20% FID reduction)

#### Option B: Aggressive Improvement (500k steps)
```bash
modal run src/train_dit.py data/cats \
  --steps 500000 \
  --batch-size 256 \
  --gradient-accumulation-steps 2 \
  --lr 3e-5 \
  --warmup-steps 20000 \
  --augmentation-level full
```
- **Cost:** ~$25-35 on Modal (H100/A10G)
- **Time:** ~30-40 hours
- **Expected improvement:** Significant (30-50% FID reduction)

#### Option C: Maximum Quality (1M steps, research-grade)
```bash
modal run src/train_dit.py data/cats \
  --steps 1000000 \
  --batch-size 256 \
  --gradient-accumulation-steps 2 \
  --lr 2e-5 \
  --warmup-steps 25000 \
  --augmentation-level full \
  --image-size 256
```
- **Cost:** ~$50-70 on Modal (H100/A10G)
- **Time:** ~60-80 hours
- **Expected improvement:** Substantial (50-70% FID reduction)
- **Note:** Requires image size increase for full benefit

## Implementation

### Phase 1: Preparation

#### 1.1 Dataset Validation
```bash
# Verify dataset integrity
python -c "
from dataset import CatsDataset
from torch.utils.data import DataLoader

dataset = CatsDataset('data/cats')
loader = DataLoader(dataset, batch_size=32)

print(f'Total samples: {len(dataset)}')
print(f'Batches per epoch: {len(loader)}')

# Check breed distribution
from collections import Counter
breeds = Counter([dataset.samples[i][1] for i in range(len(dataset))])
print('Breed distribution:', dict(breeds))
"
```

#### 1.2 Baseline Evaluation
```bash
# Evaluate current model before retraining
python src/evaluate_full.py \
  --checkpoint checkpoints/tinydit_final.pt \
  --generate-samples \
  --num-samples 50 \
  --report-path evaluation_baseline.json
```

### Phase 2: Training Execution

#### 2.1 Start Training (Modal GPU)
```bash
# Primary training command
modal run src/train_dit.py data/cats \
  --steps 400000 \
  --batch-size 256 \
  --gradient-accumulation-steps 2 \
  --lr 5e-5 \
  --warmup-steps 15000 \
  --augmentation-level full \
  --checkpoint-interval 50000
```

#### 2.2 Monitor Training
```bash
# Watch training logs (in separate terminal)
tail -f outputs/training.log

# Or use Modal dashboard
# https://modal.com/dashboard
```

#### 2.3 Checkpoint Management
```bash
# Checkpoints saved to:
# - checkpoints/dit_step_50000.pt
# - checkpoints/dit_step_100000.pt
# - ...
# - checkpoints/dit_step_400000.pt (final)
# - checkpoints/dit_step_400000_ema.pt (EMA weights)

# Resume from checkpoint if interrupted
modal run src/train_dit.py data/cats \
  --resume checkpoints/dit_step_200000.pt \
  --steps 400000
```

### Phase 3: Post-Training Evaluation

#### 3.1 Generate Samples
```bash
# Generate evaluation samples
python src/eval_dit.py \
  --checkpoint checkpoints/dit_step_400000_ema.pt \
  --all-breeds \
  --num-samples 16 \
  --output-dir samples/high_accuracy
```

#### 3.2 Compute Metrics
```bash
# Full evaluation (FID, IS, Precision/Recall)
python src/evaluate_full.py \
  --checkpoint checkpoints/dit_step_400000_ema.pt \
  --generate-samples \
  --num-samples 500 \
  --compute-fid \
  --real-dir data/cats/test \
  --fake-dir samples/high_accuracy/evaluation \
  --report-path evaluation_high_accuracy.json
```

#### 3.3 Run Benchmarks
```bash
# Inference benchmarks
python src/benchmark_inference.py \
  --model checkpoints/dit_step_400000_ema.pt \
  --device cuda \
  --num-warmup 10 \
  --num-runs 100 \
  --benchmark-throughput \
  --batch-sizes 1,4,8,16 \
  --report-path benchmark_high_accuracy.json
```

### Phase 4: Deployment

#### 4.1 Export to ONNX
```bash
# Export high-accuracy generator
python src/export_dit_onnx.py \
  --checkpoint checkpoints/dit_step_400000_ema.pt \
  --output frontend/public/models/generator_ha.onnx \
  --quantize
```

#### 4.2 Upload to HuggingFace
```bash
# Upload new model version
python src/upload_to_huggingface.py \
  --generator checkpoints/dit_step_400000_ema.pt \
  --onnx-generator frontend/public/models/generator_ha.onnx \
  --evaluation-report evaluation_high_accuracy.json \
  --benchmark-report benchmark_high_accuracy.json \
  --samples-dir samples/high_accuracy \
  --repo-id d4oit/tiny-cats-model \
  --commit-message "Upload high-accuracy model (400k steps, HA config)"
```

## Expected Improvements

### Quantitative Metrics

| Metric | Current | Expected (400k) | Expected (1M) |
|--------|---------|-----------------|---------------|
| **FID** | ~50-70 | ~30-40 | ~20-30 |
| **Inception Score** | ~3-4 | ~5-6 | ~7-8 |
| **Precision** | ~0.6 | ~0.75 | ~0.85 |
| **Recall** | ~0.4 | ~0.55 | ~0.65 |
| **Training Loss** | ~0.34 | ~0.25 | ~0.20 |

### Qualitative Improvements

| Aspect | Current | Expected |
|--------|---------|----------|
| **Image Sharpness** | Moderate | High |
| **Breed Features** | Some blur | Clear distinctions |
| **Artifacts** | Visible | Minimal |
| **Color Accuracy** | Good | Excellent |
| **Pose Variety** | Limited | More diverse |

### Cost Analysis

| Configuration | Modal Cost (H100) | Time | Recommendation |
|---------------|-------------------|------|----------------|
| 300k steps | ~$10-15 | 18-24h | Budget option |
| **400k steps** | **~$15-25** | **24-36h** | **Recommended** |
| 500k steps | ~$25-35 | 30-40h | High quality |
| 1M steps | ~$50-70 | 60-80h | Research-grade |

**Note:** Costs are estimates based on Modal's H100 pricing (~$2-3/hour). Actual costs may vary.

## Trade-offs

### Positive Consequences
- ✅ **Better sample quality:** More realistic cat images
- ✅ **Improved metrics:** Lower FID, higher IS
- ✅ **Production ready:** More reliable for deployment
- ✅ **Research value:** Better baseline for comparisons
- ✅ **Breed discrimination:** Clearer breed-specific features

### Negative Consequences
- ⚠️ **Higher cost:** $15-70 vs $5-10 for current training
- ⚠️ **Longer training time:** 24-80 hours vs 12-24 hours
- ⚠️ **Larger checkpoints:** More storage required
- ⚠️ **Diminishing returns:** Each 100k steps gives smaller improvements

### Neutral Consequences
- ℹ️ **Same architecture:** No model size increase
- ℹ️ **Same inference cost:** No impact on frontend performance
- ℹ️ **Backward compatible:** Existing code works unchanged

## Alternatives Considered

### Alternative 1: Data Augmentation Only
**Proposal:** Keep 200k steps, add full augmentation.

**Rejected because:**
- Augmentation alone insufficient for major improvements
- Training steps are primary driver of quality
- Minimal extra cost to extend training

### Alternative 2: Larger Model Architecture
**Proposal:** Increase embed_dim (384→512), depth (12→16).

**Rejected because:**
- Increases inference latency (bad for browser)
- Larger checkpoints (bandwidth concerns)
- Current size already good for web deployment
- Can be revisited for server-side deployment

### Alternative 3: Multi-Scale Training
**Proposal:** Train at 128x128 and 256x256 simultaneously.

**Rejected because:**
- Significant complexity increase
- 4x memory requirement at 256x256
- Marginal benefit for current use case
- Can be added in future work

### Alternative 4: Progressive Growing
**Proposal:** Start at 64x64, progressively increase to 128x128.

**Rejected because:**
- Complex implementation
- Longer total training time
- Benefits unclear for 128x128 target
- Better suited for very high resolutions

## Testing Plan

### Unit Tests
```bash
# Test training with minimal steps
python src/train_dit.py data/cats --steps 10 --batch-size 8

# Test gradient accumulation
python src/train_dit.py data/cats --steps 10 --batch-size 8 --gradient-accumulation-steps 2

# Test augmented data loading
python -c "
from dataset import CatsDataset
dataset = CatsDataset('data/cats', augmentation_level='full')
print('Augmentation enabled:', dataset.augmentation is not None)
"
```

### Integration Tests
```bash
# Test checkpoint resume
modal run src/train_dit.py data/cats --steps 100 --checkpoint-interval 50
# Then resume from checkpoint

# Test evaluation script
python src/evaluate_full.py --checkpoint checkpoints/tinydit_final.pt --num-samples 4
```

### End-to-End Test
```bash
# Full pipeline test (small scale)
modal run src/train_dit.py data/cats --steps 1000
python src/eval_dit.py --checkpoint checkpoints/dit_step_1000.pt --num-samples 2
python src/evaluate_full.py --checkpoint checkpoints/dit_step_1000.pt --num-samples 8
```

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Training completes | 400k steps | Checkpoint saved |
| Final loss | < 0.30 | Training logs |
| FID improvement | > 20% reduction | evaluation_report.json |
| Sample quality | Subjective improvement | Visual inspection |
| HuggingFace upload | Success | Repository updated |

## Migration Guide

### For Existing Users

If you have the current model (`tinydit_final.pt`):
- **Keep using it:** Still functional for most applications
- **Upgrade path:** New model is drop-in replacement
- **Frontend:** No changes required

### For New Users

1. **Download high-accuracy model:**
   ```bash
   huggingface-cli download d4oit/tiny-cats-model generator/model.pt --local-dir checkpoints
   ```

2. **Use in frontend:**
   - Models auto-loaded from HuggingFace Hub
   - No local setup required

3. **Generate samples:**
   ```bash
   python src/eval_dit.py --checkpoint checkpoints/model.pt --all-breeds
   ```

## Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| **Phase 1: Preparation** | 1-2 hours | Dataset validation, baseline evaluation |
| **Phase 2: Training** | 24-36 hours | Modal GPU training (400k steps) |
| **Phase 3: Evaluation** | 2-4 hours | Metrics computation, benchmarks |
| **Phase 4: Deployment** | 1-2 hours | ONNX export, HuggingFace upload |
| **Total** | **28-44 hours** | **~$15-25 Modal cost** |

## References

- ADR-017: TinyDiT Training Infrastructure
- ADR-035: Full Model Training & HuggingFace Upload Plan
- DiT Paper: https://arxiv.org/pdf/2212.09748
- Flow Matching: https://arxiv.org/pdf/2210.02747
- Modal GPU Pricing: https://modal.com/pricing

## Appendix: Quick Start Commands

### Start High-Accuracy Training
```bash
modal run src/train_dit.py data/cats \
  --steps 400000 \
  --batch-size 256 \
  --gradient-accumulation-steps 2 \
  --lr 5e-5 \
  --warmup-steps 15000 \
  --augmentation-level full
```

### Monitor Training
```bash
# Watch logs
tail -f outputs/training_*.log

# Check checkpoints
ls -lh checkpoints/dit_step_*.pt
```

### Evaluate Results
```bash
# Generate samples
python src/eval_dit.py --checkpoint checkpoints/dit_step_400000_ema.pt --all-breeds

# Compute FID
python src/evaluate_full.py --checkpoint checkpoints/dit_step_400000_ema.pt --compute-fid
```

### Upload to HuggingFace
```bash
python src/upload_to_huggingface.py \
  --generator checkpoints/dit_step_400000_ema.pt \
  --repo-id d4oit/tiny-cats-model
```
