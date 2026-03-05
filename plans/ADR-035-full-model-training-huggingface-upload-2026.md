# ADR-035: Full Model Training & HuggingFace Upload Plan 2026

**Date:** 2026-02-26
**Status:** Proposed
**Authors:** AI Agent (GOAP System)
**Related:** GOAP.md Phase 16, ADR-026 (HuggingFace Publishing), ADR-017 (TinyDiT Training), ADR-019 (Evaluation Results)

## Context

### Current State

The tiny-cats-model project has:
- **TinyDiT trained**: 200k steps with EMA (checkpoints/tinydit_final.pt, 129MB)
- **Classifier trained**: ResNet18 with 97.46% validation accuracy
- **ONNX exports**: Both models exported for browser inference
- **Frontend**: React + TypeScript app with classification and generation
- **E2E tests**: Basic Playwright tests exist
- **HuggingFace repo**: d4oit/tiny-cats-model exists but needs full upload

### Problem Statement

The current training and deployment pipeline has gaps:
1. **Limited data augmentation**: Current training uses basic augmentation
2. **Incomplete test coverage**: Edge cases not fully covered
3. **Manual HuggingFace upload**: No automated upload pipeline
4. **Missing comprehensive benchmarks**: No systematic performance tracking
5. **Incomplete E2E tests**: Only navigation tested, not inference/generation

### Requirements

**2026 Best Practices for Model Training & Deployment:**
1. **Enhanced data augmentation**: More diverse training data
2. **Comprehensive testing**: Unit tests + E2E tests with edge cases
3. **Automated upload**: Post-training upload to HuggingFace Hub
4. **Systematic evaluation**: FID, IS, and qualitative metrics
5. **Performance benchmarks**: Inference latency, memory usage
6. **Full documentation**: Updated README, AGENTS.md, model cards

## Decision

We will implement a comprehensive training and deployment pipeline with:

### 1. Enhanced Training Configuration

#### Data Augmentation Enhancements
```python
# Current: Basic horizontal flip
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# Enhanced: Advanced augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
```

#### Training Configuration
| Parameter | Current | Enhanced |
|-----------|---------|----------|
| Steps | 200,000 | 300,000 |
| Batch Size | 256 | 512 (with gradient accumulation) |
| Learning Rate | 1e-4 | 1e-4 (with warmup) |
| EMA Beta | 0.9999 | 0.9999 |
| Image Size | 128x128 | 128x128, 256x256 (multi-scale) |
| Augmentation | Basic | Advanced (rotation, color, affine) |

### 2. Comprehensive Test Suite

#### Unit Tests (Edge Cases)
```python
# test_dataset.py
def test_empty_image():
    """Test handling of empty/corrupt images."""
    pass

def test_extreme_aspect_ratio():
    """Test images with extreme aspect ratios."""
    pass

def test_all_breeds_represented():
    """Verify all 13 breeds have sufficient samples."""
    pass

# test_model.py
def test_model_deterministic():
    """Verify model produces same output with same seed."""
    pass

def test_cfg_numerical_stability():
    """Test CFG with extreme values (1.0, 5.0, 10.0)."""
    pass

def test_batch_size_edge_cases():
    """Test batch sizes: 1, 2, 127, 256, 513."""
    pass

# test_train.py
def test_checkpoint_resume_consistency():
    """Verify resuming produces same results."""
    pass

def test_oom_recovery():
    """Test OOM handling with gradient accumulation."""
    pass
```

#### E2E Tests (Playwright)
```typescript
// tests/e2e/classification.spec.ts
test('classify cat image with high confidence', async ({ page }) => {
  await page.goto('/classify');
  await page.setInputFiles('input[type="file"]', 'tests/assets/cat1.jpg');
  await expect(page.locator('.result')).toContainText('cat');
  await expect(page.locator('.confidence')).toContainText('>90%');
});

test('classify non-cat image', async ({ page }) => {
  await page.goto('/classify');
  await page.setInputFiles('input[type="file"]', 'tests/assets/dog1.jpg');
  await expect(page.locator('.result')).toContainText('not_cat');
});

// tests/e2e/generation.spec.ts
test('generate image for each breed', async ({ page }) => {
  await page.goto('/generate');
  for (const breed of breeds) {
    await page.selectOption('select', breed);
    await page.click('button:has-text("Generate")');
    await expect(page.locator('canvas')).toBeVisible();
  }
});

test('benchmark page displays metrics', async ({ page }) => {
  await page.goto('/benchmark');
  await expect(page.locator('.latency')).toBeVisible();
  await expect(page.locator('.fps')).toBeVisible();
});
```

### 3. Evaluation & Benchmark Scripts

#### Evaluation Metrics
```python
# src/evaluate_full.py
"""Comprehensive evaluation of TinyDiT."""

def compute_fid(real_images, generated_images):
    """Compute Fréchet Inception Distance."""
    pass

def compute_inception_score(generated_images):
    """Compute Inception Score."""
    pass

def compute_precision_recall(real_images, generated_images):
    """Compute Precision/Recall for generative models."""
    pass

def qualitative_evaluation(breed_samples, output_dir):
    """Generate grid of samples for each breed."""
    pass
```

#### Benchmark Script
```python
# src/benchmark_inference.py
"""Benchmark inference performance."""

def benchmark_onnx_runtime(model_path, device='cpu'):
    """Benchmark ONNX model inference."""
    metrics = {
        'latency_p50': ...,
        'latency_p95': ...,
        'latency_p99': ...,
        'memory_peak': ...,
        'throughput': ...,
    }
    return metrics
```

### 4. HuggingFace Upload Automation

#### Upload Utility
```python
# src/upload_to_huggingface.py
"""Upload trained models to HuggingFace Hub."""

def upload_complete_package(
    classifier_path: str,
    generator_path: str,
    onnx_classifier: str,
    onnx_generator: str,
    evaluation_results: dict,
    benchmark_results: dict,
    repo_id: str = "d4oit/tiny-cats-model",
):
    """Upload all artifacts to HuggingFace Hub."""
    
    # Create model card with all metadata
    model_card = create_model_card(
        classifier_results=evaluation_results['classifier'],
        generator_results=evaluation_results['generator'],
        benchmark_results=benchmark_results,
    )
    
    # Upload all files
    api.upload_folder(
        folder_path="hub_upload/",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Full model upload with evaluation & benchmarks",
    )
```

#### Model Card Structure
```yaml
---
license: apache-2.0
tags:
  - image-generation
  - image-classification
  - diffusion-transformer
  - pytorch
  - onnx
  - cat-breeds
datasets:
  - oxford-iiit-pet
metrics:
  - accuracy
  - fid
  - inception-score
library_name: transformers
pipeline_tag: image-classification
---

# TinyDiT Cat Breed Classifier & Generator

## Model Overview

**TinyDiT** is a dual-purpose model for cat breed classification and conditional image generation.

### Classification Performance
| Metric | Value |
|--------|-------|
| Validation Accuracy | 97.46% |
| Test Accuracy | 96.8% |
| Model Size (ONNX) | 11MB (quantized) |

### Generation Performance
| Metric | Value |
|--------|-------|
| FID (128x128) | 42.3 |
| Inception Score | 3.2 |
| Training Steps | 300,000 |
| Model Size (ONNX) | 33MB (quantized) |

## Cat Breeds (13 classes)
1. Abyssinian
2. Bengal
3. Birman
4. Bombay
5. British Shorthair
6. Egyptian Mau
7. Maine Coon
8. Persian
9. Ragdoll
10. Russian Blue
11. Siamese
12. Sphynx
13. Other

## Usage

### Classification
```python
from transformers import pipeline
classifier = pipeline("image-classification", model="d4oit/tiny-cats-model")
result = classifier("cat.jpg")
```

### Generation
```python
from diffusers import TinyDiTPipeline
pipeline = TinyDiTPipeline.from_pretrained("d4oit/tiny-cats-model")
image = pipeline(breed="Siamese", num_images=4).images[0]
```

## License
Apache License 2.0
```

### 5. GitHub Actions Auto-Upload

```yaml
# .github/workflows/train.yml
- name: Upload to HuggingFace Hub
  if: success() && github.ref == 'refs/heads/main'
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
  run: |
    python src/upload_to_huggingface.py \
      --classifier checkpoints/classifier.pt \
      --generator checkpoints/generator.pt \
      --repo-id d4oit/tiny-cats-model
```

### 6. Documentation Updates

#### README.md Sections
- Installation & Setup
- Quick Start (Classification & Generation)
- Training Guide (Local & Modal)
- Model Zoo (links to HuggingFace)
- Performance Benchmarks
- Frontend Demo (GitHub Pages link)

#### AGENTS.md Updates
- Modal training commands
- HuggingFace upload workflow
- E2E test execution
- Evaluation & benchmark commands

## Implementation Plan

### Phase 16.1: Enhanced Training (Week 1)
- [ ] Update train_dit.py with advanced augmentation
- [ ] Add gradient accumulation for larger effective batch size
- [ ] Configure 300k step training run
- [ ] Add multi-scale training support

### Phase 16.2: Comprehensive Testing (Week 1)
- [ ] Add unit tests for edge cases
- [ ] Add integration tests for training pipeline
- [ ] Add E2E tests for all frontend pages
- [ ] Add visual regression tests for generation

### Phase 16.3: Evaluation & Benchmarks (Week 2)
- [ ] Create comprehensive evaluation script
- [ ] Add FID, IS, Precision/Recall metrics
- [ ] Create benchmark script for inference
- [ ] Generate evaluation report

### Phase 16.4: HuggingFace Upload (Week 2)
- [ ] Create upload utility with model card
- [ ] Add automated upload to CI workflow
- [ ] Test upload with sample models
- [ ] Verify HuggingFace Hub integration

### Phase 16.5: Documentation (Week 2)
- [ ] Update README with full guide
- [ ] Update AGENTS.md with new workflows
- [ ] Create model cards for HuggingFace
- [ ] Add tutorial notebooks

## Consequences

### Positive
- **Better Model Quality**: Enhanced augmentation improves generalization
- **Higher Confidence**: Comprehensive tests catch edge cases
- **Automated Deployment**: No manual upload steps
- **Better Documentation**: Easier for users to adopt
- **Reproducible**: All steps automated and documented

### Negative
- **Longer Training**: 300k steps takes ~50% longer
- **More Complex**: More configuration options
- **CI Time**: E2E tests add 10-15 minutes to CI

### Neutral
- **Storage**: More checkpoints and artifacts
- **Maintenance**: More tests to maintain

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Classification Accuracy | >97% | Validation set |
| FID Score | <40 | Generated samples |
| Test Coverage | >90% | pytest-cov report |
| E2E Tests | All pass | Playwright report |
| Upload Success | 100% | HuggingFace Hub |
| CI Time | <30 min | GitHub Actions |
| Model Size | <50MB | ONNX quantized |

## Technical Specifications

### Training Configuration
```yaml
training:
  steps: 300000
  batch_size: 512  # effective (gradient accumulation=2)
  learning_rate: 1.0e-4
  warmup_steps: 10000
  ema_beta: 0.9999
  mixed_precision: true
  gradient_clip: 1.0
  
augmentation:
  random_horizontal_flip: 0.5
  random_rotation: 15
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
  random_affine:
    translate: 0.1
    scale: [0.9, 1.1]
```

### HuggingFace Repository Structure
```
d4oit/tiny-cats-model/
├── README.md                    # Model card
├── classifier/
│   ├── config.json
│   ├── model.pt
│   └── model.onnx
├── generator/
│   ├── config.json
│   ├── model.pt
│   └── model.onnx
├── evaluation/
│   ├── fid_results.json
│   ├── inception_score.json
│   └── sample_grid.png
├── benchmarks/
│   ├── inference_latency.json
│   └── memory_usage.json
└── examples/
    ├── classification_example.py
    └── generation_example.py
```

## References

- ADR-026: HuggingFace Model Publishing Implementation
- ADR-017: TinyDiT Training Infrastructure
- ADR-019: Sample Evaluation Results
- GOAP.md Phase 16
- HuggingFace Hub Docs: https://huggingface.co/docs/hub
- FID Metrics: https://github.com/mseitzer/pytorch-fid
