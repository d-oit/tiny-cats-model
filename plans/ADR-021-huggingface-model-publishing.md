# ADR-021: HuggingFace Model Publishing

**Date:** 2026-02-25
**Status:** Proposed
**Authors:** AI Agent
**Related:** GOAP.md Phase 9, ADR-017 (TinyDiT Training), ADR-019 (Evaluation Results)

## Context

### Current State

The project has completed:
- TinyDiT model training (200k steps, EMA enabled)
- ONNX export for browser inference
- Frontend integration with generation and benchmark pages
- 104 generated samples across 13 cat breeds

The model and artifacts exist locally:
- Checkpoint: `checkpoints/tinydit_final.pt` (129MB)
- ONNX models: `frontend/public/models/` (classifier + generator)
- Samples: `samples/generated/step_200000/`

### Problem Statement

The model is not accessible to the broader community. Publishing to HuggingFace Hub will:
1. Make the model discoverable and accessible
2. Provide a model card with documentation
3. Enable inference API for quick testing
4. Allow others to build upon our work

## Decision

We will publish the model to HuggingFace Hub with:

### 1. Model Card (README.md)
```markdown
# TinyDiT Cat Breed Generator

## Model Description
- **Architecture**: TinyDiT (Diffusion Transformer)
- **Task**: Conditional cat image generation
- **Breeds**: 12 cat breeds + other class

## Training Details
- **Dataset**: Oxford IIIT Pet
- **Steps**: 200,000
- **Batch Size**: 256
- **GPU**: Modal A10G

## Usage
python inference.py --breed "Siamese" --num-images 4
```

### 2. Model Files
- PyTorch checkpoint (for fine-tuning)
- ONNX model (for inference)
- Sample images

### 3. Inference Script
- Simple CLI for generating samples
- HuggingFace Spaces compatible

## Consequences

### Positive
- Model accessible to community
- Documentation automatically generated
- Easy integration with other projects

### Negative
- Requires HuggingFace account
- Model storage counts against HF quota

### Neutral
- Model is open source (Apache 2.0)

## Implementation Plan

1. Install `huggingface_hub`
2. Create model card with all details
3. Export PyTorch checkpoint with metadata
4. Create inference script
5. Push to Hub
6. Verify model page

## References

- HuggingFace Hub: https://huggingface.co/docs/hub
- Model Card: https://huggingface.co/docs/hub/model-cards
- ADR-017: TinyDiT Training Infrastructure
- ADR-019: Sample Evaluation Results
