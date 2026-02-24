# GOAP: Cats Classifier with Web Frontend

## Goal
Build a cats classifier and generator with web frontend, following the architecture of https://github.com/amins01/tiny-models/ but for cat breeds. Train with Modal GPU, run inference in-browser via ONNX.

## Objectives

1. **Dataset Preparation** - Oxford IIIT Pet + supplementary cat breed datasets
2. **Model Development** - TinyDiT with breed conditioning for generation
3. **Modal Training** - GPU training with proper checkpointing
4. **ONNX Export** - Export model for browser inference
5. **Web Frontend** - React + TypeScript app with ONNX Runtime Web
6. **Documentation** - Maintain ADRs and update AGENTS.md

## Actions

### Phase 1: Dataset Preparation
- [x] Download Oxford IIIT Pet dataset
- [x] Create breed-to-index mapping (12 breeds + other)
- [x] Implement PyTorch dataset loader
- [x] Test preprocessing pipeline (resize, normalize, augment)
- [x] Add Python download script for container compatibility (ADR-011)

### Phase 2: Model Development
- [x] Implement TinyDiT architecture (similar to tiny-models)
- [x] Add breed conditioning (one-hot embeddings)
- [x] Implement flow matching training loop
- [x] Implement AdaLN modulation for conditioning
- [x] Add EMA weight averaging for inference
- [x] Add classifier-free guidance (CFG) sampling

### Phase 3: Modal Training
- [x] Configure Modal GPU training (ADR-007)
- [x] Set up volume for checkpoints
- [x] Implement error handling and logging (ADR-010)
- [x] Add mixed precision training (ADR-010)
- [x] Add memory management (ADR-010)
- [x] Add learning rate warmup with cosine annealing
- [x] Add gradient clipping and OOM recovery
- [ ] Train full model (200k steps, EMA)
- [ ] Evaluate generated samples

### Phase 4: ONNX Export
- [x] Export model to ONNX format (export_onnx.py)
- [x] Add dynamic axes support for batch size
- [ ] Test ONNX inference (Python)
- [ ] Optimize model (quantization if needed)
- [ ] Deploy to frontend/public/models/

### Phase 5: Frontend Development
- [x] Set up React + TypeScript + Vite
- [x] Implement breed selector component
- [x] Implement image upload for classification
- [ ] Implement generation canvas
- [x] Add inference dashboard (step time, latency)
- [x] Integrate ONNX Runtime Web + web workers
- [ ] Test and optimize inference latency

### Phase 6: Documentation & CI/CD
- [x] Update AGENTS.md with new workflows
- [x] Create ADRs for architectural decisions (ADR-008 to ADR-012)
- [x] Fix CI flake8 linting errors (ADR-012)
- [ ] Add frontend build to CI pipeline
- [ ] Configure GitHub Pages deployment
- [x] Write comprehensive README

## Priorities
1. Dataset preparation (high) - foundation for training
2. Model development (high) - core architecture
3. Modal training (high) - GPU access
4. ONNX export (medium) - browser deployment
5. Frontend development (high) - user interface
6. Documentation (medium) - maintain knowledge

## Timeline
- Phase 1: Week 1
- Phase 2: Week 2
- Phase 3: Week 3-4
- Phase 4: Week 5
- Phase 5: Week 6-7
- Phase 6: Week 8

## Current Action Items
- [x] Create ADR-008 for architecture decision
- [x] Create ADR-010 for Modal training improvements
- [x] Create ADR-011 for Modal container dependencies fix
- [x] Download Oxford IIIT Pet dataset (download.py)
- [x] Implement cat breed dataset loader
- [x] Implement TinyDiT model
- [x] Implement error handling in train.py (ADR-010)
- [x] Add logging infrastructure (ADR-010)
- [x] Add mixed precision training (ADR-010)
- [x] Add memory management (ADR-010)
- [ ] Train full model on Modal GPU
- [ ] Complete frontend generation canvas
- [ ] Add frontend build to CI pipeline
- [ ] Configure GitHub Pages deployment

## Success Metrics
- Dataset: 12 cat breeds + other class ready
- Model: <100MB for browser deployment
- Training: Converges with good sample quality
- Inference: <2s for full generation in browser
- Frontend: Responsive, intuitive UI
- CI: All checks pass

## Technical Specifications

### Model Architecture (TinyDiT for Cats)
| Parameter | Value |
|-----------|-------|
| Parameters | ~22M |
| Patches | 256 (16x16) |
| Layers | 12 |
| Hidden Dim | 384 |
| Attention Heads | 6 |
| Image Size | 128x128 or 256x256 |
| Patch Size | 8 or 16 |
| Conditioning | Breed one-hot (13 classes) |

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Steps | 200,000 |
| Batch Size | 256 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW (β1=0.9, β2=0.95) |
| Weight Decay | 0.05 |
| EMA Beta | 0.9999 |
| Loss | Flow matching (v or x prediction) |

### Frontend Stack
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **UI Library**: Material UI
- **Model Runtime**: ONNX Runtime Web + WASM
- **Deployment**: GitHub Pages

## Dataset Details

### Oxford IIIT Pet Dataset
- **Source**: https://www.robots.ox.ac.uk/~vgg/data/pets/
- **Cat Breeds** (12): Abyssinian, Bengal, Birman, Bombay, British_Shorthair, Egyptian_Mau, Maine_Coon, Persian, Ragdoll, Russian_Blue, Siamese, Sphynx
- **Images per breed**: ~100-200
- **Total cat images**: ~2,000
- **Format**: JPG with breed labels

### Preprocessing
1. Resize to 128x128 or 256x256
2. Normalize to [-1, 1]
3. Horizontal flip augmentation
4. One-hot encode breed labels

## References
- ADR-008: Adapt tiny-models Architecture for Cats
- ADR-007: Modal GPU Training Fix
- tiny-models: https://github.com/amins01/tiny-models/
- DiT Paper: https://arxiv.org/pdf/2212.09748
- Flow Matching: https://arxiv.org/pdf/2210.02747
- Oxford IIIT Pet: https://www.robots.ox.ac.uk/~vgg/data/pets/
