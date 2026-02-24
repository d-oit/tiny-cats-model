# ADR-008: Adapt tiny-models Architecture for Cats Classifier with Web Frontend

## Status
Proposed

## Date
2026-02-24

## Context
The goal is to create a cats classifier similar to https://github.com/amins01/tiny-models/ but focused on cat breeds instead of landscapes. The system should:
1. Train a DiT (Diffusion Transformer) model on cat breed images
2. Run inference fully in-browser via ONNX + WebAssembly
3. Allow users to upload cat images and classify/design different cat breeds
4. Use Modal for GPU training
5. Get datasets from existing resources (Oxford IIIT Pet, etc.)

## Decision
We will adapt the tiny-models architecture with the following modifications:

### 1. Dataset Strategy
- **Primary**: Oxford IIIT Pet Dataset (12 cat breeds + "other" class)
- **Supplementary**: Additional cat breed datasets from Kaggle/HuggingFace
- **Preprocessing**: Extract breed labels, resize to 128x128 or 256x256
- **Conditioning**: Breed labels (one-hot or text embeddings)

### 2. Model Architecture
- **Backbone**: TinyDiT (similar to tiny-models)
- **Input**: Noise + breed conditioning
- **Conditioning Options**:
  - One-hot breed embeddings (simpler, no extra encoder)
  - Text embeddings via lightweight text encoder
  - AdaLN-Zero for timestep conditioning
- **Output**: Generated cat image matching the breed

### 3. Training Pipeline
- Use Modal GPU (T4/A10G) for training
- Flow matching objective (like tiny-models)
- EMA weights for inference
- Export to ONNX for browser deployment

### 4. Web Frontend
- **Framework**: React + TypeScript (Vite)
- **Model Runtime**: ONNX Runtime Web + WASM
- **UI Components**:
  - Breed selector (dropdown or visual picker)
  - Image upload for classification
  - Generation canvas (show generated cat)
  - Inference dashboard (step time, total latency)
- **Features**:
  - Upload cat image → classify breed
  - Select breed → generate cat image
  - Adjust sampling steps
  - Reset noise for variation

### 5. Project Structure
```
tiny-cats-model/
├── src/                    # Training code (PyTorch)
│   ├── model.py            # DiT architecture
│   ├── train.py            # Training loop (Modal-compatible)
│   ├── dataset.py          # Cat breed datasets
│   ├── export_onnx.py      # ONNX export
│   └── eval.py             # Evaluation
├── frontend/               # React + TypeScript web app
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── hooks/          # Inference hooks
│   │   ├── utils/          # Image/tensor utilities
│   │   └── pages/          # Main pages
│   └── public/             # ONNX models, assets
├── pipelines/              # Training pipelines
│   └── flow/               # Modeling, training, sampling
├── data/                   # Datasets
├── plans/                  # GOAP, ADRs
└── scripts/                # Automation scripts
```

## Consequences

### Positive
- Clear architecture based on proven tiny-models design
- Browser-based inference (no server costs, privacy-preserving)
- GPU training via Modal (fast, cost-effective)
- Well-documented with ADRs and GOAP
- Automation scripts for documentation updates

### Negative
- Requires learning ONNX export and Web Assembly deployment
- Model size constraints for browser (<100MB or split files)
- May need model quantization for faster inference

### Neutral
- Need to adapt dataset loaders for cat breeds
- Frontend needs customization for cat-specific UI

## Alternatives Considered

1. **Keep current ResNet classifier**: Rejected - doesn't support generation, only classification
2. **Server-side inference**: Rejected - defeats privacy goal, adds latency
3. **Use GAN instead of DiT**: Rejected - DiT has better quality and stability
4. **Text-only conditioning**: Rejected - breed labels are simpler and more controlled

## Implementation Plan

### Phase 1: Dataset Preparation
- [ ] Download Oxford IIIT Pet dataset
- [ ] Create breed-to-index mapping
- [ ] Implement dataset loader (PyTorch)
- [ ] Test data preprocessing pipeline

### Phase 2: Model Development
- [ ] Implement TinyDiT for cat generation
- [ ] Add breed conditioning (one-hot)
- [ ] Implement flow matching training loop
- [ ] Test training on small subset

### Phase 3: Modal Training
- [ ] Configure Modal GPU training
- [ ] Set up checkpointing to volumes
- [ ] Train full model (200k steps)
- [ ] Export EMA weights

### Phase 4: ONNX Export
- [ ] Export model to ONNX format
- [ ] Test ONNX inference (Python)
- [ ] Optimize model (quantization if needed)
- [ ] Deploy to frontend/public/models/

### Phase 5: Frontend Development
- [ ] Set up React + TypeScript project
- [ ] Implement breed selector component
- [ ] Implement image upload + classification
- [ ] Implement generation canvas
- [ ] Add inference dashboard
- [ ] Integrate ONNX runtime + web workers
- [ ] Test and optimize latency

### Phase 6: Documentation & CI/CD
- [ ] Update AGENTS.md with new workflows
- [ ] Add frontend build to CI
- [ ] Configure GitHub Pages deployment
- [ ] Write README with usage examples

## Related
- ADR-004: Frontend Cat Model Update (previous frontend work)
- ADR-007: Modal GPU Training Fix
- GOAP.md: Current action items
- Reference: https://github.com/amins01/tiny-models/

## References
- DiT Paper: https://arxiv.org/pdf/2212.09748
- Flow Matching: https://arxiv.org/pdf/2210.02747
- JiT Paper (x prediction): https://arxiv.org/pdf/2511.13720
- tiny-models GitHub: https://github.com/amins01/tiny-models/
- Oxford IIIT Pet Dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/
