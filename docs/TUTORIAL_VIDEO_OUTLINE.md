# Tutorial Video Outline

## Series Overview

A 3-part video tutorial series for the tiny-cats-model project.

---

## Episode 1: Quickstart Classification (5-7 minutes)

### Intro (0:00-0:30)
- Welcome and series overview
- What you'll learn: Classify cat breeds in 5 minutes
- Show final result: Upload cat photo → Get breed prediction

### Setup (0:30-1:30)
- Option A: Google Colab (recommended)
  - Click Colab badge in notebook
  - Runtime automatically configured
- Option B: Local setup
  - `pip install onnxruntime huggingface_hub pillow matplotlib`

### Load Model (1:30-2:30)
- Download from HuggingFace Hub
- Load ONNX model with onnxruntime
- Show model info (input/output shape)

### Classify Image (2:30-4:00)
- Load and preprocess image
- Run inference
- Display top-5 predictions with confidence
- Visualize results with matplotlib

### Try Your Own Images (4:00-5:30)
- Upload images in Colab
- Classify multiple images
- Discuss common issues (low confidence, wrong breed)

### Wrap-up (5:30-6:00)
- Recap what was learned
- Link to notebook for practice
- Tease Episode 2: Generation

---

## Episode 2: Conditional Generation (8-10 minutes)

### Intro (0:00-0:45)
- What is diffusion model?
- TinyDiT architecture overview
- Show examples: Generated cats for all 13 breeds

### Setup (0:45-1:30)
- Google Colab with GPU (Runtime → Change runtime type → GPU)
- Install: `pip install torch huggingface_hub pillow matplotlib`
- Verify GPU: `torch.cuda.is_available()`

### Load Generator (1:30-2:30)
- Download PyTorch checkpoint from HuggingFace
- Load model weights
- Show model size (126MB) and parameters (22M)

### Generate First Cat (2:30-4:00)
- Sample noise vector
- Create breed one-hot encoding
- Run flow matching sampling (50 steps)
- Display generated image
- Measure generation time

### Experiment with CFG (4:00-6:00)
- What is Classifier-Free Guidance?
- Generate with CFG=0.5, 1.0, 1.5, 2.0, 3.0
- Compare results side-by-side
- Discuss trade-offs (quality vs diversity)

### Generate All Breeds (6:00-8:00)
- Loop through all 13 breeds
- Create breed grid visualization
- Show variety within same breed (multiple samples)

### Performance Tips (8:00-9:00)
- GPU vs CPU speed comparison
- Reduce steps for faster generation (10 vs 50 vs 100)
- Batch generation for efficiency

### Wrap-up (9:00-10:00)
- Recap CFG, sampling, performance
- Link to notebook
- Tease Episode 3: Training your own model

---

## Episode 3: Training & Fine-Tuning (10-12 minutes)

### Intro (0:00-1:00)
- Why train your own model?
- Custom breeds, better quality, research
- Training options: Local GPU vs Cloud (Modal)

### Dataset Preparation (1:00-3:00)
- Required structure (breed folders)
- Minimum images per breed (50+)
- Run dataset quality checker
- Show example dataset

### Local Training (3:00-5:00)
- Configure training parameters
  - Steps: 50k for fine-tuning
  - Batch size: 32 (adjust for VRAM)
  - Learning rate: 5e-5
- Check GPU availability
- Run training command
- Monitor progress (loss, speed)

### Modal GPU Training (5:00-7:30)
- Why use Modal? (free tier, easy setup)
- Install Modal: `pip install modal`
- Authenticate: `modal token set`
- Define training function with GPU
- Run remote training job
- Monitor in Modal dashboard

### Export Model (7:30-9:00)
- Find latest checkpoint
- Export to ONNX format
- Quantize for smaller size (75% reduction)
- Compare file sizes

### Upload to HuggingFace (9:00-10:30)
- Create HuggingFace account
- Generate API token
- Upload model files
- Create model card
- Verify upload in browser

### Wrap-up & Next Steps (10:30-12:00)
- Series recap
- Share your models with community
- Advanced topics (multi-resolution, more steps)
- Links to all resources

---

## Production Notes

### Recording Setup
- **Screen Resolution**: 1920x1080 (Full HD)
- **Microphone**: Clear audio, minimize background noise
- **Software**: OBS Studio or Loom
- **Code Editor**: VS Code with Jupyter extension

### Editing Tips
- Speed up long operations (model loading, training)
- Add text overlays for key commands
- Include timestamps in description
- Add captions for accessibility

### Distribution
- **YouTube**: Main platform with chapters
- **HuggingFace**: Embed in model cards
- **GitHub**: Link in README and notebooks
- **Social Media**: Short clips for Twitter/LinkedIn

### Follow-up Materials
- GitHub repository with all code
- Colab notebooks for hands-on practice
- Troubleshooting FAQ document
- Community Discord/Slack for questions

---

## Bonus Content Ideas

### Short-form Content (1-2 minutes each)
1. **Quick Tip**: GPU vs CPU performance comparison
2. **Behind the Scenes**: How TinyDiT works (animation)
3. **Community Showcase**: Best generated cats from users
4. **Troubleshooting**: Common errors and fixes
5. **Advanced**: Fine-tuning on custom datasets

### Live Streams
- **Q&A Session**: Answer community questions
- **Live Training**: Train model from scratch (4-6 hours)
- **Code Review**: Review community models
- **Office Hours**: Weekly help session

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Video Views | 1,000+ | YouTube analytics |
| Notebook Opens | 500+ | Colab analytics |
| Model Downloads | 200+ | HuggingFace downloads |
| Community Models | 10+ | User uploads |
| GitHub Stars | +50 | Repository stars |

---

**Created**: 2026-02-27  
**Status**: Outline complete, ready for production
