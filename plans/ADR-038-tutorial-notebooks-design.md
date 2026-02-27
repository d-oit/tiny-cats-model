# ADR-038: Tutorial Notebooks Design

**Date:** 2026-02-27
**Status:** Proposed
**Authors:** AI Agent (GOAP System)
**Related:** GOAP.md Phase 19, ADR-035 (Full Model Training)

## Context

### Current State

The tiny-cats-model project has:
- **Trained models:** Classifier (97.46% accuracy), Generator (200k steps)
- **HuggingFace repo:** d4oit/tiny-cats-model with models uploaded
- **Frontend:** React app with classification and generation
- **Documentation:** README.md, AGENTS.md, ADRs (001-037)

### Problem Statement

Users lack **guided, interactive tutorials** for:
- Loading models from HuggingFace
- Running inference (classification, generation)
- Fine-tuning on custom datasets
- Understanding model capabilities and limitations

### Requirements

**2026 Best Practices for ML Tutorials:**
1. **Interactive notebooks** - Jupyter/Colab format
2. **Step-by-step guidance** - Clear instructions with explanations
3. **Copy-paste examples** - Working code snippets
4. **Visual outputs** - Images, charts, diagrams
5. **Progressive complexity** - Beginner to advanced
6. **Cloud execution** - Google Colab, Kaggle Kernels

## Decision

We will create **3 tutorial notebooks** covering the full user journey.

### Notebook 1: Quickstart Classification

**File:** `notebooks/01_quickstart_classification.ipynb`

**Learning Objectives:**
- Load classifier model from HuggingFace
- Classify cat breed images
- Interpret prediction results
- Handle common errors

**Structure:**
```markdown
# Quickstart: Cat Breed Classification

## Introduction
Learn how to classify cat breeds using our pre-trained model.

## Prerequisites
- Python 3.8+
- pip install onnxruntime transformers huggingface_hub

## Step 1: Install Dependencies
```python
!pip install onnxruntime transformers huggingface_hub pillow numpy
```

## Step 2: Load Model from HuggingFace
```python
from huggingface_hub import hf_hub_download
import onnxruntime as ort

# Download model
model_path = hf_hub_download(
    repo_id="d4oit/tiny-cats-model",
    filename="classifier/model.onnx"
)

# Load ONNX model
session = ort.InferenceSession(model_path)
print(f"Model loaded: {session.get_inputs()[0].shape}")
```

## Step 3: Prepare Image
```python
from PIL import Image
import numpy as np

# Load and preprocess image
image = Image.open("cat.jpg").convert("RGB")
image = image.resize((224, 224))

# Normalize
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image_array = np.array(image) / 255.0
image_array = (image_array - mean) / std

# Add batch dimension
input_tensor = np.expand_dims(image_array, axis=0).astype(np.float32)
```

## Step 4: Run Inference
```python
# Get input name
input_name = session.get_inputs()[0].name

# Run inference
outputs = session.run(None, {input_name: input_tensor})
probabilities = outputs[0][0]

# Get predicted breed
predicted_idx = np.argmax(probabilities)
confidence = probabilities[predicted_idx] * 100

print(f"Predicted breed: {breed_names[predicted_idx]}")
print(f"Confidence: {confidence:.2f}%")
```

## Step 5: Visualize Results
```python
import matplotlib.pyplot as plt

# Show image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(Image.open("cat.jpg"))
plt.title("Input Image")
plt.axis("off")

# Show top 5 predictions
plt.subplot(1, 2, 2)
top_5_idx = np.argsort(probabilities)[::-1][:5]
top_5_probs = probabilities[top_5_idx]
top_5_names = [breed_names[i] for i in top_5_idx]

plt.barh(range(5), top_5_probs * 100)
plt.yticks(range(5), top_5_names)
plt.xlabel("Confidence (%)")
plt.title("Top 5 Predictions")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
```

## Exercise: Try Your Own Images
Upload your cat photos and classify them!

```python
# Your code here
image_path = "your_cat_photo.jpg"
# ... (use code from above)
```

## Common Issues

### Issue: "Model not found"
**Solution:** Check HuggingFace token permissions

### Issue: "Invalid image size"
**Solution:** Ensure image is resized to 224x224

### Issue: "Low confidence predictions"
**Solution:** Image may not contain a cat, or cat is occluded

## Next Steps
- Try Notebook 02: Conditional Generation
- Learn about model architecture in ADR-008
```

---

### Notebook 2: Conditional Generation

**File:** `notebooks/02_conditional_generation.ipynb`

**Learning Objectives:**
- Load generator model from HuggingFace
- Generate cat images by breed
- Adjust classifier-free guidance (CFG)
- Understand trade-offs

**Structure:**
```markdown
# Tutorial: Conditional Cat Image Generation

## Introduction
Generate realistic cat images of specific breeds using our diffusion model.

## Prerequisites
- Python 3.8+
- pip install torch huggingface_hub pillow numpy
- GPU recommended (but CPU works)

## Step 1: Install Dependencies
```python
!pip install torch huggingface_hub pillow numpy matplotlib
```

## Step 2: Load Generator Model
```python
import torch
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="d4oit/tiny-cats-model",
    filename="generator/model.pt"
)

# Load checkpoint
checkpoint = torch.load(model_path, map_location="cpu")
model_state = checkpoint.get("model_state_dict", checkpoint)

# Initialize model (import from src)
from dit import tinydit_128
model = tinydit_128(image_size=128, num_classes=13)
model.load_state_dict(model_state)
model.eval()

print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
```

## Step 3: Generate Your First Cat Image
```python
from flow_matching import sample

# Sample noise
z = torch.randn(1, 3, 128, 128)

# Select breed (0 = Abyssinian)
breed_index = 0
y = torch.zeros(1, 13)
y[0, breed_index] = 1  # One-hot encoding

# Generate image
with torch.no_grad():
    x_generated = sample(
        model,
        z,
        y,
        num_steps=50,
        cfg_scale=1.5
    )

# Display
from PIL import Image
import numpy as np

image = Image.fromarray(
    ((x_generated[0].permute(1, 2, 0).numpy() + 1) / 2 * 255)
    .clip(0, 255).astype(np.uint8)
)
display(image)
```

## Step 4: Generate All Breeds
```python
breed_names = [
    "Abyssinian", "Bengal", "Birman", "Bombay",
    "British_Shorthair", "Egyptian_Mau", "Maine_Coon",
    "Persian", "Ragdoll", "Russian_Blue", "Siamese", "Sphynx", "Other"
]

# Generate one image per breed
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()

for i, breed_name in enumerate(breed_names):
    if i >= len(axes):
        break
    
    # One-hot vector for this breed
    y = torch.zeros(1, 13)
    y[0, i] = 1
    
    # Generate
    z = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        x = sample(model, z, y, num_steps=50, cfg_scale=1.5)
    
    # Display
    image = Image.fromarray(
        ((x[0].permute(1, 2, 0).numpy() + 1) / 2 * 255)
        .clip(0, 255).astype(np.uint8)
    )
    axes[i].imshow(image)
    axes[i].set_title(breed_name)
    axes[i].axis("off")

plt.tight_layout()
plt.show()
```

## Step 5: Adjust CFG Scale
```python
# Compare different CFG values
cfg_values = [0.5, 1.0, 1.5, 2.0, 3.0]

fig, axes = plt.subplots(1, len(cfg_values), figsize=(20, 4))

for idx, cfg in enumerate(cfg_values):
    z = torch.randn(1, 3, 128, 128)
    y = torch.zeros(1, 13)
    y[0, 0] = 1  # Abyssinian
    
    with torch.no_grad():
        x = sample(model, z, y, num_steps=50, cfg_scale=cfg)
    
    image = Image.fromarray(
        ((x[0].permute(1, 2, 0).numpy() + 1) / 2 * 255)
        .clip(0, 255).astype(np.uint8)
    )
    axes[idx].imshow(image)
    axes[idx].set_title(f"CFG = {cfg}")
    axes[idx].axis("off")

plt.tight_layout()
plt.show()

print("Observation: Higher CFG = more detailed but less diverse")
```

## Step 6: Save Generated Images
```python
# Save to directory
import os
os.makedirs("generated_cats", exist_ok=True)

for i, breed_name in enumerate(breed_names):
    y = torch.zeros(1, 13)
    y[0, i] = 1
    z = torch.randn(1, 3, 128, 128)
    
    with torch.no_grad():
        x = sample(model, z, y, num_steps=50, cfg_scale=1.5)
    
    image = Image.fromarray(
        ((x[0].permute(1, 2, 0).numpy() + 1) / 2 * 255)
        .clip(0, 255).astype(np.uint8)
    )
    image.save(f"generated_cats/{breed_name}.png")

print(f"Saved {len(breed_names)} images to generated_cats/")
```

## Advanced: Batch Generation
```python
# Generate multiple images per breed
batch_size = 8
z = torch.randn(batch_size, 3, 128, 128)
y = torch.zeros(batch_size, 13)
y[:, 0] = 1  # All Abyssinian

with torch.no_grad():
    x_batch = sample(model, z, y, num_steps=50, cfg_scale=1.5)

# Display grid
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for idx, ax in enumerate(axes.flatten()):
    image = Image.fromarray(
        ((x_batch[idx].permute(1, 2, 0).numpy() + 1) / 2 * 255)
        .clip(0, 255).astype(np.uint8)
    )
    ax.imshow(image)
    ax.axis("off")

plt.suptitle("Abyssinian Variations")
plt.tight_layout()
plt.show()
```

## Performance Tips

### GPU Acceleration
```python
# Move model to GPU
if torch.cuda.is_available():
    model = model.cuda()
    z = z.cuda()
    y = y.cuda()
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU (slower)")
```

### Reduce Steps for Faster Generation
```python
# Fewer steps = faster but lower quality
x_fast = sample(model, z, y, num_steps=10, cfg_scale=1.5)  # Fast
x_high = sample(model, z, y, num_steps=100, cfg_scale=1.5)  # High quality
```

## Next Steps
- Try Notebook 03: Training & Fine-Tuning
- Explore model architecture in ADR-017
- Read about flow matching in ADR-008
```

---

### Notebook 3: Training & Fine-Tuning

**File:** `notebooks/03_training_fine_tuning.ipynb`

**Learning Objectives:**
- Understand training configuration
- Fine-tune on custom dataset
- Export trained model
- Upload to HuggingFace

**Structure:**
```markdown
# Tutorial: Training & Fine-Tuning

## Introduction
Learn how to train or fine-tune the TinyDiT model on your own dataset.

## Prerequisites
- GPU access (recommended: NVIDIA with 8GB+ VRAM)
- pip install torch torchvision modal huggingface_hub
- Basic PyTorch knowledge

## Part 1: Local Training (CPU/GPU)

### Setup Dataset
```python
from pathlib import Path
import os

# Organize your dataset
dataset_dir = Path("my_custom_cats")
dataset_dir.mkdir(exist_ok=True)

# Expected structure:
# my_custom_cats/
# ├── breed1/
# │   ├── image1.jpg
# │   └── image2.jpg
# ├── breed2/
# │   └── ...
# └── other/
```

### Training Configuration
```python
training_config = {
    "data_dir": "my_custom_cats",
    "steps": 50000,  # Fine-tuning
    "batch_size": 32,
    "learning_rate": 5e-5,
    "image_size": 128,
    "gradient_accumulation_steps": 2,
    "warmup_steps": 5000,
    "augmentation_level": "full",
}

print(f"Training configuration:")
for key, value in training_config.items():
    print(f"  {key}: {value}")

# Estimated training time
gpu_hours = training_config["steps"] * training_config["batch_size"] / 10000
print(f"\nEstimated GPU hours: {gpu_hours:.1f}")
```

### Run Training
```python
import subprocess

# Build command
cmd = [
    "python", "src/train_dit.py",
    training_config["data_dir"],
    "--steps", str(training_config["steps"]),
    "--batch-size", str(training_config["batch_size"]),
    "--lr", str(training_config["learning_rate"]),
    "--gradient-accumulation-steps", str(training_config["gradient_accumulation_steps"]),
    "--warmup-steps", str(training_config["warmup_steps"]),
    "--augmentation-level", training_config["augmentation_level"],
]

# Run training
print("Starting training...")
subprocess.run(cmd)
print("Training complete!")
```

## Part 2: Modal GPU Training (Recommended)

### Setup Modal
```python
# Install Modal
!pip install modal

# Authenticate (run once)
# !modal token set

import modal

# Define Modal app
app = modal.App("tiny-cats-training")

# Define GPU image
image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "modal", "huggingface_hub"
)

# Define training function
@app.function(gpu="T4", timeout=7200)
def train_on_modal():
    import subprocess
    subprocess.run([
        "python", "src/train_dit.py",
        "data/cats",
        "--steps", "100000",
        "--batch-size", "256",
    ])

# Run training
with app.run():
    train_on_modal.remote()
```

## Part 3: Export Model

### Export to ONNX
```python
import torch
import onnx

# Load trained model
checkpoint_path = "checkpoints/dit_model.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Export
from export_dit_onnx import export_generator_onnx

export_generator_onnx(
    checkpoint_path=checkpoint_path,
    output_path="custom_generator.onnx",
    opset=17,
)

print(f"Exported model to custom_generator.onnx")
print(f"Model size: {os.path.getsize('custom_generator.onnx') / 1e6:.2f} MB")
```

### Quantization (Optional)
```python
from optimize_onnx import quantize_model_dynamic

# Quantize
quantize_model_dynamic(
    input_path="custom_generator.onnx",
    output_path="custom_generator_quantized.onnx",
)

print(f"Quantized model: custom_generator_quantized.onnx")
```

## Part 4: Upload to HuggingFace

### Setup Repository
```python
from huggingface_hub import HfApi, create_repo

# Initialize API
api = HfApi()

# Create repo (if doesn't exist)
create_repo(
    repo_id="your-username/my-custom-cats-model",
    repo_type="model",
    exist_ok=True,
)

print("Repository created!")
```

### Upload Files
```python
# Upload model
api.upload_file(
    path_or_fileobj="custom_generator.onnx",
    path_in_repo="generator/model.onnx",
    repo_id="your-username/my-custom-cats-model",
)

# Upload config
api.upload_file(
    path_or_fileobj="config.json",
    path_in_repo="config.json",
    repo_id="your-username/my-custom-cats-model",
)

# Upload samples
for breed in breed_names:
    api.upload_file(
        path_or_fileobj=f"samples/{breed}.png",
        path_in_repo=f"samples/{breed}.png",
        repo_id="your-username/my-custom-cats-model",
    )

print("Upload complete!")
print(f"View at: https://huggingface.co/your-username/my-custom-cats-model")
```

## Exercise: Train on Your Dataset

1. Collect images for your custom breeds
2. Organize in the expected directory structure
3. Run training (local or Modal)
4. Export and upload your model
5. Share with the community!

## Troubleshooting

### Issue: Out of Memory
**Solution:** Reduce batch_size or use gradient accumulation

### Issue: Slow Training
**Solution:** Use Modal GPU or increase num_workers

### Issue: Poor Sample Quality
**Solution:** Increase training steps or check data quality

## Next Steps
- Share your model on HuggingFace
- Read ADR-036 for high-accuracy training
- Join the community discussion
```

## Implementation

### Phase 1: Notebook Creation (Pending)
- [ ] Create `notebooks/01_quickstart_classification.ipynb`
- [ ] Create `notebooks/02_conditional_generation.ipynb`
- [ ] Create `notebooks/03_training_fine_tuning.ipynb`
- [ ] Add sample test images to `notebooks/assets/`

### Phase 2: Testing (Pending)
- [ ] Test all notebooks end-to-end
- [ ] Verify code snippets work
- [ ] Check estimated runtimes
- [ ] Validate outputs

### Phase 3: Documentation (Pending)
- [ ] Add notebooks to README.md
- [ ] Create notebooks README
- [ ] Add Colab badges
- [ ] Link from AGENTS.md

### Phase 4: Distribution (Pending)
- [ ] Upload to HuggingFace datasets
- [ ] Create Google Colab versions
- [ ] Share on social media
- [ ] Add to project website

## Consequences

### Positive
- ✅ **Lower barrier to entry** - Easier for beginners
- ✅ **Interactive learning** - Learn by doing
- ✅ **Cloud execution** - No local setup needed
- ✅ **Shareable** - Easy to distribute
- ✅ **Living documentation** - Always up to date

### Negative
- ⚠️ **Maintenance overhead** - Notebooks need updates
- ⚠️ **Execution environment** - Need to test in multiple environments
- ⚠️ **File size** - Notebooks with outputs can be large

### Neutral
- ℹ️ **Multiple formats** - May want PDF, HTML versions
- ℹ️ **Versioning** - Need to version with model releases

## Alternatives Considered

### Alternative 1: Documentation Only
**Proposal:** Just write text documentation.

**Rejected because:**
- Less engaging than interactive notebooks
- Can't execute code directly
- Harder to follow for beginners

### Alternative 2: Video Tutorials
**Proposal:** Create YouTube video series.

**Partially adopted:**
- Videos are great for visual learners
- Will consider for future
- Notebooks are easier to update

### Alternative 3: Interactive Web App
**Proposal:** Build interactive tutorial website.

**Rejected because:**
- Higher development cost
- Notebooks can be converted to web later
- Jupyter ecosystem is mature

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Notebooks created | 3 | File count |
| Code execution | 100% working | Manual testing |
| User feedback | Positive | GitHub issues, discussions |
| Colab downloads | 100+ | Colab analytics |
| Time to first result | <10 minutes | User testing |

## Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Phase 1 | 6-8 hours | Create notebooks |
| Phase 2 | 2-3 hours | Test notebooks |
| Phase 3 | 1-2 hours | Documentation |
| Phase 4 | 2-3 hours | Distribution |
| **Total** | **11-16 hours** | **~2 days** |

## References

- Jupyter Best Practices: https://jupyter.org/guide
- Google Colab: https://colab.research.google.com/
- HuggingFace Course: https://huggingface.co/course
- ADR-035: Full Model Training Plan
