# Tutorial Notebooks

Interactive Jupyter notebooks for learning the tiny-cats-model.

## Quick Start

| Notebook | Description | Runtime |
|----------|-------------|---------|
| [01_quickstart_classification.ipynb](01_quickstart_classification.ipynb) | Classify cat breeds using ONNX model | CPU/GPU |
| [02_conditional_generation.ipynb](02_conditional_generation.ipynb) | Generate cat images with diffusion model | GPU recommended |
| [03_training_fine_tuning.ipynb](03_training_fine_tuning.ipynb) | Train or fine-tune your own model | GPU required |

## Google Colab

Run these notebooks directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d-oit/tiny-cats-model/blob/main/notebooks/01_quickstart_classification.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d-oit/tiny-cats-model/blob/main/notebooks/02_conditional_generation.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d-oit/tiny-cats-model/blob/main/notebooks/03_training_fine_tuning.ipynb)

## Setup

### Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install jupyter torch torchvision huggingface_hub onnxruntime pillow numpy matplotlib
```

### Run Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

## Notebook Details

### 01: Quickstart Classification

**What you'll learn:**
- Load classifier from HuggingFace Hub
- Preprocess images for inference
- Run classification and interpret results
- Visualize predictions

**Requirements:**
- onnxruntime
- huggingface_hub
- pillow
- matplotlib

**Runtime:** CPU (fast)

### 02: Conditional Generation

**What you'll learn:**
- Load diffusion generator from HuggingFace
- Generate cat images by breed (13 breeds)
- Adjust CFG scale for quality/variety
- Batch generation for efficiency
- Speed vs quality trade-offs

**Requirements:**
- torch
- huggingface_hub
- pillow
- matplotlib

**Runtime:** GPU recommended (10-50x faster than CPU)

**Generation Times:**
| Steps | GPU (H100) | CPU |
|-------|------------|-----|
| 10 | ~0.5s | ~10s |
| 50 | ~2s | ~50s |
| 100 | ~4s | ~100s |

### 03: Training & Fine-Tuning

**What you'll learn:**
- Prepare custom datasets
- Configure training parameters
- Run local GPU training
- Use Modal for cloud GPU training
- Export to ONNX
- Upload to HuggingFace

**Requirements:**
- torch
- torchvision
- modal (for cloud training)
- huggingface_hub

**Runtime:** GPU required for training

**Training Times:**
| Steps | Batch Size | GPU (H100) | GPU (T4) |
|-------|------------|------------|----------|
| 50k | 32 | ~2 hours | ~6 hours |
| 100k | 256 | ~4 hours | ~12 hours |
| 400k | 256 | ~16 hours | ~48 hours |

## Troubleshooting

### Issue: CUDA out of memory

**Solution:** Reduce batch size or use gradient accumulation:
```python
# In training config
batch_size = 16  # Reduce from 32
gradient_accumulation_steps = 4  # Effective batch = 64
```

### Issue: Slow generation on CPU

**Solution:** Use Google Colab with GPU:
1. Go to Runtime → Change runtime type
2. Select GPU
3. Re-run all cells

### Issue: Model download fails

**Solution:** Check internet connection and HuggingFace access:
```python
from huggingface_hub import login
login()  # Enter your HF token
```

### Issue: Modal authentication fails

**Solution:** Authenticate Modal CLI:
```bash
modal token set
```

## Resources

- **Model Repository:** https://huggingface.co/d4oit/tiny-cats-model
- **Documentation:** https://github.com/d-oit/tiny-cats-model
- **Issues:** https://github.com/d-oit/tiny-cats-model/issues
- **Discussions:** https://github.com/d-oit/tiny-cats-model/discussions

## Citation

If you use these notebooks in your research:

```bibtex
@misc{tiny-cats-model,
  title = {Tiny-Cats-Model: Cat Breed Classification and Generation},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/d-oit/tiny-cats-model}
}
```

## License

MIT License - See [LICENSE](../LICENSE) for details.
