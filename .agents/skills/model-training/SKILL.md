---
name: model-training
description: Use for model training, hyperparameter tuning, and Modal GPU training.
---

# Skill: model-training

This skill covers model training workflows for tiny-cats-model.

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

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--backbone` | resnet18 | Model architecture |
| `--device` | cuda/cpu | Compute device |

## Modal GPU Training

```bash
# Set credentials
export MODAL_TOKEN_ID=your_token_id
export MODAL_TOKEN_SECRET=your_token_secret

# Run training on GPU
modal run src/train.py

# Run with custom settings
modal run src/train.py --data-dir data/cats --epochs 20
```

## Model Evaluation

```bash
# Evaluate default model
python src/eval.py

# Evaluate custom checkpoint
python src/eval.py \
  --data-dir data/cats \
  --checkpoint cats_model.pt \
  --backbone resnet18

# Evaluate with metrics
python src/eval.py --data-dir data/cats --checkpoint best_model.pt
```

## Checkpoint Management

```bash
# Default checkpoint: cats_model.pt
# Best practice: Use semantic names
cp cats_model.pt cats_model_resnet18_v1.pt

# List checkpoints
ls -la *.pt
```

## Dataset Preparation

```bash
# Download dataset
bash data/download.sh

# Dataset structure
data/cats/
├── cat/        # Cat images
└── other/      # Non-cat images
```

## Training Tips

1. **Start small** - 5-10 epochs for quick iteration
2. **Use pretrained** - Faster convergence
3. **Monitor loss** - Stop if no improvement
4. **Save best** - Track validation accuracy
5. **GPU for speed** - Use Modal for training

## Common Issues

| Issue | Solution |
|-------|----------|
| OOM errors | Reduce batch-size |
| Slow training | Use GPU or Modal |
| Poor accuracy | Increase epochs, tune lr |
| CUDA error | Use `--device cpu` |

## Modal Configuration

Edit `modal.yml` for GPU settings:

```yaml
training:
  gpu: gpu-t4      # Cost-efficient
  # gpu: gpu-a10g  # Faster
  timeout: 3600    # 1 hour max
```
