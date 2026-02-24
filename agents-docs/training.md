# Modal GPU Training

## Setup

Modal tokens are configured globally via `modal token set`. Do not set via environment variables.

```bash
# One-time setup
modal token set
```

## Running Training

```bash
# Default training (GPU)
modal run src/train.py

# With custom options
modal run src/train.py -- --epochs 20 --batch-size 64 --lr 0.0001

# Local CPU training (slower, for debugging)
python src/train.py data/cats
```

## Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--backbone` | resnet18 | Model backbone (resnet18/34/50, mobilenet_v3_small) |
| `--output` | cats_model.pt | Output checkpoint path |
| `--no-pretrained` | false | Disable pretrained weights |

## Configuration

- **Config file**: `modal.yml`
- **GPU**: T4 (cost-efficient) or A10G (speed)
- **Timeout**: 1 hour max
- **Checkpoint volume**: `cats-model-outputs` mounted at `/outputs`
