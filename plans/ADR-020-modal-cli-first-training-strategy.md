# ADR-020: Modal CLI-First Training Strategy

**Date:** 2026-02-25
**Status:** Implemented (updated status check)
**Authors:** AI Agent
**Related:** ADR-007 (Modal GPU Training Fix), ADR-010 (Modal Training Improvements), ADR-011 (Modal Container Dependencies), ADR-017 (TinyDiT Training Infrastructure)

## Context

### Current State

The codebase has two training scripts with Modal support:
1. `src/train.py` - ResNet classifier training
2. `src/train_dit.py` - TinyDiT training

Both scripts use `@stub.function()` decorators for Modal integration. However, the `modal.yml` file suggests a YAML-based configuration approach, which is not Modal's recommended pattern.

### Problem Statement

The current `modal.yml` creates confusion about the training workflow:

1. **YAML vs. Python Configuration**: Modal's modern approach uses Python decorators (`@app.function()`) directly in scripts, not YAML configuration files
2. **Execution Model**: `modal.yml` is not executed directly - it was intended as documentation, but this is unclear
3. **Inconsistent Documentation**: Some docs reference `modal.yml` while scripts use decorators
4. **Limited Flexibility**: YAML configs are less flexible than Python for complex training logic

### Modal's Recommended Pattern

Modal's official documentation recommends:
- Direct script execution with `modal run src/script.py`
- Python decorators for configuration (`@app.function()`)
- CLI arguments for runtime options
- No YAML configuration file execution

## Decision

We will adopt a **Modal CLI-First Training Strategy**:

### 1. Primary Training Interface

All training will use the Modal CLI directly with Python scripts:

```bash
# Classifier training
modal run src/train.py data/cats --epochs 20 --batch-size 64

# DiT training
modal run src/train_dit.py data/cats --steps 200000 --batch-size 256

# With custom options
modal run src/train.py -- --epochs 30 --lr 0.0001 --backbone resnet34
modal run src/train_dit.py -- --steps 100000 --image-size 256
```

### 2. modal.yml as Documentation Reference

The `modal.yml` file will be repurposed as **documentation only**:
- Not executed by Modal
- Serves as a quick reference for common commands
- Contains environment setup reminders
- Links to ADR-020 for full details

### 3. Script Configuration via Decorators

Both training scripts use Python decorators for Modal configuration:

```python
import modal

app = modal.App("tiny-cats-model")

@app.function(
    image=modal.Image.from_registry("pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime")
        .pip_install("torch", "torchvision", "pillow", "tqdm"),
    gpu="T4",  # or "A10G" for faster training
    timeout=7200,  # 2 hours
    volumes={"/outputs": output_volume, "/data": data_volume},
)
def train_on_gpu(...):
    # Training logic here
    pass
```

### 4. Local Testing Support

Scripts support both Modal GPU and local CPU execution:

```bash
# Local CPU testing (debug)
python src/train.py data/cats --epochs 1 --batch-size 8
python src/train_dit.py data/cats --steps 100

# Modal GPU training (production)
modal run src/train.py data/cats --epochs 20
modal run src/train_dit.py data/cats --steps 200000
```

## Benefits

### 1. Flexibility
- **Python over YAML**: Full programming language for configuration
- **Dynamic Configuration**: Conditional logic based on environment
- **Type Safety**: Python type hints vs. untyped YAML
- **IDE Support**: Autocomplete, linting, refactoring

### 2. Better Debugging
- **Local Execution**: Test locally before running on Modal
- **Seamless Transition**: Same script runs locally or on GPU
- **Direct Error Messages**: Python stack traces, not YAML parsing errors
- **Interactive Development**: Use Python REPL for experimentation

### 3. Modern Modal Pattern
- **Official Recommendation**: Matches Modal's documented approach
- **Community Standard**: Aligns with Modal examples and best practices
- **Future-Proof**: Modal's development focus is on Python APIs

### 4. Clearer Intent
- **Self-Documenting**: Code shows exactly what happens
- **No Ambiguity**: Scripts are executable, YAML is reference only
- **Version Control**: Python changes are easier to review than YAML

## Migration Guide

### For Existing Users

#### Before (Confusing)
```bash
# Unclear if modal.yml is used
modal run src/train.py  # Does it read modal.yml?
```

#### After (Clear)
```bash
# Explicit: script runs on Modal GPU
modal run src/train.py data/cats --epochs 20

# Explicit: script runs locally on CPU
python src/train.py data/cats --epochs 20
```

### Command Reference

| Task | Command |
|------|---------|
| **Classifier Training** | |
| Modal GPU (default) | `modal run src/train.py data/cats` |
| Modal GPU (custom epochs) | `modal run src/train.py data/cats --epochs 30` |
| Modal GPU (all options) | `modal run src/train.py -- --epochs 20 --batch-size 64 --backbone resnet34` |
| Local CPU (debug) | `python src/train.py data/cats --epochs 1 --batch-size 8` |
| **DiT Training** | |
| Modal GPU (default) | `modal run src/train_dit.py data/cats` |
| Modal GPU (custom steps) | `modal run src/train_dit.py data/cats --steps 100000` |
| Modal GPU (all options) | `modal run src/train_dit.py -- --steps 200000 --batch-size 256 --lr 0.0001` |
| Local CPU (debug) | `python src/train_dit.py data/cats --steps 100 --batch-size 8` |
| **Authentication** | |
| Setup Modal tokens | `modal token set` |
| Check status | `modal app list` |

### Environment Setup

```bash
# 1. Install Modal CLI
pip install modal

# 2. Authenticate (one-time)
modal token set

# 3. Install project dependencies
pip install -r requirements.txt

# 4. Download dataset
bash data/download.sh

# 5. Train on Modal GPU
modal run src/train.py data/cats --epochs 20
```

## Implementation Details

### Script Structure

Both training scripts follow this pattern:

```python
"""Docstring with usage examples."""

import argparse
import modal

# Modal app definition
app = modal.App("tiny-cats-model")

# GPU function with decorator configuration
@app.function(
    image=modal.Image.from_registry("pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime")
        .pip_install_from_requirements("requirements.txt"),
    gpu="T4",
    timeout=7200,
    volumes={"/outputs": output_volume},
)
def train_on_gpu(data_dir: str, **kwargs):
    """Training logic runs on Modal GPU."""
    # ... training code ...
    pass

# Local entry point
@app.local_entrypoint()
def main(
    data_dir: str = "data/cats",
    epochs: int = 20,
    batch_size: int = 64,
):
    """CLI entry point - runs locally, calls GPU function."""
    train_on_gpu.remote(data_dir, epochs=epochs, batch_size=batch_size)
```

### CLI Argument Passing

Modal CLI passes arguments to scripts:

```bash
# Arguments after -- go to the script
modal run src/train.py -- --epochs 20 --batch-size 64

# Or positional arguments directly
modal run src/train.py data/cats
```

### Volume Mounting

```python
# Create or reference volumes
output_volume = modal.Volume.from_name("cats-model-outputs", create_if_missing=True)
data_volume = modal.Volume.from_name("cats-dataset", create_if_missing=True)

# Mount in function decorator
@app.function(volumes={"/outputs": output_volume, "/data": data_volume})
```

## Consequences

### Positive
- ✅ **Clear Workflow**: `modal run` = GPU, `python` = CPU
- ✅ **Better Documentation**: Scripts are self-documenting
- ✅ **Easier Maintenance**: Python is easier to maintain than YAML + Python
- ✅ **Modern Pattern**: Aligns with Modal's direction
- ✅ **Flexible Configuration**: Python enables complex logic

### Negative
- ⚠️ **Documentation Update**: Need to update all references to modal.yml
- ⚠️ **Learning Curve**: Users unfamiliar with Modal CLI need guidance
- ⚠️ **Legacy Confusion**: Old docs may reference modal.yml as executable

### Neutral
- ℹ️ **modal.yml Retained**: Kept as documentation reference, not deleted
- ℹ️ **Same Functionality**: Training behavior unchanged, only interface clarified

## Alternatives Considered

### Alternative 1: Keep modal.yml as Executable Config
**Proposal**: Use modal.yml with `modal deploy` or similar.

**Rejected Because**:
- Modal's modern approach is Python-first
- YAML is less expressive than Python
- Doesn't match Modal's official examples
- Limited flexibility for complex training logic

### Alternative 2: Hybrid Approach (YAML + Python)
**Proposal**: Use modal.yml for simple config, Python for complex logic.

**Rejected Because**:
- Two sources of truth creates confusion
- Debugging becomes harder (which config is used?)
- Modal doesn't officially support this pattern
- More documentation overhead

### Alternative 3: Pure Python (No modal.yml)
**Proposal**: Delete modal.yml entirely, use only Python.

**Rejected Because**:
- modal.yml is useful as quick reference
- Helps users discover common commands
- Serves as documentation anchor
- Low maintenance cost if marked as reference-only

## Technical Specifications

### Modal CLI Commands

```bash
# Authentication
modal token set                    # Configure tokens
modal token list                   # Show configured tokens

# App Management
modal app list                     # List deployed apps
modal app stop tiny-cats-model     # Stop running app

# Training
modal run src/train.py             # Run classifier training
modal run src/train_dit.py         # Run DiT training
modal shell src/train.py           # Open shell in container

# Volume Management
modal volume ls cats-model-outputs # List volume contents
modal volume get cats-model-outputs /outputs/model.pt ./model.pt  # Download

# Monitoring
modal app logs tiny-cats-model     # View logs
```

### Script Configuration Options

#### Classifier (src/train.py)
| Option | Default | Description |
|--------|---------|-------------|
| `data_dir` | `data/cats` | Dataset path |
| `--epochs` | `10` | Training epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--backbone` | `resnet18` | Model backbone |
| `--mixed-precision` | `True` | Enable AMP |
| `--gradient-clip` | `1.0` | Gradient clipping norm |

#### DiT (src/train_dit.py)
| Option | Default | Description |
|--------|---------|-------------|
| `data_dir` | `data/cats` | Dataset path |
| `--steps` | `200000` | Training steps |
| `--batch-size` | `256` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--image-size` | `128` | Output image size |
| `--warmup-steps` | `10000` | LR warmup steps |
| `--ema-beta` | `0.9999` | EMA decay rate |
| `--resume` | `None` | Checkpoint to resume |

### GPU Configuration

| GPU Type | Cost/Hour | Use Case |
|----------|-----------|----------|
| `T4` | ~$0.35 | Standard training, cost-effective |
| `A10G` | ~$1.00 | Faster training, larger models |
| `A100` | ~$3.00 | Large-scale training (if needed) |

Configure in script:
```python
@app.function(gpu="T4")  # or "A10G", "A100", "H100"
```

## Success Metrics

- [x] ADR-020 created and documented
- [x] modal.yml updated to documentation-only
- [x] GOAP.md updated with CLI commands
- [x] ADR-017 updated with CLI emphasis
- [x] ADR-010 status → Implemented
- [x] AGENTS.md updated with CLI commands
- [x] All training uses Modal CLI consistently

## References

- Modal Documentation: https://modal.com/docs
- Modal GPU Guide: https://modal.com/docs/guide/gpu
- Modal CLI Reference: https://modal.com/docs/reference/cli
- ADR-007: Modal GPU Training Fix
- ADR-010: Modal Training Improvements
- ADR-017: TinyDiT Training Infrastructure

## Appendix: Example Training Session

```bash
# 1. Setup (one-time)
$ modal token set
Token set successfully.

# 2. Download dataset
$ bash data/download.sh
Downloading Oxford IIIT Pet dataset...
Dataset ready at data/cats/

# 3. Test locally (CPU)
$ python src/train.py data/cats --epochs 1 --batch-size 8
2026-02-25 10:00:00 | INFO     | Starting training...
2026-02-25 10:05:00 | INFO     | Epoch 1/1 complete. Loss: 1.234
Training complete!

# 4. Train on Modal GPU
$ modal run src/train.py data/cats --epochs 20
🔨 Running src/train.py
⠦ Building image...
⠦ Pulling code...
⠦ Running training...
2026-02-25 10:30:00 | INFO     | Starting training on GPU (T4)...
2026-02-25 11:30:00 | INFO     | Epoch 20/20 complete. Loss: 0.234
✅ Training complete! Model saved to /outputs/cats_model.pt

# 5. Download model
$ modal volume get cats-model-outputs /outputs/cats_model.pt ./cats_model.pt
Downloaded 45.2 MB
```
