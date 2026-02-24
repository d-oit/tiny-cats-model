# ADR-007: Modal GPU Training Fix

## Status
Proposed

## Context
The current Modal training setup in `src/train.py` is broken:
1. Hardcoded paths don't match `modal.yml` configuration
2. Missing volume setup for data and outputs
3. Data is local, not accessible in Modal container
4. No proper image with dependencies

Training locally on CPU times out. Need proper Modal GPU training.

## Decision
Fix Modal training with proper setup:

1. **Use Modal Volume** for data and outputs:
   - Create volume `cats-model-data` for dataset
   - Use existing `cats-model-outputs` for checkpoints

2. **Proper Modal Image**:
   - Use `modal.Image.debian_slim()` with pip installs
   - Include torch, torchvision, Pillow, tqdm

3. **Data handling**:
   - Download dataset inside Modal container via `download.sh`
   - Or mount from local after upload

4. **Training function**:
   - Use `@stub.function(gpu="T4", timeout=3600)`
   - Download data at start of function
   - Save outputs to volume

## Implementation Plan

```python
# Proper Modal setup
stub = modal.App("tiny-cats-model")
volume = modal.Volume.from_name("cats-model-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.0.0", "torchvision>=0.15.0", "Pillow>=9.0.0", "tqdm>=4.65.0")
)

@stub.function(
    image=image,
    volumes={"/outputs": volume},
    gpu="T4",
    timeout=3600,
)
def train_modal(...):
    # Download data
    subprocess.run(["bash", "/root/tiny-cats-model/data/download.sh"])
    # Train
    train(data_dir="/root/tiny-cats-model/data/cats", ...)
    # Save to volume (automatic)
```

## Consequences
- **Positive**: Proper GPU training via Modal
- **Positive**: Persistent storage for checkpoints
- **Negative**: Requires Modal token setup

## Alternatives Considered
1. Upload data to volume first - more complex
2. Use S3/cloud storage - requires extra setup
3. Fix local training only - too slow

## Related
- ADR-006: CI Fix Workflow
- GOAP.md
- model-training skill

## References
- Modal docs: https://modal.com/docs/examples/long-training
- Modal GPU: https://modal.com/docs/guide/gpu
