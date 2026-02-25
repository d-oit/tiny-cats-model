# ADR-025: Modal Cold Start Optimization

**Date:** 2026-02-25
**Status:** Proposed
**Authors:** AI Agent
**Related:** ADR-010 (Modal Training Improvements), ADR-017 (TinyDiT Training Infrastructure), ADR-020 (Modal CLI-First Training Strategy)

## Context

### Current State

Current training scripts initialize everything in the function body:

```python
@app.function(...)
def train_modal(...):
    sys.path.insert(0, "/app/src")
    os.chdir("/app")
    
    # Setup logging
    logger = setup_logging(log_file)
    logger.info("Starting Modal GPU training")
    
    # Download dataset if needed
    if not Path(data_dir).exists():
        # ... download logic
    
    # Train
    val_acc = train(...)
    return {"status": "completed", "output": output, "val_acc": val_acc}
```

### Issues Identified

1. **No Initialization Caching**: Model/dataloader setup runs on every invocation
2. **No GPU Warm-up**: First inference/training step slower due to CUDA initialization
3. **No Memory Snapshots**: Modal's GPU memory snapshot feature not utilized
4. **Cold Start Latency**: Each training run starts from scratch
5. **No Container Reuse**: No explicit container reuse configuration

### Modal Cold Start Best Practices (2025-2026)

Based on Modal documentation and examples:

1. **Use `@modal.enter()`**: One-time initialization when container starts
2. **GPU Memory Snapshots**: Modal can snapshot GPU memory for instant resume
3. **Pre-load Models**: Load models in `@enter()` to cache on GPU
4. **Warm-up Computations**: Run dummy computation to initialize CUDA
5. **Container Reuse**: Modal reuses containers for subsequent invocations
6. **Image Optimization**: Smaller images = faster pulls = faster cold starts

### Cold Start Latency Breakdown

| Phase | Typical Time | Optimization |
|-------|--------------|--------------|
| Image Pull | 5-30s | Smaller images, layer caching |
| Code Download | 1-5s | Minimal code in image |
| Dependencies Import | 2-10s | Lazy imports |
| GPU Initialization | 1-5s | Warm-up in @enter() |
| Model Load | 1-60s | Pre-load in @enter() |
| **Total** | **10-110s** | **Can reduce to 1-5s** |

## Decision

We will implement cold start optimizations following Modal best practices:

### 1. Use `@modal.enter()` for One-Time Initialization

```python
from modal import App, enter

app = App("tiny-cats-model")

class TrainingContainer:
    """Container with pre-initialized training environment."""
    
    @enter()
    def enter(self):
        """One-time initialization when container starts."""
        import sys
        from pathlib import Path
        
        # Setup paths
        sys.path.insert(0, "/app/src")
        os.chdir("/app")
        
        # Pre-import heavy modules (cached in container)
        import torch
        import torchvision
        
        # Warm up CUDA (first allocation is slow)
        if torch.cuda.is_available():
            # Allocate and free small tensor to initialize CUDA
            _ = torch.zeros(1).cuda()
            torch.cuda.empty_cache()
        
        # Pre-load dataset metadata (not full dataset)
        self.dataset_info = self._load_dataset_metadata()
    
    @app.function(
        gpu="T4",
        timeout=3600,
    )
    def train(self, **kwargs):
        """Training uses pre-initialized container."""
        # Container already initialized, start training immediately
        return train_classifier(**kwargs)
    
    def _load_dataset_metadata(self) -> dict:
        """Load dataset metadata for quick access."""
        return {"classes": ["cat", "other"], "size": "unknown"}
```

### 2. GPU Memory Snapshots (Modal Feature)

Modal automatically snapshots GPU memory for container reuse. To leverage:

```python
@app.function(
    gpu="T4",
    # Modal automatically snapshots GPU memory between invocations
    # No special configuration needed
)
def train(...):
    # First invocation: cold start (~30s)
    # Subsequent invocations: warm start (~1s)
```

**How It Works:**
- Modal snapshots GPU memory after function completes
- Next invocation restores from snapshot
- Model weights stay on GPU between calls
- Reduces cold start from 30s to <1s for same container

### 3. Lazy Imports for Non-Critical Modules

```python
def train(...):
    # Import heavy modules only when needed
    from dataset import cats_dataloader
    from model import cats_model
    
    # Training logic
```

### 4. Pre-Warm CUDA with Dummy Computation

```python
@enter()
def enter(self):
    """Initialize CUDA for faster first training step."""
    import torch
    
    if torch.cuda.is_available():
        # Warm up CUDA kernels
        dummy = torch.randn(1, 3, 128, 128).cuda()
        dummy_conv = torch.nn.Conv2d(3, 64, 3).cuda()
        _ = dummy_conv(dummy)
        
        # Clear warm-up tensors
        del dummy, dummy_conv
        torch.cuda.empty_cache()
```

### 5. Container Reuse Configuration

```python
@app.function(
    gpu="T4",
    # Allow container to stay alive for subsequent calls
    # Modal default: containers reused within ~5 minutes
    _allow_concurrent_inputs=1,  # Single-threaded for training
)
def train(...):
```

### 6. Image Size Optimization (Related to ADR-022)

Smaller images = faster pulls = faster cold starts:

```python
# Use debian_slim (smaller) vs full PyTorch image
image = modal.Image.debian_slim(python_version="3.12")

# vs
image = modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
# This is ~5GB vs ~1GB for debian_slim + uv_pip_install
```

### 7. Minimize Code Download

```python
# Add only necessary files to image
image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file("src/train.py", "/app/train.py")
    .add_local_file("src/dataset.py", "/app/dataset.py")
    .add_local_file("src/model.py", "/app/model.py")
    # Don't add entire project if not needed
)
```

## Implementation Plan

### Phase 1: Add @enter() Initialization

1. Create container class for training
2. Move path setup to @enter()
3. Add CUDA warm-up in @enter()
4. Test cold start improvement

### Phase 2: Optimize Imports

1. Move heavy imports to function scope
2. Add lazy loading for optional features
3. Profile import times

### Phase 3: Container Reuse Testing

1. Test multiple sequential training calls
2. Measure cold vs warm start times
3. Document container reuse behavior

### Phase 4: Image Optimization

1. Implement ADR-022 image changes
2. Measure image pull time improvement
3. Document cold start metrics

## Code Changes

### src/train.py (With Cold Start Optimization)

```python
from modal import App, enter

app = App("tiny-cats-model")

# Volume definitions
output_volume = modal.Volume.from_name("cats-model-outputs", create_if_missing=True)
data_volume = modal.Volume.from_name("cats-dataset", create_if_missing=True)

# Image definition (optimized per ADR-022)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "tar", "curl", "git")
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
    })
    .uv_pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "pillow==11.0.0",
        "tqdm==4.67.1",
    )
    .add_local_file("src/train.py", "/app/train.py")
    .add_local_file("src/dataset.py", "/app/dataset.py")
    .add_local_file("src/model.py", "/app/model.py")
)


class TrainingContainer:
    """Container with pre-initialized training environment for faster cold starts."""
    
    @enter()
    def enter(self):
        """One-time initialization when container starts.
        
        This runs once when the container is created, not on every invocation.
        Reduces cold start latency by:
        - Pre-importing heavy modules (torch, torchvision)
        - Warming up CUDA (first allocation is slow)
        - Setting up paths and environment
        """
        import sys
        import os
        from pathlib import Path
        
        # Setup paths (done once)
        sys.path.insert(0, "/app/src")
        os.chdir("/app")
        
        # Pre-import heavy modules (cached in container memory)
        import torch
        import torchvision
        
        # Warm up CUDA for faster first training step
        if torch.cuda.is_available():
            # First CUDA allocation triggers driver initialization (~1-3s)
            # Do this once in @enter() instead of during training
            _ = torch.zeros(1).cuda()
            
            # Optional: Run small conv operation to warm up CUDA kernels
            dummy_input = torch.randn(1, 3, 32, 32).cuda()
            dummy_conv = torch.nn.Conv2d(3, 16, 3).cuda()
            _ = dummy_conv(dummy_input)
            
            # Clean up warm-up tensors
            del dummy_input, dummy_conv
            torch.cuda.empty_cache()
        
        # Pre-load dataset class (not data)
        from dataset import cats_dataloader
        self._dataloader_fn = cats_dataloader
        
        print("Container initialized successfully")
    
    @app.function(
        image=image,
        volumes={
            "/outputs": output_volume,
            "/data": data_volume,
        },
        gpu="T4",
        timeout=3600,
        retries=modal.Retries(
            max_retries=3,
            backoff_coefficient=2.0,
            initial_delay=10.0,
        ),
    )
    def train_classifier(
        self,
        data_dir: str = "/data/cats",
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-4,
        backbone: str = "resnet18",
        output: str | None = None,
        num_workers: int = 0,
        pretrained: bool = True,
        mixed_precision: bool = True,
        gradient_clip: float = 1.0,
        warmup_epochs: int = 2,
    ) -> dict[str, Any]:
        """Train classifier using pre-initialized container.
        
        Cold start: ~10-20s (first invocation)
        Warm start: ~1-2s (subsequent invocations, container reused)
        """
        import logging
        from datetime import datetime
        from pathlib import Path
        
        # Setup logging with dated directory
        run_date = datetime.now().strftime("%Y-%m-%d")
        checkpoint_dir = f"/outputs/checkpoints/classifier/{run_date}"
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        output = output or f"{checkpoint_dir}/cats_model.pt"
        log_file = f"{checkpoint_dir}/training.log"
        
        logger = setup_logging(log_file)
        logger.info("Starting training (container pre-initialized)")
        
        # Dataset already cached in volume (per ADR-024)
        if not Path(data_dir).exists():
            logger.error(f"Dataset not found at {data_dir}")
            raise FileNotFoundError(f"Dataset not found: {data_dir}")
        
        # Training uses pre-imported modules
        val_acc = train(
            data_dir=data_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            backbone=backbone,
            output=output,
            num_workers=num_workers,
            pretrained=pretrained,
            mixed_precision=mixed_precision,
            gradient_clip=gradient_clip,
            warmup_epochs=warmup_epochs,
            log_file=log_file,
            logger=logger,
        )
        
        # Commit volume (per ADR-024)
        output_volume.commit()
        
        return {"status": "completed", "output": output, "val_acc": val_acc}


# Convenience function for CLI usage
@app.local_entrypoint()
def main(
    data_dir: str = "/data/cats",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    backbone: str = "resnet18",
):
    """Local entrypoint for Modal CLI."""
    container = TrainingContainer()
    result = container.train_classifier.remote(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        backbone=backbone,
    )
    print(f"Training completed: {result}")
```

### src/train_dit.py (Similar Pattern)

```python
# Same pattern as train.py with:
# - @enter() for CUDA warm-up
# - Pre-import DiT modules
# - Container class for training
```

## Consequences

### Positive
- ✅ **Faster Cold Starts**: 30-60s → 10-20s with optimizations
- ✅ **Warm Start Reuse**: Subsequent calls ~1-2s
- ✅ **GPU Pre-initialized**: No CUDA warm-up delay during training
- ✅ **Better Resource Usage**: Container reuse reduces overhead
- ✅ **Improved UX**: Faster feedback during development

### Negative
- ⚠️ **Code Complexity**: Container class adds abstraction
- ⚠️ **Memory Usage**: Pre-loaded modules use container memory
- ⚠️ **Debugging**: @enter() runs once, may confuse debugging

### Neutral
- ℹ️ **Same Training Logic**: Core training unchanged
- ℹ️ **Backward Compatible**: CLI usage same for users

## Alternatives Considered

### Alternative 1: No @enter() (Current State)
**Proposal**: Keep current function-based approach.

**Rejected Because**:
- Every invocation re-initializes environment
- CUDA warm-up happens during training
- Slower cold starts (30-60s vs 10-20s)

### Alternative 2: Always Use Class Methods
**Proposal**: Convert all functions to class methods.

**Rejected Because**:
- Overkill for simple training scripts
- Adds complexity without benefit for single-invocation runs
- Function-based approach is simpler for CLI usage

### Alternative 3: Pre-build Custom Image
**Proposal**: Build custom Docker image with everything pre-loaded.

**Rejected Because**:
- Adds Docker build complexity
- Modal's image building is sufficient
- Loses Modal's build caching benefits

## Success Metrics

- [ ] Cold start time reduced from 30-60s to <20s
- [ ] Warm start time <2s for container reuse
- [ ] CUDA warm-up completed in @enter()
- [ ] Container reuse working for sequential calls
- [ ] Documentation updated with cold start tips

## References

- Modal Cold Start Guide: https://modal.com/docs/guide/cold-start
- Modal @enter(): https://modal.com/docs/reference/modal.enter
- GPU Memory Snapshots: https://modal.com/blog/gpu-mem-snapshots
- High-Performance LLM Inference: https://modal.com/docs/guide/high-performance-llm-inference
- ADR-022: Modal Container Image Optimization
