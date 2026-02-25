# ADR-022: Modal Container Image Optimization

**Date:** 2026-02-25
**Status:** Proposed
**Authors:** AI Agent
**Related:** ADR-007 (Modal GPU Training Fix), ADR-010 (Modal Training Improvements), ADR-011 (Modal Container Dependencies), ADR-020 (Modal CLI-First Training Strategy)

## Context

### Current State

The current Modal image configuration in `src/train.py` and `src/train_dit.py`:

```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "tar", "curl")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.0.0",
        "tqdm>=4.65.0",
    )
    .add_local_dir(str(LOCAL_SRC), "/app", copy=True)
)
```

### Issues Identified

1. **Unpinned Dependencies**: Using `>=` version specifiers leads to non-reproducible builds
2. **Slow Package Installation**: Standard `pip_install` is slower than `uv_pip_install`
3. **Large Image Size**: `debian_slim` is good, but PyTorch base images may be more optimized for ML workloads
4. **No Build Caching**: Source code added via `add_local_dir` rebuilds on every change
5. **Missing Environment Variables**: No optimization flags for HuggingFace downloads
6. **No Multi-Stage Build**: All dependencies installed in single layer

### Modal Best Practices (2025-2026)

Based on Modal documentation and examples:

1. **Use `uv_pip_install`**: 10-100x faster than `pip_install` using uv package manager
2. **Pin Exact Versions**: Reproducible builds with `==` version specifiers
3. **Use ML-Optimized Base Images**: `nvidia/cuda:*` or `pytorch/pytorch:*` for GPU workloads
4. **Separate Dependencies Layer**: Install dependencies before adding source code
5. **Set Environment Variables**: `HF_XET_HIGH_PERFORMANCE=1` for faster HuggingFace downloads
6. **Use `.copy()` vs `.add_local_dir()`**: Better control over what gets cached

## Decision

We will optimize Modal container images following these patterns:

### 1. Use uv_pip_install for Faster Installs

```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "tar", "curl", "git")
    .uv_pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "pillow==11.0.0",
        "tqdm==4.67.1",
        "modal==0.72.0",
    )
)
```

**Benefits:**
- 10-100x faster package installation
- Better dependency resolution
- Reduced container build time

### 2. Pin Exact Package Versions

```python
# Before (non-reproducible)
.pip_install("torch>=2.0.0", "torchvision>=0.15.0")

# After (reproducible)
.uv_pip_install(
    "torch==2.5.1",
    "torchvision==0.20.1",
    "pillow==11.0.0",
    "tqdm==4.67.1",
)
```

**Benefits:**
- Reproducible builds across environments
- Easier debugging (known versions)
- Prevents breaking changes from dependency updates

### 3. Use PyTorch Base Image for GPU Workloads

```python
# For GPU training workloads
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        add_python=False,  # Use system Python
    )
    .uv_pip_install(
        "pillow==11.0.0",
        "tqdm==4.67.1",
        "modal==0.72.0",
    )
)
```

**Benefits:**
- Pre-installed CUDA/cuDNN (faster build)
- Optimized for GPU workloads
- Smaller delta from base (faster pulls)

### 4. Separate Dependencies from Source Code

```python
# Install dependencies first (cached layer)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "pillow==11.0.0",
        "tqdm==4.67.1",
    )
    # Add source code last (changes frequently)
    .add_local_file("src/train.py", "/app/train.py")
    .add_local_file("src/dataset.py", "/app/dataset.py")
    .add_local_file("src/model.py", "/app/model.py")
)
```

**Benefits:**
- Dependency layer cached across deploys
- Only source code changes trigger rebuilds
- Faster iteration during development

### 5. Set Environment Variables for Optimization

```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",  # Faster HuggingFace downloads
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",  # Memory optimization
    })
    .uv_pip_install(...)
)
```

### 6. Use Requirements File for Dependencies

```python
# Create requirements-modal.txt for Modal-specific deps
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install_from_requirements("requirements-modal.txt")
)
```

**requirements-modal.txt:**
```
torch==2.5.1
torchvision==0.20.1
pillow==11.0.0
tqdm==4.67.1
modal==0.72.0
```

## Implementation Plan

### Phase 1: Create Optimized Image Configuration

1. Create `requirements-modal.txt` with pinned versions
2. Update `src/train.py` image configuration
3. Update `src/train_dit.py` image configuration

### Phase 2: Test and Validate

1. Test classifier training with new image
2. Test DiT training with new image
3. Measure build time improvement
4. Verify all dependencies work correctly

### Phase 3: Update Documentation

1. Update ADR-020 with new image patterns
2. Update model-training skill documentation
3. Update AGENTS.md with best practices

## Code Changes

### src/train.py (Updated Image)

```python
# Before
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "tar", "curl")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.0.0",
        "tqdm>=4.65.0",
    )
    .add_local_dir(str(LOCAL_SRC), "/app", copy=True)
)

# After
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
```

### src/train_dit.py (Updated Image)

```python
# After
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
    .add_local_file("src/train_dit.py", "/app/train_dit.py")
    .add_local_file("src/dit.py", "/app/dit.py")
    .add_local_file("src/flow_matching.py", "/app/flow_matching.py")
    .add_local_file("src/dataset.py", "/app/dataset.py")
)
```

## Consequences

### Positive
- ✅ **Faster Builds**: 10-100x faster package installation with uv
- ✅ **Reproducible**: Pinned versions ensure consistent builds
- ✅ **Smaller Images**: Better layer caching reduces image size
- ✅ **Faster Iteration**: Source code changes don't rebuild dependencies
- ✅ **Optimized Downloads**: HuggingFace downloads faster with env vars

### Negative
- ⚠️ **Version Maintenance**: Need to update pinned versions periodically
- ⚠️ **Initial Setup**: Creating requirements-modal.txt adds one more file
- ⚠️ **Breaking Changes**: May need to adjust if pinned versions have issues

### Neutral
- ℹ️ **Same Functionality**: Training behavior unchanged
- ℹ️ **Backward Compatible**: Old images still work during transition

## Alternatives Considered

### Alternative 1: Keep Current pip_install
**Proposal**: Continue using `pip_install` with `>=` versions.

**Rejected Because**:
- Slow build times (5-10 minutes for PyTorch)
- Non-reproducible builds
- Modal recommends uv_pip_install

### Alternative 2: Use modal.Image.from_registry for Everything
**Proposal**: Always use PyTorch base images.

**Rejected Because**:
- Larger base images (~5GB vs ~1GB for debian_slim)
- Overkill for simple training scripts
- debian_slim + uv_pip_install is sufficient

### Alternative 3: Use Dockerfile for Custom Image
**Proposal**: Build custom Docker image and push to registry.

**Rejected Because**:
- Adds complexity (Docker build, registry management)
- Modal's Image API is sufficient
- Loses Modal's build caching benefits

## Success Metrics

- [ ] Build time reduced from 5-10 min to <2 min
- [ ] All training scripts work with new image
- [ ] No dependency conflicts
- [ ] Documentation updated

## References

- Modal Images Guide: https://modal.com/docs/guide/images
- Modal Examples: https://github.com/modal-labs/modal-examples
- uv Package Manager: https://github.com/astral-sh/uv
- ADR-020: Modal CLI-First Training Strategy
- ADR-010: Modal Training Improvements
