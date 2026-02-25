# ADR-031: Modal Container Download Scripts Fix

**Date:** 2026-02-25
**Status:** Proposed
**Authors:** AI Agent (GOAP System)
**Related:** ADR-007 (Modal GPU Training Fix), ADR-011 (Modal Container Dependencies), ADR-022 (Modal Container Image Optimization), ADR-030 (Modal Container Python Path Fix)

## Context

### Current State

The Modal training scripts (`src/train.py` and `src/train_dit.py`) attempt to download the dataset inside the container when it's not found in the mounted volume:

**Download Logic in `train_on_gpu` (train.py):**
```python
if not Path(data_dir).exists() or not list(Path(data_dir).iterdir()):
    logger.info("Dataset not found, downloading...")
    Path("/data").mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["python", "data/download.py"],
        cwd="/app",
        env={**os.environ, "DATA_DIR": "/data", "CATS_DIR": "/data/cats"},
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        logger.warning(f"Python download failed: {result.stderr}")
        logger.info("Trying bash download script...")
        subprocess.run(
            ["bash", "data/download.sh"],
            cwd="/app",
            env={**os.environ, "DATA_DIR": "/data", "CATS_DIR": "/data/cats"},
            check=True,
            capture_output=True,
            text=True,
        )
```

**Download Logic in `train_dit_on_gpu` (train_dit.py):**
```python
if not Path(data_dir).exists() or not list(Path(data_dir).iterdir()):
    logger.info("Dataset not found, downloading...")
    import subprocess

    result = subprocess.run(
        ["python", "data/download.py"],
        cwd="/app",
        env={**os.environ, "DATA_DIR": "/data", "CATS_DIR": "/data/cats"},
        capture_output=True,
        text=True,
        timeout=600,
    )
```

### Problem Statement

**Error:** `bash: data/download.sh: No such file or directory`

**Root Cause:** The Modal container image is built with only specific source files added via `add_local_file`:

**Current Image Configuration (train.py):**
```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "tar", "curl", "git")
    .env({...})
    .uv_pip_install("torch==2.5.1", "torchvision==0.20.1", ...)
    .add_local_file("src/train.py", "/app/train.py")
    .add_local_file("src/dataset.py", "/app/dataset.py")
    .add_local_file("src/model.py", "/app/model.py")
    # ❌ Missing: data/download.py, data/download.sh
)
```

**Current Image Configuration (train_dit.py):**
```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "tar", "curl", "git")
    .env({...})
    .uv_pip_install("torch==2.5.1", "torchvision==0.20.1", ...)
    .add_local_file("src/train_dit.py", "/app/train_dit.py")
    .add_local_file("src/dit.py", "/app/dit.py")
    .add_local_file("src/flow_matching.py", "/app/flow_matching.py")
    .add_local_file("src/dataset.py", "/app/dataset.py")
    # ❌ Missing: data/download.py, data/download.sh
)
```

**Files in Local Project:**
```
data/
├── download.py    # ✅ Exists locally
└── download.sh    # ✅ Exists locally
```

**Files in Modal Container:**
```
/app/
├── train.py       # ✅ Added via add_local_file
├── dataset.py     # ✅ Added via add_local_file
├── model.py       # ✅ Added via add_local_file
├── data/          # ❌ MISSING - directory not added
│   ├── download.py
│   └── download.sh
```

### Impact

1. **Training Fails:** When dataset is not cached in volume, download scripts cannot be executed
2. **Error Message:** `bash: data/download.sh: No such file or directory`
3. **Workaround Required:** Users must manually pre-populate the volume before training
4. **Inconsistent Behavior:** Local training works (scripts available), Modal training fails

### Solution Options Considered

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **1** | Add download scripts to container image | Simple; self-contained container; works out-of-box | Slightly larger image (~50KB) |
| **2** | Download scripts from GitHub at runtime | No image size increase | Requires network; adds complexity; potential failure point |
| **3** | Embed download logic directly in train.py | No external dependencies | Code duplication; harder to maintain |
| **4** | Pre-populate volume via separate Modal function | Separates concerns | Extra step for users; more complex workflow |

## Decision

We will implement **Option 1: Add download scripts to container image** as the primary fix.

### Rationale

1. **Minimal Change**: Only requires adding 2 lines to each image configuration
2. **Self-Contained**: Container has everything needed to download dataset
3. **Consistent**: Matches existing pattern of adding source files via `add_local_file`
4. **Reliable**: No external network calls or additional Modal functions needed
5. **Small Overhead**: Download scripts are tiny (~5KB total)
6. **Preserves Workflow**: No changes to user workflow or training logic

### Why Not Other Options

**Option 2 (Download from GitHub):**
- Adds network dependency during training
- Potential failure if GitHub is unavailable
- More complex error handling needed

**Option 3 (Embed Logic):**
- Duplicates code between train.py and train_dit.py
- Harder to maintain download logic in multiple places
- Violates DRY principle

**Option 4 (Pre-populate Volume):**
- Extra step for users
- Requires separate Modal function
- More complex documentation and onboarding

## Implementation

### Change 1: `src/train.py` - Add Download Scripts to Image

**Before:**
```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "tar", "curl", "git")
    .env({...})
    .uv_pip_install("torch==2.5.1", "torchvision==0.20.1", "pillow==11.0.0", "tqdm==4.67.1")
    .add_local_file("src/train.py", "/app/train.py")
    .add_local_file("src/dataset.py", "/app/dataset.py")
    .add_local_file("src/model.py", "/app/model.py")
)
```

**After:**
```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "tar", "curl", "git")
    .env({...})
    .uv_pip_install("torch==2.5.1", "torchvision==0.20.1", "pillow==11.0.0", "tqdm==4.67.1")
    .add_local_file("src/train.py", "/app/train.py")
    .add_local_file("src/dataset.py", "/app/dataset.py")
    .add_local_file("src/model.py", "/app/model.py")
    .add_local_file("data/download.py", "/app/data/download.py")
    .add_local_file("data/download.sh", "/app/data/download.sh")
)
```

### Change 2: `src/train_dit.py` - Add Download Scripts to Image

**Before:**
```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "tar", "curl", "git")
    .env({...})
    .uv_pip_install("torch==2.5.1", "torchvision==0.20.1", "pillow==11.0.0", "tqdm==4.67.1")
    .add_local_file("src/train_dit.py", "/app/train_dit.py")
    .add_local_file("src/dit.py", "/app/dit.py")
    .add_local_file("src/flow_matching.py", "/app/flow_matching.py")
    .add_local_file("src/dataset.py", "/app/dataset.py")
)
```

**After:**
```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "tar", "curl", "git")
    .env({...})
    .uv_pip_install("torch==2.5.1", "torchvision==0.20.1", "pillow==11.0.0", "tqdm==4.67.1")
    .add_local_file("src/train_dit.py", "/app/train_dit.py")
    .add_local_file("src/dit.py", "/app/dit.py")
    .add_local_file("src/flow_matching.py", "/app/flow_matching.py")
    .add_local_file("src/dataset.py", "/app/dataset.py")
    .add_local_file("data/download.py", "/app/data/download.py")
    .add_local_file("data/download.sh", "/app/data/download.sh")
)
```

### Change 3: Update GOAP.md

Add Phase 12 tracking for this fix in GOAP.md.

## Consequences

### Positive
- ✅ **Fixes Download Error**: Scripts available in container for dataset download
- ✅ **Self-Contained Container**: No external dependencies for dataset download
- ✅ **Minimal Change**: Only 4 lines added across 2 files
- ✅ **Consistent Pattern**: Follows existing `add_local_file` pattern
- ✅ **Small Overhead**: ~5KB image size increase (negligible)
- ✅ **Better UX**: Training works out-of-box without manual volume setup

### Negative
- ⚠️ **Slightly Larger Image**: ~5KB increase (negligible impact on build time)
- ⚠️ **Download Scripts in Container**: May be considered unnecessary if volume always pre-populated (but provides fallback)

### Neutral
- ℹ️ **Same Functionality**: Download logic unchanged, just made available in container
- ℹ️ **Backward Compatible**: Works with existing volume-cached datasets

## Testing Plan

### Pre-Fix Verification
```bash
# Verify current error (dataset not cached)
modal run src/train.py data/cats --epochs 1
# Expected: bash: data/download.sh: No such file or directory
```

### Post-Fix Verification
```bash
# Verify image builds with download scripts
modal deploy src/train.py
# Expected: Image builds successfully

# Verify training works with empty volume
# (Delete volume cache first if needed)
modal run src/train.py data/cats --epochs 1
# Expected: Dataset downloads successfully, training completes

# Verify DiT training
modal run src/train_dit.py data/cats --steps 100
# Expected: Dataset downloads successfully, training starts
```

### Integration Testing
```bash
# Run linting
make lint
# Expected: No linting errors

# Run tests
make test
# Expected: All tests pass
```

## Container File Layout

After this fix, the Modal container will have:

```
/app/
├── train.py              # Training script (classifier)
├── train_dit.py          # Training script (DiT)
├── dataset.py            # Dataset utilities
├── model.py              # Classifier model
├── dit.py                # DiT model
├── flow_matching.py      # Flow matching utilities
└── data/
    ├── download.py       # Python download script ✅ NEW
    └── download.sh       # Bash download script ✅ NEW
```

## Image Size Impact

| Component | Size | Change |
|-----------|------|--------|
| Base image (debian_slim) | ~100MB | - |
| PyTorch + torchvision | ~2GB | - |
| Source files (existing) | ~100KB | - |
| Download scripts (new) | ~5KB | +5KB |
| **Total** | **~2.1GB** | **+0.0002%** |

The image size increase is negligible and has no practical impact on build or deployment time.

## Success Metrics

- [ ] `modal run src/train.py data/cats --epochs 1` completes without download errors
- [ ] `modal run src/train_dit.py data/cats --steps 100` completes without download errors
- [ ] Dataset downloads automatically when not cached in volume
- [ ] Image builds successfully with new `add_local_file` entries
- [ ] CI/CD pipeline passes with new configuration
- [ ] ADR-031 documented in plans/ folder

## References

- ADR-007: Modal GPU Training Fix
- ADR-011: Modal Container Dependencies
- ADR-022: Modal Container Image Optimization
- ADR-030: Modal Container Python Path Fix
- Modal Images Guide: https://modal.com/docs/guide/images
- Modal add_local_file: https://modal.com/docs/guide/images#adding-files
