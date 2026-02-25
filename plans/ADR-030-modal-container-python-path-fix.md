# ADR-030: Modal Container Python Path Fix

**Date:** 2026-02-25
**Status:** Proposed
**Authors:** AI Agent (GOAP System)
**Related:** ADR-007 (Modal GPU Training Fix), ADR-010 (Modal Training Improvements), ADR-011 (Modal Container Dependencies), ADR-022 (Modal Container Image Optimization), ADR-025 (Modal Cold Start Optimization)

## Context

### Current State

The Modal training scripts (`src/train.py` and `src/train_dit.py`) have a path configuration mismatch:

**File Placement (via `add_local_file`):**
```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file("src/train.py", "/app/train.py")
    .add_local_file("src/dataset.py", "/app/dataset.py")
    .add_local_file("src/model.py", "/app/model.py")
)
```
Files are placed at `/app/` root (e.g., `/app/dataset.py`, `/app/model.py`).

**Python Path Configuration (via `_initialize_container`):**
```python
def _initialize_container():
    """Initialize container environment for faster training start (ADR-025)."""
    import sys
    import os

    sys.path.insert(0, "/app/src")  # ❌ WRONG: Points to /app/src/
    os.chdir("/app")
```

**Import Statements in Training Code:**
```python
# In train.py (inside train_on_gpu function)
from dataset import cats_dataloader
from model import cats_model

# In train_dit.py (inside train_dit_on_gpu function)
from dit import count_parameters, tinydit_128
from flow_matching import EMA, FlowMatchingLoss
```

### Problem Statement

**Error:** `ModuleNotFoundError: No module named 'dataset'`

**Root Cause:** Path mismatch between file locations and Python path:
- Files are at: `/app/dataset.py`, `/app/model.py`, `/app/dit.py`, `/app/flow_matching.py`
- Python searches: `/app/src/` (due to `sys.path.insert(0, "/app/src")`)
- Result: Python cannot find modules because they're not in `/app/src/`

### Solution Options Considered

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **1** | Change `add_local_file` to put files in `/app/src/` | Keeps sys.path as-is | Requires changing all file paths; inconsistent with working directory |
| **2** | Change `sys.path` to `/app` instead of `/app/src` | Minimal change; matches file locations | None identified |
| **3** | Use `add_local_python_source` (Modal 2026 best practice) | Most modern approach; automatic path handling | Requires Modal version check; may need code restructuring |

## Decision

We will implement **Option 2: Change `sys.path` to `/app`** as the immediate fix, with a note about Option 3 for future consideration.

### Rationale

1. **Minimal Change**: Only requires updating one line in each training script
2. **Matches File Layout**: Files are already at `/app/`, so `/app` in sys.path is correct
3. **Preserves Working Directory**: `os.chdir("/app")` remains unchanged
4. **No Breaking Changes**: Import statements remain the same
5. **Consistent with ADR-022/025**: File placement strategy already established

### Why Not Option 1

Option 1 (moving files to `/app/src/`) would require:
- Changing all `add_local_file` destinations
- Updating `os.chdir("/app")` to `os.chdir("/app/src")` or keeping both paths
- More complex path management for data/scripts at `/app/data/`

### Why Not Option 3 (Yet)

Option 3 (`add_local_python_source`) is Modal's 2026 best practice:
```python
image = modal.Image.debian_slim(python_version="3.12").add_local_python_source(
    "my_app",  # Module name
    "src/train.py",
    "src/dataset.py",
    # ...
)
```

However:
- Requires verifying Modal version supports this feature
- May require restructuring imports to use `my_app.dataset` instead of `dataset`
- Better suited for a future refactor when Modal version is confirmed

## Implementation

### Change 1: `src/train.py` - Fix Python Path

**Before:**
```python
def _initialize_container():
    """Initialize container environment for faster training start (ADR-025)."""
    import torch

    # Setup paths
    sys.path.insert(0, "/app/src")  # ❌ WRONG
    os.chdir("/app")
```

**After:**
```python
def _initialize_container():
    """Initialize container environment for faster training start (ADR-025)."""
    import torch

    # Setup paths - files are at /app/ via add_local_file (ADR-022)
    sys.path.insert(0, "/app")  # ✅ CORRECT: Matches file locations
    os.chdir("/app")
```

### Change 2: `src/train_dit.py` - Fix Python Path

**Before:**
```python
def _initialize_dit_container():
    """Initialize container environment for faster DiT training start (ADR-025)."""
    import torch

    # Setup paths
    sys.path.insert(0, "/app/src")  # ❌ WRONG
    os.chdir("/app")
```

**After:**
```python
def _initialize_dit_container():
    """Initialize container environment for faster DiT training start (ADR-025)."""
    import torch

    # Setup paths - files are at /app/ via add_local_file (ADR-022)
    sys.path.insert(0, "/app")  # ✅ CORRECT: Matches file locations
    os.chdir("/app")
```

### Change 3: Update GOAP.md

Add tracking item for this fix in GOAP.md under a new Phase 11 or as part of Phase 10.

## Consequences

### Positive
- ✅ **Fixes Import Error**: Modules will be found correctly
- ✅ **Minimal Change**: Only 2 lines changed across 2 files
- ✅ **No Breaking Changes**: All existing imports work as before
- ✅ **Consistent**: Path configuration matches file placement strategy from ADR-022
- ✅ **Documented**: ADR captures decision rationale for future reference

### Negative
- ⚠️ **None Identified**: This is a bug fix with no downsides

### Neutral
- ℹ️ **Same Functionality**: Training behavior unchanged after fix
- ℹ️ **Future Refactor**: Option 3 (`add_local_python_source`) can be considered later

## Testing Plan

### Pre-Fix Verification
```bash
# Verify current error
modal run src/train.py --help
# Expected: ModuleNotFoundError or similar import error
```

### Post-Fix Verification
```bash
# Verify fix works
modal run src/train.py --help
# Expected: Help text displays without import errors

# Test DiT training script
modal run src/train_dit.py --help
# Expected: Help text displays without import errors
```

### Integration Testing
```bash
# Run full test suite
make test
# Expected: All tests pass

# Run linting
make lint
# Expected: No linting errors
```

## Future Considerations

### Option 3: `add_local_python_source` (Modal 2026 Best Practice)

When Modal version is confirmed to support `add_local_python_source`, consider refactoring:

```python
# Future implementation (not part of this ADR)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_python_source(
        "tiny_cats",  # Package name
        "src/train.py",
        "src/dataset.py",
        "src/model.py",
        "src/dit.py",
        "src/flow_matching.py",
    )
)

def _initialize_container():
    # No sys.path manipulation needed - automatic with add_local_python_source
    os.chdir("/app")
```

**Benefits:**
- No manual path management
- Better IDE support
- Follows Modal 2026 best practices

**Requirements:**
- Modal version >= 0.72.0 (verify)
- Import changes: `from tiny_cats.dataset import ...`
- Testing to ensure compatibility

## Success Metrics

- [ ] `modal run src/train.py --help` works without import errors
- [ ] `modal run src/train_dit.py --help` works without import errors
- [ ] Training scripts can import all required modules
- [ ] CI/CD pipeline passes with new configuration
- [ ] ADR-030 documented in plans/ folder

## References

- ADR-007: Modal GPU Training Fix
- ADR-010: Modal Training Improvements
- ADR-011: Modal Container Dependencies
- ADR-022: Modal Container Image Optimization
- ADR-025: Modal Cold Start Optimization
- Modal Images Guide: https://modal.com/docs/guide/images
- Modal Python Paths: https://modal.com/docs/guide/python
