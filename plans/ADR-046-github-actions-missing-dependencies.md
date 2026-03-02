# ADR-046: GitHub Actions Missing Dependencies Fix

**Date:** 2026-03-02
**Status:** Accepted
**Authors:** AI Agent (CI Monitor)
**Related:** ADR-044 (Training Workflow), train.yml workflow

## Context

### Issue Discovered
During 400k training execution via GitHub Actions workflow_dispatch, the Train workflow failed immediately with:

```
ModuleNotFoundError: No module named 'torch'
```

### Root Cause Analysis

The `train.yml` workflow was missing the Python dependencies installation step:

**train-classifier job (lines 157-164):**
```yaml
- name: Install Modal
  run: pip install modal

- name: Run training  # ❌ Missing: pip install -r requirements.txt
  run: modal run src/train.py data/cats
```

**train-dit job (lines 251-260):**
```yaml
- name: Install Modal
  run: pip install modal

- name: Run DiT training  # ❌ Missing: pip install -r requirements.txt
  run: modal run src/train_dit.py data/cats ...
```

The workflow only installed `modal` but not the project dependencies (torch, torchvision, etc.) required to import the training scripts.

### Why This Happened

1. **Assumption error**: Assumed Modal container would handle all dependencies
2. **Local testing bias**: Local development always has requirements.txt installed
3. **Missing CI validation**: The model-import-check job imports the modules but doesn't test the actual training execution path

## Decision

Add explicit dependency installation step in both training jobs before running modal commands.

### Fix Applied

**train-classifier job:**
```yaml
- name: Install Modal
  run: pip install modal

- name: Install dependencies  # ✅ ADDED
  run: pip install -r requirements.txt

- name: Run training
  run: modal run src/train.py data/cats
```

**train-dit job:**
```yaml
- name: Install Modal
  run: pip install modal

- name: Install dependencies  # ✅ ADDED
  run: pip install -r requirements.txt

- name: Run DiT training
  run: modal run src/train_dit.py data/cats ...
```

## Consequences

### Positive
- ✅ Training scripts can now be imported and executed
- ✅ Consistent with local development workflow
- ✅ Explicit dependency declaration improves maintainability

### Negative
- ⚠️ Slightly longer workflow runtime (dependency installation)
- ⚠️ Additional network dependency on PyPI

### Neutral
- ℹ️ Modal container still handles its own dependencies separately
- ℹ️ GitHub Actions cache can mitigate install time

## Lessons Learned

1. **Always install requirements.txt** in CI workflows before running Python scripts
2. **Test the full execution path**, not just imports
3. **Don't assume containerized environments** handle all dependencies

## Verification

After fix:
```bash
gh workflow run train.yml -f steps=400000 -f batch_size=256
# Expected: Training starts without ModuleNotFoundError
```

## References

- Failed run: https://github.com/d-oit/tiny-cats-model/actions/runs/22588979491
- train.yml: .github/workflows/train.yml
- requirements.txt: Project dependencies
