# ADR-054: Fix Classifier Training Data Directory Path

## Status
Accepted

## Context
GitHub Actions workflow for classifier training was failing with:
```
DataLoadError: Dataset directory not found: data/cats
```

The error occurred because:
1. The workflow passed `--data-dir data/cats` (relative path)
2. Modal container downloads dataset to `/data/cats` (absolute path in volume)
3. The `train_on_gpu` function received `data/cats` but the dataset was at `/data/cats`

## Root Cause Analysis

The Modal container setup:
- Volume `cats-dataset` is mounted at `/data`
- Dataset download script (`data/download.py`) respects `CATS_DIR` env var, defaulting to `/data/cats`
- The training function `train_on_gpu` has default `data_dir="/data/cats"`

But the GitHub Actions workflow was passing:
```yaml
modal run src/train.py --data-dir data/cats
```

This overrode the correct default with a relative path that doesn't exist in the container.

## Decision

Change the workflow to use the correct absolute path:
```yaml
modal run src/train.py --data-dir /data/cats
```

This aligns with:
- The DiT training workflow which already uses `--data-dir /data/cats`
- The Modal volume mount at `/data`
- The dataset download location

## Consequences

- Classifier training will now find the dataset correctly
- Consistent path handling between classifier and DiT training workflows
- Dataset caching in Modal volumes will work properly

## References
- ADR-024: Modal Volume Storage
- ADR-031: Modal Container Download Scripts Fix
- Error log: https://github.com/d-oit/tiny-cats-model/actions/runs/22680614728