# ADR-051: Modal Volume Path Mismatch Fix

## Status
Accepted

## Context
After fixing ADR-050 (train_dit.py CLI argument), the training job started successfully but failed immediately with:

```
2026-03-03 08:56:40 | ERROR    | Training failed: [Errno 2] No such file or directory: 'data/cats'
```

The Modal volumes are configured as:
```python
@app.function(
    volumes={
        "/outputs": volume_outputs,
        "/data": volume_data,
    },
)
```

This means:
- Data volume is mounted at `/data` in the container
- Dataset should be at `/data/cats` (absolute path)
- But the CLI was passing `data/cats` which resolves to `/app/data/cats` (relative to working directory)

## Decision
Use absolute path `/data/cats` in the GitHub Actions workflow to match the Modal volume mount point.

## Changes

### train.yml
Changed from:
```yaml
run: |
  modal run src/train_dit.py \
    --data-dir data/cats \
    ...
```

To:
```yaml
run: |
  modal run src/train_dit.py \
    --data-dir /data/cats \
    ...
```

## Consequences

**Positive:**
- Dataset path now correctly resolves to the volume mount point
- Training can access cached dataset from previous runs
- Consistent with Modal volume configuration

**Negative:**
- Absolute paths are less portable across environments
- Local testing requires different path (`data/cats` vs `/data/cats`)

## Alternatives Considered

1. **Change container working directory**: Change `os.chdir("/app")` to `os.chdir("/data")` - rejected because it breaks other relative paths
2. **Use environment variable**: Set `DATA_DIR` env var - rejected as overkill for single path
3. **Add path mapping logic in script**: Auto-convert relative to absolute - rejected as magic/hidden behavior

## Related
- ADR-024: Modal Volume Storage Organization
- ADR-031: Modal Container Download Scripts Fix
- ADR-050: train_dit.py CLI Argument Fix
