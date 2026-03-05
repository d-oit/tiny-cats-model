# ADR-050: train_dit.py CLI Argument Consistency

## Status
Accepted

## Context

While implementing ADR-048 (Modal CLI Syntax Fix), we discovered an inconsistency between `train.py` and `train_dit.py` argument parsing:

| Script | Argument Type | CLI Usage |
|--------|--------------|-----------|
| `train.py` | `--data-dir` (optional flag) | `modal run src/train.py --data-dir data/cats` |
| `train_dit.py` | `data_dir` (positional) | `python src/train_dit.py data/cats` |

After ADR-048 updated the GitHub Actions workflow to use `--data-dir data/cats`, the DiT training job began failing with:
```
error: Got unexpected extra argument (data/cats)
```

This occurred because the workflow was updated to use `--data-dir` but `train_dit.py` still expected a positional argument.

## Decision

**Change `train_dit.py` to use `--data-dir` as a required flag** for consistency with:
1. `train.py` argument style
2. ADR-048 workflow updates
3. Modal 1.0+ CLI best practices

### Before (Positional)
```python
parser.add_argument("data_dir", type=str, help="Path to dataset root")
# Usage: python src/train_dit.py data/cats
```

### After (Required Flag)
```python
parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset root")
# Usage: python src/train_dit.py --data-dir data/cats
```

## Consequences

**Positive**:
- Consistent CLI syntax across all training scripts
- Works correctly with Modal 1.0+ `@app.local_entrypoint()`
- Matches ADR-048 GitHub Actions workflow
- More explicit and self-documenting

**Negative**:
- Breaking change for any existing scripts using positional arguments
- Requires documentation updates

## Migration Guide

Update any local scripts or commands:

```bash
# OLD (will fail)
python src/train_dit.py data/cats --steps 100

# NEW (correct)
python src/train_dit.py --data-dir data/cats --steps 100
```

## Related

- ADR-048: Modal CLI Syntax Fix
- ADR-046: GitHub Actions Missing Dependencies
- `.github/workflows/train.yml`
- `src/train_dit.py`

## Implementation

1. Update `src/train_dit.py` argument parsing
2. Update any documentation referencing the old syntax
3. Update `scripts/train_dit_high_accuracy.sh` if needed
4. Run quality gate to verify

## Verification

- [ ] Local test: `python src/train_dit.py --help` shows `--data-dir` as option
- [ ] Local test: `python src/train_dit.py --data-dir data/cats --steps 10` works
- [ ] GitHub Actions: Train DiT job passes
