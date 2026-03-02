# ADR-048: Modal CLI Syntax Fix for train.yml

**Date:** 2026-03-02
**Status:** Accepted
**Authors:** AI Agent (CI Monitor)
**Related:** ADR-046, train.yml, src/train_dit.py

## Context

### Issue Discovered
Train workflow run #22589524429 failed with:

```
Usage: modal run src/train_dit.py [OPTIONS]
Error: Got unexpected extra argument (data/cats)
```

### Root Cause Analysis

**Incorrect syntax in train.yml:**
```yaml
# WRONG - positional argument not supported by @app.local_entrypoint()
run: |
  modal run src/train_dit.py data/cats \
    --steps ${{ github.event.inputs.steps || '400000' }} \
    --batch-size ${{ github.event.inputs.batch_size || '256' }}
```

**Correct syntax:**
```yaml
# CORRECT - use --data-dir option
run: |
  modal run src/train_dit.py \
    --data-dir data/cats \
    --steps ${{ github.event.inputs.steps || '400000' }} \
    --batch-size ${{ github.event.inputs.batch_size || '256' }}
```

### Why This Happened

1. **Modal 1.0+ uses `@app.local_entrypoint()`** which expects `--option` syntax
2. **Old syntax confusion**: Previously may have used positional args with different Modal versions
3. **Missing documentation**: No clear examples of correct CLI syntax

## Decision

Fix train.yml to use `--data-dir` option instead of positional argument.

### Fix Applied

**train-dit job (lines 257-266):**
```yaml
- name: Run DiT training
  env:
    MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
    MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
  run: |
    modal run src/train_dit.py \
      --data-dir data/cats \
      --steps ${{ github.event.inputs.steps || '400000' }} \
      --batch-size ${{ github.event.inputs.batch_size || '256' }} \
      --lr ${{ github.event.inputs.lr || '5e-5' }} \
      --gradient-accumulation-steps ${{ github.event.inputs.gradient_accumulation_steps || '2' }}
```

**train-classifier job:**
```yaml
- name: Run training
  env:
    MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
    MODAL_TOKEN_SECRET: ${{ secrets.MODAL_TOKEN_SECRET }}
  run: |
    modal run src/train.py \
      --data-dir data/cats
```

## Modal CLI Usage Patterns

### Correct Syntax for Modal 1.0+

**With @app.local_entrypoint():**
```bash
# Use --option syntax
modal run src/train_dit.py \
  --data-dir data/cats \
  --steps 400000 \
  --batch-size 256

# NOT positional arguments
modal run src/train_dit.py data/cats  # ❌ WRONG
```

**With @app.function():**
```bash
# Call remote function directly
modal run src/train_dit.py::train_dit_on_gpu \
  --data-dir data/cats \
  --steps 400000
```

### Script Usage

**train_dit_high_accuracy.sh:**
```bash
# Local test (uses correct syntax)
bash scripts/train_dit_high_accuracy.sh --local
# Calls: modal run src/train_dit.py --data-dir data/cats --steps 4000 ...
```

## Consequences

### Positive
- ✅ Workflow now uses correct Modal CLI syntax
- ✅ Training jobs execute without argument errors
- ✅ Clear documentation of correct patterns

### Negative
- ⚠️ Must always use `--data-dir` not positional arg

### Neutral
- ℹ️ Modal 1.0+ is strict about option syntax

## Verification

After fix:
```bash
# Latest successful runs:
# - 22590291667: SUCCESS (4m28s)
# - 22589803563: SUCCESS (4m3s)
```

## Best Practices

1. **Always use `--option` syntax** with Modal 1.0+
2. **Test locally first**: `modal run src/train_dit.py --help`
3. **Check entrypoint signature**: Look for `@app.local_entrypoint()`

## References

- Failed run: https://github.com/d-oit/tiny-cats-model/actions/runs/22589524429
- Successful runs: 22590291667, 22589803563
- src/train_dit.py: @app.local_entrypoint() decorator
- .github/workflows/train.yml
