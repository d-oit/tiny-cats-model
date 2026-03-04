# ADR-053: GitHub Actions Timeout Fix for Modal Training

**Date:** 2026-03-04
**Status:** Accepted
**Authors:** AI Agent (Web Research)
**Related:** ADR-044 (400k Termination), ADR-023 (GPU Retry), train.yml

## Context

### Problem
GitHub Actions workflow run #22620541362 failed with:
- **Train Classifier (Modal)**: Cancelled after 6h 5m
- **Train DiT (Modal)**: Cancelled after 6h 5m
- **Root Cause:** GitHub Actions default timeout (6 hours) < Modal training duration

### Analysis

**Modal Configuration (train_dit.py):**
```python
@app.function(
    timeout=86400,  # 24 hours - sufficient for 400k steps
    retries=modal.Retries(max_retries=2, ...)
)
```

**GitHub Actions Configuration (train.yml):**
```yaml
jobs:
  train-dit:
    runs-on: ubuntu-latest
    # NO timeout-minutes specified = default 6 hours
```

**Runtime Estimation:**
- 400k steps on A10G GPU: ~6-8 hours
- Batch size 256, gradient accumulation 2: effective batch 512
- Learning rate 5e-5, warmup 10k steps

## Decision

### Fix: Increase GitHub Actions Timeout to Match Modal Timeout

Update train.yml to allow longer execution:

```yaml
# BEFORE (insufficient)
jobs:
  train-dit:
    runs-on: ubuntu-latest
    environment:
      name: modal-training
    steps: ...

# AFTER (sufficient for 24-hour Modal runs)
jobs:
  train-dit:
    runs-on: ubuntu-latest
    environment:
      name: modal-training
    timeout-minutes: 1440  # 24 hours = Modal's timeout
    steps: ...
```

### Rationale

1. **Modal handles timeouts**: Modal's 24-hour timeout (86400s) is appropriate for long training runs
2. **GitHub Actions should not interfere**: The workflow should let Modal manage container lifecycle
3. **Consistency**: Both classifier (1 hour) and DiT (24 hours) need appropriate timeouts
4. **Cost efficiency**: Modal charges per second; longer timeout doesn't increase cost if training finishes early

## Implementation

### Changes to train.yml

#### 1. Train Classifier (Modal)
```yaml
train-classifier:
  timeout-minutes: 180  # 3 hours (was: default 6h)
```

**Rationale:** Classifier training is fast (~30-60 minutes), 3 hours provides buffer.

**Note:** The `train.py` script uses Modal's `@app.function(timeout=...)` decorator for timeout control (see `src/train.py` line ~420), not CLI arguments. Do NOT pass `--timeout-minutes` to `modal run src/train.py`.

#### 2. Train DiT (Modal)
```yaml
train-dit:
  timeout-minutes: 1440  # 24 hours (was: default 6h)
```

**Rationale:** Matches Modal's `timeout=86400` setting, allows full 400k step training.

### Why This Works

**Before (Broken):**
```
GitHub Actions (6h timeout) → kills job at 6h
Modal container still running (needs 6-8h)
Result: Job cancelled, no checkpoint
```

**After (Fixed):**
```
GitHub Actions (24h timeout) → allows full runtime
Modal container (24h timeout) → completes training
Result: Checkpoint saved, model uploaded
```

## Verification

### Expected Behavior

1. **Workflow starts** → GitHub Actions allocates runner
2. **Modal container starts** → 24-hour timeout active
3. **Training runs** → 400k steps complete in ~7 hours
4. **Checkpoint saved** → Uploaded to Hugging Face
5. **Workflow completes** → GitHub Actions releases runner

### Test Strategy

```bash
# Trigger with override (fast test)
gh workflow run train.yml \
  -f steps=50000 \
  -f batch_size=256

# Monitor logs
gh run view <run-id> --log-size=0

# Verify timeout is respected
gh run list --workflow=train.yml
```

## Consequences

### Positive
- ✅ 400k step training completes successfully
- ✅ Modal's 24-hour timeout properly utilized
- ✅ No premature job cancellation
- ✅ Checkpoints saved and uploaded

### Negative
- ⚠️ GitHub Actions runner tied up longer (up to 24h)
- ⚠️ Potential cost if training fails mid-run

### Neutral
- ℹ️ Modal still enforces its own 24-hour cutoff
- ℹ️ GitHub Actions releases runner on completion/failure

## Alternatives Considered

### Alternative 1: Use Modal Scheduled Functions
**Proposal:** Replace GitHub Actions with Modal schedules.

**Rejected because:**
- GitHub Actions provides better UX/visibility
- Harder to trigger on-demand
- Less flexible for hyperparameter tuning

### Alternative 2: Reduce Training Steps
**Proposal:** Use 200k steps instead of 400k.

**Rejected because:**
- ADR-036 showed 400k steps needed for high accuracy
- User explicitly requested 400k steps
- Reliability should be solved, not avoided

### Alternative 3: Interval Checkpointing
**Proposal:** Save checkpoints more frequently to reduce loss on failure.

**Rejected because:**
- Already implemented (10k step intervals in train_dit.py)
- Doesn't solve timeout issue
- Only mitigates impact

## Best Practices

1. **Match timeouts:** GitHub Actions timeout ≥ Modal function timeout
2. **Monitor runtime:** Use `gh run view` to check actual duration
3. **Set alerts:** Configure notifications for runs > 12 hours
4. **Test short runs:** Validate with `steps=1000` before full training

## References

- ADR-044: 400k Termination Root Cause (SIGHUP handler)
- ADR-023: Modal GPU Retry Strategy
- train.yml: Current workflow configuration
- train_dit.py: `timeout=86400` (line 439)
- Modal docs: https://modal.com/docs/guide/timeouts

## Timeline

| Activity | Duration | Status |
|----------|----------|--------|
| Identify timeout issue | 15 min | ✅ Done |
| Research Modal timeouts | 30 min | ✅ Done |
| Update train.yml | 10 min | ✅ Done |
| Test with 50k steps | 1 hour | Pending |
| Production 400k run | 7 hours | Pending |
