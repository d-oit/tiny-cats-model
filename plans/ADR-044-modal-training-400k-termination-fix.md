# ADR-044: Modal Training 400k Termination Root Cause and Fix

**Date:** 2026-03-01
**Status:** Proposed
**Authors:** AI Agent (Analysis Swarm)
**Related:** ADR-043 (Early Termination Investigation), ADR-042, ADR-023, GOAP.md Phase 18

## Context

### Incident Summary

| Metric | Value |
|--------|-------|
| Training Job | 400k step high-accuracy training |
| Started | 2026-02-28 17:02 UTC |
| Stopped | ~17:34 UTC (after step 200) |
| Duration | ~32 minutes |
| Exit Code | Unknown (no error logged) |
| App State | stopped |

### Investigation Results

**Root Cause:** Missing SIGHUP signal handler combined with `nohup` background execution.

When running `nohup modal run ... &`:
1. The shell running the script terminates after launching
2. SIGHUP signal is sent to the `modal run` process
3. **No SIGHUP handler exists in train_dit.py** (only SIGINT and SIGTERM)
4. Process terminates silently without checkpoint save

### Code Analysis Findings

| Issue | Location | Severity | Impact |
|-------|----------|----------|--------|
| Missing SIGHUP handler | train_dit.py:821-822 | **CRITICAL** | Silent termination on shell exit |
| nohup with modal run | train_dit_high_accuracy.sh:42 | **HIGH** | CLI spawns separate process |
| No exit status logging | train_dit.py:1025-1027 | **MEDIUM** | No record of termination reason |
| 24h timeout borderline | train_dit.py:437 | **LOW** | Could exceed for slower training |
| Signal number not mapped | train_dit.py:818 | **LOW** | Harder to debug |

## Decision

### Fix 1: Add SIGHUP Handler (CRITICAL)

```python
# train_dit.py lines 821-822 - ADD SIGHUP handler
old_handler = signal.signal(signal.SIGINT, signal_handler)
old_handler_term = signal.signal(signal.SIGTERM, signal_handler)
old_handler_hup = signal.signal(signal.SIGHUP, signal_handler)  # ADD THIS

# In finally block (lines 1025-1027) - ADD restore
finally:
    signal.signal(signal.SIGINT, old_handler)
    signal.signal(signal.SIGTERM, old_handler_term)
    signal.signal(signal.SIGHUP, old_handler_hup)  # ADD THIS
```

### Fix 2: Add Exit Status Logging

```python
# train_dit.py finally block - ADD logging
finally:
    if shutdown_requested:
        logger.info(f"Training ended at step {step}/{steps} due to signal")
    else:
        logger.info(f"Training ended at step {step}/{steps}")
    # ... existing signal restore code
```

### Fix 3: Use Modal Cron/Schedule Instead of nohup

The `nohup modal run ... &` pattern is fundamentally flawed because:
- Modal CLI manages container lifecycle separately
- nohup only protects the outer CLI process
- Better: Use Modal's built-in scheduling or GitHub Actions

```bash
# Option A: GitHub Actions (RECOMMENDED)
gh workflow run train.yml -f steps=400000

# Option B: Modal scheduled function
# Use modal.schedule with longer timeout
```

### Fix 4: Signal Name Mapping

```python
# train_dit.py signal_handler - IMPROVE logging
import signal as signal_module

def signal_handler(signum: int, frame: Any) -> None:
    nonlocal shutdown_requested
    signal_name = signal_module.Signals(signum).name
    logger.warning(f"Signal {signum} ({signal_name}) received, finishing current step...")
    shutdown_requested = True
```

## Implementation Plan

| Priority | Fix | File | Lines | Status |
|----------|-----|------|-------|--------|
| P0 | Add SIGHUP handler | train_dit.py | 821-822 | ✅ DONE |
| P0 | Restore SIGHUP in finally | train_dit.py | 1025-1029 | ✅ DONE |
| P1 | Add exit status logging | train_dit.py | 1025-1029 | ✅ DONE |
| P1 | Update train_dit_high_accuracy.sh | train_dit_high_accuracy.sh | all | ✅ DONE |
| P1 | Update train.yml workflow | train.yml | 14-34, 231-238 | ✅ DONE |
| P2 | Map signal names | train_dit.py | 818 | Deferred |

## Alternative Approaches

### Alternative 1: Use GitHub Actions for Training (RECOMMENDED)

```bash
# Instead of nohup modal run
gh workflow run train.yml -f steps=400000 -f batch_size=256
```

**Pros:**
- Modal handles all process management
- Built-in retry and timeout handling
- Logs visible in GitHub UI
- No SIGHUP issues

**Cons:**
- Requires GitHub Actions quota
- Slightly more complex setup

### Alternative 2: Use Modal Scheduled Functions

```python
@app.function(schedule=modal.Period(days=1))
def scheduled_training():
    # Resume from last checkpoint
    ...
```

**Pros:**
- Modal-native solution
- Automatic retry

**Cons:**
- More complex checkpoint management
- Less flexible

## Consequences

### Positive
- ✅ Training will handle SIGHUP gracefully
- ✅ Exit status logged for debugging
- ✅ More reliable background execution via GitHub Actions

### Negative
- ⚠️ Requires code changes to train_dit.py
- ⚠️ Existing training script needs update

### Neutral
- ℹ️ GitHub Actions recommended over nohup

## References

- Modal Timeouts: https://modal.com/docs/guide/timeouts
- Modal Retries: https://modal.com/docs/guide/retries
- Signal handling best practices: https://docs.python.org/3/library/signal.html
- ADR-043: Previous termination investigation
- ADR-042: Modal training enhancements
- ADR-023: GPU retry strategy
