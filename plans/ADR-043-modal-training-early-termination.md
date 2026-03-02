# ADR-043: Modal Training Early Termination Investigation

**Date:** 2026-02-28
**Status:** Investigating
**Related:** ADR-036, ADR-042, GOAP.md Phase 18

## Context

The 400k step Modal training started successfully but stopped at step 200/400,000 without explicit error.

### Observed Behavior

| Metric | Value |
|--------|-------|
| Training Started | 2026-02-28 17:02 UTC |
| Last Step Logged | Step 200/400,000 |
| App State | stopped |
| Exit Code | Unknown |

### Training Log Excerpt

```
2026-02-28 17:02:53 | INFO | Starting TinyDiT Modal GPU training
2026-02-28 17:03:42 | INFO | Starting TinyDiT training with flow matching
2026-02-28 17:18:47 | INFO | Step 100/400,000 | Loss: 0.0000 | LR: 8.30e-07 | Speed: 11.7 steps/s
2026-02-28 17:34:00 | INFO | Step 200/400,000 | Loss: 0.0000 | LR: 1.16e-06 | Speed: 5.4 steps/s
```

### Investigation Checklist

- [ ] Check Modal dashboard for task status
- [ ] Check volume for checkpoint files
- [ ] Verify training script uses nohup correctly
- [ ] Check for signal handling issues (SIGTERM, etc.)
- [ ] Review GPU availability during training window

## Decision

### Possible Causes

1. **CLI Timeout**: Modal CLI may have a timeout that killed the process
2. **GPU Exhaustion**: Modal GPU quota may have been exceeded
3. **OOM Error**: GPU out of memory during training
4. **Signal Handling**: Training script not handling signals properly

### Recommended Actions

1. Run training via GitHub Actions workflow instead of CLI
2. Add better logging to capture exit reason
3. Use Modal's built-in retry mechanism
4. Consider using screen/tmux for long-running training

## References

- scripts/train_dit_high_accuracy.sh
- src/train_dit.py
- Modal docs: https://modal.com/docs
