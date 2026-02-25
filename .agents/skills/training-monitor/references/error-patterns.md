# Error Patterns Reference

## Detected Error Types

### 1. Out of Memory (OOM)

**Patterns:**
- `out of memory`
- `OOM`
- `CUDA out of memory`
- `RuntimeError: CUDA out of memory`

**Recovery Actions:**
1. Reduce batch size
2. Enable gradient accumulation
3. Use mixed precision training
4. Check for memory leaks

**Retry Command:**
```bash
python src/train.py data/cats --batch-size 4  # Reduced from default
```

### 2. Connection Errors

**Patterns:**
- `connection refused`
- `timeout`
- `network error`
- `ConnectionResetError`

**Recovery Actions:**
1. Wait and retry with exponential backoff
2. Check Modal service status
3. Verify network connectivity

**Retry Strategy:**
```python
import time
for attempt in range(3):
    try:
        # retry training
        break
    except Exception:
        time.sleep(2 ** attempt)  # Exponential backoff
```

### 3. CUDA Errors

**Patterns:**
- `CUDA error`
- `cuDNN error`
- `GPU.*error`
- `illegal memory access`

**Recovery Actions:**
1. Halt training (not recoverable)
2. Check GPU health
3. Restart Modal container
4. Report to infrastructure team

### 4. Modal Errors

**Patterns:**
- `modal.*error`
- `app.*not.*found`
- `function.*not.*found`
- `volume.*not.*mounted`

**Recovery Actions:**
1. Check Modal app configuration
2. Verify volume mounts
3. Re-deploy Modal app

## Error Severity Levels

| Severity | Errors | Action |
|----------|--------|--------|
| Critical | CUDA, OOM | Halt, alert |
| High | Connection | Retry 3x, then halt |
| Medium | Modal config | Retry 1x, then halt |
| Low | Warnings | Log, continue |

## Error Log Format

```json
{
  "type": "oom",
  "message": "RuntimeError: CUDA out of memory",
  "timestamp": "2026-02-25T10:30:45.123456",
  "severity": "critical",
  "recovery_attempted": false
}
```
