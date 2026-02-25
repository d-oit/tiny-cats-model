---
name: training-monitor
description: Monitor Modal training processes with comprehensive logging, error handling, performance tracking, and security checks
---

# Training Monitor

Comprehensive monitoring for Modal training processes with logging, error handling, performance tracking, and security.

## Quick Start

```bash
# Monitor training with automatic retry loop
python .agents/skills/training-monitor/scripts/monitor_training.py "modal run src/train.py data/cats" --timeout 3600
```

## Features

- **Real-time logging**: All output captured to `logs/training/`
- **Error detection**: Automatic detection of OOM, CUDA, network, and Modal errors
- **Performance tracking**: Loss values, iterations, GPU memory usage
- **Security scanning**: Detects secret exposure, credential leaks, path traversal
- **Auto-retry loop**: Configurable retry on recoverable errors
- **JSON reports**: Comprehensive training reports with metrics summary

## Usage

### Basic Monitoring

```bash
python .agents/skills/training-monitor/scripts/monitor_training.py "modal run src/train.py data/cats"
```

### With Timeout

```bash
python .agents/skills/training-monitor/scripts/monitor_training.py "modal run src/train.py data/cats" --timeout 7200
```

### Custom Output Path

```bash
python .agents/skills/training-monitor/scripts/monitor_training.py "modal run src/train.py data/cats" --output reports/training_report.json
```

## Error Handling

The monitor automatically detects these error types:

| Error Type | Pattern | Action |
|------------|---------|--------|
| OOM | `out of memory`, `CUDA out of memory` | Log, alert, optional retry with smaller batch |
| Connection | `connection refused`, `timeout` | Log, alert, retry with backoff |
| CUDA | `CUDA error`, `cuDNN` | Log, alert, halt |
| Modal | `modal.*error`, `app.*not.*found` | Log, alert, halt |

## Security Checks

Automatic scanning for:

- **Secret exposure**: API keys, tokens, passwords in output
- **Credential leaks**: AWS_, GCP_, AZURE_, MODAL_ environment variables
- **Path traversal**: `../` patterns
- **Shell injection**: Dangerous shell characters

## Performance Metrics

Tracked metrics:

- Start/end time, duration
- Iterations completed
- Loss values (array)
- GPU memory usage (array)
- Error/warning counts

## Output Files

- **Logs**: `logs/training/training_YYYYMMDD_HHMMSS.log`
- **Reports**: `logs/training/training_report_YYYYMMDD_HHMMSS.json`

## Retry Loop Pattern

For automatic retry on recoverable errors:

```python
from scripts.monitor_training import TrainingMonitor, TrainingStatus

max_retries = 3
retry_count = 0

while retry_count < max_retries:
    monitor = TrainingMonitor("modal run src/train.py data/cats")
    monitor.start()
    status = monitor.monitor()
    
    if status == TrainingStatus.COMPLETED:
        break
    elif status in [TrainingStatus.TIMEOUT, TrainingStatus.FAILED]:
        retry_count += 1
        logger.warning(f"Retry {retry_count}/{max_retries}")
```

## Reference Files

- **Error patterns**: See [references/error-patterns.md](references/error-patterns.md)
- **Security policy**: See [references/security-policy.md](references/security-policy.md)
- **Performance thresholds**: See [references/performance-thresholds.md](references/performance-thresholds.md)

## Integration

### With CI/CD

```yaml
# .github/workflows/train.yml
- name: Monitor Training
  run: python .agents/skills/training-monitor/scripts/monitor_training.py "modal run src/train.py data/cats" --timeout 7200
```

### With Modal

```python
# In your Modal app
@app.function(timeout=3600)
def train():
    import subprocess
    subprocess.run(["python", ".agents/skills/training-monitor/scripts/monitor_training.py", "modal run src/train.py data/cats"])
```

## Alerts

Configure alerts for:

- Loss NaN or exploding (>10x initial)
- GPU memory >90% capacity
- Error count > threshold
- Security issues detected
