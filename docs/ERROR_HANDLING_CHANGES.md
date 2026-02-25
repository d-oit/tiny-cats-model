# Modal Error Handling and Monitoring Implementation

## Summary

Added comprehensive error handling and monitoring for Modal GPU training runs to improve debugging, error recovery, and training visibility.

## Changes Made

### 1. New File: `src/modal_monitor.py`

A comprehensive monitoring module providing:

- **ModalMonitor class**: Main monitoring orchestrator
- **TrainingContext**: Tracks training run metadata
- **Error categorization**: Automatic classification into 8 categories
- **Error severity levels**: WARNING, ERROR, CRITICAL, FATAL
- **GPU health monitoring**: Memory stats and health checks
- **Recovery suggestions**: Context-aware error recovery guidance
- **Retry decorator**: Exponential backoff for retryable operations
- **Context manager**: `track_training()` for automatic tracking
- **Event logging**: JSONL format for easy parsing
- **Training summaries**: Complete run reports in JSON

### 2. Updated: `src/train.py` (Classifier Training)

**Changes:**
- Added imports from `modal_monitor`
- Removed duplicate `TrainingError` class (now imported)
- Integrated monitoring in `train_on_gpu()`:
  - Modal environment logging
  - GPU health check before training
  - Initial/final GPU memory stats recording
  - Progress tracking with epoch and accuracy
  - Error categorization (MEMORY vs TRAINING)
  - Recovery suggestions on errors
  - Training completion summaries

**Error Handling Improvements:**
```python
# Before: Generic error handling
except RuntimeError as e:
    raise TrainingError(f"Training failed: {e}") from e

# After: Categorized errors with recovery
except RuntimeError as e:
    error_category = ErrorCategory.MEMORY if "out of memory" in str(e).lower() else ErrorCategory.TRAINING
    monitor.handle_error(e, severity=ErrorSeverity.ERROR, category=error_category)
    recovery = monitor.get_recovery_suggestion(e)
    logger.info(f"Recovery suggestion: {recovery}")
    raise TrainingError(error_msg) from e
```

### 3. Updated: `src/train_dit.py` (DiT Training)

**Changes:**
- Added imports from `modal_monitor`
- Removed duplicate `TrainingError` class (now imported)
- Integrated monitoring in `train_dit_on_gpu()`:
  - Modal environment logging
  - GPU health check before training
  - Initial/final GPU memory stats recording
  - Progress tracking with steps and loss
  - Error categorization (MEMORY, DATA_LOADING, TRAINING)
  - Recovery suggestions on errors
  - Training completion summaries

### 4. Updated: Modal Image Configuration

Both training scripts now include monitoring dependencies:
```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    ...
    .add_local_file("src/modal_monitor.py", "/app/modal_monitor.py")
    .add_local_file("src/volume_utils.py", "/app/volume_utils.py")
)
```

### 5. New File: `docs/MONITORING.md`

Comprehensive documentation covering:
- Feature overview
- Error categories and severity levels
- GPU health monitoring
- Usage examples
- Log file formats
- Integration patterns
- Best practices

## Error Categories

| Category | Detection | Example |
|----------|-----------|---------|
| `DATA_LOADING` | "data", "dataset", "dataloader" | Missing dataset files |
| `MODEL_CREATION` | "model", "layer", "parameter" | Invalid architecture |
| `TRAINING` | RuntimeError default | Loss becomes NaN |
| `MEMORY` | "memory", "cuda", "oom" | GPU out of memory |
| `CHECKPOINT` | "checkpoint", "save", "load" | Corrupt checkpoint file |
| `NETWORK` | "network", "connection", "timeout" | Download failure |
| `CONFIGURATION` | "config", "argument" | Invalid hyperparameters |
| `UNKNOWN` | Default fallback | Unexpected exceptions |

## Monitoring Outputs

### Directory Structure
```
/outputs/
├── checkpoints/
│   └── classifier/
│       └── 2026-02-25/
│           └── cats_model.pt
└── logs/
    └── monitor/
        ├── monitor_2026-02-25.jsonl      # Event log
        ├── errors_2026-02-25.jsonl       # Error records
        └── summary_classifier_*.json     # Training summary
```

### Event Log Format
```jsonl
{"timestamp": "2026-02-25T10:00:00", "event_type": "training_started", "data": {"script": "classifier", "gpu_type": "T4"}}
{"timestamp": "2026-02-25T10:05:00", "event_type": "gpu_stats", "data": {"allocated_mb": 2048.5}}
{"timestamp": "2026-02-25T11:00:00", "event_type": "training_completed", "data": {"status": "completed", "final_metric": 0.95}}
```

### Error Record Format
```json
{
  "timestamp": "2026-02-25T10:30:00",
  "error_type": "RuntimeError",
  "error_message": "CUDA out of memory",
  "severity": "error",
  "category": "memory",
  "traceback": "Traceback (most recent call last):...",
  "context": {},
  "script": "classifier"
}
```

### Training Summary Format
```json
{
  "training": {
    "script": "classifier",
    "start_time": "2026-02-25T10:00:00",
    "end_time": "2026-02-25T11:00:00",
    "status": "completed",
    "epochs_completed": 10,
    "final_metric": 0.95,
    "checkpoint_path": "/outputs/checkpoints/classifier/2026-02-25/cats_model.pt"
  },
  "errors": [...],
  "gpu_stats": {
    "samples": 100,
    "avg_allocated_mb": 2048.5,
    "max_allocated_mb": 3500.0
  }
}
```

## Benefits

### 1. Improved Debugging
- Automatic error categorization reduces investigation time
- Full traceback preservation in error logs
- GPU memory tracking helps identify memory leaks

### 2. Faster Recovery
- Context-aware recovery suggestions
- Retry logic with exponential backoff
- Clear error messages with categorized context

### 3. Better Visibility
- Real-time GPU health monitoring
- Training progress tracking
- Comprehensive audit trail in JSONL format

### 4. Production Ready
- Structured logging for log aggregation systems
- Training summaries for reporting
- Error history for trend analysis

## Usage

### Running Training with Monitoring

```bash
# Classifier training
modal run src/train.py data/cats --epochs 20

# DiT training
modal run src/train_dit.py data/cats --steps 200000

# View monitoring logs
modal volume ls cats-model-outputs /outputs/logs/monitor/

# Download training summary
modal volume get cats-model-outputs /outputs/logs/monitor/summary_classifier_20260225.json ./summary.json
```

### Analyzing Training Results

```python
import json
from pathlib import Path

# Load latest summary
summaries = sorted(Path("/outputs/logs/monitor").glob("summary_*.json"))
if summaries:
    with open(summaries[-1]) as f:
        summary = json.load(f)
    
    print(f"Status: {summary['training']['status']}")
    print(f"Duration: {summary['training']['duration_seconds']}s")
    print(f"Final metric: {summary['training']['final_metric']}")
    print(f"Errors: {len(summary['errors'])}")
    print(f"Avg GPU memory: {summary['gpu_stats']['avg_allocated_mb']}MB")
```

## Testing

### Local Testing (CPU)
```bash
# Test monitoring module
python -c "from src.modal_monitor import ModalMonitor; print('OK')"

# Test classifier training (1 epoch)
python src/train.py data/cats --epochs 1 --batch-size 8

# Check logs created
ls -la /outputs/logs/monitor/ 2>/dev/null || ls -la logs/monitor/
```

### Modal GPU Testing
```bash
# Quick GPU test
modal run src/train.py data/cats --epochs 1

# Verify monitoring outputs
modal volume get cats-model-outputs /outputs/logs/monitor/ ./logs/
```

## Code Quality

All changes pass:
- ✅ Syntax check (`python -m py_compile`)
- ✅ Linting (`ruff check --fix`)
- ✅ Type hints (where applicable)
- ✅ Documentation (docstrings + MONITORING.md)

## Related ADRs

- **ADR-020**: Modal CLI-First Training Strategy
- **ADR-023**: Modal GPU Retry Strategy  
- **ADR-024**: Modal Volume and Storage Best Practices
- **ADR-025**: Modal Cold Start Optimization

## Future Enhancements

Potential improvements for future iterations:

1. **Real-time dashboards**: WebSocket-based live monitoring
2. **Alerting**: Slack/email notifications on errors
3. **Metrics export**: Prometheus/OpenTelemetry integration
4. **Automated tuning**: Suggest hyperparameter changes based on errors
5. **Historical analysis**: Trend analysis across training runs
6. **Cost tracking**: Estimate training costs from GPU time

## Files Changed

| File | Type | Lines Changed | Description |
|------|------|---------------|-------------|
| `src/modal_monitor.py` | New | 450+ | Monitoring module |
| `src/train.py` | Modified | +50 | Integrated monitoring |
| `src/train_dit.py` | Modified | +50 | Integrated monitoring |
| `docs/MONITORING.md` | New | 300+ | Documentation |

Total: ~850 lines added

## Backward Compatibility

- ✅ All existing training commands work unchanged
- ✅ `TrainingError` now imported (was previously defined locally)
- ✅ Monitoring is transparent to existing code
- ✅ No breaking changes to API or behavior

## Migration Notes

If you have custom training scripts:

1. **Import monitoring**:
   ```python
   from modal_monitor import ModalMonitor, ErrorCategory, ErrorSeverity
   ```

2. **Initialize monitor**:
   ```python
   monitor = ModalMonitor()
   monitor.start_training(script="my_script", gpu_type="T4")
   ```

3. **Track progress**:
   ```python
   monitor.update_progress(epoch=epoch, metric=loss)
   ```

4. **Handle errors**:
   ```python
   try:
       # training
   except Exception as e:
       monitor.handle_error(e, severity=ErrorSeverity.ERROR)
       raise
   finally:
       monitor.complete_training(status="completed")
   ```

## Support

For issues or questions:
1. Check `docs/MONITORING.md` for detailed documentation
2. Review error logs in `/outputs/logs/monitor/errors_*.jsonl`
3. Examine training summaries in `/outputs/logs/monitor/summary_*.json`
4. Use `monitor.get_recovery_suggestion(error)` for guidance
