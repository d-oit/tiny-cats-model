# Modal Training Monitoring and Error Handling

Comprehensive monitoring and error handling for Modal GPU training runs.

## Overview

The `modal_monitor.py` module provides:

- **Structured error logging** with categorization and severity levels
- **GPU health monitoring** with memory tracking
- **Training progress tracking** with context preservation
- **Error recovery suggestions** based on error type
- **Automatic error categorization** for better debugging
- **Training summaries** saved to JSON files

## Features

### 1. Error Categorization

Errors are automatically categorized into:

| Category | Description | Examples |
|----------|-------------|----------|
| `DATA_LOADING` | Dataset issues | Missing files, corrupt data |
| `MODEL_CREATION` | Model architecture errors | Invalid layers, parameters |
| `TRAINING` | General training errors | Loss NaN, convergence issues |
| `MEMORY` | GPU memory issues | OOM, memory fragmentation |
| `CHECKPOINT` | Save/load errors | File not found, corrupt checkpoint |
| `NETWORK` | Connectivity issues | Download failures, timeouts |
| `CONFIGURATION` | Invalid settings | Bad arguments, missing config |
| `UNKNOWN` | Unclassified errors | Unexpected exceptions |

### 2. Error Severity Levels

| Severity | When Used |
|----------|-----------|
| `WARNING` | Non-critical issues |
| `ERROR` | Recoverable errors |
| `CRITICAL` | Serious but handled |
| `FATAL` | Unrecoverable failures |

### 3. GPU Health Monitoring

```python
from modal_monitor import check_gpu_health, get_gpu_memory_stats

# Check GPU health
is_healthy, message = check_gpu_health()
print(f"GPU: {message}")

# Get detailed stats
stats = get_gpu_memory_stats()
if stats:
    print(f"Allocated: {stats.allocated_mb}MB")
    print(f"Utilization: {stats.utilization_percent}%")
```

### 4. Training Context Tracking

```python
from modal_monitor import ModalMonitor

monitor = ModalMonitor()
monitor.start_training(script="classifier", gpu_type="T4")

# During training
monitor.update_progress(epoch=5, metric=0.95)
monitor.record_gpu_stats(stats)

# On completion
monitor.complete_training(
    status="completed",
    final_metric=0.95,
    checkpoint_path="/outputs/model.pt",
)
```

### 5. Error Handling with Recovery Suggestions

```python
from modal_monitor import ModalMonitor, ErrorCategory, ErrorSeverity

monitor = ModalMonitor()

try:
    # Training code
    pass
except Exception as e:
    # Handle and categorize error
    monitor.handle_error(
        e,
        severity=ErrorSeverity.ERROR,
        category=ErrorCategory.MEMORY,  # Auto-detected if not provided
    )
    
    # Get recovery suggestion
    recovery = monitor.get_recovery_suggestion(e)
    logger.info(f"Recovery: {recovery}")
    
    raise
```

### 6. Context Manager for Automatic Tracking

```python
from modal_monitor import track_training

with track_training("classifier", gpu_type="T4") as monitor:
    # Training code
    for epoch in range(epochs):
        # ...
        monitor.update_progress(epoch=epoch, metric=loss)
        
# Automatically marks as completed on success
# Automatically handles errors on failure
```

### 7. Retry with Exponential Backoff

```python
from modal_monitor import retry_with_backoff

@retry_with_backoff(
    max_retries=3,
    initial_delay=1.0,
    max_delay=60.0,
    exceptions=(RuntimeError, ConnectionError),
)
def download_dataset():
    # Network operation that may fail
    pass
```

## Log Files

Monitoring creates the following log files in `/outputs/logs/monitor/`:

### Event Log
`monitor_YYYY-MM-DD.jsonl` - All training events
```jsonl
{"timestamp": "2026-02-25T10:00:00", "event_type": "training_started", "data": {...}}
{"timestamp": "2026-02-25T11:00:00", "event_type": "training_completed", "data": {...}}
```

### Error Log
`errors_YYYY-MM-DD.jsonl` - Error records
```jsonl
{
  "timestamp": "2026-02-25T10:30:00",
  "error_type": "RuntimeError",
  "error_message": "CUDA out of memory",
  "severity": "error",
  "category": "memory",
  "traceback": "...",
  "context": {...}
}
```

### Training Summary
`summary_{script}_{timestamp}.json` - Complete training summary
```json
{
  "training": {
    "script": "classifier",
    "start_time": "2026-02-25T10:00:00",
    "end_time": "2026-02-25T11:00:00",
    "status": "completed",
    "epochs_completed": 10,
    "final_metric": 0.95,
    "checkpoint_path": "/outputs/model.pt"
  },
  "errors": [...],
  "gpu_stats": {
    "samples": 100,
    "avg_allocated_mb": 2048.5,
    "max_allocated_mb": 3500.0
  }
}
```

## Integration with Training Scripts

### Classifier Training (`src/train.py`)

The classifier training uses comprehensive monitoring:

```python
@app.function(...)
def train_on_gpu(...):
    # Initialize monitor
    monitor = ModalMonitor(log_dir="/outputs/logs/monitor")
    monitor.start_training(script="classifier", gpu_type="T4")
    
    # Check GPU health
    gpu_healthy, gpu_message = check_gpu_health()
    logger.info(f"GPU health: {gpu_message}")
    
    try:
        # Training
        val_acc = train(...)
        
        # Record stats and complete
        monitor.update_progress(epoch=epochs, metric=val_acc)
        monitor.complete_training(
            status="completed",
            final_metric=val_acc,
            checkpoint_path=output,
        )
        
    except RuntimeError as e:
        # Categorize and handle error
        error_category = ErrorCategory.MEMORY if "out of memory" in str(e).lower() else ErrorCategory.TRAINING
        monitor.handle_error(e, severity=ErrorSeverity.ERROR, category=error_category)
        
        # Get recovery suggestion
        recovery = monitor.get_recovery_suggestion(e)
        logger.info(f"Recovery suggestion: {recovery}")
        
        raise
```

### DiT Training (`src/train_dit.py`)

Similar monitoring is integrated for DiT training with A10G GPU tracking.

## Modal Environment Logging

The `log_modal_environment()` function captures:

```python
{
    "is_modal": True,
    "app_id": "app-123",
    "function_name": "train_on_gpu",
    "container_id": "container-456",
    "gpu_type": "T4",
    "python_version": "3.12.0",
    "hostname": "modal-container",
    "timestamp": "2026-02-25T10:00:00"
}
```

## Error Recovery Suggestions

The monitor provides automatic recovery suggestions:

| Error Pattern | Suggestion |
|---------------|------------|
| "out of memory" | Reduce batch size, enable mixed precision, use gradient accumulation |
| "dataset"/"data" | Check dataset path, verify format, re-download |
| "checkpoint"/"load" | Check file exists, verify format, start from scratch |
| "network"/"connection" | Retry with backoff, check connectivity, use cache |

## Usage Examples

### Basic Monitoring

```bash
# Start training with monitoring
modal run src/train.py data/cats --epochs 20

# Check monitoring logs
modal volume ls cats-model-outputs /outputs/logs/monitor/

# Download training summary
modal volume get cats-model-outputs /outputs/logs/monitor/summary_classifier_20260225.json ./summary.json
```

### Analyzing Errors

```python
import json

# Load error log
with open("/outputs/logs/monitor/errors_2026-02-25.jsonl") as f:
    errors = [json.loads(line) for line in f]

# Find all memory errors
memory_errors = [e for e in errors if e["category"] == "memory"]
print(f"Found {len(memory_errors)} memory errors")

# Get most recent error
if errors:
    print(f"Latest error: {errors[-1]['error_message']}")
    print(f"Recovery: {errors[-1].get('recovery_suggestion', 'N/A')}")
```

### GPU Memory Analysis

```python
import json

# Load training summary
with open("/outputs/logs/monitor/summary.json") as f:
    summary = json.load(f)

# Analyze GPU usage
gpu_stats = summary["gpu_stats"]
print(f"Average GPU memory: {gpu_stats['avg_allocated_mb']}MB")
print(f"Peak GPU memory: {gpu_stats['max_allocated_mb']}MB")
print(f"Total samples: {gpu_stats['samples']}")
```

## Best Practices

1. **Always initialize monitor** at the start of training functions
2. **Record GPU stats** before and after training
3. **Categorize errors** for better debugging
4. **Save recovery suggestions** in logs
5. **Review training summaries** after each run
6. **Monitor GPU memory trends** across runs
7. **Use context manager** for automatic tracking when possible

## Files

- `src/modal_monitor.py` - Main monitoring module
- `src/train.py` - Classifier with monitoring integration
- `src/train_dit.py` - DiT with monitoring integration
- `src/volume_utils.py` - Volume management utilities

## Related Documentation

- ADR-020: Modal CLI-First Training Strategy
- ADR-023: Modal GPU Retry Strategy
- ADR-024: Modal Volume and Storage Best Practices
- ADR-025: Modal Cold Start Optimization
