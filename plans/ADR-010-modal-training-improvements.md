# ADR-010: Modal Training Improvements - Error Handling, Logging, Performance, and Memory Management

## Status
Implemented

## Date
2026-02-24 (Proposed) | 2026-02-25 (Implemented)

## Context
The current Modal training setup in `src/train.py` works but lacks production-ready features:

1. **Error Handling**: No try-catch blocks, no graceful failure handling, no retry logic
2. **Logging**: Only basic print statements, no structured logging, no log levels
3. **Performance**: No mixed precision training, no gradient accumulation, no learning rate warmup
4. **Memory Management**: No explicit garbage collection, no gradient clipping, potential memory leaks during long training runs

Training on Modal GPU should be robust, efficient, and production-ready.

## Decision
We will implement comprehensive improvements to the training pipeline:

### 1. Error Handling
- Wrap training in try-except blocks with specific exception types
- Add retry logic for transient failures (network, GPU)
- Implement graceful shutdown on SIGINT/SIGTERM
- Add validation for all inputs and configuration
- Save checkpoint on error for recovery

### 2. Logging
- Use Python `logging` module with structured format
- Add log levels: INFO for progress, WARNING for issues, ERROR for failures
- Include timestamps, epoch, batch info in log messages
- Log to both console and file (saved in outputs)
- Add JSON logging for metrics (compatible with MLflow/W&B)

### 3. Performance Optimizations
- **Mixed Precision Training**: Use `torch.cuda.amp` for 2-3x speedup on T4
- **Gradient Accumulation**: Support larger effective batch sizes
- **Learning Rate Warmup**: Prevent early instability
- **DataLoader Optimization**: `pin_memory=True`, `num_workers`, `prefetch_factor`
- **Gradient Checkpointing**: For larger models (optional)

### 4. Memory Leak Prevention
- **Explicit Garbage Collection**: Call `gc.collect()` between epochs
- **Gradient Clipping**: Prevent exploding gradients (`torch.nn.utils.clip_grad_norm_`)
- **Clear CUDA Cache**: `torch.cuda.empty_cache()` after validation
- **Detach Tensors**: Ensure no gradient accumulation in metrics
- **Context Managers**: Use `with torch.no_grad()` appropriately
- **Memory Monitoring**: Log GPU memory usage during training

## Implementation Plan

### Scripts Implemented

The improvements from this ADR are now implemented in:
- **`src/train.py`** - ResNet classifier training with all improvements
- **`src/train_dit.py`** - TinyDiT training with flow matching and EMA

### Usage (Modal CLI)

```bash
# Classifier training (Modal GPU)
modal run src/train.py data/cats --epochs 20 --batch-size 64

# DiT training (Modal GPU)
modal run src/train_dit.py data/cats --steps 200000 --batch-size 256

# Local testing (CPU)
python src/train.py data/cats --epochs 1 --batch-size 8
python src/train_dit.py data/cats --steps 100 --batch-size 8
```

> **Note:** See ADR-020 for the complete Modal CLI-First Training Strategy.

### train.py Changes
```python
import logging
import gc
import signal
from contextlib import nullcontext
from torch.cuda.amp import GradScaler, autocast

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/outputs/training.log')
    ]
)
logger = logging.getLogger(__name__)

# Memory monitoring
def log_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU Memory: {allocated:.1f}MB allocated, {reserved:.1f}MB reserved")

# Mixed precision training
scaler = GradScaler()
for xb, yb in loader:
    with autocast():
        pred = model(xb)
        loss = loss_fn(pred, yb)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Memory cleanup
gc.collect()
torch.cuda.empty_cache()
```

### Configuration Options
Add new CLI arguments:
- `--mixed-precision`: Enable AMP training
- `--gradient-clip`: Max gradient norm (default 1.0)
- `--warmup-epochs`: LR warmup period
- `--grad-accum-steps`: Gradient accumulation steps
- `--log-file`: Path to log file
- `--save-interval`: Checkpoint save interval (epochs)

## Consequences

### Positive
- **Robustness**: Training recovers from transient errors
- **Observability**: Clear logging for debugging and monitoring
- **Performance**: 2-3x faster training with mixed precision
- **Stability**: No memory leaks during long runs
- **Recovery**: Can resume from checkpoint after failure

### Negative
- Slightly more complex code
- Mixed precision may require tuning for some models
- Additional CLI options to manage

### Neutral
- Requires testing on Modal GPU
- May need adjustment of hyperparameters

## Alternatives Considered

1. **Use PyTorch Lightning**: Rejected - adds dependency, overkill for this project
2. **Use Weights & Biases for logging**: Deferred - can add later, logging module is sufficient
3. **Skip mixed precision**: Rejected - significant speedup on T4 GPUs
4. **Manual memory management only**: Rejected - need both gc and CUDA cache management

## Related
- ADR-007: Modal GPU Training Fix
- ADR-017: TinyDiT Training Infrastructure
- ADR-020: Modal CLI-First Training Strategy
- GOAP.md: Modal Training phase
- Modal GPU docs: https://modal.com/docs/guide/gpu

## References
- PyTorch AMP: https://pytorch.org/docs/stable/amp.html
- Gradient Clipping: https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
- Memory Management: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
- Mixed Precision Training Guide: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
