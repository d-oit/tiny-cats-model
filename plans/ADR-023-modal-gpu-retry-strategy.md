# ADR-023: Modal GPU Resource Configuration and Retry Strategy

**Date:** 2026-02-25
**Status:** Proposed
**Authors:** AI Agent
**Related:** ADR-007 (Modal GPU Training Fix), ADR-010 (Modal Training Improvements), ADR-017 (TinyDiT Training Infrastructure), ADR-020 (Modal CLI-First Training Strategy)

## Context

### Current State

Current GPU configuration in training scripts:

**src/train.py (Classifier):**
```python
@stub.function(
    image=image,
    volumes={
        "/outputs": volume_outputs,
        "/data": volume_data,
    },
    gpu="T4",
    timeout=3600,
)
def train_modal(...):
```

**src/train_dit.py (DiT):**
```python
@stub.function(
    image=image,
    volumes={
        "/outputs": volume_outputs,
        "/data": volume_data,
    },
    gpu="A10G",  # Better for transformer training
    timeout=7200,  # 2 hours max
)
def train_dit_modal(...):
```

### Issues Identified

1. **No Retry Configuration**: Transient GPU failures cause complete training failure
2. **Fixed Timeout**: No dynamic timeout based on workload
3. **No Concurrency Control**: Single execution only, no parallel training
4. **No GPU Memory Optimization**: No memory snapshot or optimization settings
5. **Limited Error Recovery**: Manual intervention required for recoverable errors
6. **No Cost Optimization**: No spot/preemptible GPU configuration

### Modal Best Practices (2025-2026)

Based on Modal documentation and production examples:

1. **Use `modal.Retries`**: Automatic retry with exponential backoff
2. **Configure Appropriate Timeouts**: Match timeout to expected workload duration
3. **Use `@modal.concurrent()`**: Handle multiple concurrent inputs
4. **Use `@modal.batched()`**: Process batches efficiently
5. **GPU Selection by Workload**: Match GPU type to task requirements
6. **Use Spot Instances**: Cost savings for fault-tolerant workloads

## Decision

We will implement comprehensive GPU resource configuration and retry strategies:

### 1. Add Retry Configuration with Exponential Backoff

```python
from modal import Retries

@app.function(
    gpu="T4",
    timeout=3600,
    retries=Retries(
        max_retries=3,           # Retry up to 3 times
        backoff_coefficient=2.0, # Double delay each retry
        initial_delay=10.0,      # Start with 10 second delay
        max_delay=300.0,         # Cap at 5 minutes
    ),
)
def train_modal(...):
```

**Retry Strategy:**
- **max_retries=3**: Balance between recovery and cost
- **backoff_coefficient=2.0**: Exponential backoff (10s → 20s → 40s)
- **initial_delay=10.0**: Short initial delay for quick recovery
- **max_delay=300.0**: Prevent excessive waiting

**When Retries Help:**
- Transient GPU allocation failures
- Network timeouts during dataset download
- Temporary Modal infrastructure issues
- OOM errors that may resolve with cache clearing

### 2. GPU Selection Matrix

| Workload | GPU Type | VRAM | Cost/Hour | Use Case |
|----------|----------|------|-----------|----------|
| Classifier Training | `T4` | 16GB | ~$0.35 | Standard ResNet training |
| DiT Training | `A10G` | 24GB | ~$1.00 | Transformer training |
| Large Batch Training | `A100` | 40-80GB | ~$3.00 | Batch size > 512 |
| Fast Iteration | `L4` | 24GB | ~$0.60 | Development/testing |
| Production Inference | `T4` | 16GB | ~$0.35 | Cost-effective serving |

### 3. Timeout Configuration by Workload

```python
# Classifier training (shorter)
@app.function(timeout=3600)  # 1 hour

# DiT training (longer)
@app.function(timeout=7200)  # 2 hours

# Full DiT training (200k steps)
@app.function(timeout=86400)  # 24 hours (max allowed)
```

**Timeout Guidelines:**
- Add 20% buffer to expected duration
- Consider retry time in total budget
- Use longest timeout for critical training jobs

### 4. Concurrency for Parallel Processing

```python
@app.function(
    gpu="T4",
    concurrency_limit=2,  # Allow 2 concurrent executions
)
@modal.concurrent(max_inputs=10)  # Queue up to 10 inputs
def train_batch(configs: list[dict]) -> list[dict]:
    """Process multiple training configurations."""
    results = []
    for config in configs:
        results.append(train_single(**config))
    return results
```

### 5. Batched Processing for Efficiency

```python
@app.function(
    gpu="A10G",
    batch_sizes=(16, 64),  # Min and max batch size
    batch_wait_ms=100,     # Wait up to 100ms for batch
)
def train_batched(batch: list[dict]) -> list[dict]:
    """Process training requests in batches."""
    # Process entire batch on GPU
    return [train_single(**config) for config in batch]
```

### 6. Cost Optimization with Spot Instances

```python
@app.function(
    gpu=modal.GpuSpec("T4", count=1),
    spot=True,  # Use spot/preemptible instances
)
def train_spot(...):
    """Training with spot instances (60-70% cost savings)."""
```

**Spot Instance Considerations:**
- **Pros**: 60-70% cost savings
- **Cons**: May be preempted (use checkpoints!)
- **Best For**: Fault-tolerant workloads with checkpointing

### 7. GPU Memory Optimization

```python
@app.function(
    gpu="A10G",
    memory=16384,  # 16GB system memory
)
def train_with_memory(...):
    """Training with explicit memory configuration."""
```

## Implementation Plan

### Phase 1: Add Retry Configuration

1. Add `from modal import Retries` to training scripts
2. Configure retries for `train_modal` and `train_dit_modal`
3. Test retry behavior with simulated failures

### Phase 2: Update GPU Configuration

1. Document GPU selection in modal.yml
2. Add GPU type constants for consistency
3. Update timeout values based on workload

### Phase 3: Add Concurrency Support

1. Add batch training function for hyperparameter sweeps
2. Configure concurrency limits
3. Test parallel training scenarios

### Phase 4: Cost Optimization

1. Document spot instance usage
2. Add checkpoint-based recovery for spot preemption
3. Create cost estimation guide

## Code Changes

### src/train.py (Updated Function Decorator)

```python
from modal import Retries

# Add retry configuration
@app.function(
    image=image,
    volumes={
        "/outputs": volume_outputs,
        "/data": volume_data,
    },
    gpu="T4",
    timeout=3600,
    retries=Retries(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=10.0,
        max_delay=300.0,
    ),
)
def train_modal(
    data_dir: str = "/data/cats",
    epochs: int = 10,
    batch_size: int = 32,
    # ... other params
) -> dict[str, Any]:
```

### src/train_dit.py (Updated Function Decorator)

```python
from modal import Retries

# Add retry configuration for DiT training
@app.function(
    image=image,
    volumes={
        "/outputs": volume_outputs,
        "/data": volume_data,
    },
    gpu="A10G",
    timeout=86400,  # 24 hours for full 200k step training
    retries=Retries(
        max_retries=2,  # Fewer retries for long jobs
        backoff_coefficient=2.0,
        initial_delay=30.0,  # Longer initial delay
        max_delay=600.0,     # 10 minute max delay
    ),
)
def train_dit_modal(
    data_dir: str = "/data/cats",
    steps: int = 200_000,
    # ... other params
) -> dict[str, Any]:
```

### New: Batch Training Function

```python
@app.function(
    image=image,
    gpu="T4",
    concurrency_limit=2,
)
@modal.concurrent(max_inputs=10)
def train_hyperparameter_sweep(
    configs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run multiple training jobs with different hyperparameters."""
    results = []
    for config in configs:
        try:
            result = train_modal.remote(**config)
            results.append({"config": config, "result": result, "status": "success"})
        except Exception as e:
            results.append({"config": config, "error": str(e), "status": "failed"})
    return results
```

## Consequences

### Positive
- ✅ **Automatic Recovery**: Transient failures handled automatically
- ✅ **Cost Savings**: Spot instances reduce training costs 60-70%
- ✅ **Better Resource Utilization**: Concurrency for parallel workloads
- ✅ **Predictable Costs**: Clear GPU pricing by type
- ✅ **Improved Reliability**: Exponential backoff prevents cascade failures

### Negative
- ⚠️ **Increased Complexity**: More configuration options to manage
- ⚠️ **Retry Costs**: Failed retries still incur GPU costs
- ⚠️ **Spot Preemption**: Spot instances may be interrupted mid-training

### Neutral
- ℹ️ **Same Training Logic**: Core training unchanged
- ℹ️ **Backward Compatible**: Existing calls still work

## Alternatives Considered

### Alternative 1: No Retries (Current State)
**Proposal**: Keep current configuration without retries.

**Rejected Because**:
- Transient GPU failures cause complete training loss
- Manual intervention required for recoverable errors
- Production systems need automatic recovery

### Alternative 2: Infinite Retries
**Proposal**: Retry indefinitely until success.

**Rejected Because**:
- Unbounded cost risk
- May retry on permanent errors
- Better to fail fast for debugging

### Alternative 3: Always Use A100
**Proposal**: Use A100 for all training for maximum speed.

**Rejected Because**:
- 8-10x cost increase vs T4
- Overkill for ResNet classifier
- T4 sufficient for most workloads

## Success Metrics

- [ ] Retry configuration added to both training scripts
- [ ] Transient failures automatically recovered
- [ ] GPU cost documented and optimized
- [ ] Concurrency tested for batch training

## References

- Modal Retries: https://modal.com/docs/guide/retries
- Modal GPU Guide: https://modal.com/docs/guide/gpu
- Modal Timeouts: https://modal.com/docs/guide/timeouts
- Modal Concurrency: https://modal.com/docs/guide/concurrency
- ADR-010: Modal Training Improvements
- ADR-017: TinyDiT Training Infrastructure
