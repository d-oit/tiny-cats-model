# Performance Thresholds Reference

## Loss Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Loss NaN | Immediate | Immediate | Halt, check data |
| Loss > 10x initial | Yes | Yes | Check learning rate |
| Loss plateau (>100 iters) | Yes | No | Reduce LR |
| Loss exploding (>1000) | Yes | Yes | Gradient clipping |

## GPU Memory Thresholds

| Usage | Status | Action |
|-------|--------|--------|
| <70% | Normal | Continue |
| 70-90% | Warning | Monitor closely |
| >90% | Critical | Reduce batch size |
| OOM | Fatal | Halt, retry smaller batch |

## Iteration Speed

| Metric | Expected | Warning | Action |
|--------|----------|---------|--------|
| Iterations/sec | >1 | <0.5 | Check GPU utilization |
| Data loading | <20% time | >50% time | Increase workers |
| GPU utilization | >80% | <50% | Check bottleneck |

## Recommended Configurations

### Small Dataset (<1000 images)

```python
{
    "batch_size": 8,
    "learning_rate": 1e-4,
    "epochs": 50,
    "grad_clip": 1.0
}
```

### Medium Dataset (1000-10000 images)

```python
{
    "batch_size": 16,
    "learning_rate": 3e-4,
    "epochs": 100,
    "grad_clip": 1.0
}
```

### Large Dataset (>10000 images)

```python
{
    "batch_size": 32,
    "learning_rate": 1e-3,
    "epochs": 200,
    "grad_clip": 0.5
}
```

## Alert Configuration

```python
ALERT_THRESHOLDS = {
    'loss_max': 100.0,
    'loss_nan': True,
    'gpu_memory_percent': 90,
    'error_count_max': 5,
    'security_issues': True
}
```

## Performance Optimization Tips

1. **Batch Size**: Largest that fits in GPU memory
2. **Mixed Precision**: Enable for 2x speedup
3. **Data Loading**: Use `num_workers > 0`
4. **Gradient Accumulation**: Simulate larger batches
5. **Checkpointing**: Save every N epochs, not every iter

## Monitoring Dashboard Metrics

Track these in real-time:

- Current loss (plot over time)
- GPU memory usage (%)
- Iterations per second
- Error/warning count
- Security check status
