# ADR-027: Experiment Tracking with MLflow

**Date:** 2026-02-25
**Status:** Proposed
**Authors:** AI Agent
**Related:** GOAP.md Phase 9, ADR-026 (HuggingFace Publishing), ADR-017 (TinyDiT Training)

## Context

### Current State

Training logs are written to files only:
- No centralized experiment tracking
- No hyperparameter comparison across runs
- No metric visualization dashboard
- No artifact lineage tracking

### Problem Statement

Without experiment tracking:
1. Cannot compare different training runs easily
2. Hyperparameters not linked to results
3. No audit trail for model versions
4. Difficult to reproduce experiments
5. No visibility into training progress remotely

## Decision

Implement MLflow integration for experiment tracking with:

### 1. Tracking Server Setup

**Option A: Local Tracking (Default)**
```python
import mlflow

# Track locally
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("tiny-cats-model")
```

**Option B: Remote Tracking (Production)**
```python
# Track to remote server
mlflow.set_tracking_uri("https://mlflow.example.com")
mlflow.set_experiment("tiny-cats-model")
```

### 2. Training Integration

```python
import mlflow

class TrainingTracker:
    def __init__(self, experiment_name: str = "tiny-cats-model"):
        mlflow.set_experiment(experiment_name)
        self.run = None
    
    def start_run(self, config: dict):
        """Start MLflow run with hyperparameters."""
        self.run = mlflow.start_run()
        mlflow.log_params(config)
        
        # Log code version
        mlflow.log_param("git_commit", get_git_commit())
        mlflow.log_param("timestamp", datetime.now().isoformat())
        
        return self.run
    
    def log_metrics(self, metrics: dict, step: int = None):
        """Log metrics during training."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model_path: str, artifact_path: str = "model"):
        """Log model artifact."""
        mlflow.pytorch.log_model(model_path, artifact_path)
    
    def log_image(self, image: np.ndarray, name: str):
        """Log generated sample."""
        mlflow.log_image(image, f"samples/{name}.png")
    
    def end_run(self):
        """End MLflow run."""
        if self.run:
            mlflow.end_run()
```

### 3. Hyperparameter Tracking

```python
# Log all training config
config = {
    "model_type": "TinyDiT",
    "image_size": 128,
    "patch_size": 16,
    "embed_dim": 384,
    "depth": 12,
    "num_heads": 6,
    "batch_size": 256,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "ema_beta": 0.9999,
    "warmup_steps": 10000,
    "total_steps": 200000,
    "mixed_precision": True,
    "gradient_clip": 1.0,
}

mlflow.log_params(config)
```

### 4. Metrics Tracking

```python
# During training
for step in range(total_steps):
    # ... training ...
    
    if step % log_interval == 0:
        mlflow.log_metrics({
            "train_loss": loss.item(),
            "learning_rate": current_lr,
        }, step=step)

# At end
mlflow.log_metrics({
    "final_loss": final_loss,
    "best_loss": best_loss,
    "training_duration_hours": duration / 3600,
})
```

### 5. Artifact Logging

```python
# Log checkpoint
mlflow.log_artifact("checkpoints/dit_model.pt", "checkpoints")

# Log generated samples
mlflow.log_artifacts("samples/generated/", "samples")

# Log training curves
mlflow.log_artifact("training_log.json", "logs")
```

### 6. Model Registry Integration

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
model_uri = f"runs:/{run_id}/model"
model_version = client.create_model_version(
    name="tinydit-cats",
    source=model_uri,
    run_id=run_id,
)

# Add metadata
client.set_model_version_tag(
    name="tinydit-cats",
    version=model_version.version,
    key="breeds",
    value="12_cat_breeds_plus_other",
)
```

## Implementation Plan

### Phase 1: Core Integration (Week 1)
1. Add `mlflow` to `requirements-modal.txt`
2. Create `src/experiment_tracker.py` wrapper
3. Integrate into `train.py`
4. Integrate into `train_dit.py`

### Phase 2: Advanced Features (Week 2)
1. Add automatic hyperparameter logging
2. Add training curve visualization
3. Add sample image logging
4. Add model registry integration

### Phase 3: Dashboard Setup (Week 3)
1. Set up MLflow UI
2. Create dashboard views
3. Add alerting for anomalies

## Consequences

### Positive
- **Reproducibility**: All experiments tracked with hyperparameters
- **Comparison**: Easy to compare different runs
- **Visibility**: Real-time training progress
- **Audit Trail**: Complete history of model versions
- **Collaboration**: Team can view all experiments

### Negative
- **Storage**: MLflow artifacts consume disk space
- **Complexity**: Additional dependency and setup
- **Overhead**: Small performance impact from logging

### Neutral
- **Open Source**: MLflow is Apache 2.0 licensed
- **Vendor Neutral**: Can use local or hosted tracking

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Hyperparameters logged | 100% | All runs have params |
| Metrics logged | Every N steps | Configurable interval |
| Artifacts logged | Checkpoint + samples | Per run |
| Tracking overhead | <5% | Time impact |

## Dependencies

- `mlflow>=2.19.0`
- Optional: `mlflow[extras]` for additional features

## References

- MLflow Docs: https://mlflow.org/docs
- ADR-017: TinyDiT Training Infrastructure
- ADR-026: HuggingFace Model Publishing
