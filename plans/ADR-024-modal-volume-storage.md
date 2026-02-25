# ADR-024: Modal Volume and Storage Best Practices

**Date:** 2026-02-25
**Status:** Proposed
**Authors:** AI Agent
**Related:** ADR-007 (Modal GPU Training Fix), ADR-010 (Modal Training Improvements), ADR-020 (Modal CLI-First Training Strategy)

## Context

### Current State

Current volume configuration in training scripts:

```python
# src/train.py and src/train_dit.py
volume_outputs = modal.Volume.from_name("cats-model-outputs", create_if_missing=True)
volume_data = modal.Volume.from_name("cats-model-data", create_if_missing=True)

@app.function(
    volumes={
        "/outputs": volume_outputs,
        "/data": volume_data,
    },
    ...
)
def train_modal(...):
```

### Issues Identified

1. **No Explicit Commit**: Volumes not explicitly committed after writes
2. **No Volume Organization**: Flat directory structure for checkpoints
3. **No Cleanup Strategy**: Old checkpoints accumulate indefinitely
4. **No Backup Strategy**: Single volume, no redundancy
5. **No Performance Optimization**: No read-ahead or caching configuration
6. **Dataset Redownload**: Dataset downloaded every run instead of caching

### Modal Volumes Best Practices (2025-2026)

Based on Modal documentation:

1. **Explicit Commits**: Call `volume.commit()` after important writes
2. **Organized Structure**: Use dated/versioned directories
3. **Automatic Cleanup**: Remove old checkpoints based on retention policy
4. **Read Performance**: Modal Volumes provide 1-2 GB/s read speeds
5. **Write Buffering**: Writes are buffered, commit flushes to storage
6. **Cross-Function Sharing**: Same volume can be mounted by multiple functions

## Decision

We will implement comprehensive volume and storage best practices:

### 1. Explicit Volume Commits

```python
@app.function(volumes={"/outputs": output_volume})
def train_modal(...):
    # Training logic
    torch.save(checkpoint, "/outputs/checkpoints/epoch_10.pt")
    
    # Explicit commit after important writes
    output_volume.commit()
```

**Why Commit:**
- Ensures data is persisted across function invocations
- Makes checkpoints immediately available to other functions
- Prevents data loss on unexpected termination

### 2. Organized Directory Structure

```
/outputs/
├── checkpoints/
│   ├── classifier/
│   │   ├── 2026-02-25/
│   │   │   ├── epoch_10.pt
│   │   │   ├── best_model.pt
│   │   │   └── training_log.json
│   │   └── 2026-02-26/
│   └── dit/
│       ├── 2026-02-25/
│       │   ├── step_10000.pt
│       │   ├── step_20000.pt
│       │   └── final_model.pt
│       └── samples/
├── logs/
│   ├── classifier/
│   └── dit/
└── metadata/
    └── training_runs.json
```

### 3. Checkpoint Retention Policy

```python
def cleanup_old_checkpoints(
    volume: modal.Volume,
    base_path: str,
    keep_last_n: int = 5,
) -> None:
    """Remove old checkpoints, keeping only the last N."""
    import subprocess
    
    # List checkpoints sorted by date
    result = subprocess.run(
        ["ls", "-t", base_path],
        capture_output=True,
        text=True,
    )
    checkpoints = result.stdout.strip().split("\n")
    
    # Remove old checkpoints
    for old_checkpoint in checkpoints[keep_last_n:]:
        checkpoint_path = f"{base_path}/{old_checkpoint}"
        subprocess.run(["rm", "-rf", checkpoint_path])
        logger.info(f"Removed old checkpoint: {checkpoint_path}")
    
    volume.commit()
```

### 4. Dataset Caching Strategy

```python
def ensure_dataset_cached(data_volume: modal.Volume, data_dir: str) -> bool:
    """Check if dataset is cached, download if needed."""
    from pathlib import Path
    
    cats_dir = Path(data_dir) / "cats"
    if cats_dir.exists() and len(list(cats_dir.iterdir())) > 0:
        logger.info("Dataset already cached in volume")
        return True
    
    logger.info("Dataset not found, downloading...")
    # Download dataset
    subprocess.run(["python", "data/download.py"], check=True)
    
    # Copy to volume and commit
    volume.copy_local_dir("data/cats", f"{data_dir}/cats")
    volume.commit()
    return True
```

### 5. Volume Mounting Best Practices

```python
# Define volumes at module level (reusable)
output_volume = modal.Volume.from_name("cats-model-outputs", create_if_missing=True)
data_volume = modal.Volume.from_name("cats-dataset", create_if_missing=True)

# Mount with clear paths
@app.function(
    volumes={
        "/outputs": output_volume,  # Checkpoints and logs
        "/data": data_volume,       # Dataset cache
    }
)
```

### 6. Cross-Function Volume Sharing

```python
# Training function writes checkpoints
@app.function(volumes={"/outputs": output_volume})
def train(...):
    torch.save(model, "/outputs/checkpoints/model.pt")
    output_volume.commit()

# Evaluation function reads same checkpoints
@app.function(volumes={"/outputs": output_volume})
def evaluate(...):
    model = torch.load("/outputs/checkpoints/model.pt")
```

### 7. Volume CLI Management

```bash
# List volume contents
modal volume ls cats-model-outputs

# Download specific file
modal volume get cats-model-outputs /outputs/checkpoints/best_model.pt ./best_model.pt

# Upload file to volume
modal volume put cats-model-outputs ./model.pt /outputs/checkpoints/model.pt

# Delete volume (careful!)
modal volume delete cats-model-outputs
```

## Implementation Plan

### Phase 1: Add Volume Commits

1. Add `volume.commit()` calls after checkpoint saves
2. Add `volume.commit()` after log writes
3. Test checkpoint persistence across runs

### Phase 2: Organize Directory Structure

1. Create dated checkpoint directories
2. Separate classifier and DiT checkpoints
3. Add metadata JSON for training runs

### Phase 3: Implement Cleanup Strategy

1. Add cleanup function for old checkpoints
2. Configure retention policy (keep last 5)
3. Run cleanup at start of training

### Phase 4: Dataset Caching

1. Check for cached dataset before download
2. Copy downloaded dataset to volume
3. Commit volume after dataset download

## Code Changes

### src/train.py (Updated Volume Handling)

```python
from datetime import datetime

# Volume definitions
output_volume = modal.Volume.from_name("cats-model-outputs", create_if_missing=True)
data_volume = modal.Volume.from_name("cats-dataset", create_if_missing=True)

@app.function(
    image=image,
    volumes={
        "/outputs": output_volume,
        "/data": data_volume,
    },
    gpu="T4",
    timeout=3600,
    retries=Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=10.0),
)
def train_modal(
    data_dir: str = "/data/cats",
    epochs: int = 10,
    # ... other params
) -> dict[str, Any]:
    """Modal function with improved volume handling."""
    sys.path.insert(0, "/app/src")
    os.chdir("/app")
    
    logger = setup_logging()
    logger.info("Starting Modal GPU training")
    
    # Ensure dataset is cached
    if not Path(data_dir).exists() or not list(Path(data_dir).iterdir()):
        logger.info("Downloading dataset...")
        Path("/data").mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["python", "data/download.py"],
            cwd="/app",
            env={**os.environ, "DATA_DIR": "/data"},
            check=True,
            timeout=600,
        )
        # Copy to volume and commit
        data_volume.copy_local_dir("/data/cats", f"{data_dir}/cats")
        data_volume.commit()
        logger.info("Dataset cached in volume")
    
    # Create dated checkpoint directory
    run_date = datetime.now().strftime("%Y-%m-%d")
    checkpoint_dir = f"/outputs/checkpoints/classifier/{run_date}"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Update output path to dated directory
    output = f"{checkpoint_dir}/cats_model.pt"
    log_file = f"{checkpoint_dir}/training.log"
    logger = setup_logging(log_file)
    
    try:
        val_acc = train(
            data_dir=data_dir,
            epochs=epochs,
            output=output,
            log_file=log_file,
            logger=logger,
            # ... other params
        )
        
        # Commit volume after successful training
        output_volume.commit()
        logger.info("Checkpoint committed to volume")
        
        # Cleanup old checkpoints (keep last 5)
        cleanup_old_checkpoints(
            output_volume,
            "/outputs/checkpoints/classifier",
            keep_last_n=5,
        )
        
        return {"status": "completed", "output": output, "val_acc": val_acc}
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Commit partial checkpoint on error
        output_volume.commit()
        raise
```

### src/train_dit.py (Similar Updates)

```python
# Same pattern as train.py with:
# - Dated checkpoint directories
# - Volume commits after saves
# - Dataset caching
# - Cleanup policy
```

### New: Volume Utility Functions

```python
# src/volume_utils.py
"""Volume management utilities for Modal training."""

import subprocess
from pathlib import Path
import modal


def cleanup_old_checkpoints(
    volume: modal.Volume,
    base_path: str,
    keep_last_n: int = 5,
) -> None:
    """Remove old checkpoint directories, keeping only the last N."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # List directories sorted by modification time (newest first)
        result = subprocess.run(
            ["ls", "-t", base_path],
            capture_output=True,
            text=True,
            check=True,
        )
        directories = [d for d in result.stdout.strip().split("\n") if d]
        
        # Remove old directories
        for old_dir in directories[keep_last_n:]:
            dir_path = f"{base_path}/{old_dir}"
            subprocess.run(["rm", "-rf", dir_path], check=True)
            logger.info(f"Removed old checkpoint directory: {dir_path}")
        
        # Commit volume changes
        volume.commit()
        logger.info(f"Volume committed after cleanup (kept {keep_last_n} latest)")
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"Cleanup failed: {e}")


def ensure_directory_exists(volume: modal.Volume, path: str) -> None:
    """Ensure a directory exists in the volume."""
    import os
    try:
        os.makedirs(path, exist_ok=True)
        volume.commit()
    except Exception as e:
        # Directory may already exist in volume
        pass


def get_checkpoint_metadata(volume: modal.Volume, checkpoint_path: str) -> dict:
    """Get metadata about a checkpoint file."""
    import os
    import json
    
    try:
        stat = os.stat(checkpoint_path)
        return {
            "path": checkpoint_path,
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime,
        }
    except FileNotFoundError:
        return {"path": checkpoint_path, "exists": False}
```

## Consequences

### Positive
- ✅ **Data Persistence**: Explicit commits ensure checkpoints are saved
- ✅ **Organized Storage**: Dated directories make finding checkpoints easy
- ✅ **Cost Control**: Cleanup policy prevents unlimited storage growth
- ✅ **Faster Starts**: Cached dataset avoids redownload
- ✅ **Cross-Function Sharing**: Multiple functions can access same checkpoints

### Negative
- ⚠️ **Commit Overhead**: Volume commits add small latency
- ⚠️ **Storage Costs**: Modal Volumes have storage costs (~$0.10/GB/month)
- ⚠️ **Complexity**: More code for volume management

### Neutral
- ℹ️ **Same Checkpoint Files**: Format unchanged
- ℹ️ **Backward Compatible**: Old checkpoints still accessible

## Alternatives Considered

### Alternative 1: No Explicit Commits
**Proposal**: Rely on automatic volume persistence.

**Rejected Because**:
- Data may be lost on unexpected termination
- Checkpoints not immediately available to other functions
- Modal recommends explicit commits for important data

### Alternative 2: External Storage (S3/GCS)
**Proposal**: Use cloud storage instead of Modal Volumes.

**Rejected Because**:
- Adds complexity (credentials, SDK)
- Slower than Modal Volumes (1-2 GB/s vs network speed)
- Modal Volumes are simpler for this use case

### Alternative 3: No Cleanup
**Proposal**: Keep all checkpoints indefinitely.

**Rejected Because**:
- Storage costs accumulate
- Hard to find relevant checkpoints
- Good practice to manage storage

## Success Metrics

- [ ] All checkpoint saves followed by volume.commit()
- [ ] Dated directory structure implemented
- [ ] Cleanup policy removes old checkpoints
- [ ] Dataset cached in volume (no redownload)
- [ ] Cross-function checkpoint sharing works

## References

- Modal Volumes: https://modal.com/docs/guide/volumes
- Modal CLI Volume Commands: https://modal.com/docs/reference/cli
- ADR-007: Modal GPU Training Fix
- ADR-010: Modal Training Improvements
