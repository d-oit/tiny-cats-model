# MLflow Setup Guide

This guide describes how to set up and configure MLflow for experiment tracking in the `tiny-cats-model` project.

## Overview

We use MLflow to track:
- Hyperparameters (learning rate, batch size, steps, etc.)
- Training metrics (loss, accuracy)
- Model artifacts (checkpoints, generated samples)
- Environment details

The `ExperimentTracker` class in `src/experiment_tracker.py` handles the integration with MLflow, providing a graceful fallback if the server is unavailable.

## Option 1: Local MLflow Server (Recommended for Development)

### 1. Install MLflow
Included in `requirements.txt`:
```bash
pip install mlflow
```

### 2. Start the Server
Run the MLflow UI in your terminal:
```bash
mlflow ui --port 5000
```
This will create a `mlruns` directory in your current folder and start the UI at [http://localhost:5000](http://localhost:5000).

### 3. Configure Project
Set the tracking URI as an environment variable:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

## Option 2: Remote MLflow Server (Managed)

If you have a shared MLflow server (e.g., hosted on Databricks, AWS, or a custom VM):

### 1. Set Tracking URI
```bash
export MLFLOW_TRACKING_URI=https://your-mlflow-server.com
```

### 2. Configure Authentication (if required)
```bash
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=password
# OR
export MLFLOW_TRACKING_TOKEN=your-auth-token
```

## Option 3: DagsHub Integration (Easy Managed MLflow)

DagsHub provides a managed MLflow server for free for open-source projects.

1. Create a repository on [DagsHub](https://dagshub.com).
2. Go to "Remote" -> "Experiments".
3. Copy the MLflow tracking URI.
4. Set the environment variables:
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/username/repo.mlflow
export MLFLOW_TRACKING_USERNAME=username
export MLFLOW_TRACKING_PASSWORD=dagshub_token
```

## Using with Modal

When training on Modal, the `ExperimentTracker` will attempt to connect to the URI specified in `MLFLOW_TRACKING_URI`.

To pass these to Modal:
1. Add them as secrets in Modal dashboard.
2. The training scripts (`src/train.py`, `src/train_dit.py`) will automatically use them if present.

## Verify Setup

Run a short test training:
```bash
python src/train.py data/cats --epochs 1 --batch-size 8
```
Then check your MLflow UI for the new run.
