"""src/experiment_tracker.py

MLflow experiment tracking wrapper for tiny-cats-model.

Usage:
    from experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker("tiny-cats-model")
    tracker.start_run({"lr": 1e-4, "batch_size": 64})
    tracker.log_metrics({"loss": 0.5}, step=100)
    tracker.log_artifact("checkpoints/model.pt")
    tracker.end_run()
"""

from __future__ import annotations

import contextlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

try:
    import mlflow

    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


class ExperimentTracker:
    """MLflow experiment tracker wrapper."""

    def __init__(
        self,
        experiment_name: str = "tiny-cats-model",
        tracking_uri: str | None = None,
    ):
        """Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment.
            tracking_uri: MLflow tracking URI (default: file:./mlruns).
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run: Any = None
        self._enabled = HAS_MLFLOW

        if self._enabled:
            uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment_name)

    def start_run(
        self,
        params: dict[str, Any] | None = None,
        run_name: str | None = None,
    ) -> Any:
        """Start an MLflow run.

        Args:
            params: Hyperparameters to log.
            run_name: Optional name for the run.

        Returns:
            MLflow run object.
        """
        if not self._enabled:
            return None

        self.run = mlflow.start_run(run_name=run_name)

        if params:
            self.log_params(params)

        mlflow.log_param("start_time", datetime.now().isoformat())

        try:
            import git

            repo = git.Repo(search_parent_directories=True)
            mlflow.log_param("git_commit", repo.head.commit.hexsha[:8])
        except Exception:
            pass

        return self.run

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters.

        Args:
            params: Dictionary of parameters.
        """
        if not self._enabled or self.run is None:
            return

        for key, value in params.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(key, value)
            else:
                mlflow.log_param(key, str(value))

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log training metrics.

        Args:
            metrics: Dictionary of metric values.
            step: Optional step number.
        """
        if not self._enabled or self.run is None:
            return

        mlflow.log_metrics(metrics, step=step)

    def log_model(
        self,
        model: torch.nn.Module,
        artifact_path: str = "model",
        **kwargs,
    ) -> None:
        """Log a PyTorch model.

        Args:
            model: PyTorch model to log.
            artifact_path: Path to save the model.
            **kwargs: Additional arguments for mlflow.pytorch.log_model.
        """
        if not self._enabled or self.run is None:
            return

        with contextlib.suppress(Exception):
            mlflow.pytorch.log_model(model, artifact_path, **kwargs)

    def log_artifact(
        self, local_path: str | Path, artifact_path: str | None = None
    ) -> None:
        """Log an artifact.

        Args:
            local_path: Path to the artifact.
            artifact_path: Optional subdirectory in MLflow.
        """
        if not self._enabled or self.run is None:
            return

        local_path = Path(local_path)
        if not local_path.exists():
            return

        with contextlib.suppress(Exception):
            mlflow.log_artifact(str(local_path), artifact_path)

    def log_image(self, image: torch.Tensor | Any, name: str) -> None:
        """Log an image tensor.

        Args:
            image: Image tensor (CHW format).
            name: Name for the image.
        """
        if not self._enabled or self.run is None:
            return

        try:
            from PIL import Image

            if isinstance(image, torch.Tensor):
                img_array = image.cpu().numpy()
                if img_array.ndim == 3:
                    img_array = img_array.transpose(1, 2, 0)
                img_array = (img_array * 255).clip(0, 255).astype("uint8")
                img = Image.fromarray(img_array)
            else:
                img = image

            mlflow.log_image(img, f"samples/{name}.png")
        except Exception:
            pass

    def end_run(self) -> None:
        """End the current MLflow run."""
        if not self._enabled or self.run is None:
            return

        mlflow.log_param("end_time", datetime.now().isoformat())
        mlflow.end_run()
        self.run = None

    def __enter__(self) -> ExperimentTracker:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.end_run()
