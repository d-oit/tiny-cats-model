"""Modal training monitoring and error handling utilities.

Provides:
- Structured error logging with context
- GPU memory and health monitoring
- Training progress tracking
- Error recovery strategies
- Modal log aggregation

Usage:
    from modal_monitor import ModalMonitor, TrainingError

    monitor = ModalMonitor()
    with monitor.track_training("classifier", epochs=20):
        # Training code
        pass
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Custom exception for training errors."""

    pass


class ErrorSeverity(Enum):
    """Error severity levels."""

    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(Enum):
    """Error categories for better classification."""

    DATA_LOADING = "data_loading"
    MODEL_CREATION = "model_creation"
    TRAINING = "training"
    MEMORY = "memory"
    CHECKPOINT = "checkpoint"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class TrainingContext:
    """Context information for training runs."""

    script: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    status: str = "running"
    error: str | None = None
    error_category: ErrorCategory | None = None
    error_severity: ErrorSeverity | None = None
    gpu_type: str | None = None
    epochs_completed: int = 0
    steps_completed: int = 0
    final_metric: float | None = None
    checkpoint_path: str | None = None
    log_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "script": self.script,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "error": self.error,
            "error_category": self.error_category.value
            if self.error_category
            else None,
            "error_severity": self.error_severity.value
            if self.error_severity
            else None,
            "gpu_type": self.gpu_type,
            "epochs_completed": self.epochs_completed,
            "steps_completed": self.steps_completed,
            "final_metric": self.final_metric,
            "checkpoint_path": self.checkpoint_path,
            "log_file": self.log_file,
            "duration_seconds": (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time
                else (datetime.now() - self.start_time).total_seconds()
            ),
        }


@dataclass
class GPUMemoryStats:
    """GPU memory statistics."""

    allocated_mb: float
    reserved_mb: float
    free_mb: float
    total_mb: float
    utilization_percent: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocated_mb": round(self.allocated_mb, 2),
            "reserved_mb": round(self.reserved_mb, 2),
            "free_mb": round(self.free_mb, 2),
            "total_mb": round(self.total_mb, 2),
            "utilization_percent": round(self.utilization_percent, 2),
        }


class ModalMonitor:
    """Monitor and handle Modal training runs."""

    def __init__(self, log_dir: str = "/outputs/logs"):
        """Initialize monitor.

        Args:
            log_dir: Directory for storing monitoring logs.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.context: TrainingContext | None = None
        self.error_history: list[dict[str, Any]] = []
        self.gpu_stats_history: list[dict[str, Any]] = []

    def start_training(
        self,
        script: str,
        gpu_type: str | None = None,
        log_file: str | None = None,
    ) -> TrainingContext:
        """Start tracking a training run.

        Args:
            script: Name of training script.
            gpu_type: GPU type being used.
            log_file: Path to training log file.

        Returns:
            TrainingContext for the run.
        """
        self.context = TrainingContext(
            script=script,
            gpu_type=gpu_type,
            log_file=log_file,
        )
        self._log_event(
            "training_started",
            {"script": script, "gpu_type": gpu_type},
        )
        return self.context

    def update_progress(
        self,
        epoch: int | None = None,
        step: int | None = None,
        metric: float | None = None,
    ) -> None:
        """Update training progress.

        Args:
            epoch: Current epoch number.
            step: Current step number.
            metric: Current metric value (loss/accuracy).
        """
        if self.context:
            if epoch is not None:
                self.context.epochs_completed = epoch
            if step is not None:
                self.context.steps_completed = step
            if metric is not None:
                self.context.final_metric = metric

    def record_gpu_stats(self, stats: GPUMemoryStats) -> None:
        """Record GPU memory statistics.

        Args:
            stats: GPU memory statistics.
        """
        self.gpu_stats_history.append(stats.to_dict())

        # Keep only last 100 entries
        if len(self.gpu_stats_history) > 100:
            self.gpu_stats_history = self.gpu_stats_history[-100:]

    def handle_error(
        self,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Handle and log an error.

        Args:
            error: Exception that occurred.
            severity: Error severity level.
            category: Error category.
            context: Additional context information.

        Returns:
            Error record dictionary.
        """
        # Auto-categorize if not provided
        if category is None:
            category = self._categorize_error(error)

        error_record = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": severity.value,
            "category": category.value,
            "traceback": traceback.format_exc(),
            "context": context or {},
            "script": self.context.script if self.context else "unknown",
        }

        self.error_history.append(error_record)

        # Keep only last 50 errors
        if len(self.error_history) > 50:
            self.error_history = self.error_history[-50:]

        # Log error
        log_method = getattr(logger, severity.value, logger.error)
        log_method(f"Error [{category.value}]: {error}")

        # Save error to file
        self._save_error_log(error_record)

        return error_record

    def complete_training(
        self,
        status: str = "completed",
        final_metric: float | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        """Mark training as complete.

        Args:
            status: Final status (completed/failed/interrupted).
            final_metric: Final metric value.
            checkpoint_path: Path to saved checkpoint.
        """
        if self.context:
            self.context.end_time = datetime.now()
            self.context.status = status
            self.context.final_metric = final_metric
            self.context.checkpoint_path = checkpoint_path

            self._log_event(
                "training_completed",
                self.context.to_dict(),
            )

            # Save training summary
            self._save_training_summary()

    def get_recovery_suggestion(self, error: Exception) -> str:
        """Get recovery suggestion for an error.

        Args:
            error: Exception that occurred.

        Returns:
            Suggested recovery action.
        """
        error_str = str(error).lower()

        if "out of memory" in error_str or "cuda out of memory" in error_str:
            return (
                "OOM detected. Try: "
                "1) Reduce batch size, "
                "2) Enable mixed precision, "
                "3) Use gradient accumulation, "
                "4) Clear CUDA cache more frequently"
            )

        if "dataset" in error_str or "data" in error_str:
            return (
                "Data loading error. Try: "
                "1) Check dataset path, "
                "2) Verify dataset format, "
                "3) Re-download dataset, "
                "4) Check file permissions"
            )

        if "checkpoint" in error_str or "load" in error_str:
            return (
                "Checkpoint error. Try: "
                "1) Check checkpoint file exists, "
                "2) Verify checkpoint format, "
                "3) Start from scratch without resume"
            )

        if "network" in error_str or "connection" in error_str:
            return (
                "Network error. Try: "
                "1) Retry with exponential backoff, "
                "2) Check network connectivity, "
                "3) Use cached data if available"
            )

        return "No specific recovery suggestion. Check logs for details."

    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Automatically categorize an error."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        if "data" in error_str or "dataset" in error_str or "dataloader" in error_str:
            return ErrorCategory.DATA_LOADING

        if "model" in error_str or "layer" in error_str or "parameter" in error_str:
            return ErrorCategory.MODEL_CREATION

        if "memory" in error_str or "cuda" in error_str or "oom" in error_str:
            return ErrorCategory.MEMORY

        if "checkpoint" in error_str or "save" in error_str or "load" in error_str:
            return ErrorCategory.CHECKPOINT

        if (
            "network" in error_str
            or "connection" in error_str
            or "timeout" in error_str
        ):
            return ErrorCategory.NETWORK

        if "config" in error_str or "argument" in error_str or "parameter" in error_str:
            return ErrorCategory.CONFIGURATION

        if error_type == "runtimeerror":
            return ErrorCategory.TRAINING

        return ErrorCategory.UNKNOWN

    def _log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log an event to file."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self.log_dir / f"monitor_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(event) + "\n")

    def _save_error_log(self, error_record: dict[str, Any]) -> None:
        """Save error record to file."""
        error_file = (
            self.log_dir / f"errors_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
        )
        with open(error_file, "a") as f:
            f.write(json.dumps(error_record) + "\n")

    def _save_training_summary(self) -> None:
        """Save training summary to file."""
        if not self.context:
            return

        summary_file = (
            self.log_dir
            / f"summary_{self.context.script}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        summary = {
            "training": self.context.to_dict(),
            "errors": self.error_history[-10:],  # Last 10 errors
            "gpu_stats": {
                "samples": len(self.gpu_stats_history),
                "avg_allocated_mb": (
                    sum(s["allocated_mb"] for s in self.gpu_stats_history)
                    / len(self.gpu_stats_history)
                    if self.gpu_stats_history
                    else 0
                ),
                "max_allocated_mb": (
                    max(s["allocated_mb"] for s in self.gpu_stats_history)
                    if self.gpu_stats_history
                    else 0
                ),
            },
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training summary saved to {summary_file}")


@contextmanager
def track_training(
    script: str,
    gpu_type: str | None = None,
    log_file: str | None = None,
) -> Generator[ModalMonitor, None, None]:
    """Context manager for tracking training runs.

    Args:
        script: Name of training script.
        gpu_type: GPU type being used.
        log_file: Path to training log file.

    Yields:
        ModalMonitor instance.

    Example:
        with track_training("classifier", gpu_type="T4"):
            # Training code
            monitor.update_progress(epoch=5)
    """
    monitor = ModalMonitor()
    monitor.start_training(script, gpu_type, log_file)

    try:
        yield monitor
        monitor.complete_training(status="completed")
    except Exception as e:
        monitor.handle_error(e, severity=ErrorSeverity.ERROR)
        monitor.complete_training(status="failed")
        raise


def check_gpu_health() -> tuple[bool, str]:
    """Check GPU health status.

    Returns:
        Tuple of (is_healthy, message).
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return True, "CUDA not available, using CPU"

        # Check GPU memory
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**2)

        utilization = (allocated / total) * 100

        # Check for unhealthy states
        if utilization > 95:
            return False, f"GPU memory critically high: {utilization:.1f}%"

        if reserved - allocated > total * 0.5:
            return False, "Large gap between reserved and allocated memory"

        # Try a small CUDA operation
        _ = torch.zeros(1).cuda()
        del _
        torch.cuda.empty_cache()

        return True, f"GPU healthy: {allocated:.0f}/{total:.0f}MB ({utilization:.1f}%)"

    except Exception as e:
        return False, f"GPU health check failed: {e}"


def get_gpu_memory_stats() -> GPUMemoryStats | None:
    """Get current GPU memory statistics.

    Returns:
        GPUMemoryStats or None if CUDA not available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        free = total - reserved
        utilization = (allocated / total) * 100

        return GPUMemoryStats(
            allocated_mb=allocated,
            reserved_mb=reserved,
            free_mb=free,
            total_mb=total,
            utilization_percent=utilization,
        )

    except Exception:
        return None


def log_modal_environment() -> dict[str, Any]:
    """Log Modal environment information.

    Returns:
        Dictionary with environment info.
    """
    env_info = {
        "is_modal": os.environ.get("MODAL_APP_ID") is not None,
        "app_id": os.environ.get("MODAL_APP_ID"),
        "function_name": os.environ.get("MODAL_FUNCTION_NAME"),
        "container_id": os.environ.get("MODAL_CONTAINER_ID"),
        "gpu_type": os.environ.get("MODAL_GPU_TYPE"),
        "python_version": sys.version,
        "hostname": os.environ.get("HOSTNAME", "unknown"),
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(f"Modal environment: {env_info}")
    return env_info


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_multiplier: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
):
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.
        backoff_multiplier: Multiplier for delay increase.
        exceptions: Exception types to catch.

    Returns:
        Decorated function.

    Example:
        @retry_with_backoff(max_retries=3, exceptions=(RuntimeError,))
        def download_dataset():
            # ...
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay = min(delay * backoff_multiplier, max_delay)
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed. Last error: {e}"
                        )

            raise last_error

        return wrapper

    return decorator
