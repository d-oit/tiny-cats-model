"""Volume management utilities for Modal training (ADR-024).

Provides utilities for:
- Checkpoint cleanup (retention policy)
- Directory management in volumes
- Checkpoint metadata inspection

Usage:
    from volume_utils import cleanup_old_checkpoints, ensure_directory_exists

    # Cleanup old checkpoints, keeping last 5
    cleanup_old_checkpoints(volume, "/outputs/checkpoints/classifier", keep_last_n=5)

    # Ensure directory exists
    ensure_directory_exists(volume, "/outputs/checkpoints/classifier/2026-02-25")
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any

import modal

logger = logging.getLogger(__name__)


def cleanup_old_checkpoints(
    volume: modal.Volume,
    base_path: str,
    keep_last_n: int = 5,
) -> None:
    """Remove old checkpoint directories, keeping only the last N.

    Args:
        volume: Modal volume to clean up.
        base_path: Base directory containing dated checkpoint subdirectories.
        keep_last_n: Number of recent checkpoint directories to keep.

    Example:
        cleanup_old_checkpoints(volume_outputs, "/outputs/checkpoints/classifier", 5)
    """
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
    except FileNotFoundError:
        logger.warning(f"Base path not found: {base_path}")


def ensure_directory_exists(volume: modal.Volume, path: str) -> None:
    """Ensure a directory exists in the volume.

    Args:
        volume: Modal volume.
        path: Directory path to create.
    """
    try:
        os.makedirs(path, exist_ok=True)
        volume.commit()
        logger.debug(f"Created directory: {path}")
    except Exception as e:
        # Directory may already exist in volume
        logger.debug(f"Directory may already exist: {e}")


def get_checkpoint_metadata(checkpoint_path: str) -> dict[str, Any]:
    """Get metadata about a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        Dictionary with checkpoint metadata (path, size, modified time).
        Returns {"exists": False} if file not found.
    """
    try:
        stat = os.stat(checkpoint_path)
        return {
            "path": checkpoint_path,
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": stat.st_mtime,
        }
    except FileNotFoundError:
        return {"path": checkpoint_path, "exists": False}


def list_checkpoints(base_path: str) -> list[dict[str, Any]]:
    """List all checkpoints in a directory with metadata.

    Args:
        base_path: Base directory containing checkpoints.

    Returns:
        List of checkpoint metadata dictionaries.
    """
    checkpoints = []
    try:
        result = subprocess.run(
            ["find", base_path, "-name", "*.pt"],
            capture_output=True,
            text=True,
            check=True,
        )
        for path in result.stdout.strip().split("\n"):
            if path:
                metadata = get_checkpoint_metadata(path)
                if metadata.get("exists"):
                    checkpoints.append(metadata)
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to list checkpoints: {e}")

    return checkpoints


def get_volume_usage(volume: modal.Volume, path: str = "/") -> dict[str, Any]:
    """Get volume usage statistics.

    Args:
        volume: Modal volume.
        path: Path to analyze (default: root).

    Returns:
        Dictionary with usage statistics.
    """
    try:
        result = subprocess.run(
            ["du", "-sh", path],
            capture_output=True,
            text=True,
            check=True,
        )
        size = result.stdout.strip().split()[0]

        result = subprocess.run(
            ["find", path, "-type", "f", "|", "wc", "-l"],
            capture_output=True,
            text=True,
            shell=True,
            check=True,
        )
        file_count = int(result.stdout.strip())

        return {"path": path, "size": size, "file_count": file_count}

    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to get volume usage: {e}")
        return {"path": path, "error": str(e)}
