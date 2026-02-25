#!/usr/bin/env python3
"""
Modal Training Monitor - Monitors training process with logging, error handling,
performance tracking, and security checks.
"""

import contextlib
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Configure logging
LOG_DIR = Path("logs/training")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class PerformanceMetrics:
    """Training performance metrics"""

    start_time: str
    end_time: str | None = None
    duration_seconds: float | None = None
    iterations_completed: int = 0
    loss_values: list[float] = None
    gpu_memory_mb: list[float] = None
    errors_count: int = 0
    warnings_count: int = 0

    def __post_init__(self):
        if self.loss_values is None:
            self.loss_values = []
        if self.gpu_memory_mb is None:
            self.gpu_memory_mb = []


@dataclass
class SecurityCheck:
    """Security check result"""

    check_name: str
    passed: bool
    details: str
    severity: str = "info"  # info, warning, critical


class TrainingMonitor:
    """Monitor Modal training process with comprehensive tracking"""

    # Performance patterns to extract
    LOSS_PATTERN = re.compile(r"loss[=:\s]+([0-9.]+)", re.I)
    ITER_PATTERN = re.compile(r"(?:iteration|step|epoch)[=:\s]+(\d+)", re.I)
    GPU_PATTERN = re.compile(r"GPU.*?(\d+)\s*MB", re.I)

    def __init__(self, command: str, timeout: int | None = None):
        self.command = command
        self.timeout = timeout
        self.process: subprocess.Popen | None = None
        self.metrics = PerformanceMetrics(start_time=datetime.now().isoformat())
        self.security_checks: list[SecurityCheck] = []
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []

    def run_security_checks(self, line: str) -> list[SecurityCheck]:
        """Run security checks on output line"""
        checks = []

        for check_name, pattern in self.SECURITY_PATTERNS.items():
            if pattern.search(line):
                check = SecurityCheck(
                    check_name=check_name,
                    passed=False,
                    details=f"Potential security issue detected: {line.strip()[:100]}",
                    severity="critical"
                    if check_name in ["secret_exposure", "credential_leak"]
                    else "warning",
                )
                checks.append(check)
                logger.warning(f"SECURITY [{check_name}]: {check.details}")

        if not checks:
            checks.append(
                SecurityCheck(
                    check_name="output_scan",
                    passed=True,
                    details="No security issues detected",
                    severity="info",
                )
            )

        return checks

    def extract_metrics(self, line: str):
        """Extract performance metrics from output line"""
        # Extract loss
        loss_match = self.LOSS_PATTERN.search(line)
        if loss_match:
            try:
                loss = float(loss_match.group(1))
                self.metrics.loss_values.append(loss)
            except ValueError:
                pass

        # Extract iteration
        iter_match = self.ITER_PATTERN.search(line)
        if iter_match:
            with contextlib.suppress(ValueError):
                self.metrics.iterations_completed = int(iter_match.group(1))

        # Extract GPU memory
        gpu_match = self.GPU_PATTERN.search(line)
        if gpu_match:
            try:
                gpu_mem = float(gpu_match.group(1))
                self.metrics.gpu_memory_mb.append(gpu_mem)
            except ValueError:
                pass

    def detect_errors(self, line: str) -> dict[str, Any] | None:
        """Detect errors in output line"""
        for error_type, pattern in self.ERROR_PATTERNS.items():
            if pattern.search(line):
                error = {
                    "type": error_type,
                    "message": line.strip()[:200],
                    "timestamp": datetime.now().isoformat(),
                }
                self.metrics.errors_count += 1
                self.errors.append(error)
                logger.error(f"ERROR [{error_type}]: {error['message']}")
                return error
        return None

    def detect_warnings(self, line: str) -> dict[str, Any] | None:
        """Detect warnings in output line"""
        if re.search(r"(warning|warn|deprecated)", line, re.I):
            warning = {
                "message": line.strip()[:200],
                "timestamp": datetime.now().isoformat(),
            }
            self.metrics.warnings_count += 1
            self.warnings.append(warning)
            logger.warning(f"WARNING: {warning['message']}")
            return warning
        return None

    def start(self):
        """Start the training process"""
        logger.info(f"Starting training: {self.command}")
        self.metrics.start_time = datetime.now().isoformat()

        try:
            self.process = subprocess.Popen(
                self.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            logger.info(f"Process started with PID: {self.process.pid}")
        except Exception as e:
            logger.error(f"Failed to start process: {e}")
            raise

    def monitor(self) -> TrainingStatus:
        """Monitor the training process until completion"""
        if not self.process:
            raise RuntimeError("Process not started. Call start() first.")

        start_time = time.time()

        try:
            while True:
                # Check timeout
                if self.timeout and (time.time() - start_time) > self.timeout:
                    logger.error("Training timed out")
                    self.process.kill()
                    return TrainingStatus.TIMEOUT

                # Read output
                line = self.process.stdout.readline()
                if line:
                    line = line.strip()
                    if line:
                        logger.info(line)

                        # Run all monitoring
                        self.run_security_checks(line)
                        self.extract_metrics(line)
                        self.detect_errors(line)
                        self.detect_warnings(line)

                # Check if process completed
                poll = self.process.poll()
                if poll is not None:
                    self.metrics.end_time = datetime.now().isoformat()
                    self.metrics.duration_seconds = time.time() - start_time

                    if poll == 0:
                        logger.info("Training completed successfully")
                        return TrainingStatus.COMPLETED
                    else:
                        logger.error(f"Training failed with exit code: {poll}")
                        return TrainingStatus.FAILED

                # Small delay to prevent CPU spinning
                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            self.process.kill()
            return TrainingStatus.FAILED
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            if self.process and self.process.poll() is None:
                self.process.kill()
            return TrainingStatus.FAILED

    def get_report(self) -> dict[str, Any]:
        """Generate comprehensive training report"""
        return {
            "status": self.monitor.__name__,
            "metrics": asdict(self.metrics),
            "security_checks": [asdict(c) for c in self.security_checks],
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": {
                "total_errors": len(self.errors),
                "total_warnings": len(self.warnings),
                "security_issues": sum(1 for c in self.security_checks if not c.passed),
                "final_loss": self.metrics.loss_values[-1]
                if self.metrics.loss_values
                else None,
                "avg_gpu_memory_mb": sum(self.metrics.gpu_memory_mb)
                / len(self.metrics.gpu_memory_mb)
                if self.metrics.gpu_memory_mb
                else None,
            },
        }

    def save_report(self, output_path: Path | None = None):
        """Save training report to file"""
        if output_path is None:
            output_path = (
                LOG_DIR
                / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        report = self.get_report()
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Training report saved to: {output_path}")
        return output_path


def main():
    """Main entry point for training monitor"""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Modal training process")
    parser.add_argument("command", help="Training command to execute")
    parser.add_argument("--timeout", type=int, help="Timeout in seconds")
    parser.add_argument("--output", type=Path, help="Output report path")

    args = parser.parse_args()

    monitor = TrainingMonitor(args.command, timeout=args.timeout)
    monitor.start()
    status = monitor.monitor()
    monitor.save_report(args.output)

    print(f"\n{'=' * 60}")
    print(f"Training Status: {status.value}")
    print(f"Report: {monitor.get_report()['summary']}")
    print(f"{'=' * 60}")

    sys.exit(0 if status == TrainingStatus.COMPLETED else 1)


if __name__ == "__main__":
    main()
