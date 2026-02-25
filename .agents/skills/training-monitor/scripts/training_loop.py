#!/usr/bin/env python3
"""
Training Loop - Monitor training with automatic retry until completion.

Usage:
    python .agents/skills/training-monitor/scripts/training_loop.py "modal run src/train.py data/cats"
    python .agents/skills/training-monitor/scripts/training_loop.py "modal run src/train.py data/cats" --max-retries 5
    python .agents/skills/training-monitor/scripts/training_loop.py "modal run src/train.py data/cats" --timeout-per-run 3600
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Setup logging
LOG_DIR = Path("logs/training")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / f"training_loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class TrainingLoop:
    """Monitor and retry training until completion"""

    def __init__(
        self,
        command: str,
        max_retries: int = 5,
        timeout_per_run: int | None = None,
        backoff_factor: float = 2.0,
        initial_delay: float = 60.0,
    ):
        self.command = command
        self.max_retries = max_retries
        self.timeout_per_run = timeout_per_run
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.attempts = []
        self.start_time = datetime.now()

    def is_recoverable_error(self, output: str) -> tuple[bool, str]:
        # Recoverable errors (can retry)
        self.RECOVERABLE_ERRORS = [
            "connection refused",
            "timeout",
            "network error",
            "ConnectionResetError",
            "modal.*timeout",
        ]

        # Non-recoverable errors (stop immediately)
        self.FATAL_ERRORS = [
            "CUDA out of memory",
            "cuDNN error",
            "illegal memory access",
            "app.*not.*found",
        ]
        """Check if error is recoverable"""
        import re

        for pattern in self.FATAL_ERRORS:
            if re.search(pattern, output, re.I):
                return False, f"Fatal error detected: {pattern}"

        for pattern in self.RECOVERABLE_ERRORS:
            if re.search(pattern, output, re.I):
                return True, f"Recoverable error detected: {pattern}"

        # Default: treat unknown errors as recoverable
        return True, "Unknown error (treating as recoverable)"

    def run_attempt(self, attempt_num: int) -> tuple[bool, str]:
        """Run a single training attempt"""
        logger.info(f"{'=' * 60}")
        logger.info(f"Training Attempt {attempt_num}/{self.max_retries + 1}")
        logger.info(f"{'=' * 60}")
        logger.info(f"Command: {self.command}")

        attempt_start = datetime.now()
        output_lines = []

        try:
            # Build timeout command
            if self.timeout_per_run:
                full_command = f"timeout {self.timeout_per_run}s {self.command}"
            else:
                full_command = self.command

            process = subprocess.Popen(
                full_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Stream output
            for line in process.stdout:
                line = line.strip()
                if line:
                    logger.info(line)
                    output_lines.append(line)

            process.wait()
            output = "\n".join(output_lines)

            if process.returncode == 0:
                duration = (datetime.now() - attempt_start).total_seconds()
                logger.info(f"✓ Training completed successfully in {duration:.1f}s")
                return True, output
            else:
                logger.error(f"✗ Training failed with exit code {process.returncode}")

                # Check if recoverable
                recoverable, reason = self.is_recoverable_error(output)
                if not recoverable:
                    logger.error(f"Non-recoverable error: {reason}")
                    return False, output

                logger.warning(f"Recoverable error: {reason}")
                return False, output

        except subprocess.TimeoutExpired:
            logger.error(f"✗ Training timed out after {self.timeout_per_run}s")
            output_lines.append(f"TIMEOUT after {self.timeout_per_run}s")
            return False, "\n".join(output_lines)

        except Exception as e:
            logger.error(f"✗ Unexpected error: {e}")
            return False, str(e)

    def run(self) -> bool:
        """Run training with retry loop"""
        logger.info(f"Starting training loop at {self.start_time.isoformat()}")
        logger.info(f"Max retries: {self.max_retries}")
        logger.info(f"Timeout per run: {self.timeout_per_run}s")
        logger.info(f"Backoff factor: {self.backoff_factor}")

        attempt = 0
        delay = self.initial_delay

        while attempt <= self.max_retries:
            attempt += 1
            attempt_start = datetime.now()

            success, _output = self.run_attempt(attempt)

            attempt_duration = (datetime.now() - attempt_start).total_seconds()
            self.attempts.append(
                {
                    "attempt": attempt,
                    "success": success,
                    "duration_seconds": attempt_duration,
                    "timestamp": attempt_start.isoformat(),
                }
            )

            if success:
                total_duration = (datetime.now() - self.start_time).total_seconds()
                logger.info(f"{'=' * 60}")
                logger.info("✓ TRAINING COMPLETED SUCCESSFULLY")
                logger.info(f"Total attempts: {attempt}")
                logger.info(f"Total duration: {total_duration:.1f}s")
                logger.info(f"{'=' * 60}")
                self.save_report(success=True)
                return True

            if attempt <= self.max_retries:
                logger.warning(
                    f"Retrying in {delay:.0f}s... (attempt {attempt}/{self.max_retries})"
                )
                time.sleep(delay)
                delay *= self.backoff_factor  # Exponential backoff

        # All attempts failed
        total_duration = (datetime.now() - self.start_time).total_seconds()
        logger.error(f"{'=' * 60}")
        logger.error(f"✗ TRAINING FAILED AFTER {self.max_retries + 1} ATTEMPTS")
        logger.error(f"Total duration: {total_duration:.1f}s")
        logger.error(f"{'=' * 60}")
        self.save_report(success=False)
        return False

    def save_report(self, success: bool, output_path: Path | None = None):
        """Save training loop report"""
        if output_path is None:
            output_path = (
                LOG_DIR
                / f"training_loop_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        report = {
            "success": success,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_duration_seconds": (
                datetime.now() - self.start_time
            ).total_seconds(),
            "command": self.command,
            "config": {
                "max_retries": self.max_retries,
                "timeout_per_run": self.timeout_per_run,
                "backoff_factor": self.backoff_factor,
                "initial_delay": self.initial_delay,
            },
            "attempts": self.attempts,
            "summary": {
                "total_attempts": len(self.attempts),
                "successful": success,
                "avg_attempt_duration": sum(
                    a["duration_seconds"] for a in self.attempts
                )
                / len(self.attempts)
                if self.attempts
                else 0,
            },
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Training loop with automatic retry")
    parser.add_argument("command", help="Training command to execute")
    parser.add_argument(
        "--max-retries", type=int, default=5, help="Maximum retry attempts"
    )
    parser.add_argument(
        "--timeout-per-run", type=int, help="Timeout per attempt (seconds)"
    )
    parser.add_argument("--backoff", type=float, default=2.0, help="Backoff multiplier")
    parser.add_argument(
        "--initial-delay",
        type=float,
        default=60.0,
        help="Initial delay between retries (seconds)",
    )

    args = parser.parse_args()

    loop = TrainingLoop(
        command=args.command,
        max_retries=args.max_retries,
        timeout_per_run=args.timeout_per_run,
        backoff_factor=args.backoff,
        initial_delay=args.initial_delay,
    )

    success = loop.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
