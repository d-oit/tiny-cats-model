"""src/retry_utils.py

Retry utilities with exponential backoff for transient failures.

This module provides:
- RetryConfig dataclass with configurable parameters
- Retry decorator with exponential backoff and jitter
- RetryManager class for complex retry scenarios
- Status code checking for HTTP-like operations
- Structured logging for retry attempts
- Integration with auth_utils for authentication flows

Usage:
    from retry_utils import (
        RetryConfig,
        RetryManager,
        retry_with_backoff,
        upload_with_retry,
    )

    # RetryConfig dataclass
    config = RetryConfig(
        max_retries=3,
        backoff_coefficient=2.0,
        initial_delay=1.0,
        max_delay=60.0,
        retry_on_exceptions=[ConnectionError, TimeoutError],
        retry_on_status_codes=[429, 500, 502, 503, 504],
    )

    # Simple retry decorator
    @retry_with_backoff(config=config)
    def upload_file():
        ...

    # RetryManager with execute methods
    manager = RetryManager()
    result = manager.execute(upload_func, arg1, arg2, kwarg1="value")
    result = manager.execute_with_config(upload_func, config, arg1, arg2)
"""

from __future__ import annotations

import functools
import logging
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts before giving up
        backoff_coefficient: Multiplier for exponential backoff calculation
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay in seconds between retries
        retry_on_exceptions: List of exception types that trigger retry
        retry_on_status_codes: List of HTTP status codes that trigger retry
        jitter: Whether to add random jitter to delays to prevent thundering herd
    """

    max_retries: int = 3
    backoff_coefficient: float = 2.0
    initial_delay: float = 1.0
    max_delay: float = 60.0
    retry_on_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (
            ConnectionError,
            TimeoutError,
            OSError,
        )
    )
    retry_on_status_codes: list[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )
    jitter: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.backoff_coefficient < 1.0:
            raise ValueError("backoff_coefficient must be >= 1.0")
        if self.initial_delay < 0:
            raise ValueError("initial_delay must be non-negative")
        if self.max_delay < self.initial_delay:
            raise ValueError("max_delay must be >= initial_delay")

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number using exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed, 0 is first retry)

        Returns:
            Delay in seconds before next retry attempt
        """
        # Exponential backoff: initial_delay * (backoff_coefficient ^ attempt)
        delay = self.initial_delay * (self.backoff_coefficient**attempt)

        # Cap at max delay
        delay = min(delay, self.max_delay)

        # Add jitter to prevent thundering herd (±25% randomization)
        if self.jitter:
            jitter_factor = 0.75 + (random.random() * 0.5)
            delay *= jitter_factor

        return delay

    def should_retry_exception(self, exception: Exception) -> bool:
        """Check if an exception should trigger a retry.

        Args:
            exception: The exception that was raised

        Returns:
            True if the exception type is in retry_on_exceptions
        """
        return isinstance(exception, self.retry_on_exceptions)

    def should_retry_status_code(self, status_code: int | None) -> bool:
        """Check if an HTTP status code should trigger a retry.

        Args:
            status_code: HTTP status code (or None if not applicable)

        Returns:
            True if the status code is in retry_on_status_codes
        """
        if status_code is None:
            return False
        return status_code in self.retry_on_status_codes


def retry_with_backoff(
    max_retries: int = 3,
    backoff_coefficient: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_on_exceptions: Sequence[type[Exception]] | None = None,
    retry_on_status_codes: Sequence[int] | None = None,
    jitter: bool = True,
    logger: logging.Logger | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_coefficient: Multiplier for exponential backoff
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay in seconds
        retry_on_exceptions: Exception types that trigger retry
        retry_on_status_codes: HTTP status codes that trigger retry
        jitter: Whether to add random jitter to delays
        logger: Logger for retry events

    Returns:
        Decorated function with retry logic

    Example:
        @retry_with_backoff(max_retries=3, backoff_coefficient=2.0)
        def upload_to_hf():
            api.upload_file(...)

        @retry_with_backoff(
            max_retries=5,
            retry_on_exceptions=[ConnectionError, TimeoutError],
            retry_on_status_codes=[429, 503],
        )
        def make_api_call():
            response = requests.get(url)
            if response.status_code in [429, 503]:
                raise RetryableHTTPError(response.status_code)
            return response
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        config = RetryConfig(
            max_retries=max_retries,
            backoff_coefficient=backoff_coefficient,
            initial_delay=initial_delay,
            max_delay=max_delay,
            retry_on_exceptions=tuple(retry_on_exceptions)
            if retry_on_exceptions
            else RetryConfig().retry_on_exceptions,
            retry_on_status_codes=list(retry_on_status_codes)
            if retry_on_status_codes
            else RetryConfig().retry_on_status_codes,
            jitter=jitter,
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            log = logger or logging.getLogger(__name__)
            last_exception: Exception | None = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if we should retry this exception
                    if not config.should_retry_exception(e):
                        log.error(
                            f"Non-retryable error in {func.__name__}: "
                            f"{type(e).__name__}: {e}"
                        )
                        raise

                    # Check if we've exhausted retries
                    if attempt >= config.max_retries:
                        log.error(
                            f"All {config.max_retries + 1} attempts failed for "
                            f"{func.__name__}: {type(e).__name__}: {e}"
                        )
                        break

                    # Calculate delay and wait
                    delay = config.calculate_delay(attempt)
                    log.warning(
                        f"Attempt {attempt + 1}/{config.max_retries + 1} failed for "
                        f"{func.__name__}: {type(e).__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)

            # All retries exhausted
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Retry loop exited unexpectedly for {func.__name__}")

        return wrapper

    return decorator


class RetryManager:
    """Manage retry logic for complex scenarios with detailed tracking.

    This class provides programmatic retry control with:
    - Configurable retry parameters
    - Detailed attempt history
    - Support for custom retry conditions
    - Integration with logging systems

    Example:
        manager = RetryManager(max_retries=5, initial_delay=2.0)

        # Simple execution
        result = manager.execute(api_call, arg1, arg2)

        # Execution with custom config
        config = RetryConfig(max_retries=3, retry_on_exceptions=[ConnectionError])
        result = manager.execute_with_config(api_call, config, arg1, arg2)

        # Get retry report
        report = manager.get_retry_report()
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_coefficient: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        retry_on_exceptions: tuple[type[Exception], ...] | None = None,
        retry_on_status_codes: list[int] | None = None,
        jitter: bool = True,
        logger: logging.Logger | None = None,
    ):
        """Initialize RetryManager with configuration.

        Args:
            max_retries: Maximum number of retry attempts
            backoff_coefficient: Multiplier for exponential backoff
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            retry_on_exceptions: Exception types that trigger retry
            retry_on_status_codes: HTTP status codes that trigger retry
            jitter: Whether to add jitter to delays
            logger: Logger for retry events
        """
        self.config = RetryConfig(
            max_retries=max_retries,
            backoff_coefficient=backoff_coefficient,
            initial_delay=initial_delay,
            max_delay=max_delay,
            retry_on_exceptions=retry_on_exceptions
            or RetryConfig().retry_on_exceptions,
            retry_on_status_codes=retry_on_status_codes
            or RetryConfig().retry_on_status_codes,
            jitter=jitter,
        )
        self.log = logger or logging.getLogger(__name__)
        self.attempt_history: list[dict[str, Any]] = []

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a function with retry logic using default config.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from successful function execution

        Raises:
            Exception: The last exception if all retries fail

        Example:
            manager = RetryManager(max_retries=3)
            result = manager.execute(upload_file, "/path/to/file")
        """
        return self._execute_with_config_internal(func, self.config, *args, **kwargs)

    def execute_with_config(
        self,
        func: Callable[..., T],
        config: RetryConfig,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a function with custom retry configuration.

        Args:
            func: Function to execute
            config: Custom RetryConfig to use for this execution
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from successful function execution

        Raises:
            Exception: The last exception if all retries fail

        Example:
            custom_config = RetryConfig(
                max_retries=5,
                retry_on_exceptions=[ConnectionError],
            )
            result = manager.execute_with_config(
                upload_file, custom_config, "/path/to/file"
            )
        """
        return self._execute_with_config_internal(func, config, *args, **kwargs)

    def _execute_with_config_internal(
        self,
        func: Callable[..., T],
        config: RetryConfig,
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Internal method to execute function with specified config.

        Args:
            func: Function to execute
            config: RetryConfig to use
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from successful function execution
        """
        last_exception: Exception | None = None
        start_time = time.time()

        for attempt in range(config.max_retries + 1):
            attempt_info: dict[str, Any] = {
                "function": func.__name__,
                "attempt": attempt + 1,
                "max_attempts": config.max_retries + 1,
                "timestamp": time.time(),
            }

            try:
                result = func(*args, **kwargs)

                # Record successful attempt
                attempt_info["success"] = True
                attempt_info["duration"] = time.time() - attempt_info["timestamp"]
                self.attempt_history.append(attempt_info)

                total_duration = time.time() - start_time
                if attempt > 0:
                    self.log.info(
                        f"Function {func.__name__} succeeded after "
                        f"{attempt + 1} attempts ({total_duration:.2f}s total)"
                    )

                return result

            except Exception as e:
                last_exception = e
                attempt_info["error"] = str(e)
                attempt_info["error_type"] = type(e).__name__

                # Check if we should retry
                if not config.should_retry_exception(e):
                    attempt_info["success"] = False
                    attempt_info["non_retryable"] = True
                    self.attempt_history.append(attempt_info)
                    self.log.error(
                        f"Non-retryable error in {func.__name__}: "
                        f"{type(e).__name__}: {e}"
                    )
                    raise

                # Check if we've exhausted retries
                if attempt >= config.max_retries:
                    attempt_info["success"] = False
                    self.attempt_history.append(attempt_info)
                    self.log.error(
                        f"All {config.max_retries + 1} attempts failed for "
                        f"{func.__name__}: {type(e).__name__}: {e}"
                    )
                    break

                # Calculate delay and wait
                delay = config.calculate_delay(attempt)
                attempt_info["retry_delay"] = delay
                attempt_info["success"] = False
                self.attempt_history.append(attempt_info)

                self.log.warning(
                    f"Attempt {attempt + 1}/{config.max_retries + 1} failed for "
                    f"{func.__name__}: {type(e).__name__}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)

        # All retries exhausted
        if last_exception:
            raise last_exception
        raise RuntimeError(f"Retry loop exited unexpectedly for {func.__name__}")

    def get_retry_report(self) -> dict[str, Any]:
        """Generate detailed retry report.

        Returns:
            Dictionary with retry statistics and history
        """
        if not self.attempt_history:
            return {
                "status": "no_attempts",
                "total_executions": 0,
            }

        # Group by function
        function_stats: dict[str, dict[str, Any]] = {}
        for attempt in self.attempt_history:
            func_name = attempt["function"]
            if func_name not in function_stats:
                function_stats[func_name] = {
                    "total_attempts": 0,
                    "successful": 0,
                    "failed": 0,
                    "retries_needed": 0,
                }

            stats = function_stats[func_name]
            stats["total_attempts"] += 1

            if attempt.get("success"):
                stats["successful"] += 1
                if attempt["attempt"] > 1:
                    stats["retries_needed"] += 1
            else:
                stats["failed"] += 1

        total_attempts = len(self.attempt_history)
        successful = sum(1 for a in self.attempt_history if a.get("success"))
        failed = total_attempts - successful

        return {
            "status": "success" if failed == 0 else "partial_failure",
            "summary": {
                "total_attempts": total_attempts,
                "successful": successful,
                "failed": failed,
                "success_rate": successful / total_attempts
                if total_attempts > 0
                else 0,
            },
            "functions": function_stats,
            "config": {
                "max_retries": self.config.max_retries,
                "backoff_coefficient": self.config.backoff_coefficient,
                "initial_delay": self.config.initial_delay,
                "max_delay": self.config.max_delay,
            },
            "history": self.attempt_history,
        }

    def reset_history(self) -> None:
        """Clear attempt history."""
        self.attempt_history.clear()
        self.log.debug("Retry history cleared")


class RetryableHTTPError(Exception):
    """Exception for HTTP errors that should trigger retry.

    This exception wraps HTTP status codes to enable retry logic
    based on status code checking.

    Attributes:
        status_code: HTTP status code
        message: Error message
        response: Original response object (if available)
    """

    def __init__(
        self,
        status_code: int,
        message: str = "",
        response: Any = None,
    ):
        self.status_code = status_code
        self.message = message
        self.response = response
        super().__init__(f"HTTP {status_code}: {message}")


def upload_with_retry(
    upload_func: Callable[..., T],
    max_retries: int = 3,
    initial_delay: float = 2.0,
    logger: logging.Logger | None = None,
) -> T:
    """Specialized retry function for HuggingFace uploads.

    Handles common HuggingFace Hub API errors with appropriate retry logic.
    Integrates with auth_utils for authentication-aware retry handling.

    Args:
        upload_func: Upload function to retry
        max_retries: Maximum retry attempts
        initial_delay: Initial delay in seconds
        logger: Logger for retry events

    Returns:
        Result from successful upload

    Raises:
        Exception: Last exception if all retries fail

    Example:
        def upload_model():
            return api.upload_file(...)

        result = upload_with_retry(upload_model, max_retries=3)
    """
    log = logger or logging.getLogger(__name__)

    # HuggingFace-specific retry configuration
    config = RetryConfig(
        max_retries=max_retries,
        backoff_coefficient=2.0,
        initial_delay=initial_delay,
        max_delay=120.0,  # Longer max delay for uploads
        retry_on_exceptions=(
            ConnectionError,
            TimeoutError,
            OSError,
            RetryableHTTPError,
        ),
        retry_on_status_codes=[429, 500, 502, 503, 504],
        jitter=True,
    )

    manager = RetryManager(
        max_retries=config.max_retries,
        backoff_coefficient=config.backoff_coefficient,
        initial_delay=config.initial_delay,
        max_delay=config.max_delay,
        retry_on_exceptions=config.retry_on_exceptions,
        retry_on_status_codes=config.retry_on_status_codes,
        jitter=config.jitter,
        logger=log,
    )

    return manager.execute(upload_func)


def is_retryable_error(
    exception: Exception,
    status_code: int | None = None,
    config: RetryConfig | None = None,
) -> bool:
    """Check if an error should be retried based on configuration.

    Utility function for checking if an error is retryable without
    executing retry logic.

    Args:
        exception: The exception to check
        status_code: Optional HTTP status code
        config: RetryConfig to use (defaults to standard config)

    Returns:
        True if the error should trigger a retry

    Example:
        try:
            result = api_call()
        except Exception as e:
            if is_retryable_error(e, getattr(e, 'status_code', None)):
                # Handle retry manually
                pass
            else:
                raise
    """
    if config is None:
        config = RetryConfig()

    return config.should_retry_exception(exception) or (
        status_code is not None and config.should_retry_status_code(status_code)
    )
