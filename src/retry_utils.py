"""src/retry_utils.py

Retry utilities with exponential backoff for transient failures.

This module provides:
- Retry decorator with configurable backoff
- Retry manager for complex retry scenarios
- Specialized retry logic for HuggingFace uploads
- Structured logging for retry attempts

Usage:
    from retry_utils import retry_with_backoff, RetryConfig, RetryManager

    # Simple retry decorator
    @retry_with_backoff(max_retries=3, backoff=2.0)
    def upload_file():
        ...

    # Retry manager for complex scenarios
    manager = RetryManager(max_retries=5)
    result = manager.execute_with_retry(upload_function)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

from requests.exceptions import RequestException  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = (
        RequestException,
        ConnectionError,
        TimeoutError,
    )

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = self.initial_delay * (self.exponential_base**attempt)

        # Cap at max delay
        delay = min(delay, self.max_delay)

        # Add jitter to prevent thundering herd
        if self.jitter:
            import random

            delay = delay * (0.5 + random.random())

        return delay


T = TypeVar("T")


def retry_with_backoff(
    config: RetryConfig | None = None,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
    logger: logging.Logger | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying functions with exponential backoff.

    Args:
        config: RetryConfig object (overrides other parameters if provided)
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
        retryable_exceptions: Tuple of exception types to retry on
        logger: Logger for retry events

    Returns:
        Decorated function with retry logic

    Example:
        @retry_with_backoff(max_retries=3, backoff=2.0)
        def upload_to_hf():
            api.upload_file(...)
    """
    if config is None:
        config = RetryConfig(
            max_retries=max_retries,
            initial_delay=initial_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
            retryable_exceptions=retryable_exceptions
            or RetryConfig().retryable_exceptions,
        )

    log = logger or logging.getLogger(__name__)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(config.max_retries + 1):
                try:
                    if attempt > 0:
                        delay = config.get_delay(attempt - 1)
                        log.info(
                            f"Retry attempt {attempt}/{config.max_retries} for {func.__name__}, "
                            f"waiting {delay:.2f}s"
                        )
                        time.sleep(delay)

                    return func(*args, **kwargs)

                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt < config.max_retries:
                        log.warning(
                            f"Attempt {attempt + 1}/{config.max_retries + 1} failed for "
                            f"{func.__name__}: {type(e).__name__}: {e}"
                        )
                    else:
                        log.error(
                            f"All {config.max_retries + 1} attempts failed for "
                            f"{func.__name__}: {type(e).__name__}: {e}"
                        )

                except Exception as e:
                    # Non-retryable exception
                    log.error(
                        f"Non-retryable error in {func.__name__}: {type(e).__name__}: {e}"
                    )
                    raise

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Retry loop exited unexpectedly for {func.__name__}")

        return wrapper

    return decorator


class RetryManager:
    """Manage retry logic for complex scenarios."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        logger: logging.Logger | None = None,
    ):
        self.config = RetryConfig(
            max_retries=max_retries,
            initial_delay=initial_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
        )
        self.log = logger or logging.getLogger(__name__)
        self.attempt_history: list[dict[str, Any]] = []

    def execute_with_retry(
        self,
        func: Callable[..., T],
        *args: Any,
        retryable_exceptions: tuple[type[Exception], ...] | None = None,
        on_retry: Callable[[int, Exception, float], None] | None = None,
        **kwargs: Any,
    ) -> T:
        """Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            retryable_exceptions: Override default retryable exceptions
            on_retry: Callback(attempt, exception, delay) called before each retry
            **kwargs: Keyword arguments for func

        Returns:
            Result from successful function execution

        Raises:
            Exception: The last exception if all retries fail
        """
        exceptions = retryable_exceptions or self.config.retryable_exceptions
        last_exception: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            attempt_info = {
                "function": func.__name__,
                "attempt": attempt + 1,
                "max_attempts": self.config.max_retries + 1,
                "timestamp": time.time(),
            }

            try:
                if attempt > 0:
                    delay = self.config.get_delay(attempt - 1)
                    self.log.info(
                        f"Retry attempt {attempt + 1}/{self.config.max_retries + 1} "
                        f"for {func.__name__}, waiting {delay:.2f}s"
                    )

                    if on_retry:
                        on_retry(attempt, last_exception, delay)  # type: ignore[arg-type]

                    time.sleep(delay)

                result = func(*args, **kwargs)

                attempt_info["success"] = True
                attempt_info["duration"] = time.time() - attempt_info["timestamp"]  # type: ignore[operator]
                self.attempt_history.append(attempt_info)

                return result

            except exceptions as e:
                last_exception = e
                attempt_info["success"] = False
                attempt_info["error"] = str(e)
                attempt_info["error_type"] = type(e).__name__
                self.attempt_history.append(attempt_info)

                if attempt < self.config.max_retries:
                    self.log.warning(
                        f"Attempt {attempt + 1}/{self.config.max_retries + 1} failed for "
                        f"{func.__name__}: {type(e).__name__}: {e}"
                    )
                else:
                    self.log.error(
                        f"All {self.config.max_retries + 1} attempts failed for "
                        f"{func.__name__}: {type(e).__name__}: {e}"
                    )

            except Exception as e:
                # Non-retryable exception
                attempt_info["success"] = False
                attempt_info["error"] = str(e)
                attempt_info["error_type"] = type(e).__name__
                attempt_info["non_retryable"] = True
                self.attempt_history.append(attempt_info)

                self.log.error(
                    f"Non-retryable error in {func.__name__}: {type(e).__name__}: {e}"
                )
                raise

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError(f"Retry loop exited unexpectedly for {func.__name__}")

    def get_retry_report(self) -> dict[str, Any]:
        """Generate retry report.

        Returns:
            Dictionary with retry statistics
        """
        if not self.attempt_history:
            return {"status": "no_attempts"}

        successful = [a for a in self.attempt_history if a.get("success")]
        failed = [a for a in self.attempt_history if not a.get("success")]

        return {
            "status": "success" if successful else "failed",
            "total_attempts": len(self.attempt_history),
            "successful_attempts": len(successful),
            "failed_attempts": len(failed),
            "max_retries": self.config.max_retries,
            "history": self.attempt_history,
        }


def upload_with_retry(
    upload_func: Callable[..., Any],
    max_retries: int = 3,
    initial_delay: float = 2.0,
    logger: logging.Logger | None = None,
) -> Any:
    """Specialized retry function for HuggingFace uploads.

    Handles common HuggingFace Hub API errors with appropriate retry logic.

    Args:
        upload_func: Upload function to retry
        max_retries: Maximum retry attempts
        initial_delay: Initial delay in seconds
        logger: Logger for retry events

    Returns:
        Result from successful upload

    Raises:
        Exception: Last exception if all retries fail
    """
    log = logger or logging.getLogger(__name__)

    # HuggingFace-specific retryable errors
    hf_retryable = (
        RequestException,
        ConnectionError,
        TimeoutError,
    )

    config = RetryConfig(
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=120.0,  # Longer max delay for uploads
        exponential_base=2.0,
        jitter=True,
        retryable_exceptions=hf_retryable,
    )

    manager = RetryManager(
        max_retries=config.max_retries,
        initial_delay=config.initial_delay,
        max_delay=config.max_delay,
        exponential_base=config.exponential_base,
        jitter=config.jitter,
        logger=log,
    )

    return manager.execute_with_retry(upload_func)


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascade failures.

    When too many failures occur in a short time, the circuit "opens"
    and prevents further attempts for a cooldown period.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        logger: logging.Logger | None = None,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.log = logger or logging.getLogger(__name__)

        self.failures = 0
        self.last_failure_time: float | None = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpen: If circuit is open
            Exception: Function exception if it fails
        """
        if self.state == "open":
            if (
                self.last_failure_time is not None
                and time.time() - self.last_failure_time > self.recovery_timeout
            ):
                self.log.info("Circuit breaker transitioning to half-open")
                self.state = "half-open"
            else:
                raise CircuitBreakerOpen(
                    f"Circuit breaker open, retry after {self.recovery_timeout}s"
                )

        try:
            result = func(*args, **kwargs)

            if self.state == "half-open":
                self.log.info("Circuit breaker closed after successful call")
                self.state = "closed"
                self.failures = 0

            return result

        except Exception:
            self.failures += 1
            self.last_failure_time = time.time()

            if self.failures >= self.failure_threshold:
                self.log.error(f"Circuit breaker opened after {self.failures} failures")
                self.state = "open"

            raise

    def reset(self) -> None:
        """Manually reset circuit breaker."""
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"
        self.log.info("Circuit breaker manually reset")


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""

    pass
