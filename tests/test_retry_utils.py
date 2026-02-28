"""tests/test_retry_utils.py

Tests for retry utilities with exponential backoff.
"""

from __future__ import annotations

# Import directly from the module
import sys
import time
from pathlib import Path

import pytest
from requests.exceptions import RequestException  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retry_utils import (
    CircuitBreaker,
    CircuitBreakerOpen,
    RetryConfig,
    RetryManager,
    retry_with_backoff,
)


class TestRetryConfig:
    """Test RetryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_get_delay_exponential(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, jitter=False)

        assert config.get_delay(0) == 1.0  # 1 * 2^0 = 1
        assert config.get_delay(1) == 2.0  # 1 * 2^1 = 2
        assert config.get_delay(2) == 4.0  # 1 * 2^2 = 4
        assert config.get_delay(3) == 8.0  # 1 * 2^3 = 8

    def test_get_delay_capped(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(
            initial_delay=1.0, max_delay=10.0, exponential_base=2.0, jitter=False
        )

        # 1 * 2^10 = 1024, but should be capped at 10
        delay = config.get_delay(10)
        assert delay == 10.0

    def test_get_delay_with_jitter(self):
        """Test that jitter adds randomness."""
        config = RetryConfig(initial_delay=10.0, jitter=True)

        # Run multiple times to check for variation
        delays = [config.get_delay(0) for _ in range(10)]

        # All delays should be in range [5, 15] (10 * 0.5 to 10 * 1.5)
        for delay in delays:
            assert 5.0 <= delay <= 15.0


class TestRetryWithBackoff:
    """Test retry_with_backoff decorator."""

    def test_success_no_retry(self):
        """Test successful function doesn't retry."""
        call_count = [0]

        @retry_with_backoff(max_retries=3)
        def decorated():
            call_count[0] += 1
            return "success"

        result = decorated()

        assert result == "success"
        assert call_count[0] == 1

    def test_retry_on_failure(self):
        """Test retry on transient failure."""
        call_count = [0]

        @retry_with_backoff(max_retries=3, initial_delay=0.001, jitter=False)
        def decorated():
            call_count[0] += 1
            if call_count[0] < 3:
                raise RequestException("fail")
            return "success"

        result = decorated()

        assert result == "success"
        assert call_count[0] == 3

    def test_exhaust_retries(self):
        """Test function that fails after all retries."""
        call_count = [0]

        @retry_with_backoff(max_retries=2, initial_delay=0.001, jitter=False)
        def decorated():
            call_count[0] += 1
            raise RequestException("always fails")

        with pytest.raises(RequestException):
            decorated()

        assert call_count[0] == 3  # Initial + 2 retries

    def test_non_retryable_exception(self):
        """Test non-retryable exception raises immediately."""
        call_count = [0]

        @retry_with_backoff(max_retries=3)
        def decorated():
            call_count[0] += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            decorated()

        assert call_count[0] == 1

    def test_custom_retryable_exceptions(self):
        """Test custom retryable exceptions."""
        call_count = [0]

        @retry_with_backoff(
            max_retries=3,
            initial_delay=0.001,
            retryable_exceptions=(ValueError,),
        )
        def decorated():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("retry this")
            return "success"

        result = decorated()

        assert result == "success"
        assert call_count[0] == 2


class TestRetryManager:
    """Test RetryManager class."""

    def test_execute_success(self):
        """Test successful execution."""
        manager = RetryManager()
        call_count = [0]

        def func():
            call_count[0] += 1
            return "result"

        result = manager.execute_with_retry(func)

        assert result == "result"
        assert call_count[0] == 1

    def test_execute_with_retry(self):
        """Test execution with retries."""
        manager = RetryManager(max_retries=3, initial_delay=0.001, jitter=False)
        call_count = [0]

        def func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise RequestException("fail")
            return "success"

        result = manager.execute_with_retry(func)

        assert result == "success"
        assert call_count[0] == 3

    def test_execute_exhaust_retries(self):
        """Test execution that exhausts retries."""
        manager = RetryManager(max_retries=2, initial_delay=0.001, jitter=False)
        call_count = [0]

        def func():
            call_count[0] += 1
            raise RequestException("always fails")

        with pytest.raises(RequestException):
            manager.execute_with_retry(func)

        assert call_count[0] == 3  # Initial + 2 retries

    def test_get_retry_report_no_attempts(self):
        """Test report with no attempts."""
        manager = RetryManager()
        report = manager.get_retry_report()

        assert report["status"] == "no_attempts"


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    def test_initial_state_closed(self):
        """Test circuit breaker starts closed."""
        breaker = CircuitBreaker()
        assert breaker.state == "closed"
        assert breaker.failures == 0

    def test_call_success(self):
        """Test successful call keeps circuit closed."""
        breaker = CircuitBreaker()

        def func():
            return "success"

        result = breaker.call(func)

        assert result == "success"
        assert breaker.state == "closed"
        assert breaker.failures == 0

    def test_call_failure_increments_count(self):
        """Test failed call increments failure count."""
        # Test circuit breaker with generic failure
        breaker = CircuitBreaker(failure_threshold=3)

        def func():
            raise RuntimeError("fail")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                breaker.call(func)

        assert breaker.failures == 2
        assert breaker.state == "closed"

    def test_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        breaker = CircuitBreaker(failure_threshold=2)

        def func():
            raise RuntimeError("fail")

        # First failure
        with pytest.raises(RuntimeError):
            breaker.call(func)

        # Second failure should open circuit
        with pytest.raises(RuntimeError):
            breaker.call(func)

        assert breaker.state == "open"

    def test_open_circuit_raises(self):
        """Test calling open circuit raises CircuitBreakerOpen."""
        breaker = CircuitBreaker(failure_threshold=1)

        def func():
            raise RuntimeError("fail")

        # Open the circuit
        with pytest.raises(RuntimeError):
            breaker.call(func)

        # Next call should raise CircuitBreakerOpen
        with pytest.raises(CircuitBreakerOpen):
            breaker.call(func)

    def test_recovery_timeout(self):
        """Test circuit transitions to half-open after timeout."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)

        def func():
            raise RuntimeError("fail")

        # Open the circuit
        with pytest.raises(RuntimeError):
            breaker.call(func)

        assert breaker.state == "open"

        # Wait for recovery timeout
        time.sleep(0.1)

        # Next call should transition to half-open
        def success_func():
            return "success"

        result = breaker.call(success_func)

        assert result == "success"
        assert breaker.state == "closed"
        assert breaker.failures == 0

    def test_manual_reset(self):
        """Test manual reset of circuit breaker."""
        breaker = CircuitBreaker(failure_threshold=1)

        def func():
            raise RuntimeError("fail")

        # Open the circuit
        with pytest.raises(RuntimeError):
            breaker.call(func)

        assert breaker.state == "open"

        # Manual reset
        breaker.reset()

        assert breaker.state == "closed"
        assert breaker.failures == 0

    def test_half_open_success_closes(self):
        """Test successful call in half-open state closes circuit."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)

        def fail_func():
            raise RuntimeError("fail")

        # Open the circuit
        with pytest.raises(RuntimeError):
            breaker.call(fail_func)

        # Wait for recovery
        time.sleep(0.1)

        # Successful call should close circuit
        def success_func():
            return "success"

        breaker.call(success_func)

        assert breaker.state == "closed"

    def test_half_open_fail_reopens(self):
        """Test failed call in half-open state reopens circuit."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)

        def func():
            raise RuntimeError("fail")

        # Open the circuit
        with pytest.raises(RuntimeError):
            breaker.call(func)

        # Wait for recovery
        time.sleep(0.1)

        # Failed call should reopen circuit
        with pytest.raises(RuntimeError):
            breaker.call(func)

        assert breaker.state == "open"
