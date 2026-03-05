"""tests/test_retry_utils.py

Comprehensive tests for retry utilities with exponential backoff.

Based on GOAP-AUTH-PLAN A04 requirements:
- Test retry_with_backoff decorator with various scenarios
- Test RetryManager.execute() with successful execution
- Test RetryManager.execute() with retries and eventual success
- Test RetryManager.execute() with permanent failure
- Test exponential backoff timing
- Test exception filtering
"""

from __future__ import annotations

# Import directly from the module
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retry_utils import (
    RetryableHTTPError,
    RetryConfig,
    RetryManager,
    is_retryable_error,
    retry_with_backoff,
    upload_with_retry,
)


class TestRetryConfig:
    """Test RetryConfig dataclass - Requirement: test_retry_config_defaults."""

    def test_retry_config_defaults(self):
        """Verify sensible default configuration values."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.backoff_coefficient == 2.0
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter is True
        assert ConnectionError in config.retry_on_exceptions
        assert TimeoutError in config.retry_on_exceptions
        assert OSError in config.retry_on_exceptions
        assert 429 in config.retry_on_status_codes
        assert 500 in config.retry_on_status_codes
        assert 502 in config.retry_on_status_codes
        assert 503 in config.retry_on_status_codes
        assert 504 in config.retry_on_status_codes

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = RetryConfig(
            max_retries=5,
            backoff_coefficient=3.0,
            initial_delay=2.0,
            max_delay=120.0,
            retry_on_exceptions=(ValueError,),
            retry_on_status_codes=[500, 503],
            jitter=False,
        )

        assert config.max_retries == 5
        assert config.backoff_coefficient == 3.0
        assert config.initial_delay == 2.0
        assert config.max_delay == 120.0
        assert config.jitter is False
        assert config.retry_on_exceptions == (ValueError,)
        assert config.retry_on_status_codes == [500, 503]

    def test_config_validation(self):
        """Test configuration validation in __post_init__."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            RetryConfig(max_retries=-1)

        with pytest.raises(ValueError, match=r"backoff_coefficient must be >= 1.0"):
            RetryConfig(backoff_coefficient=0.5)

        with pytest.raises(ValueError, match="initial_delay must be non-negative"):
            RetryConfig(initial_delay=-1.0)

        with pytest.raises(ValueError, match="max_delay must be >= initial_delay"):
            RetryConfig(initial_delay=10.0, max_delay=5.0)

    def test_calculate_delay_exponential(self):
        """Test exponential backoff calculation without jitter."""
        config = RetryConfig(
            initial_delay=1.0,
            backoff_coefficient=2.0,
            jitter=False,
        )

        # Verify exponential growth
        assert config.calculate_delay(0) == 1.0  # 1.0 * 2^0 = 1.0
        assert config.calculate_delay(1) == 2.0  # 1.0 * 2^1 = 2.0
        assert config.calculate_delay(2) == 4.0  # 1.0 * 2^2 = 4.0
        assert config.calculate_delay(3) == 8.0  # 1.0 * 2^3 = 8.0

    def test_calculate_delay_capped(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(
            initial_delay=1.0,
            max_delay=10.0,
            backoff_coefficient=2.0,
            jitter=False,
        )

        # 1.0 * 2^10 = 1024, but should be capped at 10
        delay = config.calculate_delay(10)
        assert delay == 10.0

    def test_calculate_delay_with_jitter(self):
        """Test that jitter adds randomness (±25% range)."""
        config = RetryConfig(
            initial_delay=10.0,
            jitter=True,
        )

        # Run multiple times to check for variation
        delays = [config.calculate_delay(0) for _ in range(20)]

        # All delays should be in range [7.5, 12.5] (10 * 0.75 to 10 * 1.25)
        for delay in delays:
            assert 7.5 <= delay <= 12.5

        # Not all delays should be identical (jitter introduces variation)
        assert len(set(delays)) > 1

    def test_should_retry_exception(self):
        """Test exception retry checking."""
        config = RetryConfig()

        assert config.should_retry_exception(ConnectionError()) is True
        assert config.should_retry_exception(TimeoutError()) is True
        assert config.should_retry_exception(OSError()) is True
        assert config.should_retry_exception(ValueError()) is False
        assert config.should_retry_exception(RuntimeError()) is False

    def test_should_retry_status_code(self):
        """Test status code retry checking."""
        config = RetryConfig()

        assert config.should_retry_status_code(429) is True  # Rate limited
        assert config.should_retry_status_code(500) is True  # Server error
        assert config.should_retry_status_code(502) is True  # Bad gateway
        assert config.should_retry_status_code(503) is True  # Service unavailable
        assert config.should_retry_status_code(504) is True  # Gateway timeout
        assert config.should_retry_status_code(200) is False  # OK
        assert config.should_retry_status_code(404) is False  # Not found
        assert config.should_retry_status_code(None) is False


class TestRetryWithBackoff:
    """Test retry_with_backoff decorator - Requirement: test_retry_with_backoff_*."""

    def test_retry_with_backoff_success(self):
        """Test successful function doesn't retry - no retries when successful."""
        call_count = [0]

        @retry_with_backoff(max_retries=3)
        def decorated():
            call_count[0] += 1
            return "success"

        result = decorated()

        assert result == "success"
        assert call_count[0] == 1

    def test_retry_with_backoff_eventual_success(self):
        """Test retry until success - RetryManager.execute() with retries."""
        call_count = [0]

        @retry_with_backoff(
            max_retries=3,
            initial_delay=0.001,
            retry_on_exceptions=(ConnectionError,),
            jitter=False,
        )
        def decorated():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError(f"fail #{call_count[0]}")
            return "success"

        result = decorated()

        assert result == "success"
        assert call_count[0] == 3

    def test_retry_with_backoff_max_retries_exceeded(self):
        """Test fail after max retries - RetryManager.execute() with permanent failure."""
        call_count = [0]

        @retry_with_backoff(
            max_retries=2,
            initial_delay=0.001,
            retry_on_exceptions=(ConnectionError,),
            jitter=False,
        )
        def decorated():
            call_count[0] += 1
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError):
            decorated()

        assert call_count[0] == 3  # Initial + 2 retries

    def test_non_retryable_exception(self):
        """Test non-retryable exception raises immediately."""
        call_count = [0]

        @retry_with_backoff(max_retries=3, retry_on_exceptions=(ConnectionError,))
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
            retry_on_exceptions=(ValueError,),
            jitter=False,
        )
        def decorated():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("retry this")
            return "success"

        result = decorated()

        assert result == "success"
        assert call_count[0] == 2

    def test_retry_on_specific_exceptions(self):
        """Only retry on specified exceptions."""
        call_count = [0]

        @retry_with_backoff(
            max_retries=3,
            initial_delay=0.001,
            retry_on_exceptions=(ConnectionError, TimeoutError),
            jitter=False,
        )
        def decorated():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("retry this")
            elif call_count[0] == 2:
                raise TimeoutError("retry this too")
            elif call_count[0] == 3:
                raise ValueError("don't retry this")
            return "success"

        # ValueError should not be retried and should raise immediately
        with pytest.raises(ValueError):
            decorated()

        assert call_count[0] == 3

    def test_zero_retries(self):
        """Test decorator with max_retries=0 (no retries)."""
        call_count = [0]

        @retry_with_backoff(
            max_retries=0,
            retry_on_exceptions=(ValueError,),
        )
        def decorated():
            call_count[0] += 1
            raise ValueError("fail")

        with pytest.raises(ValueError):
            decorated()

        assert call_count[0] == 1  # Only initial attempt

    def test_function_arguments_preserved(self):
        """Test that function arguments are preserved through retry."""
        call_count = [0]
        received_args = []
        received_kwargs = {}

        @retry_with_backoff(
            max_retries=2,
            initial_delay=0.001,
            retry_on_exceptions=(ConnectionError,),
            jitter=False,
        )
        def decorated(arg1, arg2, kwarg1=None, **kwargs):
            call_count[0] += 1
            received_args.clear()
            received_args.extend([arg1, arg2])
            received_kwargs.clear()
            received_kwargs.update({"kwarg1": kwarg1, **kwargs})
            if call_count[0] < 2:
                raise ConnectionError("fail")
            return "success"

        result = decorated("pos1", "pos2", kwarg1="kw1", extra="extra_val")

        assert result == "success"
        assert received_args == ["pos1", "pos2"]
        assert received_kwargs["kwarg1"] == "kw1"
        assert received_kwargs["extra"] == "extra_val"


class TestRetryManager:
    """Test RetryManager class - Requirement: test_retry_manager_execute."""

    def test_retry_manager_execute_success(self):
        """Test RetryManager.execute() with successful execution."""
        manager = RetryManager()
        call_count = [0]

        def func():
            call_count[0] += 1
            return "result"

        result = manager.execute(func)

        assert result == "result"
        assert call_count[0] == 1
        assert len(manager.attempt_history) == 1
        assert manager.attempt_history[0]["success"] is True

    def test_retry_manager_execute_eventual_success(self):
        """Test RetryManager.execute() with retries and eventual success."""
        manager = RetryManager(
            max_retries=3,
            initial_delay=0.001,
            jitter=False,
        )
        call_count = [0]

        def func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("fail")
            return "success"

        result = manager.execute(func)

        assert result == "success"
        assert call_count[0] == 3
        assert len(manager.attempt_history) == 3
        # First two attempts failed
        assert manager.attempt_history[0]["success"] is False
        assert manager.attempt_history[1]["success"] is False
        # Third attempt succeeded
        assert manager.attempt_history[2]["success"] is True

    def test_retry_manager_execute_permanent_failure(self):
        """Test RetryManager.execute() with permanent failure."""
        manager = RetryManager(
            max_retries=2,
            initial_delay=0.001,
            jitter=False,
        )
        call_count = [0]

        def func():
            call_count[0] += 1
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError):
            manager.execute(func)

        assert call_count[0] == 3  # Initial + 2 retries
        assert len(manager.attempt_history) == 3
        # All attempts failed
        for attempt in manager.attempt_history:
            assert attempt["success"] is False
            assert attempt["error_type"] == "ConnectionError"

    def test_retry_manager_execute_with_config(self):
        """Test RetryManager.execute() with custom configuration."""
        custom_config = RetryConfig(
            max_retries=5,
            initial_delay=0.01,
            max_delay=1.0,
            backoff_coefficient=1.5,
            jitter=False,
            retry_on_exceptions=(ConnectionError,),
        )
        manager = RetryManager()
        call_count = [0]

        def func():
            call_count[0] += 1
            if call_count[0] < 4:
                raise ConnectionError("fail")
            return "success"

        result = manager.execute_with_config(func, custom_config)

        assert result == "success"
        assert call_count[0] == 4
        assert len(manager.attempt_history) == 4

    def test_retry_manager_non_retryable_exception(self):
        """Test RetryManager with non-retryable exception."""
        manager = RetryManager(
            retry_on_exceptions=(ConnectionError,),
        )
        call_count = [0]

        def func():
            call_count[0] += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            manager.execute(func)

        assert call_count[0] == 1
        assert len(manager.attempt_history) == 1
        assert manager.attempt_history[0]["non_retryable"] is True

    def test_get_retry_report_no_attempts(self):
        """Test report with no attempts."""
        manager = RetryManager()
        report = manager.get_retry_report()

        assert report["status"] == "no_attempts"
        assert report["total_executions"] == 0

    def test_get_retry_report_after_success(self):
        """Test retry report generation after successful execution."""
        manager = RetryManager(
            max_retries=3,
            initial_delay=0.001,
            jitter=False,
        )

        def func():
            return "success"

        manager.execute(func)
        report = manager.get_retry_report()

        assert report["status"] == "success"
        assert report["summary"]["total_attempts"] == 1
        assert report["summary"]["successful"] == 1
        assert report["summary"]["failed"] == 0
        assert report["summary"]["success_rate"] == 1.0
        assert report["config"]["max_retries"] == 3

    def test_get_retry_report_after_failure(self):
        """Test retry report generation after failed execution."""
        manager = RetryManager(
            max_retries=2,
            initial_delay=0.001,
            jitter=False,
        )

        def func():
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            manager.execute(func)

        report = manager.get_retry_report()

        assert report["status"] == "partial_failure"
        assert report["summary"]["total_attempts"] == 3
        assert report["summary"]["successful"] == 0
        assert report["summary"]["failed"] == 3
        assert report["summary"]["success_rate"] == 0.0
        assert len(report["history"]) == 3

    def test_reset_history(self):
        """Test clearing attempt history."""
        manager = RetryManager()

        def func():
            return "success"

        manager.execute(func)
        assert len(manager.attempt_history) == 1

        manager.reset_history()
        assert len(manager.attempt_history) == 0

    def test_execute_with_args_and_kwargs(self):
        """Test execute with positional and keyword arguments."""
        manager = RetryManager()
        received_args = []
        received_kwargs = {}

        def func(arg1, arg2, kwarg1=None, **kwargs):
            received_args.extend([arg1, arg2])
            received_kwargs.update({"kwarg1": kwarg1, **kwargs})
            return "result"

        result = manager.execute(
            func,
            "pos1",
            "pos2",
            kwarg1="kw1",
            extra="extra_val",
        )

        assert result == "result"
        assert received_args == ["pos1", "pos2"]
        assert received_kwargs["kwarg1"] == "kw1"
        assert received_kwargs["extra"] == "extra_val"


class TestExponentialBackoffTiming:
    """Test exponential backoff timing - Requirement: test_exponential_backoff_timing."""

    def test_exponential_backoff_timing(self):
        """Verify delay increases exponentially with each retry attempt."""
        config = RetryConfig(
            initial_delay=0.1,
            backoff_coefficient=2.0,
            max_delay=10.0,
            jitter=False,
        )

        # Test delay progression
        delays = [config.calculate_delay(i) for i in range(5)]

        # Verify exponential growth: 0.1, 0.2, 0.4, 0.8, 1.6
        assert delays[0] == 0.1  # 0.1 * 2^0
        assert delays[1] == 0.2  # 0.1 * 2^1
        assert delays[2] == 0.4  # 0.1 * 2^2
        assert delays[3] == 0.8  # 0.1 * 2^3
        assert delays[4] == 1.6  # 0.1 * 2^4

    def test_backoff_with_time_mock(self, monkeypatch):
        """Test backoff timing with mocked time.sleep."""
        sleep_times = []

        def mock_sleep(seconds):
            sleep_times.append(seconds)

        monkeypatch.setattr(time, "sleep", mock_sleep)

        @retry_with_backoff(
            max_retries=3,
            initial_delay=0.1,
            backoff_coefficient=2.0,
            retry_on_exceptions=(ConnectionError,),
            jitter=False,
        )
        def always_fails():
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            always_fails()

        # Should have 3 sleep calls (between attempts)
        assert len(sleep_times) == 3
        # Verify exponential progression
        assert sleep_times[0] == 0.1  # After 1st failure
        assert sleep_times[1] == 0.2  # After 2nd failure
        assert sleep_times[2] == 0.4  # After 3rd failure


class TestRetryOnStatusCodes:
    """Test HTTP status code filtering - Requirement: test_retry_on_status_codes."""

    def test_retryable_http_error(self):
        """Test RetryableHTTPError with status codes."""
        error = RetryableHTTPError(503, "Service Unavailable")
        assert error.status_code == 503
        assert error.message == "Service Unavailable"
        assert "503" in str(error)

    def test_is_retryable_error_with_status_code(self):
        """Test is_retryable_error with status codes."""
        config = RetryConfig()

        # Test with status codes that should trigger retry
        assert is_retryable_error(Exception(), 429, config) is True
        assert is_retryable_error(Exception(), 500, config) is True
        assert is_retryable_error(Exception(), 502, config) is True
        assert is_retryable_error(Exception(), 503, config) is True
        assert is_retryable_error(Exception(), 504, config) is True

        # Test with status codes that should not trigger retry
        assert is_retryable_error(Exception(), 200, config) is False
        assert is_retryable_error(Exception(), 404, config) is False
        assert is_retryable_error(Exception(), 401, config) is False

    def test_is_retryable_error_with_exception(self):
        """Test is_retryable_error with exceptions."""
        config = RetryConfig()

        assert is_retryable_error(ConnectionError(), None, config) is True
        assert is_retryable_error(TimeoutError(), None, config) is True
        assert is_retryable_error(ValueError(), None, config) is False


class TestUploadWithRetry:
    """Test upload_with_retry function."""

    def test_upload_with_retry_success(self):
        """Test upload_with_retry with successful upload."""
        call_count = [0]

        def upload_func():
            call_count[0] += 1
            return "uploaded"

        result = upload_with_retry(upload_func, max_retries=3, initial_delay=0.001)

        assert result == "uploaded"
        assert call_count[0] == 1

    def test_upload_with_retry_eventual_success(self):
        """Test upload_with_retry with eventual success."""
        call_count = [0]

        def upload_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("connection failed")
            return "uploaded"

        result = upload_with_retry(upload_func, max_retries=5, initial_delay=0.001)

        assert result == "uploaded"
        assert call_count[0] == 3

    def test_upload_with_retry_permanent_failure(self):
        """Test upload_with_retry with permanent failure."""
        call_count = [0]

        def upload_func():
            call_count[0] += 1
            raise TimeoutError("timeout")

        with pytest.raises(TimeoutError):
            upload_with_retry(upload_func, max_retries=2, initial_delay=0.001)

        assert call_count[0] == 3  # Initial + 2 retries

    def test_upload_with_retry_custom_logger(self):
        """Test upload_with_retry with custom logger."""
        import logging

        custom_logger = logging.getLogger("test_logger")
        call_count = [0]

        def upload_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("fail")
            return "uploaded"

        result = upload_with_retry(
            upload_func,
            max_retries=3,
            initial_delay=0.001,
            logger=custom_logger,
        )

        assert result == "uploaded"
