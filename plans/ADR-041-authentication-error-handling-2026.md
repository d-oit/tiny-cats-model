# ADR-041: Authentication Error Handling and Token Validation

**Date:** 2026-02-28
**Status:** Proposed
**Authors:** AI Agent (GOAP System)
**Related:** ADR-039 (Automated HuggingFace CI Upload), ADR-026 (HuggingFace Publishing), ADR-023 (Modal GPU Retry Strategy)

## Context

### Current State

The tiny-cats-model project has:
- **HuggingFace upload workflow:** `.github/workflows/upload-hub.yml` (created, ADR-039)
- **Upload script:** `src/upload_to_huggingface.py` (660 lines)
- **HuggingFace repo:** d4oit/tiny-cats-model (models uploaded manually)
- **Modal training:** `src/train_dit.py` with GPU support
- **GitHub Secrets:** HF_TOKEN configured but not validated

### Problem Statement

**Primary Blocking Issues:**

1. **HF_TOKEN not validated in GitHub Secrets**
   - Upload workflow references `${{ secrets.HF_TOKEN }}`
   - No validation that secret exists or is valid
   - Workflow fails silently or with cryptic 401 errors
   - No pre-flight checks before upload attempts

2. **Modal tokens need verification**
   - Modal authentication configured globally via `modal token set`
   - No programmatic validation in training scripts
   - Training fails mid-execution if token expired
   - No clear error messages for authentication failures

3. **Missing error handling in training scripts**
   - `src/train_dit.py` assumes Modal authentication works
   - No try/catch for authentication-related exceptions
   - No graceful degradation or retry logic
   - Logs don't capture authentication state

4. **Missing token validation utilities**
   - No centralized auth validation module
   - Each script implements ad-hoc token checks
   - Inconsistent error messages across scripts
   - No token expiry detection

5. **Missing retry logic for HuggingFace uploads**
   - Upload fails on transient network errors
   - No exponential backoff
   - No partial upload recovery
   - No upload progress tracking

6. **Missing comprehensive logging for authentication flows**
   - Authentication state not logged
   - Token validation results not captured
   - No audit trail for auth failures
   - Debugging requires manual log inspection

### Root Cause Analysis

| Issue | Root Cause | Impact |
|-------|------------|--------|
| HF_TOKEN not validated | No pre-flight check in workflow | Workflow fails after 30min runtime |
| Modal token expiry | No validation before training | Training fails mid-execution |
| Missing error handling | Assumption of valid auth | Poor user experience, unclear errors |
| No validation utilities | Ad-hoc implementation | Code duplication, inconsistency |
| No retry logic | Single-attempt uploads | Fragile upload process |
| Missing auth logging | Focus on functional logging | Hard to debug auth issues |

### Requirements

**2026 Best Practices for Authentication:**

1. **Pre-flight validation** - Check tokens before long-running operations
2. **Clear error messages** - Actionable feedback for auth failures
3. **Retry with backoff** - Handle transient failures gracefully
4. **Comprehensive logging** - Audit trail for debugging
5. **Token expiry detection** - Warn before tokens expire
6. **Centralized utilities** - Single source of truth for auth logic
7. **Secret rotation support** - Easy token updates without code changes

## Decision

We will implement **comprehensive authentication error handling** with validation utilities, retry logic, and enhanced logging.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Authentication Layer                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │  Token Validator │    │  Retry Manager   │              │
│  │  - HF_TOKEN      │    │  - Exponential   │              │
│  │  - MODAL_TOKEN   │    │    Backoff       │              │
│  │  - Expiry Check  │    │  - Max Attempts  │              │
│  └────────┬─────────┘    │  - Jitter        │              │
│           │              └────────┬─────────┘              │
│           ▼                       ▼                        │
│  ┌──────────────────────────────────────────┐              │
│  │         Auth Logger (Structured)          │              │
│  │  - Token status (masked)                  │              │
│  │  - Validation results                     │              │
│  │  - Retry attempts                         │              │
│  │  - Failure reasons                        │              │
│  └──────────────────────────────────────────┘              │
│           │                       │                        │
│           ▼                       ▼                        │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │  Training Scripts│    │  Upload Scripts  │              │
│  │  (train_dit.py)  │    │  (upload_*.py)   │              │
│  └──────────────────┘    └──────────────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Components

#### 1. Token Validation Module (`src/auth_utils.py`)

```python
"""src/auth_utils.py

Centralized authentication utilities for token validation and management.

Features:
- HF_TOKEN validation with API call
- MODAL_TOKEN validation via Modal CLI
- Token expiry detection
- Structured logging for auth events
- Environment variable validation
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, HfFolder


class TokenStatus(Enum):
    """Token validation status."""
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    MISSING = "missing"
    UNKNOWN = "unknown"


@dataclass
class TokenValidationResult:
    """Result of token validation."""
    token_type: str  # "HF_TOKEN", "MODAL_TOKEN"
    status: TokenStatus
    message: str
    expires_at: Optional[datetime] = None
    validated_at: datetime = None

    def __post_init__(self):
        if self.validated_at is None:
            self.validated_at = datetime.utcnow()


class AuthValidator:
    """Centralized authentication validator."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def validate_hf_token(self, token: Optional[str] = None) -> TokenValidationResult:
        """Validate HuggingFace token.

        Args:
            token: HF token (uses HF_TOKEN env var if not provided)

        Returns:
            TokenValidationResult with status and details
        """
        token = token or os.environ.get("HF_TOKEN")

        if not token:
            self.logger.warning("HF_TOKEN not set in environment")
            return TokenValidationResult(
                token_type="HF_TOKEN",
                status=TokenStatus.MISSING,
                message="HF_TOKEN environment variable not set"
            )

        # Validate token format
        if not token.startswith("hf_"):
            self.logger.warning("HF_TOKEN has invalid format (should start with 'hf_')")
            return TokenValidationResult(
                token_type="HF_TOKEN",
                status=TokenStatus.INVALID,
                message="HF_TOKEN has invalid format"
            )

        # Validate token with HuggingFace API
        try:
            api = HfApi()
            user_info = api.whoami(token=token)

            self.logger.info(
                f"HF_TOKEN validated for user: {user_info.get('name', 'unknown')}"
            )

            return TokenValidationResult(
                token_type="HF_TOKEN",
                status=TokenStatus.VALID,
                message=f"Valid token for user {user_info.get('name', 'unknown')}",
                # Note: HF doesn't expose expiry via API
            )

        except Exception as e:
            error_msg = str(e)

            if "Invalid credentials" in error_msg:
                self.logger.error("HF_TOKEN is invalid or expired")
                return TokenValidationResult(
                    token_type="HF_TOKEN",
                    status=TokenStatus.INVALID,
                    message="HF_TOKEN is invalid or expired"
                )

            self.logger.error(f"HF_TOKEN validation failed: {error_msg}")
            return TokenValidationResult(
                token_type="HF_TOKEN",
                status=TokenStatus.UNKNOWN,
                message=f"Validation error: {error_msg}"
            )

    def validate_modal_token(self) -> TokenValidationResult:
        """Validate Modal token via CLI.

        Returns:
            TokenValidationResult with status and details
        """
        # Check if Modal CLI is installed
        modal_path = shutil.which("modal")
        if not modal_path:
            self.logger.warning("Modal CLI not found in PATH")
            return TokenValidationResult(
                token_type="MODAL_TOKEN",
                status=TokenStatus.UNKNOWN,
                message="Modal CLI not installed"
            )

        # Run modal token status
        try:
            result = subprocess.run(
                ["modal", "token", "status"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                self.logger.info("MODAL_TOKEN validated via CLI")
                return TokenValidationResult(
                    token_type="MODAL_TOKEN",
                    status=TokenStatus.VALID,
                    message="Modal token is valid"
                )
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                self.logger.error(f"Modal token validation failed: {error_msg}")
                return TokenValidationResult(
                    token_type="MODAL_TOKEN",
                    status=TokenStatus.INVALID,
                    message=f"Modal token invalid: {error_msg}"
                )

        except subprocess.TimeoutExpired:
            self.logger.error("Modal token validation timed out")
            return TokenValidationResult(
                token_type="MODAL_TOKEN",
                status=TokenStatus.UNKNOWN,
                message="Validation timeout"
            )
        except Exception as e:
            self.logger.error(f"Modal token validation error: {e}")
            return TokenValidationResult(
                token_type="MODAL_TOKEN",
                status=TokenStatus.UNKNOWN,
                message=f"Validation error: {e}"
            )

    def validate_all_tokens(self) -> dict[str, TokenValidationResult]:
        """Validate all required tokens.

        Returns:
            Dict mapping token type to validation result
        """
        results = {
            "HF_TOKEN": self.validate_hf_token(),
            "MODAL_TOKEN": self.validate_modal_token(),
        }

        # Log summary
        valid_count = sum(1 for r in results.values() if r.status == TokenStatus.VALID)
        total_count = len(results)

        self.logger.info(f"Token validation: {valid_count}/{total_count} valid")

        return results


def setup_auth_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Setup structured logging for authentication events.

    Args:
        log_file: Optional log file path
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("auth")
    logger.setLevel(level)
    logger.handlers.clear()

    # Structured formatter for auth events
    formatter = logging.Formatter(
        "%(asctime)s | AUTH | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
```

#### 2. Retry Manager (`src/retry_utils.py`)

```python
"""src/retry_utils.py

Retry utilities with exponential backoff for network operations.

Features:
- Exponential backoff with jitter
- Max attempts configuration
- Retry on specific exceptions
- Progress logging
"""

from __future__ import annotations

import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type


class RetryConfig:
    """Retry configuration."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (Exception,)


def retry_with_backoff(config: Optional[RetryConfig] = None):
    """Decorator for retry with exponential backoff.

    Args:
        config: RetryConfig instance (uses defaults if not provided)

    Returns:
        Decorated function
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        logger = logging.getLogger(f"retry.{func.__name__}")

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(1, config.max_attempts + 1):
                try:
                    logger.info(f"Attempt {attempt}/{config.max_attempts}")
                    return func(*args, **kwargs)

                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt == config.max_attempts:
                        logger.error(f"All {config.max_attempts} attempts failed")
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.exponential_base ** (attempt - 1)),
                        config.max_delay
                    )

                    # Add jitter
                    if config.jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"Attempt {attempt} failed: {e}. Retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)

                except Exception as e:
                    # Non-retryable exception
                    logger.error(f"Non-retryable error: {e}")
                    raise

            # Should not reach here, but just in case
            raise last_exception

        return wrapper
    return decorator


class RetryManager:
    """Programmatic retry manager for complex operations."""

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.logger = logging.getLogger("retry_manager")

    def execute(
        self,
        operation: Callable,
        *args,
        on_retry: Optional[Callable[[int, Exception, float], None]] = None,
        **kwargs
    ) -> Any:
        """Execute operation with retry logic.

        Args:
            operation: Function to execute
            *args: Positional arguments for operation
            on_retry: Optional callback(attempt, exception, delay)
            **kwargs: Keyword arguments for operation

        Returns:
            Result of operation
        """
        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                self.logger.info(f"Executing operation (attempt {attempt})")
                return operation(*args, **kwargs)

            except self.config.retryable_exceptions as e:
                last_exception = e

                if attempt == self.config.max_attempts:
                    self.logger.error(f"All attempts failed: {e}")
                    raise

                # Calculate delay
                delay = min(
                    self.config.base_delay * (self.config.exponential_base ** (attempt - 1)),
                    self.config.max_delay
                )

                if self.config.jitter:
                    delay = delay * (0.5 + random.random())

                # Callback
                if on_retry:
                    on_retry(attempt, e, delay)

                self.logger.warning(
                    f"Retry {attempt}/{self.config.max_attempts} after {delay:.1f}s"
                )
                time.sleep(delay)

            except Exception as e:
                self.logger.error(f"Non-retryable error: {e}")
                raise

        raise last_exception
```

#### 3. Enhanced Upload Script Integration

Update `src/upload_to_huggingface.py` to use auth utilities:

```python
# Add at imports
from auth_utils import AuthValidator, TokenStatus, setup_auth_logging
from retry_utils import RetryConfig, retry_with_backoff, RetryManager

# Add pre-flight validation
def upload_to_huggingface(...):
    # Setup auth logging
    logger = setup_auth_logging(log_file="logs/auth.log")

    # Pre-flight token validation
    validator = AuthValidator(logger)
    validation = validator.validate_hf_token(token)

    if validation.status != TokenStatus.VALID:
        logger.error(f"Upload aborted: {validation.message}")
        raise ValueError(f"HF_TOKEN validation failed: {validation.message}")

    logger.info("Pre-flight validation passed, starting upload")

    # Use retry for upload operations
    config = RetryConfig(
        max_attempts=3,
        base_delay=2.0,
        max_delay=30.0,
        retryable_exceptions=(ConnectionError, TimeoutError)
    )

    retry_manager = RetryManager(config)

    def upload_operation():
        # ... existing upload logic ...
        pass

    retry_manager.execute(upload_operation)
```

#### 4. Enhanced Training Script Integration

Update `src/train_dit.py` to validate Modal token:

```python
# Add pre-flight validation in main()
def main():
    logger = setup_logging()

    # Validate Modal token before training
    validator = AuthValidator(logger)
    modal_validation = validator.validate_modal_token()

    if modal_validation.status != TokenStatus.VALID:
        logger.error(f"Training aborted: {modal_validation.message}")
        logger.error("Run 'modal token set' to configure authentication")
        sys.exit(1)

    logger.info("Modal authentication validated, starting training")

    # ... rest of training logic ...
```

#### 5. GitHub Actions Pre-flight Check

Add pre-flight validation to `.github/workflows/upload-hub.yml`:

```yaml
- name: Validate HF_TOKEN
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
  run: |
    python -c "
    import os
    from huggingface_hub import HfApi

    token = os.environ.get('HF_TOKEN')
    if not token:
        print('❌ HF_TOKEN not set')
        exit(1)

    if not token.startswith('hf_'):
        print('❌ HF_TOKEN has invalid format')
        exit(1)

    try:
        api = HfApi()
        user = api.whoami(token=token)
        print(f'✅ HF_TOKEN valid for user: {user[\"name\"]}')
    except Exception as e:
        print(f'❌ HF_TOKEN validation failed: {e}')
        exit(1)
    "
```

### Testing Strategy

#### Unit Tests

```python
# tests/test_auth_utils.py

import pytest
from src.auth_utils import AuthValidator, TokenStatus, TokenValidationResult


class TestAuthValidator:
    """Test authentication validator."""

    def test_validate_hf_token_missing(self, monkeypatch):
        """Test validation when token is missing."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        validator = AuthValidator()
        result = validator.validate_hf_token()

        assert result.status == TokenStatus.MISSING
        assert "not set" in result.message

    def test_validate_hf_token_invalid_format(self, monkeypatch):
        """Test validation with invalid token format."""
        monkeypatch.setenv("HF_TOKEN", "invalid_token")
        validator = AuthValidator()
        result = validator.validate_hf_token()

        assert result.status == TokenStatus.INVALID
        assert "format" in result.message

    def test_validate_hf_token_valid(self, monkeypatch, mock_hf_api):
        """Test validation with valid token."""
        monkeypatch.setenv("HF_TOKEN", "hf_valid_token")
        validator = AuthValidator()
        result = validator.validate_hf_token()

        assert result.status == TokenStatus.VALID
```

#### Integration Tests

```python
# tests/test_upload_retry.py

import pytest
from src.retry_utils import RetryConfig, RetryManager


def test_upload_retry_on_connection_error():
    """Test retry logic on connection errors."""
    config = RetryConfig(
        max_attempts=3,
        base_delay=0.01,  # Fast for tests
        retryable_exceptions=(ConnectionError,)
    )
    manager = RetryManager(config)

    attempt_count = 0

    def failing_operation():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError("Network error")
        return "success"

    result = manager.execute(failing_operation)
    assert result == "success"
    assert attempt_count == 3
```

#### E2E Tests

```bash
# Test auth validation end-to-end
python -m pytest tests/test_auth_e2e.py -v

# Test upload with retry
python -m pytest tests/test_upload_e2e.py -v

# Test training with auth validation
modal run src/train_dit.py --help  # Should validate token
```

## Implementation Plan

### Phase 1: Auth Utilities (P0 - 2 hours)
- [ ] Create `src/auth_utils.py` with AuthValidator
- [ ] Create `src/retry_utils.py` with RetryManager
- [ ] Add unit tests for auth utilities
- [ ] Add integration tests for retry logic

### Phase 2: Upload Script Enhancement (P1 - 2 hours)
- [ ] Update `src/upload_to_huggingface.py` with auth validation
- [ ] Add retry logic for upload operations
- [ ] Add structured logging for auth events
- [ ] Test upload with invalid token (should fail fast)
- [ ] Test upload with network error (should retry)

### Phase 3: Training Script Enhancement (P1 - 1 hour)
- [ ] Update `src/train_dit.py` with Modal token validation
- [ ] Add pre-flight auth check
- [ ] Add structured logging
- [ ] Test training with invalid Modal token

### Phase 4: CI/CD Enhancement (P1 - 1 hour)
- [ ] Add HF_TOKEN pre-flight check to upload-hub.yml
- [ ] Add auth validation logging to CI
- [ ] Test workflow with missing HF_TOKEN
- [ ] Test workflow with invalid HF_TOKEN

### Phase 5: Documentation (P2 - 1 hour)
- [ ] Update AGENTS.md with auth validation commands
- [ ] Update learnings.md with auth troubleshooting patterns
- [ ] Create auth troubleshooting guide
- [ ] Update ADR-039 with new validation steps

## Consequences

### Positive
- ✅ **Fast failure** - Auth errors detected before long operations
- ✅ **Clear errors** - Actionable messages for token issues
- ✅ **Resilient uploads** - Retry logic handles transient failures
- ✅ **Audit trail** - Structured logging for debugging
- ✅ **Centralized logic** - Single source of truth for auth
- ✅ **Better UX** - Users know exactly what to fix

### Negative
- ⚠️ **Code complexity** - Additional abstraction layer
- ⚠️ **Dependencies** - New modules to maintain
- ⚠️ **Testing overhead** - More test cases needed

### Neutral
- ℹ️ **Token validation** - Adds ~1-2 seconds to operations
- ℹ️ **Retry delays** - Failed uploads take longer (by design)
- ℹ️ **Log volume** - More auth-related log entries

## Alternatives Considered

### Alternative 1: Inline Validation Only
**Proposal:** Add validation directly in each script without utilities.

**Rejected because:**
- Code duplication across scripts
- Inconsistent error messages
- Harder to maintain and update
- No centralized logging

### Alternative 2: External Auth Service
**Proposal:** Use external service (e.g., Vault) for token management.

**Rejected because:**
- Overkill for this project scale
- Adds infrastructure complexity
- GitHub Secrets sufficient for current needs
- Can revisit at larger scale

### Alternative 3: No Retry Logic
**Proposal:** Fail immediately on upload errors, require manual retry.

**Rejected because:**
- Poor user experience for transient failures
- Wastes compute on re-runs
- Industry standard is automatic retry
- Simple to implement with decorator

## Security Considerations

### Token Handling

**Do:**
- ✅ Log token status, never token value
- ✅ Mask tokens in structured logs
- ✅ Use environment variables for token storage
- ✅ Validate token format before API calls

**Don't:**
- ❌ Log full token values
- ❌ Store tokens in code
- ❌ Pass tokens via command line args
- ❌ Commit tokens to version control

### Secret Rotation

**Rotate HF_TOKEN every 90 days:**
1. Generate new token at https://huggingface.co/settings/tokens
2. Update GitHub secret: `gh secret set HF_TOKEN`
3. Update local environment: `export HF_TOKEN=hf_new`
4. Verify with: `python -c "from auth_utils import AuthValidator; AuthValidator().validate_hf_token()"`

## Troubleshooting

### Issue: "HF_TOKEN not set"
**Cause:** Environment variable not configured
**Solution:**
```bash
# Local
export HF_TOKEN=hf_xxx

# GitHub Secrets
gh secret set HF_TOKEN --body "hf_xxx"
```

### Issue: "HF_TOKEN has invalid format"
**Cause:** Token doesn't start with `hf_`
**Solution:**
1. Verify token at https://huggingface.co/settings/tokens
2. Ensure copying full token (not truncated)
3. Regenerate token if needed

### Issue: "Modal token invalid"
**Cause:** Modal CLI not authenticated
**Solution:**
```bash
modal token set
modal token status  # Verify
```

### Issue: Upload fails after retry
**Cause:** Persistent network issue or invalid token
**Solution:**
1. Check auth logs: `cat logs/auth.log`
2. Verify token: `python -c "from auth_utils import AuthValidator; print(AuthValidator().validate_hf_token())"`
3. Check network connectivity
4. Contact HuggingFace support if token valid but upload fails

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Auth validation coverage | 100% | Scripts using auth_utils |
| Pre-flight checks | 100% | Upload/training scripts |
| Retry success rate | >80% | Upload retry logs |
| Auth error clarity | Subjective | User feedback |
| Debug time reduction | >50% | Time to resolve auth issues |

## Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Phase 1 | 2 hours | Auth utilities + tests |
| Phase 2 | 2 hours | Upload script enhancement |
| Phase 3 | 1 hour | Training script enhancement |
| Phase 4 | 1 hour | CI/CD enhancement |
| Phase 5 | 1 hour | Documentation |
| **Total** | **7 hours** | **~1 day** |

## References

- ADR-039: Automated HuggingFace CI Upload
- ADR-026: HuggingFace Model Publishing
- ADR-023: Modal GPU Retry Strategy
- HuggingFace Hub API: https://huggingface.co/docs/huggingface_hub
- Modal CLI: https://modal.com/docs/reference/cli

## Appendix: File Locations

| File | Purpose |
|------|---------|
| `src/auth_utils.py` | Token validation utilities |
| `src/retry_utils.py` | Retry with backoff |
| `src/upload_to_huggingface.py` | Enhanced upload script |
| `src/train_dit.py` | Enhanced training script |
| `.github/workflows/upload-hub.yml` | Enhanced workflow |
| `tests/test_auth_utils.py` | Auth unit tests |
| `tests/test_retry_utils.py` | Retry unit tests |
| `logs/auth.log` | Auth event logs |
