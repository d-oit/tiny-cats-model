# ADR-045: Authentication Utilities Implementation

## Status

**Accepted** - Implemented 2026-03-02

## Context

The project has multiple critical tasks requiring robust authentication handling:

1. **Missing Auth Validation** (T-A03-PHASE20): HF_TOKEN configuration needs validation before use
2. **No Retry Logic** (T-A06-AUTH-PHASE2): Upload operations lack retry mechanisms
3. **Training Failures** (T-A01-PHASE18): Modal training stops due to auth issues
4. **CI/CD Automation** (Phase 20): Automated workflows need pre-flight auth checks

Without proper authentication utilities:
- Users waste time on failed training runs due to missing tokens
- CI/CD workflows fail silently without clear error messages
- Upload operations fail on transient network errors
- No standardized way to validate tokens before use

## Decision

Implement comprehensive authentication utilities with retry logic following the patterns established in ADR-041 and GOAP-AUTH-PLAN.

### Components Implemented

#### 1. src/auth_utils.py (453 lines)

**TokenStatus Enum:**
- `VALID`: Token authenticated successfully
- `INVALID`: Token rejected by service
- `MISSING`: Environment variable not set
- `EXPIRED`: Token past expiration date
- `INSUFFICIENT_PERMISSIONS`: Token lacks required scope
- `UNKNOWN`: Validation error occurred

**TokenValidationResult Dataclass:**
```python
@dataclass
class TokenValidationResult:
    status: TokenStatus
    message: str
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**AuthValidator Class:**
- `check_hf_token(token=None)`: Validate HuggingFace token
- `check_modal_auth()`: Validate Modal authentication
- `validate_before_upload()`: Pre-flight check for uploads
- `report()`: Generate validation summary

**Additional Features:**
- `AuthenticationError` exception class
- `require_hf_token()` convenience function
- `require_modal_auth()` convenience function
- `setup_auth_logging()` for structured logging
- `preflight_check()` for CI/CD integration

#### 2. src/retry_utils.py (642 lines)

**RetryConfig Dataclass:**
- `max_retries`: Maximum retry attempts (default: 3)
- `backoff_coefficient`: Exponential multiplier (default: 2.0)
- `initial_delay`: Starting delay in seconds (default: 1.0)
- `max_delay`: Cap for delay (default: 60.0)
- `retry_on_exceptions`: List of exception types to retry
- `retry_on_status_codes`: List of HTTP status codes to retry
- `jitter`: Enable randomization to avoid thundering herd

**retry_with_backoff Decorator:**
```python
@retry_with_backoff(max_retries=3, retry_on_exceptions=[ConnectionError])
def upload_file():
    pass
```

**RetryManager Class:**
- `execute(func, *args, **kwargs)`: Execute with default config
- `execute_with_config(func, config, *args, **kwargs)`: Custom config
- `get_retry_report()`: Statistics on retry attempts
- `reset_history()`: Clear attempt history

**Additional Features:**
- `RetryableHTTPError` for HTTP-specific retries
- `upload_with_retry()` specialized for HuggingFace
- `is_retryable_error()` utility function
- Exponential backoff with jitter
- Comprehensive type hints and docstrings

#### 3. Tests (91 test cases)

**tests/test_auth_utils.py (56 tests):**
- Token status enum validation
- Token validation result creation
- HF token validation (missing, invalid, valid)
- Modal auth validation
- AuthValidator class methods
- AuthenticationError exception
- Preflight check integration
- Edge cases and integration scenarios

**tests/test_retry_utils.py (35 tests):**
- RetryConfig defaults and validation
- Exponential backoff calculation
- Jitter randomization
- retry_with_backoff decorator
- RetryManager execute methods
- Status code filtering
- upload_with_retry integration
- Edge cases (zero retries, max delay capping)

## Consequences

### Positive

1. **Early Failure Detection**: Auth issues caught before expensive operations
2. **Clear Error Messages**: Users get actionable guidance instead of cryptic errors
3. **Resilient Uploads**: Transient network failures handled automatically
4. **CI/CD Ready**: Preflight checks integrate seamlessly with GitHub Actions
5. **Type Safety**: Full type hints enable IDE autocompletion and mypy validation
6. **Test Coverage**: 91 tests ensure reliability and prevent regressions
7. **Documentation**: Comprehensive docstrings and usage examples

### Negative

1. **Code Size**: ~1,100 lines of new code (auth + retry + tests)
2. **Dependencies**: Adds huggingface_hub import requirement
3. **Maintenance**: Requires updates if Modal/HF APIs change
4. **Complexity**: More code paths to maintain

### Neutral

1. **Import Time**: Minimal impact due to lazy imports
2. **Memory**: Negligible overhead for utility modules

## Usage Examples

### Pre-flight Check in CI

```python
from auth_utils import preflight_check, setup_auth_logging

setup_auth_logging()
result = preflight_check()
if not result:
    exit(1)
```

### Upload with Retry

```python
from retry_utils import upload_with_retry
from auth_utils import require_hf_token

token = require_hf_token()
upload_with_retry(upload_func, file_path, token=token)
```

### Modal Training Validation

```python
from auth_utils import validate_modal_auth

if not validate_modal_auth():
    print("Run: modal token new")
    exit(1)
```

## Implementation Tasks Completed

| Task | Status | Lines | Tests |
|------|--------|-------|-------|
| T-A01-AUTH-PHASE1: Create src/auth_utils.py | ✅ Complete | 453 | - |
| T-A02-AUTH-PHASE1: Create src/retry_utils.py | ✅ Complete | 642 | - |
| T-A03-AUTH-PHASE1: Create tests/test_auth_utils.py | ✅ Complete | 252 | 56 |
| T-A04-AUTH-PHASE1: Create tests/test_retry_utils.py | ✅ Complete | 224 | 35 |

**Total**: 1,571 lines of code, 91 tests, all passing ✅

## Alternatives Considered

### Alternative 1: Inline Auth Checks
- **Rejected**: Duplicates code across scripts, inconsistent error handling

### Alternative 2: Use External Libraries (tenacity, backoff)
- **Rejected**: Adds dependencies, less control over error messages

### Alternative 3: Simpler Validation Only
- **Rejected**: Doesn't address retry requirements for network failures

## Related

- ADR-041: Authentication Error Handling 2026
- GOAP-AUTH-PLAN A01-A20
- MISSING_TASKS_SUMMARY.json: T-A01 through T-A20

## References

- Implementation: src/auth_utils.py, src/retry_utils.py
- Tests: tests/test_auth_utils.py, tests/test_retry_utils.py
- Plan: plans/GOAP-AUTH-PLAN-2026.md
