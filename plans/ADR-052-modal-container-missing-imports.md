# ADR-052: Fix Modal Container Missing Module Imports in train.py

## Status
Accepted

## Context
The 400k training job (Run 22620541362) failed with:
```
ModuleNotFoundError: No module named 'auth_utils'
```

The error occurred because `train.py`:
1. Had direct imports without fallback handling for `auth_utils` and `experiment_tracker`
2. Missing `add_local_file` entries in Modal container image configuration

While `train_dit.py` had proper try/except fallback pattern and complete `add_local_file` entries (ADR-042), `train.py` was not updated with the same pattern.

## Root Cause Analysis

| Component | train.py | train_dit.py |
|-----------|----------|--------------|
| Import pattern | Direct import | try/except fallback ✅ |
| auth_utils in container | ❌ Missing | ✅ Added (ADR-042) |
| retry_utils in container | ❌ Missing | ✅ Added (ADR-042) |
| experiment_tracker in container | ❌ Missing | ✅ Added (ADR-042) |

## Decision
1. Add try/except fallback pattern for optional imports (matching `train_dit.py`)
2. Add missing `add_local_file` entries for `auth_utils.py`, `retry_utils.py`, `experiment_tracker.py`

## Changes

### src/train.py
```python
# BEFORE: Direct imports (breaks if module not in container)
from auth_utils import AuthenticationError, require_modal_auth, setup_auth_logging
from experiment_tracker import ExperimentTracker

# AFTER: Try/except with fallbacks
try:
    from auth_utils import AuthenticationError, require_modal_auth, setup_auth_logging
    AUTH_UTILS_AVAILABLE = True
except ImportError:
    AUTH_UTILS_AVAILABLE = False
    # Fallback classes defined...
```

### Modal container configuration
```python
# Added to image.add_local_file():
.add_local_file("src/auth_utils.py", "/app/auth_utils.py")
.add_local_file("src/retry_utils.py", "/app/retry_utils.py")
.add_local_file("src/experiment_tracker.py", "/app/experiment_tracker.py")
```

## Consequences

**Positive:**
- Training scripts have consistent import patterns
- Both `train.py` and `train_dit.py` work correctly in Modal container
- Graceful degradation when optional modules unavailable

**Negative:**
- Slightly more complex import logic
- Must maintain fallback classes in sync with actual implementations

## Related
- ADR-042: Modal Training Enhancement (original fix for train_dit.py)
- ADR-030: Modal Container Python Path Fix
- ADR-031: Modal Container Download Scripts Fix

## Date
2026-03-03