# ADR-005: CI Pipeline Fixes

## Status
Accepted

## Date
2026-02-23

## Context
GitHub Actions CI was failing with multiple issues:
- Flake8/F401: Unused imports in src files
- Flake8/F541: F-strings without placeholders
- Mypy: Type errors in train.py, eval.py
- Test failures: requirements-dev.txt missing
- Configuration: Inconsistent line-length settings

## Decision
We implemented fixes for all identified CI issues:

### 1. Lint Fixes (via ruff --fix and manual edits)
- Removed unused imports (os, Optional, tempfile)
- Fixed f-strings without placeholders in eval.py, export_onnx.py

### 2. Mypy Fixes
- Fixed return type annotation in train.py (tuple[float, float])
- Added `# type: ignore[attr-defined]` for PyTorch dataset patterns
- Rewrote test_train.py to use actual train.py exports

### 3. Configuration Fixes
- Created requirements-dev.txt
- Added ruff and mypy config to pyproject.toml
- Fixed Makefile line-length from 120 to 88
- Added E501 to ruff ignore (black handles formatting)

## Consequences
- **Positive**: All 36 tests pass
- **Positive**: CI lint checks pass
- **Positive**: Consistent configuration across project
- **Negative**: Some LSP type warnings remain (acceptable)

## Alternatives Considered
1. Disable lint checks in CI - rejected, defeats purpose
2. Use only ruff, remove flake8 - deferred for future
3. Disable all mypy errors - rejected, type safety important

## Related
- ADR-002: CI Workflow Optimization
- ADR-003: AGENTS.md Structure
