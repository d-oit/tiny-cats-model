# ADR-012: Fix Flake8 Linting Errors in CI Pipeline

**Date:** 2026-02-24  
**Status:** Superseded by ADR-016 (Modern Python Code Quality 2026)  
**Authors:** AI Agent  
**Related:** ADR-005 (CI Pipeline Fixes), ADR-006 (CI Fix Workflow)

## Context

The CI pipeline is failing on the `flake8` linting step with multiple E501 (line too long) and E402 (import order) errors across multiple files:

- `scripts/commit-msg-hook.py`: 1 E501
- `scripts/pre-commit-docs.py`: 5 E501
- `scripts/update-goap.py`: 4 E501
- `src/dataset.py`: 5 E501
- `src/dit.py`: 9 E501
- `src/eval.py`: 2 E402, 3 E501
- `src/export_onnx.py`: 1 E501
- `src/flow_matching.py`: 1 E501
- `src/model.py`: 2 E501
- `src/train.py`: 20 E501
- `tests/test_dataset.py`: 2 E402, 2 E501
- `tests/test_model.py`: 1 E501
- `tests/test_train.py`: 5 E402, 1 E501

Total: **66 errors** (59 E501, 7 E402)

## Decision

We will fix these linting errors systematically:

1. **E501 (line too long)**: Break long lines at 88 characters using:
   - Parenthesized continuations (preferred over backslashes)
   - Named parameter splits for function calls
   - String concatenation or f-string breaks

2. **E402 (import order)**: Move all imports to the top of the file, before any code statements

3. **Automated fixes first**: Use `ruff` and `black` to auto-fix what they can, then manually fix remaining issues

## Consequences

### Positive
- CI pipeline will pass
- Code quality improves with consistent formatting
- Better maintainability with proper import organization
- Aligns with PEP 8 style guide

### Negative
- Time investment to fix all errors
- Some code may look slightly more verbose

### Risks
- Manual fixes might introduce bugs if not careful
- Need to verify tests still pass after changes

## Implementation Plan

1. Run `ruff check . --fix` to auto-fix what's possible
2. Run `black .` to format code
3. Manually fix remaining E501 and E402 errors
4. Run `flake8 . --max-line-length=88 --extend-ignore=E203,W503` locally to verify
5. Commit and push changes
6. Monitor CI to confirm all checks pass

## Success Criteria

- [ ] All flake8 errors resolved
- [ ] CI Lint job passes
- [ ] All tests still pass
- [ ] No functional changes to code logic

## References

- PEP 8: https://pep8.org/
- Flake8 documentation: https://flake8.pycqa.org/
- Ruff documentation: https://docs.astral.sh/ruff/
- ADR-005: CI Pipeline Fixes
- ADR-006: CI Fix Workflow
