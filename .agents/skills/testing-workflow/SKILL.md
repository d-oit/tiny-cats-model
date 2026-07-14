---
name: testing-workflow
description: Use when verifying CI, tests, and training integration. Runs the full test and lint suite and reports results.
---

# Skill: testing-workflow

This skill verifies the repository's CI pipeline, tests, and training integration.

## When to Use

- Before and after any code change
- When debugging failing CI
- To confirm the repository is in a healthy state
- Before opening a pull request

## Commands

Run in this exact order:

```bash
# 1. Format check
ruff format --check .

# 2. Lint check
ruff check .

# 3. Type check
mypy . --ignore-missing-imports

# 4. Unit tests (includes GPU pool, train chain, fallback)
pytest tests/ -v --tb=short

# 5. Fallback chain simulation (38 checks)
python scripts/test_fallback_chain.py
```

Alternatively, use the quality gate:

```bash
bash scripts/quality-gate.sh
```

## Expected Output

- All `ruff check` and `flake8` commands exit with code 0
- All `pytest` tests pass (exit code 0)
- No import errors

## Failure Handling

- If lint fails: fix code style issues before running tests
- If tests fail: read the traceback carefully, fix the source code, then re-run
- Never skip failing tests - fix them first

## Integration with CI

Same commands run automatically in `.github/workflows/ci.yml` on every push and PR.
The CI pipeline also includes a benchmark estimate drift check and
a GPU pool fallback chain simulation step.
