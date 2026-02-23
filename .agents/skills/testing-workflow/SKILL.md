---
name: testing-workflow
description: Use when verifying CI, tests, and training integration. Runs the full test and lint suite and reports results.
triggers:
  - "run tests"
  - "verify ci"
  - "check tests"
  - "test pipeline"
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
# 1. Lint check
ruff check .
flake8 . --max-line-length=88 --extend-ignore=E203,W503

# 2. Format check
black --check .

# 3. Unit tests
pytest tests/ -v --tb=short

# 4. Module import sanity check
python -c "import sys; sys.path.insert(0, 'src'); from model import cats_model; m = cats_model(pretrained=False); print('Model OK')"
```

Alternatively, use the verify script:

```bash
bash .agents/skills/testing-workflow/verify.sh
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

Same commands run automatically in `.github/workflows/train.yml` on every push and PR.
