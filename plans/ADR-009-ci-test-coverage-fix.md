# ADR-009: Fix CI Test Coverage Dependencies

## Status

Proposed

## Context

The CI workflow `.github/workflows/ci.yml` runs pytest with coverage options:
```bash
pytest tests/ -v --cov=. --cov-report=xml --cov-report=term-missing
```

However, the `pytest-cov` package is not installed in the CI test job. The test job only installs `requirements.txt`, which does not include `pytest-cov`.

`pytest-cov` is currently listed in `requirements-dev.txt`, but that file is not installed in CI.

Error observed:
```
pytest: error: unrecognized arguments: --cov=. --cov-report=xml --cov-report=term-missing
```

## Decision

Install `pytest-cov` in the CI test job by adding it to the install dependencies step.

Two options considered:
1. Install `requirements-dev.txt` in CI test job
2. Add `pytest-cov` directly to the test job install step

We choose option 2 to keep CI explicit and minimal - only install what's needed for tests.

## Consequences

- **Positive**: CI tests will run successfully with coverage reporting
- **Positive**: Clear separation between runtime deps (requirements.txt) and test deps
- **Negative**: Need to maintain pytest-cov version in two places if we also add to requirements-dev.txt

## Alternatives Considered

1. **Move pytest-cov to requirements.txt**: Rejected - coverage is only needed for CI/testing, not runtime
2. **Remove coverage from CI**: Rejected - coverage reporting is valuable for code quality
3. **Install requirements-dev.txt in CI**: Valid alternative, but less explicit about what's needed

## Related

- ADR-006: CI fix workflow
- `.github/workflows/ci.yml`
- `requirements-dev.txt`
