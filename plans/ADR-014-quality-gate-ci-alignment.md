# ADR-014: Quality Gate and CI Pipeline Alignment

**Date:** 2026-02-24
**Status:** Superseded by ADR-016 (Modern Python Code Quality 2026)
**Authors:** AI Agent
**Related:** ADR-005 (CI Pipeline Fixes), ADR-006 (CI Fix Workflow), ADR-012 (Flake8 Linting Fixes), ADR-013 (GitHub Actions Workflow Optimization)

## Context

### Problem Statement

Commits pass the local `scripts/quality-gate.sh` script but fail in GitHub Actions CI with flake8 errors. This creates a frustrating development loop where:

1. Developer runs `bash scripts/quality-gate.sh` ✅ passes
2. Developer commits and pushes
3. GitHub Actions CI runs ❌ fails with flake8 E402/E501 errors
4. Developer must iterate remotely instead of catching issues locally

### Root Cause Analysis

The local quality gate and CI pipeline have **different tool configurations**:

| Check | Local Quality Gate | GitHub Actions CI | Match? |
|-------|-------------------|-------------------|--------|
| **Black** | ✅ `black --check .` | ✅ `black --check .` | ✅ Yes |
| **Ruff** | ✅ `ruff check .` | ✅ `ruff check .` | ✅ Yes |
| **Mypy** | ✅ `mypy . --ignore-missing-imports` | ✅ `mypy . --ignore-missing-imports` | ✅ Yes |
| **Flake8** | ❌ **NOT RUN** | ✅ `flake8 . --max-line-length=88 --extend-ignore=E203,W503,E402,E501` | ❌ **NO** |
| **Pytest** | ✅ `pytest tests/ -v` | ✅ `pytest tests/ -v --cov=.` | ⚠️ Partial |

**Key Finding**: The local quality gate **does not run flake8**, but CI does. This is the primary mismatch.

### Configuration Differences

#### Local Ruff Config (`pyproject.toml`)
```toml
[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
ignore = ["E203", "E501"]  # Ignores line length
```

#### CI Flake8 Config (`.github/workflows/ci.yml`)
```yaml
- name: Run flake8
  run: flake8 . --max-line-length=88 --extend-ignore=E203,W503,E402,E501
```

#### Local Black Config (`pyproject.toml`)
```toml
[tool.black]
line-length = 120  # ⚠️ DIFFERENT FROM CI!
```

**Secondary Issue**: Black is configured for 120 chars locally, but CI flake8 expects 88 chars.

### Impact

1. **Developer Experience**: Frustrating feedback loop with remote CI failures
2. **Productivity**: Time wasted on CI iterations instead of local fixes
3. **Confidence**: Unclear if local tests are trustworthy
4. **Merge Delays**: PRs blocked by preventable CI failures

## Decision

We will **align the local quality gate with CI exactly** to ensure "what passes locally passes in CI."

### 1. Add Flake8 to Local Quality Gate

Update `scripts/quality-gate.sh` to include flake8 with the **same configuration** as CI:

```bash
# Add after ruff check
log_info "Running flake8 linter..."

if FLAKE8_OUTPUT=$(flake8 . --max-line-length=88 --extend-ignore=E203,W503,E402,E501 2>&1); then
    log_success "Flake8 check passed"
else
    log_error "Flake8 check failed"
    echo "$FLAKE8_OUTPUT" | head -20
    echo "   Fix flake8 errors before committing"
    FAILURES=$((FAILURES + 1))
    if [[ "$STRICT" == true ]]; then
        exit 1
    fi
fi
```

**Rationale**: Flake8 catches different issues than ruff (e.g., code complexity, stricter PEP 8).

### 2. Align Black Configuration

Update `pyproject.toml` to use **88 characters** (black default) instead of 120:

```toml
[tool.black]
line-length = 88  # Changed from 120
target-version = ["py310", "py311"]
```

**Rationale**: CI flake8 expects 88 chars. Black should match.

### 3. Update isort Configuration

Align isort with black's new line length:

```toml
[tool.isort]
profile = "black"
line_length = 88  # Changed from 120
```

### 4. Add Coverage Check to Local Quality Gate

CI runs `pytest --cov=.` but local quality gate doesn't check coverage. Add optional coverage:

```bash
# Optional coverage check
log_info "Running tests with coverage..."

if PYTEST_OUTPUT=$(python -m pytest tests/ -v --tb=short --cov=. --cov-report=term-missing 2>&1); then
    log_success "All tests passed"
else
    log_error "Tests failed"
    echo "$PYTEST_OUTPUT" | tail -30
    FAILURES=$((FAILURES + 1))
    if [[ "$STRICT" == true ]]; then
        exit 1
    fi
fi
```

### 5. Create `.flake8` Configuration File

Create a `.flake8` file in the project root to centralize flake8 config:

```ini
[flake8]
max-line-length = 88
extend-ignore = E203, W503, E402, E501
exclude = .git,__pycache__,build,dist,.venv,node_modules,notebooks
```

**Rationale**: Single source of truth for flake8 config, used by both local and CI.

### 6. Update CI to Use `.flake8` File

Update `.github/workflows/ci.yml` to read from `.flake8`:

```yaml
- name: Run flake8
  run: flake8 .
```

**Rationale**: Removes duplication, ensures local and CI use identical config.

### 7. Add Pre-commit Hook Integration

Update `.agents/skills/git-workflow/SKILL.md` to recommend pre-commit hook:

```bash
# Install pre-commit hook
bash scripts/install-hooks.sh
```

This ensures quality gate runs **automatically** before every commit.

## Implementation Plan

### Phase 1: Fix Quality Gate Script (Immediate)
- [x] Add flake8 check to `scripts/quality-gate.sh`
- [x] Use same flake8 flags as CI

### Phase 2: Align Configurations (Today)
- [ ] Update `pyproject.toml` black line-length to 88
- [ ] Update `pyproject.toml` isort line-length to 88
- [ ] Create `.flake8` configuration file

### Phase 3: Update CI (Today)
- [ ] Update `.github/workflows/ci.yml` to use `.flake8` file
- [ ] Remove inline flake8 flags from CI

### Phase 4: Documentation Updates (Today)
- [ ] Update `AGENTS.md` with quality gate commands
- [ ] Update `agents-docs/ci-cd.md` with alignment guidance
- [ ] Update `.agents/skills/git-workflow/SKILL.md`
- [ ] Update `plans/GOAP.md` with action items
- [ ] Update `agents-docs/learnings.md` with pattern

### Phase 5: Verification (After Implementation)
- [ ] Run `bash scripts/quality-gate.sh` locally
- [ ] Commit and push
- [ ] Verify CI passes with same checks

## Consequences

### Positive
- ✅ **Single Source of Truth**: `.flake8` file used by both local and CI
- ✅ **Consistent Feedback**: What passes locally passes in CI
- ✅ **Better Developer Experience**: No surprise CI failures
- ✅ **Faster Iteration**: Catch issues before push
- ✅ **Clear Configuration**: 88 chars everywhere (black default)

### Negative
- Slightly longer local quality gate runtime (~10-20 seconds for flake8)
- May require fixing existing code that passes with 120-char lines

### Risks
- **Breaking Changes**: Code formatted for 120 chars will need reformatting to 88
  - **Mitigation**: Run `black .` once to reformat all files
- ** Flake8 Errors**: Existing code may have flake8 violations
  - **Mitigation**: Use `--extend-ignore` for E402/E501 initially, fix incrementally

## Alternatives Considered

### Alternative 1: Remove Flake8 from CI
**Proposal**: Only use ruff, remove flake8 from CI.

**Rejected Because**:
- Flake8 catches different issues than ruff
- Team prefers flake8's stricter PEP 8 enforcement
- Existing CI pipeline already uses flake8

### Alternative 2: Separate Quality Gate and CI
**Proposal**: Keep them different, document the differences.

**Rejected Because**:
- Creates cognitive load for developers
- Still results in CI failures
- Defeats the purpose of a quality gate

### Alternative 3: Use Pre-commit Framework
**Proposal**: Use https://pre-commit.com/ instead of custom script.

**Rejected Because**:
- Adds external dependency
- Current script works well, just needs alignment
- Can adopt pre-commit in future if needed

## 2026 Best Practices Applied

1. **Single Source of Truth**: Configuration files (`.flake8`) over inline flags
2. **Local-First Development**: Catch issues before push
3. **Parity Principle**: Local and CI should match exactly
4. **Automation**: Pre-commit hooks for consistency
5. **Documentation**: Clear guidance in AGENTS.md and skills

## Updated Quality Gate Flow

```
┌─────────────────────────────────────────────────────┐
│  Local Quality Gate (scripts/quality-gate.sh)       │
├─────────────────────────────────────────────────────┤
│  1. Black (format) - 88 chars                       │
│  2. isort (imports) - 88 chars                      │
│  3. Ruff (lint) - E,F,W,I codes                     │
│  4. Flake8 (lint) - .flake8 config                  │
│  5. Mypy (types) - ignore missing imports           │
│  6. Pytest (tests) - with coverage                  │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│  GitHub Actions CI (.github/workflows/ci.yml)       │
├─────────────────────────────────────────────────────┤
│  1. Black (format) - 88 chars                       │
│  2. Ruff (lint) - E,F,W,I codes                     │
│  3. Flake8 (lint) - .flake8 config                  │
│  4. Mypy (types) - ignore missing imports           │
│  5. Pytest (tests) - with coverage                  │
└─────────────────────────────────────────────────────┘
```

**Key**: Both use **identical** configurations via shared config files.

## Success Metrics

- [ ] Zero CI failures due to issues that passed local quality gate
- [ ] Quality gate runtime < 2 minutes (acceptable overhead)
- [ ] Developer satisfaction with local feedback loop
- [ ] All config files aligned (black, isort, ruff, flake8)
- [ ] Documentation updated and clear

## References

- ADR-005: CI Pipeline Fixes
- ADR-006: CI Fix Workflow
- ADR-012: Flake8 Linting Fixes
- ADR-013: GitHub Actions Workflow Optimization
- [Flake8 Configuration](https://flake8.pycqa.org/en/latest/user/configuration.html)
- [Black Configuration](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html)
- [Pre-commit Framework](https://pre-commit.com/)
