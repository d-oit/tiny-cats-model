# CI/CD Guide

## Quality Gate Parity (ADR-014)

**Golden Rule**: The local quality gate must match CI exactly.

```bash
# Run local quality gate before every commit
bash scripts/quality-gate.sh
```

This runs the **same checks** as GitHub Actions CI:
- ✅ Black format check (88 chars)
- ✅ isort import order (88 chars)
- ✅ Ruff linting (E,F,W,I codes)
- ✅ Flake8 linting (reads `.flake8` config)
- ✅ Mypy type checking
- ✅ Pytest tests

**Configuration Files**:
- `.flake8` - Flake8 configuration (single source of truth)
- `pyproject.toml` - Black, isort, mypy configuration

## Workflows

- **Files**: `.github/workflows/ci.yml`, `train.yml`, `deploy.yml`
- **Trigger**: push + PR to `main` + `workflow_dispatch`
- **Jobs**: lint → test → type-check → build-frontend
- **Never merge if CI fails**

## GitHub CLI Commands

```bash
# View CI status
gh run list --repo owner/tiny-cats-model
gh run view <run-id>

# Re-run failed
gh run rerun <run-id> --failed

# Trigger workflow
gh workflow run train.yml
```

## Complete CI/CD Fix Workflow

After every commit, follow this atomic loop:

```
1. git commit → git push
2. gh run list → get run-id
3. gh run view <id> → identify failures
4. FOR EACH failure:
   a. Analyze error type → determine skill needed
   b. Spawn specialist agent with @skill
   c. Agent fixes → commits → pushes
   d. Repeat from step 2 until all pass
5. NEVER skip: each fix must go through full cycle
```

## Example Fix Cycle

```bash
# After push fails:
gh run list --repo owner/tiny-cats-model
gh run view <run-id>  # See failures

# Spawn specialist:
# (Agent uses code-quality skill to fix lint)

# Re-push and re-check:
gh run view <run-id>  # Check again

# Loop until SUCCESS
```

## Local-First Development

**Before pushing**:
```bash
# 1. Run quality gate
bash scripts/quality-gate.sh

# 2. If it passes, commit and push
git add -A && git commit -m "fix: resolve linting"
git push

# 3. Monitor CI
gh run watch
```

**If CI fails anyway**:
1. Check if local quality gate was run
2. Check if config files are in sync (`.flake8`, `pyproject.toml`)
3. Use `gh run view <id>` to see exact error
4. Fix locally, verify with quality gate, push again
