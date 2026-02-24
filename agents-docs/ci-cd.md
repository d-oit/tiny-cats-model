# CI/CD Guide

## Workflows

- **Files**: `.github/workflows/ci.yml`, `train.yml`, `deploy.yml`
- **Trigger**: push + PR to `main`
- **Jobs**: lint → test → type-check (parallel)
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
