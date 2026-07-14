---
name: gh-actions
description: Use when interacting with GitHub Actions CI/CD automation. Covers workflow status, triggering, debugging, and configuration.
---

# Skill: gh-actions

This skill covers interacting with GitHub Actions for this repository.

## Workflow Location

`.github/workflows/ci.yml` (main CI) and `.github/workflows/train-pool.yml` (GPU pool training)

## What the Workflows Do

| Job | Description |
|-----|-------------|
| `lint` | Runs `ruff format --check` and `ruff check` |
| `test` | Runs `pytest tests/`, fallback chain simulation, benchmark drift check |
| `type-check` | Runs `mypy` for type checking |
| `build-frontend` | Builds frontend if lint/test/type pass |
| `guardrail-deps` | Forbids bad dependency versions |

### GPU Pool Workflow (`train-pool.yml`)

| Job | Description |
|-----|-------------|
| `train-modal` | Modal GPU training with HF Hub checkpoint sync |
| `train-pool-runner` | CPU fallback training on pool runner |
| `pool-summary` | Prints pool status, cost estimates, and result summary |

## Trigger Conditions

- Runs on every `push` to `main`
- Runs on every `pull_request` targeting `main`
- Concurrency: cancels in-progress runs on new pushes
- `train-pool.yml`: workflow_dispatch (manual) or schedule (every 6h)

## Checking CI Status (gh CLI)

```bash
# View latest workflow runs
gh run list --repo owner/tiny-cats-model

# View a specific run
gh run view <run-id>

# Watch a run in progress
gh run watch <run-id>

# View run logs
gh run view <run-id> --log

# Re-run failed jobs
gh run rerun <run-id> --failed
```

## Manually Triggering

```bash
# Trigger CI workflow
gh workflow run ci.yml

# Trigger GPU pool training
gh workflow run train-pool.yml -f steps=20000 -f batch_size=256
gh workflow run train-pool.yml -f provider=all -f steps=50000
```

## Adding Secrets (gh CLI)

```bash
# Add Modal token secrets
gh secret set MODAL_TOKEN_ID --body "your_token_id"
gh secret set MODAL_TOKEN_SECRET --body "your_token_secret"

# List secrets
gh secret list
```

Or via GitHub UI: **Settings** → **Secrets and variables** → **Actions**

## Common Issues

| Issue | Fix |
|-------|-----|
| Lint fails | Run `ruff check . --fix` and `ruff format .` locally |
| Tests fail | Run `pytest tests/ -v` locally |
| Import error | Check dependencies in `requirements.txt` |
| Timeout | Reduce epochs or use `gpu-t4` in `modal.yml` |

## CI Best Practices

1. **Never merge if CI fails** - Required checks must pass
2. **Run locally first** - Use verify script before push
3. **Use concurrency** - Cancels stale runs automatically
4. **Follow complete workflow** - See git-workflow skill for fix loop

## Integration with Complete CI Fix Workflow

When CI fails after push:

1. **Check status**: `gh run list`
2. **Identify failures**: `gh run view <run-id>`
3. **Determine skill**:
   - Lint → `@skill code-quality`
   - Test → `@skill testing-workflow`
   - Type → `@skill code-quality`
   - Workflow → `@skill gh-actions`
4. **Spawn specialist** - Agent fixes and pushes
5. **Loop** - Repeat until all pass

See `plans/ADR-006-ci-fix-workflow.md` for full workflow details.
