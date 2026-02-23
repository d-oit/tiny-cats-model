---
name: gh-actions
description: Use when interacting with GitHub Actions CI/CD automation. Covers workflow status, triggering, debugging, and configuration.
---

# Skill: gh-actions

This skill covers interacting with GitHub Actions for this repository.

## Workflow Location

`.github/workflows/train.yml` and `.github/workflows/ci.yml`

## What the Workflows Do

| Job | Description |
|-----|-------------|
| `lint` | Runs `ruff`, `black --check`, `flake8` |
| `test` | Runs `pytest tests/` after lint passes |
| `type-check` | Runs `mypy` for type checking |
| `model-import-check` | Verifies model and dataset modules |

## Trigger Conditions

- Runs on every `push` to `main`
- Runs on every `pull_request` targeting `main`
- Concurrency: cancels in-progress runs on new pushes

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
# Trigger workflow_dispatch
gh workflow run train.yml

# Trigger with inputs
gh workflow run train.yml -f epochs=20
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
| Lint fails | Run `ruff check . --fix` and `black .` locally |
| Tests fail | Run `pytest tests/ -v` locally |
| Import error | Check dependencies in `requirements.txt` |
| Timeout | Reduce epochs or use `gpu-t4` in `modal.yml` |

## CI Best Practices

1. **Never merge if CI fails** - Required checks must pass
2. **Run locally first** - Use verify script before push
3. **Use concurrency** - Cancels stale runs automatically
