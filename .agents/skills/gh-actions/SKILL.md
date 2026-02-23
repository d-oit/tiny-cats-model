---
name: gh-actions
description: Use when interacting with GitHub Actions CI/CD automation. Covers workflow status, triggering, debugging, and configuration.
triggers:
  - "check ci"
  - "github actions"
  - "ci status"
  - "workflow"
  - "pipeline"
---

# Skill: gh-actions

This skill covers interacting with GitHub Actions for this repository.

## Workflow Location

`.github/workflows/train.yml`

## What the Workflow Does

| Job | Description |
|-----|-------------|
| `lint` | Runs `ruff`, `black --check`, `flake8` |
| `test` | Runs `pytest tests/` after lint passes |
| `model-import-check` | Verifies model and dataset modules import correctly |

## Trigger Conditions

- Runs on every `push` to `main`
- Runs on every `pull_request` targeting `main`

## Checking CI Status

```bash
# View latest workflow runs via GitHub CLI
gh run list --repo d-oit/tiny-cats-model

# View a specific run
gh run view <run-id>

# Watch a run in progress
gh run watch <run-id>

# Re-run failed jobs
gh run rerun <run-id> --failed
```

## Manually Triggering

```bash
# Trigger workflow_dispatch (add dispatch trigger to train.yml first)
gh workflow run train.yml
```

## Adding Secrets

```bash
# Add Modal token secrets via GitHub CLI
gh secret set MODAL_TOKEN_ID --body "your_token_id"
gh secret set MODAL_TOKEN_SECRET --body "your_token_secret"
```

Or via GitHub UI: **Settings** > **Secrets and variables** > **Actions** > **New repository secret**

## Common Issues

| Issue | Fix |
|-------|-----|
| Lint fails | Run `ruff check . --fix` and `black .` locally |
| Tests fail | Run `pytest tests/ -v` locally and fix failures |
| Import error | Check that all dependencies are in `requirements.txt` |
| Timeout | Reduce epochs or use `gpu-t4` in `modal.yml` |

## Key Rule

**Never merge a PR if CI is failing.**
