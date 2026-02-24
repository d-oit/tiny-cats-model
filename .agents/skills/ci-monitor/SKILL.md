---
name: ci-monitor
description: Monitor GitHub Actions CI runs, identify failures, and coordinate specialist agent fixes using GOAP/ADR workflow
---

# Skill: ci-monitor

This skill monitors CI/CD pipelines and orchestrates fixes using specialist agents.

## When to Use

- After every `git push`
- When GitHub Actions show failures
- Coordinating multiple specialist fixes
- Tracking CI fix progress

## Workflow

### 1. Monitor CI Status

```bash
# Get latest runs
gh run list --repo <owner>/<repo> --limit 5

# Watch in real-time
gh run watch --repo <owner>/<repo>

# View specific run
gh run view <run-id> --repo <owner>/<repo>
```

### 2. Identify Failures

```bash
# Get failed run ID
gh run list --json databaseId,status,conclusion --jq '.[] | select(.conclusion == "failure") | .databaseId'

# View failure details
gh run view <run-id> --log-failed

# Get job-level failures
gh run view <run-id> --json jobs --jq '.jobs[] | select(.conclusion == "failure")'
```

### 3. Categorize Failure Type

| Failure Pattern | Specialist Agent | Skill |
|-----------------|------------------|-------|
| `flake8`, `ruff`, `black` errors | code-quality | `@skill code-quality` |
| `pytest` failures | testing-workflow | `@skill testing-workflow` |
| `mypy` type errors | code-quality | `@skill code-quality` |
| `npm`, `tsc`, frontend build | frontend (if exists) | Create skill |
| `modal`, training, GPU | model-training | `@skill model-training` |
| Workflow config, runner issues | gh-actions | `@skill gh-actions` |
| Secrets, tokens, credentials | security | `@skill security` |

### 4. Spawn Specialist Agent

```markdown
@task
**Description**: Fix <failure type> in CI run <run-id>
**Prompt**: 
CI run <run-id> failed with:
```
<error logs from gh run view>
```

Fix this issue using <specialist skill>. After fixing:
1. Commit with conventional commit message
2. Push to trigger new CI
3. Report back the new commit SHA
**Subagent**: general-purpose (or specialist)
```

### 5. Track Progress

Create/update GOAP action item:

```markdown
## Current Action Items
- [x] <completed fix>
- [ ] <current fix in progress>
- [ ] <remaining fixes>
```

### 6. Verify Fix

```bash
# Wait for new run
sleep 30 && gh run list --limit 1

# Check if passed
gh run view <new-run-id>

# If still failing, repeat from step 2
```

## Atomic Loop

```
1. git commit → git push
2. gh run list → get run-id
3. gh run view <id> → identify failures
4. FOR EACH failure:
   a. Analyze error type → determine specialist needed
   b. Spawn specialist agent with @task
   c. Agent fixes → commits → pushes
   d. gh run view <new-id> → verify
   e. Repeat until all pass
5. Update GOAP.md with completed items
6. NEVER skip: each fix must go through full cycle
```

## Example Session

```bash
# 1. Push code
git add -A && git commit -m "feat: add new feature" && git push

# 2. Monitor CI
gh run watch

# 3. CI fails - check what failed
gh run view 123456 --log-failed

# Output shows: flake8 E501 errors in src/model.py

# 4. Spawn specialist
@task
**Description**: Fix flake8 E501 errors
**Prompt**: CI run 123456 failed with flake8 E501 errors in src/model.py. Fix using @skill code-quality.
**Subagent**: general-purpose

# 5. Agent fixes and pushes
# Agent reports: "Fixed, pushed commit abc123"

# 6. Verify
gh run view 123457  # New run ID

# 7. If passed, update GOAP
# Mark action complete in plans/GOAP.md
```

## Integration with GOAP

### Update GOAP.md

```markdown
## Current Action Items
- [x] Fix flake8 E501 errors (CI run 123456)
- [x] Fix mypy type errors (CI run 123457)
- [ ] Fix frontend build errors (CI run 123458)
```

### Create ADR if Significant

```bash
# If architectural decision needed
python scripts/adr-scaffold.py "Title"

# Edit plans/ADR-XXX-*.md
# Document problem, decision, consequences
```

## Best Practices

1. **One failure at a time** - Fix failures in order (lint → test → type)
2. **Atomic commits** - Each fix is a separate commit
3. **Track everything** - Update GOAP after each fix
4. **Use specialists** - Match failure type to specialist skill
5. **Verify each fix** - Don't batch multiple fixes without verification
6. **Document patterns** - Create ADR for recurring issues

## Commands Reference

| Command | Purpose |
|---------|---------|
| `gh run list` | List recent runs |
| `gh run view <id>` | View run details |
| `gh run view <id> --log-failed` | View failure logs |
| `gh run watch` | Watch in real-time |
| `gh run rerun <id>` | Re-run failed run |
| `gh workflow list` | List workflows |
| `gh workflow run <name>` | Trigger workflow |

## Related Skills

- `git-workflow` - Git operations, commits, pushes
- `code-quality` - Linting, formatting, type checking
- `testing-workflow` - Test failures, coverage
- `gh-actions` - Workflow configuration, runners
- `goap` - Planning, ADR, action items
