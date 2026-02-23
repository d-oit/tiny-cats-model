---
name: git-workflow
description: Use for branch management, commits, PRs, and git operations via gh CLI.
---

# Skill: git-workflow

This skill covers git operations using GitHub CLI (gh).

## Branch Management

```bash
# Create and switch to new branch
git checkout -b feature/my-feature

# List branches
git branch -a

# Delete local branch
git branch -d feature/my-feature

# Delete remote branch
git push origin --delete feature/my-feature
```

## Commits

```bash
# Stage all changes
git add -A

# Stage specific file
git add src/model.py

# Commit with message
git commit -m "feat: add new model layer"

# Amend last commit (ONLY if not pushed)
git commit --amend

# View commit history
git log --oneline -10
```

## Pull Requests (gh CLI)

```bash
# Create PR from current branch
gh pr create --title "feat: new feature" --body "## Summary

- Added new feature"

# List PRs
gh pr list

# View PR
gh pr view <pr-number>

# Check PR status
gh pr status

# Merge PR
gh pr merge <pr-number> --squash

# Add PR reviewers
gh pr edit <pr-number> --reviewer username
```

## Code Reviews

```bash
# View PR diff
gh pr diff <pr-number>

# Checkout PR locally
gh pr checkout <pr-number>

# Approve PR
gh pr review <pr-number> --approve

# Request changes
gh pr review <pr-number> --body "Please fix..."
```

## Sync & Rebase

```bash
# Fetch all branches
git fetch --all

# Rebase onto main
git rebase main

# Stash changes
git stash push -m "work in progress"

# Apply stash
git stash pop
```

## Common Patterns

```bash
# Quick fix workflow
git checkout -b fix/issue-123
# make changes
git add -A && git commit -m "fix: resolve issue"
git push -u origin fix/issue-123
gh pr create --title "fix: resolve issue" --body "## Summary

- Fixed the bug"

# Feature workflow
git checkout main && git pull
git checkout -b feature/new-feature
# develop...
git push -u origin feature/new-feature
gh pr create
```

## Quality Gate

Run quality checks before every commit:

```bash
bash .agents/skills/git-workflow/quality-gate.sh
```

This runs: ruff, flake8, black, mypy, and pytest.

### Pre-commit Hook (Optional)

To run automatically before each commit:

```bash
# Create .git/hooks/pre-commit
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
exec bash .agents/skills/git-workflow/quality-gate.sh
EOF
chmod +x .git/hooks/pre-commit
```

### Key Rules

- **Never force push to main**
- **Never amend pushed commits**
- **Always run quality gate before commit**
- **Use conventional commits**: `feat:`, `fix:`, `docs:`, `refactor:`
- **PR must pass CI before merge**

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

### Check CI Status

```bash
# View latest workflow runs
gh run list --repo owner/tiny-cats-model

# View specific run details
gh run view <run-id>

# Get failure summary
gh run view <run-id> --log | grep "^ERR"

# Check if passed
gh run view <run-id> --json conclusion
```

### Specialist Agent Selection

| Failure Type | Use Skill |
|--------------|-----------|
| Lint error | `@skill code-quality` |
| Test failure | `@skill testing-workflow` |
| Type error | `@skill code-quality` |
| CI/workflow | `@skill gh-actions` |
| Model/training | `@skill model-training` |
| Security | `@skill security` |

### 2026 Best Practices Integration

Before implementing fixes:
1. Use `websearch` for latest solutions
2. Use `codesearch` for API patterns
3. Document in `plans/ADR-*.md` if significant
4. Update `plans/GOAP.md` with action items
5. Use `@skill goap` for planning tasks

### Example Fix Cycle

```bash
# After push fails:
gh run list
gh run view <run-id>  # See failures

# Identify: lint error → use code-quality
# @skill code-quality

# Agent fixes → commits → pushes

# Re-check:
gh run view <run-id>  # If still failing, repeat

# Loop until SUCCESS
```
