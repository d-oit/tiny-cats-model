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

## Key Rules

- **Never force push to main**
- **Never amend pushed commits**
- **Always run tests before PR**
- **Use conventional commits**: `feat:`, `fix:`, `docs:`, `refactor:`
- **PR must pass CI before merge**
