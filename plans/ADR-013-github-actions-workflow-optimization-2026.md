# ADR-013: GitHub Actions Workflow Optimization for 2026

**Date:** 2026-02-24  
**Status:** Proposed  
**Authors:** AI Agent  
**Related:** ADR-005 (CI Pipeline Fixes), ADR-006 (CI Fix Workflow)

## Context

### Current Issue

GitHub PR status shows "3/12 checks failing" even though all required checks pass successfully. This is caused by:

1. **Concurrency cancellation**: Workflow configured with `cancel-in-progress: true`
2. **GitHub UI bug**: Cancelled runs displayed as "failing" instead of "cancelled"
3. **No branch protection**: Required status checks not enforced on `main` branch
4. **Misleading status indicators**: PR mergeability unclear to developers

### Repository Settings Analysis

```json
{
  "allow_auto_merge": true,
  "delete_branch_on_merge": true,
  "has_issues": true,
  "has_projects": false,
  "web_commit_signoff_required": false
}
```

**Branch Protection**: Not configured (feature branch unprotected)

### Current Workflow Configuration

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true  # ← Causes cancelled runs
```

## Decision

We will implement **2026 GitHub Actions Best Practices** to prevent cancelled run confusion and optimize CI/CD workflows.

### 1. Update Concurrency Strategy

**Problem**: Cancelling in-progress runs creates misleading "failing" status indicators.

**Solution**: Use smarter concurrency groups that don't cancel important runs:

```yaml
concurrency:
  # Group by workflow + PR (not branch)
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}
  # Only cancel for same PR, not for main branch
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}
```

**Rationale**: 
- Prevents cancellation confusion on PRs
- Allows main branch builds to complete
- Uses PR number for better grouping (2026 standard)

### 2. Add Workflow Run Names

**Problem**: Generic workflow names make it hard to identify runs in GitHub UI.

**Solution**: Add descriptive `name` field with dynamic values:

```yaml
name: CI - ${{ github.event.pull_request.title || github.ref_name }}
```

**Result**: Shows as "CI - Fix flake8 linting errors" instead of just "CI"

### 3. Configure Branch Protection Rules

**Problem**: No required status checks enforced, allowing merges with failing CI.

**Solution**: Enable branch protection on `main`:

```bash
# Via GitHub CLI (requires admin)
gh api repos/d-oit/tiny-cats-model/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"contexts":["CI / Lint","CI / Test (Python 3.10)","CI / Test (Python 3.11)","CI / Type Check"],"strict":true}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true}' \
  --field allow_deletions=false \
  --field block_creations=false
```

**Via GitHub UI**:
1. Settings → Branches → Add branch protection rule
2. Branch name pattern: `main`
3. Require status checks to pass before merging: ✅
4. Require branches to be up to date before merging: ✅
5. Require pull request reviews before merging: ✅ (1 reviewer)
6. Include administrators: ✅

### 4. Add Job Summaries

**Problem**: Hard to see test results at a glance.

**Solution**: Use GitHub Actions job summaries (2026 feature):

```yaml
- name: Upload test summary
  if: always()
  run: |
    echo "## Test Results" >> $GITHUB_STEP_SUMMARY
    echo "✅ Passed: $PASSED" >> $GITHUB_STEP_SUMMARY
    echo "❌ Failed: $FAILED" >> $GITHUB_STEP_SUMMARY
```

### 5. Add Workflow Badge to README

**Problem**: No visibility of CI status on repo homepage.

**Solution**: Add dynamic badge:

```markdown
![CI Status](https://github.com/d-oit/tiny-cats-model/actions/workflows/ci.yml/badge.svg?branch=main)
```

### 6. Implement Smart Retry Logic

**Problem**: Flaky tests cause CI failures requiring manual retries.

**Solution**: Add automatic retry for known flaky tests:

```yaml
- name: Run tests with retry
  uses: nick-fields/retry@v3
  with:
    timeout_minutes: 5
    max_attempts: 2
    command: pytest tests/ -v --tb=short
```

### 7. Add Workflow Dispatch for Manual Runs

**Problem**: Can't easily re-run specific workflows without pushing code.

**Solution**: Add `workflow_dispatch` trigger:

```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:
    inputs:
      debug_enabled:
        description: 'Run with debug logging'
        required: false
        default: 'false'
```

## Implementation Plan

### Phase 1: Immediate Fixes (This Week)
- [ ] Update concurrency strategy in `.github/workflows/ci.yml`
- [ ] Add workflow run names
- [ ] Add `workflow_dispatch` trigger

### Phase 2: Repository Configuration (This Week)
- [ ] Enable branch protection on `main`
- [ ] Configure required status checks
- [ ] Add workflow badge to README

### Phase 3: Enhanced Features (Next Sprint)
- [ ] Add job summaries
- [ ] Implement smart retry for flaky tests
- [ ] Add debug mode via workflow_dispatch

## Consequences

### Positive
- ✅ Clear CI status indicators (no more "cancelled = failing" confusion)
- ✅ Better workflow organization with descriptive names
- ✅ Protected main branch from broken merges
- ✅ Easier debugging with manual workflow triggers
- ✅ Improved developer experience

### Negative
- Slightly longer CI queue (not cancelling in-progress on main)
- Requires admin access for branch protection setup

### Risks
- Branch protection may block urgent hotfixes (mitigation: emergency override process)
- Retry logic may hide flaky test issues (mitigation: alert on retry)

## 2026 Best Practices Applied

1. **Concurrency Groups by PR Number** - GitHub recommended pattern
2. **Descriptive Workflow Names** - Improves UX in Actions tab
3. **Branch Protection** - Security best practice
4. **Job Summaries** - Native GitHub feature (2025+)
5. **Smart Retry** - Handles transient failures
6. **Manual Triggers** - Operational flexibility
7. **Status Badges** - Project transparency

## Updated Workflow Example

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:
    inputs:
      debug_enabled:
        description: 'Run with debug logging'
        required: false
        default: 'false'

# 2026 Best Practice: Smart concurrency
concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}

jobs:
  lint:
    name: Lint
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      # ... existing steps ...
      
  test:
    name: Test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    timeout-minutes: 10
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4
      
      # ... existing steps ...
      
      # 2026: Smart retry for flaky tests
      - name: Run tests with retry
        if: matrix.python-version == '3.11'
        uses: nick-fields/retry@v3
        with:
          timeout_minutes: 5
          max_attempts: 2
          command: pytest tests/ -v --tb=short
          
      # 2026: Job summary
      - name: Upload test summary
        if: always()
        run: |
          echo "## Test Results" >> $GITHUB_STEP_SUMMARY
          echo "- Python ${{ matrix.python-version }}" >> $GITHUB_STEP_SUMMARY
          echo "- Status: ${{ job.status }}" >> $GITHUB_STEP_SUMMARY
```

## Success Metrics

- [ ] Zero "cancelled" runs showing as "failing" in PR status
- [ ] All required status checks clearly visible and passing
- [ ] Branch protection prevents merges with failing CI
- [ ] Developers can manually trigger workflows for debugging
- [ ] README shows current CI status badge

## References

- [GitHub Actions Concurrency](https://docs.github.com/en/actions/using-jobs/using-concurrency)
- [Branch Protection Rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)
- [Workflow Dispatch](https://docs.github.com/en/actions/using-workflows/manually-running-a-workflow)
- [GitHub Actions Summaries](https://github.blog/2022-05-09-improving-your-development-workflow-with-github-actions-job-summaries/)
- ADR-005: CI Pipeline Fixes
- ADR-006: CI Fix Workflow
