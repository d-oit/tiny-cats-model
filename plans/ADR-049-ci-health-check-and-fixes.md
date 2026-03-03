# ADR-049: CI Health Check and Fixes (March 2026)

**Date:** 2026-03-02
**Status:** Accepted
**Authors:** AI Agent (CI Monitor)
**Related:** ADR-046 (Missing Dependencies), ADR-048 (Modal CLI Syntax)

## Context

### Background
During the execution of the 400k training workflow via GitHub Actions, the CI pipeline experienced consecutive failures that prevented training from starting. A comprehensive health check was conducted to identify and resolve all blocking issues.

### Health Check Process
The CI monitoring agent performed a systematic analysis of recent workflow runs to identify failure patterns and root causes. The investigation revealed two critical issues that needed immediate resolution.

## Issues Found

### Issue 1: Missing Dependencies (ADR-046)
**Failure:** Train workflow run #22588979491 failed immediately with:
```
ModuleNotFoundError: No module named 'torch'
```

**Root Cause:** The `train.yml` workflow was missing the Python dependencies installation step. The workflow only installed `modal` but not project dependencies (torch, torchvision, etc.) required to import training scripts.

**Details documented in:** [ADR-046: GitHub Actions Missing Dependencies Fix](ADR-046-github-actions-missing-dependencies.md)

### Issue 2: Modal CLI Syntax (ADR-048)
**Failure:** Train workflow run #22589524429 failed with:
```
Usage: modal run src/train_dit.py [OPTIONS]
Error: Got unexpected extra argument (data/cats)
```

**Root Cause:** The workflow used positional arguments (`data/cats`) instead of the required `--data-dir` option syntax expected by Modal 1.0+'s `@app.local_entrypoint()` decorator.

**Details documented in:** [ADR-048: Modal CLI Syntax Fix for train.yml](ADR-048-modal-cli-syntax-fix.md)

## Decision

Apply fixes for both issues using specialist agent coordination via the GOAP (Goal-Oriented Action Planning) workflow.

### Resolution Strategy

1. **Specialist Agent Deployment:**
   - CI monitor agent identified and triaged issues
   - Individual ADRs created for each fix (ADR-046, ADR-048)
   - Specialist agents assigned to implement fixes

2. **Fixes Applied:**
   - ADR-046: Added `pip install -r requirements.txt` to both training jobs
   - ADR-048: Changed positional argument `data/cats` to `--data-dir data/cats`

3. **Coordination Pattern:**
   - Issues tracked via GOAP workflow
   - Sequential fix application to avoid conflicts
   - Verification after each fix before proceeding

## Consequences

### Positive
- ✅ CI pipeline now passes all checks
- ✅ Training jobs execute without errors
- ✅ 400k training successfully launched
- ✅ Specialist agent coordination pattern validated
- ✅ Documentation captured for future reference

### Negative
- ⚠️ Initial training launch delayed by CI failures
- ⚠️ Multiple workflow runs required for full verification

### Neutral
- ℹ️ Health check process now established for CI monitoring
- ℹ️ GOAP workflow effective for coordinating complex fixes

## Verification

### Successful Runs After Fixes

```bash
# Latest successful runs:
# - 22590291667: SUCCESS (4m28s) - DiT training
# - 22589803563: SUCCESS (4m3s)  - DiT training
# - 22590583421: SUCCESS (400k training initiated)
```

### Current Status
- **400k training:** In progress (initiated via GitHub Actions)
- **CI Health:** All workflows passing
- **Pipeline:** Ready for future training runs

## Lessons Learned

### CI Monitoring Pattern
1. **Proactive monitoring:** Regular CI health checks catch issues early
2. **Systematic triage:** Use GOAP workflow to track and coordinate fixes
3. **Documentation:** Create ADRs for each issue to build organizational knowledge

### Specialist Agent Coordination
1. **Issue isolation:** Separate ADRs for distinct problems enable parallel work
2. **Sequential validation:** Verify each fix before applying the next
3. **Clear handoffs:** Document fixes for future CI monitoring agents

### Prevention Strategies
1. **Always install dependencies:** `pip install -r requirements.txt` before running Python scripts
2. **Use correct CLI syntax:** Modal 1.0+ requires `--option` format, not positional args
3. **Test locally first:** Validate commands before committing to CI

## References

- ADR-046: [GitHub Actions Missing Dependencies Fix](ADR-046-github-actions-missing-dependencies.md)
- ADR-048: [Modal CLI Syntax Fix for train.yml](ADR-048-modal-cli-syntax-fix.md)
- Workflow: `.github/workflows/train.yml`
- GOAP: Goal-Oriented Action Planning workflow for issue tracking
