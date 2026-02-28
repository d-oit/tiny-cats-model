# GOAP: Authentication Error Handling Plan

**Created:** 2026-02-28
**Related ADR:** ADR-041 (Authentication Error Handling and Token Validation)
**Status:** Proposed

## Goal

Implement comprehensive authentication error handling with token validation, retry logic, and enhanced logging for Modal and HuggingFace operations.

## Objectives

1. **Token Validation** - Pre-flight checks for HF_TOKEN and MODAL_TOKEN
2. **Error Handling** - Clear, actionable error messages for auth failures
3. **Retry Logic** - Automatic retry with exponential backoff for uploads
4. **Logging** - Structured logging for authentication events
5. **CI/CD Integration** - Pre-flight validation in GitHub Actions workflows

## Actions

### Phase 1: Authentication Utilities (P0 - 2 hours)

#### A01: Create `src/auth_utils.py`
- [ ] Implement `TokenStatus` enum (VALID, INVALID, EXPIRED, MISSING, UNKNOWN)
- [ ] Implement `TokenValidationResult` dataclass
- [ ] Implement `AuthValidator` class with:
  - [ ] `validate_hf_token()` - HF token validation with API call
  - [ ] `validate_modal_token()` - Modal token validation via CLI
  - [ ] `validate_all_tokens()` - Batch validation
- [ ] Implement `setup_auth_logging()` for structured auth logging
- [ ] Add docstrings and type hints
- [ ] Format with ruff

**Skill:** `model-training` + `code-quality`
**Dependencies:** None
**Completed At:** -

#### A02: Create `src/retry_utils.py`
- [ ] Implement `RetryConfig` class
- [ ] Implement `retry_with_backoff()` decorator
- [ ] Implement `RetryManager` class
- [ ] Support exponential backoff with jitter
- [ ] Support configurable retryable exceptions
- [ ] Add progress logging
- [ ] Format with ruff

**Skill:** `model-training` + `code-quality`
**Dependencies:** A01
**Completed At:** -

#### A03: Create Unit Tests for Auth Utilities
- [ ] Create `tests/test_auth_utils.py`
- [ ] Test `validate_hf_token()` with missing token
- [ ] Test `validate_hf_token()` with invalid format
- [ ] Test `validate_hf_token()` with valid token (mocked)
- [ ] Test `validate_modal_token()` with CLI not found
- [ ] Test `validate_modal_token()` with valid token (mocked)
- [ ] Run tests: `pytest tests/test_auth_utils.py -v`

**Skill:** `testing-workflow`
**Dependencies:** A01
**Completed At:** -

#### A04: Create Unit Tests for Retry Utilities
- [ ] Create `tests/test_retry_utils.py`
- [ ] Test `retry_with_backoff()` decorator
- [ ] Test `RetryManager.execute()` with success
- [ ] Test `RetryManager.execute()` with retry
- [ ] Test `RetryManager.execute()` with max attempts exceeded
- [ ] Run tests: `pytest tests/test_retry_utils.py -v`

**Skill:** `testing-workflow`
**Dependencies:** A02
**Completed At:** -

### Phase 2: Upload Script Enhancement (P1 - 2 hours)

#### A05: Update `src/upload_to_huggingface.py` with Auth Validation
- [ ] Import `auth_utils` module
- [ ] Add pre-flight token validation in `upload_to_huggingface()`
- [ ] Fail fast with clear error if token invalid
- [ ] Log validation results
- [ ] Test: `python src/upload_to_huggingface.py --help`

**Skill:** `model-training`
**Dependencies:** A01
**Completed At:** -

#### A06: Add Retry Logic to Upload Operations
- [ ] Import `retry_utils` module
- [ ] Wrap upload operations with `retry_with_backoff()` or `RetryManager`
- [ ] Configure retry for ConnectionError, TimeoutError
- [ ] Log retry attempts
- [ ] Test with simulated network error

**Skill:** `model-training`
**Dependencies:** A02, A05
**Completed At:** -

#### A07: Add Structured Logging to Upload Script
- [ ] Import `setup_auth_logging()` from `auth_utils`
- [ ] Setup auth logger at script start
- [ ] Log token validation status (masked)
- [ ] Log upload progress
- [ ] Log retry events
- [ ] Test: Run upload and verify logs

**Skill:** `model-training`
**Dependencies:** A05
**Completed At:** -

#### A08: Test Upload with Invalid Token
- [ ] Set invalid HF_TOKEN: `export HF_TOKEN=hf_invalid`
- [ ] Run upload script
- [ ] Verify fast failure (< 5 seconds)
- [ ] Verify clear error message
- [ ] Verify error logged to `logs/auth.log`

**Skill:** `testing-workflow`
**Dependencies:** A05, A07
**Completed At:** -

#### A09: Test Upload with Network Error (Retry)
- [ ] Mock network error in upload
- [ ] Run upload script
- [ ] Verify retry attempts (3 attempts)
- [ ] Verify exponential backoff delays
- [ ] Verify retry events logged

**Skill:** `testing-workflow`
**Dependencies:** A06
**Completed At:** -

### Phase 3: Training Script Enhancement (P1 - 1 hour)

#### A10: Update `src/train_dit.py` with Modal Token Validation
- [ ] Import `auth_utils` module
- [ ] Add pre-flight Modal token validation in `main()`
- [ ] Fail fast with clear error if token invalid
- [ ] Log validation results
- [ ] Include setup instructions in error message
- [ ] Test: `modal run src/train_dit.py --help`

**Skill:** `model-training`
**Dependencies:** A01
**Completed At:** -

#### A11: Add Structured Logging to Training Script
- [ ] Import `setup_auth_logging()` from `auth_utils`
- [ ] Setup auth logger at script start
- [ ] Log Modal token validation status
- [ ] Log training start/stop events
- [ ] Test: Run training and verify logs

**Skill:** `model-training`
**Dependencies:** A10
**Completed At:** -

#### A12: Test Training with Invalid Modal Token
- [ ] Clear Modal token: `modal token set` (don't complete)
- [ ] Run training script
- [ ] Verify fast failure (< 5 seconds)
- [ ] Verify clear error message
- [ ] Verify setup instructions in error

**Skill:** `testing-workflow`
**Dependencies:** A10, A11
**Completed At:** -

### Phase 4: CI/CD Enhancement (P1 - 1 hour)

#### A13: Add HF_TOKEN Pre-flight Check to upload-hub.yml
- [ ] Add "Validate HF_TOKEN" step before upload
- [ ] Use Python script for validation
- [ ] Fail workflow if validation fails
- [ ] Log validation result
- [ ] Test: Trigger workflow with missing HF_TOKEN

**Skill:** `gh-actions`
**Dependencies:** A01
**Completed At:** -

#### A14: Add Auth Validation Logging to CI
- [ ] Add structured logging to workflow steps
- [ ] Capture auth validation output
- [ ] Include in workflow logs
- [ ] Test: Trigger workflow and verify logs

**Skill:** `gh-actions`
**Dependencies:** A13
**Completed At:** -

#### A15: Test Workflow with Missing HF_TOKEN
- [ ] Temporarily remove HF_TOKEN secret (or use test repo)
- [ ] Trigger upload workflow
- [ ] Verify workflow fails at validation step
- [ ] Verify clear error in logs
- [ ] Restore HF_TOKEN secret

**Skill:** `testing-workflow` + `gh-actions`
**Dependencies:** A13
**Completed At:** -

#### A16: Test Workflow with Invalid HF_TOKEN
- [ ] Set invalid HF_TOKEN secret
- [ ] Trigger upload workflow
- [ ] Verify workflow fails at validation step
- [ ] Verify clear error in logs
- [ ] Restore valid HF_TOKEN secret

**Skill:** `testing-workflow` + `gh-actions`
**Dependencies:** A13
**Completed At:** -

### Phase 5: Documentation (P2 - 1 hour)

#### A17: Update AGENTS.md with Auth Validation Commands
- [ ] Add auth validation section
- [ ] Document `python -c "from auth_utils import AuthValidator; ..."` commands
- [ ] Document troubleshooting steps
- [ ] Link to ADR-041

**Skill:** `agents-md`
**Dependencies:** A01
**Completed At:** -

#### A18: Update learnings.md with Auth Troubleshooting Patterns
- [ ] Add authentication troubleshooting section
- [ ] Document common errors and solutions
- [ ] Include code patterns for validation
- [ ] Link to ADR-041

**Skill:** `agents-md`
**Dependencies:** A01
**Completed At:** -

#### A19: Create Auth Troubleshooting Guide
- [ ] Create `agents-docs/auth-troubleshooting.md`
- [ ] Document all auth error types
- [ ] Provide step-by-step resolution
- [ ] Include FAQ section

**Skill:** `agents-md`
**Dependencies:** A01
**Completed At:** -

#### A20: Update ADR-039 with New Validation Steps
- [ ] Add reference to ADR-041
- [ ] Update implementation plan with validation steps
- [ ] Update workflow file reference
- [ ] Mark ADR-041 as related

**Skill:** `agents-md`
**Dependencies:** A01
**Completed At:** -

## GOAP Action Status

| Action | Status | Phase | Skill | Priority | Completed At |
|--------|--------|-------|-------|----------|--------------|
| A01: Create auth_utils.py | ⏳ PENDING | 1 | model-training | P0 | - |
| A02: Create retry_utils.py | ⏳ PENDING | 1 | model-training | P0 | - |
| A03: Test auth utilities | ⏳ PENDING | 1 | testing-workflow | P0 | - |
| A04: Test retry utilities | ⏳ PENDING | 1 | testing-workflow | P0 | - |
| A05: Update upload script | ⏳ PENDING | 2 | model-training | P1 | - |
| A06: Add retry to upload | ⏳ PENDING | 2 | model-training | P1 | - |
| A07: Add logging to upload | ⏳ PENDING | 2 | model-training | P1 | - |
| A08: Test invalid token | ⏳ PENDING | 2 | testing-workflow | P1 | - |
| A09: Test network retry | ⏳ PENDING | 2 | testing-workflow | P1 | - |
| A10: Update training script | ⏳ PENDING | 3 | model-training | P1 | - |
| A11: Add logging to training | ⏳ PENDING | 3 | model-training | P1 | - |
| A12: Test invalid Modal token | ⏳ PENDING | 3 | testing-workflow | P1 | - |
| A13: Add CI pre-flight | ⏳ PENDING | 4 | gh-actions | P1 | - |
| A14: Add CI logging | ⏳ PENDING | 4 | gh-actions | P1 | - |
| A15: Test CI missing token | ⏳ PENDING | 4 | testing-workflow | P1 | - |
| A16: Test CI invalid token | ⏳ PENDING | 4 | testing-workflow | P1 | - |
| A17: Update AGENTS.md | ⏳ PENDING | 5 | agents-md | P2 | - |
| A18: Update learnings.md | ⏳ PENDING | 5 | agents-md | P2 | - |
| A19: Create troubleshooting guide | ⏳ PENDING | 5 | agents-md | P2 | - |
| A20: Update ADR-039 | ⏳ PENDING | 5 | agents-md | P2 | - |

**Progress:** 0/20 actions complete (0%)

## Dependency Graph

```
Phase 1 (Utilities)
┌─────────────────────────────────────┐
│ A01: auth_utils.py                  │
│ A02: retry_utils.py                 │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ A03: Test auth_utils                │
│ A04: Test retry_utils               │
└──────────────┬──────────────────────┘
               │
               ▼
Phase 2 (Upload)    Phase 3 (Training)    Phase 4 (CI/CD)
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ A05: Upload  │    │ A10: Train   │    │ A13: CI      │
│   validation │    │   validation │    │   validation │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ A06: Retry   │    │ A11: Logging │    │ A14: Logging │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ A07: Logging │    │ A12: Test    │    │ A15: Test    │
└──────┬───────┘    └──────────────┘    │ A16: Test    │
       │                                 └──────────────┘
       ▼
┌──────────────┐
│ A08: Test    │
│ A09: Test    │
└──────────────┘
               │
               ▼
         Phase 5 (Docs)
         ┌──────────────┐
         │ A17: AGENTS  │
         │ A18: Learning│
         │ A19: Guide   │
         │ A20: ADR-039 │
         └──────────────┘
```

## Timeline Estimate

| Week | Phase | Activities | Deliverables |
|------|-------|------------|--------------|
| **Week 1** | Phase 1-2 | Auth utilities, Upload enhancement | auth_utils.py, retry_utils.py, Enhanced upload script |
| **Week 2** | Phase 3-4 | Training enhancement, CI/CD | Enhanced training script, CI validation |
| **Week 3** | Phase 5 | Documentation | Updated docs, troubleshooting guide |

**Total Duration:** 2-3 weeks (7 hours of active work + testing buffer)

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Auth validation coverage | 100% | All scripts using auth_utils |
| Pre-flight checks | 100% | Upload + training scripts |
| Retry success rate | >80% | Upload retry logs |
| Test coverage | >90% | pytest --cov=src/auth_utils |
| Documentation complete | 100% | All A20 actions complete |

## References

- ADR-041: Authentication Error Handling and Token Validation
- ADR-039: Automated HuggingFace CI Upload
- `src/upload_to_huggingface.py`
- `src/train_dit.py`
- `.github/workflows/upload-hub.yml`
