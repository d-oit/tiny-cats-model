---
name: auth-manager
description: Use for authentication management, token validation, and credential troubleshooting.
triggers:
  - "authentication"
  - "token"
  - "credentials"
  - "HF_TOKEN"
  - "MODAL_TOKEN"
  - "HuggingFace auth"
  - "Modal auth"
  - "secret validation"
  - "auth error"
  - "401"
  - "unauthorized"
---

# Skill: auth-manager

This skill provides authentication management capabilities for HuggingFace and Modal tokens, including validation, troubleshooting, and error handling.

## Capabilities

### Token Validation

```bash
# Validate HF_TOKEN
python -c "from auth_utils import AuthValidator; print(AuthValidator().validate_hf_token())"

# Validate MODAL_TOKEN
python -c "from auth_utils import AuthValidator; print(AuthValidator().validate_modal_token())"

# Validate all tokens
python -c "from auth_utils import AuthValidator; v = AuthValidator(); print(v.validate_all_tokens())"
```

### Token Configuration

```bash
# Set HF_TOKEN (local)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Set HF_TOKEN (GitHub Secrets)
gh secret set HF_TOKEN --body "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Set Modal token (Modal 1.0+ uses 'token new')
modal token new

# Verify Modal token
modal token info
modal token list
```

### Auth Troubleshooting

```bash
# Check token format
echo $HF_TOKEN | grep -E "^hf_"

# List GitHub secrets
gh secret list

# Check auth logs
cat logs/auth.log

# Test upload with validation
python src/upload_to_huggingface.py --help
```

## When to Use

Use this skill when:

1. **Token validation needed** - Before running training or upload operations
2. **Auth errors occur** - 401 Unauthorized, Invalid credentials
3. **CI/CD failures** - Workflow fails at auth step
4. **Token expiry suspected** - Previously working token now fails
5. **Setup verification** - Verify auth configuration is correct

## Common Issues and Solutions

### Issue: HF_TOKEN not set

**Symptoms:**
```
ValueError: HF_TOKEN environment variable not set
```

**Diagnosis:**
```bash
echo $HF_TOKEN  # Should show token
```

**Solution:**
```bash
# Local
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# GitHub Secrets
gh secret set HF_TOKEN --body "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Issue: HF_TOKEN has invalid format

**Symptoms:**
```
TokenValidationResult(status=TokenStatus.INVALID, message="HF_TOKEN has invalid format")
```

**Diagnosis:**
```bash
echo $HF_TOKEN | head -c 10  # Should start with "hf_"
```

**Solution:**
1. Go to https://huggingface.co/settings/tokens
2. Create new token or regenerate existing
3. Copy full token (starts with `hf_`)
4. Update environment/secret

### Issue: Modal token invalid

**Symptoms:**
```
TokenValidationResult(status=TokenStatus.INVALID, message="Modal token invalid")
```

**Diagnosis:**
```bash
modal token info  # Should show valid token
```

**Solution:**
```bash
modal token new  # Re-authenticate (Modal 1.0+)
modal token info  # Verify
```

### Issue: 401 Unauthorized in CI

**Symptoms:**
```
huggingface_hub.utils._errors.HFValidationError: 401 Client Error
```

**Diagnosis:**
```bash
# Check if secret exists
gh secret list | grep HF_TOKEN

# Check workflow logs
gh run view <run-id> --log
```

**Solution:**
```bash
# Update secret
gh secret set HF_TOKEN --body "hf_new_token_here"

# Re-run workflow
gh workflow run upload-hub.yml
```

### Issue: Upload fails after retry

**Symptoms:**
```
WARNING: Retry 3/3 after 30.0s
ERROR: All attempts failed: Connection timeout
```

**Diagnosis:**
```bash
# Check network
curl -I https://huggingface.co

# Check token validity
python -c "from auth_utils import AuthValidator; print(AuthValidator().validate_hf_token())"
```

**Solution:**
1. Verify network connectivity
2. Verify token is valid
3. Check HuggingFace status: https://status.huggingface.co/
4. Retry after network issue resolved

## Validation Workflow

### Pre-flight Validation

Before long-running operations (training, upload):

```python
from auth_utils import AuthValidator, TokenStatus, setup_auth_logging

logger = setup_auth_logging()
validator = AuthValidator(logger)

# Validate HF_TOKEN
hf_result = validator.validate_hf_token()
if hf_result.status != TokenStatus.VALID:
    logger.error(f"HF_TOKEN validation failed: {hf_result.message}")
    sys.exit(1)

# Validate MODAL_TOKEN (for training)
modal_result = validator.validate_modal_token()
if modal_result.status != TokenStatus.VALID:
    logger.error(f"MODAL_TOKEN validation failed: {modal_result.message}")
    sys.exit(1)

logger.info("All tokens validated successfully")
```

### CI/CD Validation

In GitHub Actions workflow:

```yaml
- name: Validate HF_TOKEN
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
  run: |
    python -c "
    import os
    from huggingface_hub import HfApi

    token = os.environ.get('HF_TOKEN')
    if not token:
        print('❌ HF_TOKEN not set')
        exit(1)

    if not token.startswith('hf_'):
        print('❌ HF_TOKEN has invalid format')
        exit(1)

    try:
        api = HfApi()
        user = api.whoami(token=token)
        print(f'✅ HF_TOKEN valid for user: {user[\"name\"]}')
    except Exception as e:
        print(f'❌ HF_TOKEN validation failed: {e}')
        exit(1)
    "
```

## Retry Pattern

For operations that may fail transiently:

```python
from retry_utils import RetryConfig, RetryManager

config = RetryConfig(
    max_attempts=3,
    base_delay=2.0,
    max_delay=30.0,
    retryable_exceptions=(ConnectionError, TimeoutError)
)
manager = RetryManager(config)

def upload_operation():
    # ... upload logic ...
    pass

try:
    manager.execute(upload_operation)
except Exception as e:
    logger.error(f"Upload failed after all retries: {e}")
    sys.exit(1)
```

## Security Best Practices

### Do

- ✅ Use environment variables for tokens
- ✅ Use GitHub Secrets for CI/CD
- ✅ Validate tokens before operations
- ✅ Log token status (not values)
- ✅ Rotate tokens periodically

### Don't

- ❌ Hardcode tokens in code
- ❌ Commit tokens to version control
- ❌ Log full token values
- ❌ Pass tokens via command line args
- ❌ Share tokens in chat/logs

## Related Files

| File | Purpose |
|------|---------|
| `src/auth_utils.py` | Token validation utilities |
| `src/retry_utils.py` | Retry with exponential backoff |
| `src/upload_to_huggingface.py` | Upload script with auth validation |
| `src/train_dit.py` | Training script with auth validation |
| `.github/workflows/upload-hub.yml` | CI/CD workflow with auth check |
| `logs/auth.log` | Authentication event logs |

## Related Documentation

- [ADR-041](../plans/ADR-041-authentication-error-handling-2026.md) - Authentication Error Handling
- [ADR-042](../plans/ADR-042-modal-training-enhancement.md) - Modal Training Enhancement
- [GOAP-AUTH-PLAN-2026](../plans/GOAP-AUTH-PLAN-2026.md) - Implementation Plan
- [Authentication Troubleshooting](../agents-docs/auth-troubleshooting.md) - Detailed Guide
- [AGENTS.md](../AGENTS.md#authentication) - Quick Reference

## Example Sessions

### Session 1: Validate Tokens Before Training

```bash
# Check HF_TOKEN
python -c "from auth_utils import AuthValidator; print(AuthValidator().validate_hf_token())"
# Output: TokenValidationResult(token_type='HF_TOKEN', status=TokenStatus.VALID, ...)

# Check MODAL_TOKEN (Modal 1.0+)
modal token info
# Output: ✅ Token valid

# Start training
modal run src/train_dit.py data/cats --steps 300000
```

### Session 2: Fix CI Auth Failure

```bash
# Check workflow failure
gh run view 12345 --log | grep -i auth

# Check secret
gh secret list

# Update secret
gh secret set HF_TOKEN --body "hf_new_token"

# Re-run workflow
gh workflow run upload-hub.yml
gh run watch
```

### Session 3: Debug Upload Failure

```bash
# Check auth logs
cat logs/auth.log | tail -50

# Validate token
python -c "from auth_utils import AuthValidator; print(AuthValidator().validate_hf_token())"

# Test upload with verbose logging
python src/upload_to_huggingface.py --generator checkpoints/model.pt --repo-id test/repo 2>&1 | tee upload.log

# Check for retry events
grep -i retry upload.log
```
