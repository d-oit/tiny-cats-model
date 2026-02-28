# Authentication Troubleshooting Guide

**Related:** ADR-041, ADR-042, AGENTS.md, GOAP-AUTH-PLAN-2026.md

This guide provides comprehensive troubleshooting for authentication issues with HuggingFace and Modal tokens.

## Quick Reference

| Error | Quick Fix |
|-------|-----------|
| `HF_TOKEN not set` | `export HF_TOKEN=hf_...` |
| `Invalid format` | Regenerate token at huggingface.co/settings/tokens |
| `Modal token invalid` | `modal token new` (Modal 1.0+) |
| `401 Unauthorized` | Update token in GitHub Secrets |
| `Upload failed after retry` | Check network, verify token |

## Table of Contents

1. [HuggingFace Token Issues](#huggingface-token-issues)
2. [Modal Token Issues](#modal-token-issues)
3. [GitHub Actions Issues](#github-actions-issues)
4. [Upload Script Issues](#upload-script-issues)
5. [Training Script Issues](#training-script-issues)
6. [Logging and Debugging](#logging-and-debugging)

---

## HuggingFace Token Issues

### HF_TOKEN not set

**Symptoms:**
```
ValueError: HF_TOKEN environment variable not set
TokenValidationResult(status=TokenStatus.MISSING)
```

**Diagnosis:**
```bash
# Check if token is set
echo $HF_TOKEN

# Should show: hf_xxxxxxxxxxxx
# If empty: token not set
```

**Solution (Local):**
```bash
# Set token in current session
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Verify
echo $HF_TOKEN

# Make permanent (add to shell config)
echo 'export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx' >> ~/.bashrc
source ~/.bashrc
```

**Solution (GitHub Secrets):**
```bash
# Set secret
gh secret set HF_TOKEN --body "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Verify secret exists
gh secret list

# Re-run workflow
gh workflow run upload-hub.yml
```

**Prevention:**
- Add token to `.env` file (gitignored)
- Use secret management tool
- Document token setup in README

---

### HF_TOKEN has invalid format

**Symptoms:**
```
TokenValidationResult(status=TokenStatus.INVALID, message="HF_TOKEN has invalid format")
```

**Diagnosis:**
```bash
# Check token format
echo $HF_TOKEN | head -c 10

# Should start with: hf_
```

**Root Cause:**
- Token copied incorrectly (truncated)
- Old token format (pre-2023)
- Typo in token value

**Solution:**
1. Go to https://huggingface.co/settings/tokens
2. Click "New token" or regenerate existing
3. Select appropriate permissions (write access for upload)
4. Copy full token (starts with `hf_`)
5. Update environment/secret:
   ```bash
   export HF_TOKEN=hf_new_token_here
   ```

**Verification:**
```bash
python -c "
from auth_utils import AuthValidator
result = AuthValidator().validate_hf_token()
print(f'Status: {result.status}')
print(f'Message: {result.message}')
"
```

---

### HF_TOKEN is invalid or expired

**Symptoms:**
```
huggingface_hub.utils._errors.HFValidationError: 401 Client Error
TokenValidationResult(status=TokenStatus.INVALID)
```

**Diagnosis:**
```bash
# Test token with API
python -c "
from huggingface_hub import HfApi
import os

token = os.environ.get('HF_TOKEN')
try:
    api = HfApi()
    user = api.whoami(token=token)
    print(f'Valid for: {user[\"name\"]}')
except Exception as e:
    print(f'Invalid: {e}')
"
```

**Root Cause:**
- Token revoked
- Token expired (if using old-style tokens)
- Token from wrong account

**Solution:**
1. Generate new token at https://huggingface.co/settings/tokens
2. Ensure token has required permissions:
   - `read` - Read repos
   - `write` - Upload files
3. Update all locations:
   ```bash
   # Local
   export HF_TOKEN=hf_new_token

   # GitHub
   gh secret set HF_TOKEN --body "hf_new_token"
   ```

---

## Modal Token Issues

### Modal token invalid

**Symptoms:**
```
TokenValidationResult(status=TokenStatus.INVALID, message="Modal token invalid")
modal.error: Authentication failed
```

**Diagnosis:**
```bash
# Check token status
modal token info

# Should show: ✅ Token valid
# If error: token invalid
```

**Root Cause:**
- Token not configured
- Token expired
- Token revoked

**Solution (Modal 1.0+):**
```bash
# Re-authenticate
modal token new

# Follow prompts:
# 1. Open URL in browser
# 2. Authorize application
# 3. Copy code
# 4. Paste in terminal

# Verify
modal token info
```

**Verification:**
```bash
python -c "
from auth_utils import AuthValidator
result = AuthValidator().validate_modal_token()
print(f'Status: {result.status}')
print(f'Message: {result.message}')
"
```

---

### Modal CLI not found

**Symptoms:**
```
TokenValidationResult(status=TokenStatus.UNKNOWN, message="Modal CLI not installed")
bash: modal: command not found
```

**Solution:**
```bash
# Install Modal CLI
pip install modal

# Verify installation
modal --version

# Authenticate
modal token set
```

---

## GitHub Actions Issues

### Workflow fails at auth step

**Symptoms:**
```
❌ HF_TOKEN not set
Error: Process completed with exit code 1.
```

**Diagnosis:**
```bash
# Check if secret exists
gh secret list

# Check workflow logs
gh run view <run-id> --log | grep -i auth
```

**Root Cause:**
- Secret not configured
- Secret name mismatch
- Secret in wrong environment

**Solution:**
```bash
# Set secret
gh secret set HF_TOKEN --body "hf_xxxxxxxxxxxx"

# Verify
gh secret list

# Re-run workflow
gh workflow run upload-hub.yml
gh run watch
```

**Alternative (GitHub UI):**
1. Go to repository Settings
2. Click "Secrets and variables" → "Actions"
3. Click "New repository secret"
4. Name: `HF_TOKEN`
5. Value: `hf_xxxxxxxxxxxx`
6. Click "Add secret"

---

### Workflow has 401 error

**Symptoms:**
```
huggingface_hub.utils._errors.HFValidationError: 401 Client Error
Invalid credentials
```

**Diagnosis:**
```bash
# Check secret value (can't see value, but can update)
gh secret set HF_TOKEN --body "hf_new_token"

# Check workflow run
gh run view <run-id> --log
```

**Solution:**
1. Generate new token at HuggingFace
2. Update GitHub secret:
   ```bash
   gh secret set HF_TOKEN --body "hf_new_token"
   ```
3. Re-run workflow:
   ```bash
   gh workflow run upload-hub.yml
   ```

---

## Upload Script Issues

### Upload fails immediately

**Symptoms:**
```
ValueError: HuggingFace token required
Upload aborted: HF_TOKEN validation failed
```

**Diagnosis:**
```bash
# Check token
echo $HF_TOKEN

# Validate token
python -c "from auth_utils import AuthValidator; print(AuthValidator().validate_hf_token())"
```

**Solution:**
```bash
# Set token
export HF_TOKEN=hf_xxxxxxxxxxxx

# Verify
python -c "from auth_utils import AuthValidator; print(AuthValidator().validate_hf_token())"

# Retry upload
python src/upload_to_huggingface.py --generator checkpoints/model.pt --repo-id d4oit/tiny-cats-model
```

---

### Upload fails after retry

**Symptoms:**
```
WARNING: Retry 1/3 after 2.0s
WARNING: Retry 2/3 after 4.0s
WARNING: Retry 3/3 after 8.0s
ERROR: All attempts failed: Connection timeout
```

**Diagnosis:**
```bash
# Check network
curl -I https://huggingface.co

# Check token validity
python -c "from auth_utils import AuthValidator; print(AuthValidator().validate_hf_token())"

# Check HuggingFace status
# https://status.huggingface.co/
```

**Root Cause:**
- Network connectivity issue
- HuggingFace service outage
- Firewall blocking upload
- Large file timeout

**Solution:**
1. Check network:
   ```bash
   curl -I https://huggingface.co
   ```
2. Check HuggingFace status: https://status.huggingface.co/
3. Retry after network issue resolved
4. For large files, use resumable upload:
   ```bash
   # Split upload into smaller batches
   python src/upload_to_huggingface.py --generator checkpoints/model.pt --batch-size 5
   ```

---

## Training Script Issues

### Training fails at start

**Symptoms:**
```
Training aborted: MODAL_TOKEN validation failed
Run 'modal token new' to configure authentication (Modal 1.0+)
```

**Diagnosis:**
```bash
# Check Modal token
modal token info
```

**Solution (Modal 1.0+):**
```bash
# Re-authenticate
modal token new

# Verify
modal token info

# Retry training
modal run src/train_dit.py data/cats --steps 300000
```

---

### Training fails mid-execution

**Symptoms:**
```
modal.error: Connection lost
Training interrupted
```

**Root Cause:**
- Network issue
- Modal service issue
- Token expired during training

**Solution:**
1. Check Modal status: https://modal.com/status
2. Re-authenticate if token expired (Modal 1.0+):
   ```bash
   modal token new
   ```
3. Resume from checkpoint:
   ```bash
   modal run src/train_dit.py data/cats --resume checkpoints/dit_step_50000.pt
   ```

---

## Logging and Debugging

### Auth Logs

**Location:** `logs/auth.log`

**View logs:**
```bash
# Recent entries
tail -50 logs/auth.log

# Search for errors
grep -i error logs/auth.log

# Search for validation
grep -i validate logs/auth.log

# Real-time monitoring
tail -f logs/auth.log
```

**Log format:**
```
2026-02-28 10:30:00 | AUTH | INFO | HF_TOKEN validated for user: username
2026-02-28 10:30:01 | AUTH | WARNING | HF_TOKEN not set in environment
2026-02-28 10:30:02 | AUTH | ERROR | HF_TOKEN validation failed: Invalid credentials
```

### Enable Debug Logging

**For upload script:**
```bash
# Set log level
export AUTH_LOG_LEVEL=DEBUG

# Run with debug
python src/upload_to_huggingface.py --generator checkpoints/model.pt --verbose 2>&1 | tee upload_debug.log
```

**For training script:**
```bash
# Run with debug logging
modal run src/train_dit.py data/cats --steps 100 --log-level DEBUG
```

### Debug Checklist

- [ ] Token is set: `echo $HF_TOKEN`
- [ ] Token format valid: starts with `hf_`
- [ ] Token validated: `python -c "from auth_utils import AuthValidator; ..."`
- [ ] Modal authenticated: `modal token status`
- [ ] Network working: `curl -I https://huggingface.co`
- [ ] HuggingFace status: https://status.huggingface.co/
- [ ] GitHub secret set: `gh secret list`
- [ ] Auth logs checked: `cat logs/auth.log`

---

## FAQ

### Q: How often should I rotate tokens?

**A:** Every 90 days for security best practices. Set a calendar reminder.

### Q: Can I use the same token for local and CI/CD?

**A:** Yes, but consider using separate tokens for better access control and audit trail.

### Q: What permissions does the HF_TOKEN need?

**A:** For upload: `read` + `write`. For download only: `read`.

### Q: How do I know if my token was compromised?

**A:** Check HuggingFace account activity. If suspicious activity, revoke token immediately and generate new one.

### Q: Can I use environment-specific tokens?

**A:** Yes, use different tokens for development, staging, and production. Configure via GitHub Environments.

---

## Related Documentation

- [ADR-041](../plans/ADR-041-authentication-error-handling-2026.md) - Architecture Decision
- [AGENTS.md](../AGENTS.md#authentication) - Quick Reference
- [GOAP-AUTH-PLAN-2026](../plans/GOAP-AUTH-PLAN-2026.md) - Implementation Plan
- [auth-manager Skill](../.agents/skills/auth-manager/SKILL.md) - Skill Documentation
