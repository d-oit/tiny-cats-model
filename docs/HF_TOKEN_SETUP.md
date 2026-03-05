# HuggingFace Token Setup Guide

This guide explains how to configure HuggingFace tokens for automated model uploads.

## Overview

The tiny-cats-model project uses GitHub Actions to automatically upload trained models to HuggingFace Hub. This requires configuring a `HF_TOKEN` secret in your GitHub repository.

## Prerequisites

- GitHub account with write access to the repository
- HuggingFace account at https://huggingface.co
- HuggingFace repository created (e.g., `d4oit/tiny-cats-model`)

## Step 1: Generate HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Configure token:
   - **Name:** `tiny-cats-model-ci`
   - **Type:** Fine-grained (recommended) or classic
   - **Permissions:**
     - ✅ `read` - Read access to public repos
     - ✅ `write` - Write access to repos you own

4. Click **"Generate token"**
5. **Copy the token immediately** - you won't see it again!

Token format: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

## Step 2: Add Token to GitHub Secrets

### Option A: GitHub CLI (Recommended)

```bash
# Add HF_TOKEN secret
gh secret set HF_TOKEN --body "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Verify secret was added
gh secret list
```

### Option B: GitHub Web UI

1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Enter:
   - **Name:** `HF_TOKEN`
   - **Value:** `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
5. Click **"Add secret"**

## Step 3: Verify Secret Configuration

### Test Secret Access

Create a test workflow to verify the secret is accessible:

```yaml
# .github/workflows/test-secret.yml
name: Test HF_TOKEN Secret

on:
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Test HF_TOKEN
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          if [ -z "$HF_TOKEN" ]; then
            echo "❌ HF_TOKEN is not set"
            exit 1
          fi
          if [[ ! "$HF_TOKEN" =~ ^hf_ ]]; then
            echo "❌ HF_TOKEN has invalid format"
            exit 1
          fi
          echo "✅ HF_TOKEN is configured correctly"
```

Run the workflow:
```bash
gh workflow run test-secret.yml
gh run watch  # Watch progress
```

## Step 4: Test Upload Workflow

### Manual Test Upload

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login with token
huggingface-cli login
# Paste your token when prompted

# Test upload
python src/upload_to_huggingface.py \
  --generator checkpoints/tinydit_final.pt \
  --onnx-generator frontend/public/models/generator_quantized.onnx \
  --onnx-classifier frontend/public/models/cats_quantized.onnx \
  --repo-id d4oit/tiny-cats-model
```

### Verify Upload

```bash
# List files in repository
huggingface-cli list-files d4oit/tiny-cats-model

# Or visit in browser
# https://huggingface.co/d4oit/tiny-cats-model/tree/main
```

## Automated Upload Workflow

Once configured, the upload happens automatically after training:

```yaml
# .github/workflows/upload-hub.yml
name: Upload to HuggingFace Hub

on:
  workflow_run:
    workflows: ["Train"]
    types: [completed]
    branches: [main]

jobs:
  upload:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    steps:
      - uses: actions/checkout@v4
      
      - name: Upload to HuggingFace
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          python src/upload_to_huggingface.py \
            --generator checkpoints/tinydit_final.pt \
            --repo-id d4oit/tiny-cats-model
```

## Security Best Practices

### 1. Token Permissions

Use **minimum required permissions**:
- ✅ `read` - Read public repos
- ✅ `write` - Write to owned repos
- ❌ `delete` - Not needed
- ❌ `admin` - Not needed

### 2. Token Rotation

Rotate tokens every 90 days:

```bash
# 1. Generate new token at https://huggingface.co/settings/tokens

# 2. Update GitHub secret
gh secret set HF_TOKEN --body "hf_new_token_here"

# 3. Revoke old token at https://huggingface.co/settings/tokens
```

### 3. Secret Access Control

Limit which workflows can access the secret:

```yaml
# Only specific workflows can access secrets
permissions:
  contents: read
  # No write permissions for untrusted code
```

### 4. Never Log Tokens

The upload script never logs token values:
```python
# ✅ Good
logger.info("Uploading to HuggingFace...")

# ❌ Bad - never do this
logger.info(f"Using token: {token}")
```

## Troubleshooting

### Issue: "403 Forbidden"

**Cause:** Invalid or expired token

**Solution:**
1. Generate new token at https://huggingface.co/settings/tokens
2. Update GitHub secret: `gh secret set HF_TOKEN`
3. Re-run workflow

### Issue: "401 Unauthorized"

**Cause:** Token doesn't have write permissions

**Solution:**
1. Go to https://huggingface.co/settings/tokens
2. Check token permissions
3. Ensure `write` access is enabled
4. Generate new token if needed

### Issue: "Repository not found"

**Cause:** Repository doesn't exist or wrong name

**Solution:**
1. Create repository at https://huggingface.co/new
2. Check repo_id in upload command
3. Verify you have write access to the repo

### Issue: Secret not accessible in workflow

**Cause:** Secret not configured or workflow restrictions

**Solution:**
```bash
# Verify secret exists
gh secret list

# Check workflow has correct permissions
cat .github/workflows/upload-hub.yml
```

## Environment Variables

The upload script uses these environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | HuggingFace API token |
| `HF_HOME` | No | Custom cache directory |

## Related Documentation

- [ADR-039: Automated HuggingFace CI Upload](../plans/ADR-039-automated-huggingface-ci-upload.md)
- [Tutorial Notebook 03: Training & Fine-Tuning](../notebooks/03_training_fine_tuning.ipynb)
- [HuggingFace Security Docs](https://huggingface.co/docs/hub/security)

## Quick Reference

```bash
# Generate token
# https://huggingface.co/settings/tokens

# Add to GitHub
gh secret set HF_TOKEN --body "hf_xxxxx"

# Verify
gh secret list

# Test upload
python src/upload_to_huggingface.py --repo-id username/repo

# View uploaded model
# https://huggingface.co/username/repo
```

## Support

For issues with HuggingFace tokens:
- **HuggingFace Forum:** https://discuss.huggingface.co/
- **HuggingFace Docs:** https://huggingface.co/docs
- **GitHub Issues:** https://github.com/d-oit/tiny-cats-model/issues
