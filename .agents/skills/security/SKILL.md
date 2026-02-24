---
name: security
description: Use for secrets management, credentials handling, and security best practices.
triggers:
  - "security"
  - "secrets"
  - "credentials"
  - "api key"
  - "token"
  - "env var"
---

# Skill: security

This skill covers security best practices for handling secrets and credentials.

## Environment Variables

```bash
# Set environment variable (temporary)
export MODAL_TOKEN_ID=your_token_id

# Set in .bashrc/.zshrc for persistence (NOT recommended)
# Instead use .env file (gitignored)

# Load from .env
source .env  # Only if .env is gitignored!
```

## GitHub Secrets

```bash
# List secrets (shows only names, not values)
gh secret list

# Set a new secret
gh secret set MODAL_TOKEN_ID --body "your_token_value"

# Delete a secret
gh secret delete MODAL_TOKEN_ID

# Set secret for specific environment
gh secret set API_KEY --env production --body "value"
```

## GitHub UI Path

1. Go to **Repository** → **Settings**
2. Click **Secrets and variables** → **Actions**
3. Click **New repository secret**

## Secrets in CI/CD

```yaml
# In workflow file, access via:
env:
  MODAL_TOKEN_ID: ${{ secrets.MODAL_TOKEN_ID }}
```

## What NOT to Do

| Action | Risk | Alternative |
|--------|------|-------------|
| Hardcode tokens | Exposed in repo | Use env vars |
| Commit .env | Secret leak | Add to .gitignore |
| Log secrets | Visible in CI | Use masking: `::add-mask::value` |
| Use --body with real token | Shell history | Use file: `--body "$(cat token.txt)"` |

## .gitignore Checklist

Ensure these are ignored:

```
.env
*.pem
*.key
credentials.json
token.txt
secrets/
```

## Pre-commit Security

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    hooks:
      - id: check-json
      - id: check-yaml
      - id: detect-secrets
```

## Security Scanning

```bash
# Scan for secrets in repo
git-secrets --scan

# Scan commits
git-secrets --scan-commits

# Install pre-commit hook
git-secrets --install
```

## Modal Credentials

```bash
# Set for Modal CLI globally (recommended)
modal token set

# Verify
modal token status
```

## Key Rules

1. **Never commit secrets** - Always use .gitignore
2. **Use GitHub secrets** - For CI/CD automation
3. **Use Modal global config** - `modal token set` for local
4. **Rotate tokens regularly** - Especially if compromised
5. **Mask in logs** - Use `::add-mask::` in GitHub Actions
