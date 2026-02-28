# Security Guidelines

## Rules

- **Never hardcode tokens** - Use env vars or global config
- **Never commit `.env` files** - Already gitignored
- **Never commit secrets** - Use GitHub secrets or Modal global config

## Modal Tokens (Modal 1.0+)

Configured globally via Modal CLI:

```bash
modal token new  # Not 'token set'
modal token info
```

## GitHub Secrets

For CI/CD, use GitHub Secrets:

```bash
# Settings → Secrets → Actions
gh secret set SECRET_NAME --body "value"
```

## Required Secrets

| Secret | Purpose |
|--------|---------|
| `MODAL_TOKEN_ID` | Modal authentication |
| `MODAL_TOKEN_SECRET` | Modal authentication |
