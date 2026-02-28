# AGENTS.md

AI agent guidance for tiny-cats-model (cat image classification with DiT).

## Quick Commands

```bash
# Install & setup
pip install -r requirements.txt && bash data/download.sh

# Training (Modal GPU)
modal run src/train_dit.py data/cats --steps 300000

# Local testing
python src/train_dit.py data/cats --steps 100 --batch-size 8

# Quality gate
bash scripts/quality-gate.sh
```

## Authentication

### Modal 1.0+ (IMPORTANT: uses `token new`)
```bash
modal token new          # Configure (not token set)
modal token info         # Verify
modal run src/train_dit.py data/cats --steps 300000
```

### HuggingFace
```bash
export HF_TOKEN=hf_xxx
python src/upload_to_huggingface.py --repo-id d4oit/tiny-cats-model
```

## Code Style
- Ruff linting + formatting (88 char line)
- Type hints required
- 500 LOC max per file

## Testing
```bash
pytest tests/ -v
npx playwright test
bash scripts/quality-gate.sh
```

## Security
- Never commit tokens/secrets
- Use environment variables

## PR Rules
- Run quality gate before commit
- Never merge if CI fails

## Extended Docs
- [Training](agents-docs/training.md)
- [CI/CD](agents-docs/ci-cd.md)
- [Security](agents-docs/security.md)
- [Skills](agents-docs/skills.md)
- [Learnings](agents-docs/learnings.md)
- [Auth Troubleshooting](agents-docs/auth-troubleshooting.md)

# Token Optimization Rules
Never run raw testing, linting, or building commands directly in the terminal. Always use the provided scripts to truncate verbose output and save tokens.

- For build/test commands: `bash .agents/skills/token_safe_exec.sh "<command>"`
- For lint/format analysis: `python .agents/skills/smart_lint.py "<command>"`
