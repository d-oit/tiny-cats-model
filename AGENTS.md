# tiny-cats-model

Image classification project for cats vs other objects using PyTorch.

---

## Quick Commands

| Task | Command |
|------|---------|
| Install | `pip install -r requirements.txt` |
| Download data | `bash data/download.sh` |
| Train | `python src/train.py data/cats` |
| Evaluate | `python src/eval.py` |
| Tests | `pytest tests/ -v` |
| Lint | `ruff check . && flake8 .` |
| Format | `black .` |
| Verify | `bash .agents/skills/testing-workflow/verify.sh` |

---

## Code Style

- **PEP 8** - Python standard
- **Line length**: 88 chars (black default)
- **Linting**: `ruff check . --fix`
- **Formatting**: `black .`
- **Type hints**: Encouraged

---

## Agent Skills

Location: `.agents/skills/`

| Skill | Purpose |
|-------|---------|
| `cli-usage` | Training, evaluation, dataset commands |
| `testing-workflow` | Run tests, lint, verification |
| `gh-actions` | CI/CD status, triggers, debugging |
| `git-workflow` | Branch management, commits, PRs |
| `code-quality` | Linting, formatting, type checking |
| `security` | Secrets, credentials, safe practices |
| `model-training` | Training, hyperparameters, Modal |

---

## CI/CD

- **Workflows**: `.github/workflows/ci.yml`, `train.yml`
- **Trigger**: push + PR to `main`
- **Jobs**: lint → test → type-check (parallel)
- **Never merge if CI fails**

### GitHub CLI Commands

```bash
# View CI status
gh run list --repo owner/tiny-cats-model
gh run view <run-id>

# Re-run failed
gh run rerun <run-id> --failed

# Trigger workflow
gh workflow run train.yml

# Set secrets
gh secret set MODAL_TOKEN_ID --body "xxx"
```

---

## Security

- **Never hardcode tokens** - Use env vars
- **Never commit `.env`** - Already gitignored
- **Use GitHub secrets** - Settings → Secrets → Actions
- **Required env vars**: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`

---

## File Structure

```
tiny-cats-model/
├── src/           # train.py, eval.py, model.py, dataset.py
├── tests/         # test_dataset.py
├── data/cats/     # dataset (gitignored)
├── .agents/skills/  # agent automation
├── .github/workflows/
├── plans/         # GOAP, ADR documents
├── AGENTS.md      # this file
└── modal.yml      # GPU training config
```

---

## Modal GPU Training

```bash
export MODAL_TOKEN_ID=xxx
export MODAL_TOKEN_SECRET=xxx
modal run src/train.py
```

---

## Notes

- Run `pytest tests/` after any code change
- Dataset `data/cats/` is gitignored
- Model checkpoint `cats_model.pt` is gitignored
- Max workflow timeout: 10 minutes
