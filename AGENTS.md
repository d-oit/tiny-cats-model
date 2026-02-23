# tiny-cats-model AGENTS.md

AI agent guidance for the tiny-cats-model project. This file provides explicit instructions for AI coding agents.

---

## Quick Commands

| Task | Command |
|------|---------|
| Install | `pip install -r requirements.txt` |
| Download data | `bash data/download.sh` |
| Train | `python src/train.py data/cats` |
| Evaluate | `python src/eval.py` |
| Verify | `bash .agents/skills/git-workflow/quality-gate.sh` |
|--------|---------------------------------------------------|
| Tests | `pytest tests/ -v` |
| Lint | `ruff check . && flake8 .` |
| Format | `black .` |

---

## Code Style

- **PEP 8** - Python standard
- **Line length**: 88 chars (black default)
- **Linting**: `ruff check . --fix`
- **Formatting**: `black .`
- **Type hints**: Required for new code

### What NOT to Do

- Never hardcode tokens or secrets
- Never commit `.env` files
- Never skip CI checks before merging
- Never use `--force` with git push to main
- Never commit without running lint/test locally

---

## Agent Skills

Load skills using the `@skill` command when the task matches:

| Skill | When to Use | Triggers |
|-------|-------------|----------|
| `cli-usage` | Training, evaluation, dataset | "train", "evaluate", "download data" |
| `testing-workflow` | Running tests, verification | "test", "verify", "CI" |
| `code-quality` | Linting, formatting | "lint", "format", "style" |
| `gh-actions` | CI/CD, workflows | "CI", "GitHub Actions", "workflow" |
| `git-workflow` | Branches, commits, PRs | "commit", "branch", "PR", "quality gate", "pre-commit" |
| `goap` | Planning, ADR, project goals | "plan", "GOAP", "ADR", "action item", "priority" |
| `security` | Secrets, credentials | "secret", "token", "credential" |
| `model-training` | GPU training, Modal | "GPU", "Modal", "hyperparameter" |

---

## CI/CD

- **Workflows**: `.github/workflows/ci.yml`, `train.yml`, `deploy.yml`
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

## Complete CI/CD Fix Workflow

After every commit, follow this atomic loop:

```
1. git commit → git push
2. gh run list → get run-id
3. gh run view <id> → identify failures
4. FOR EACH failure:
   a. Analyze error type → determine skill needed
   b. Spawn specialist agent with @skill
   c. Agent fixes → commits → pushes
   d. Repeat from step 2 until all pass
5. NEVER skip: each fix must go through full cycle
```

### Specialist Agent Selection

| Failure Type | Use Skill |
|--------------|-----------|
| Lint error | `@skill code-quality` |
| Test failure | `@skill testing-workflow` |
| Type error | `@skill code-quality` |
| CI/workflow | `@skill gh-actions` |
| Model/training | `@skill model-training` |
| Security | `@skill security` |

### 2026 Best Practices

Before fixing issues:
1. Run `websearch` for latest solutions on similar issues
2. Run `codesearch` for API/pattern examples
3. Document significant decisions in `plans/ADR-*.md`
4. Update `plans/GOAP.md` with action items

### Example Fix Cycle

```bash
# After push fails:
gh run list --repo owner/tiny-cats-model
gh run view <run-id>  # See failures

# Spawn specialist:
# (Agent uses code-quality skill to fix lint)

# Re-push and re-check:
gh run view <run-id>  # Check again

# Loop until SUCCESS
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

check 

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
- Always use type hints for new functions
