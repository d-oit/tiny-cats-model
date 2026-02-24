# tiny-cats-model AGENTS.md

AI agent guidance for the tiny-cats-model project.

## Quick Commands

| Task | Command |
|------|---------|
| Install | `pip install -r requirements.txt` |
| Download data | `bash data/download.sh` |
| Train (Modal GPU) | `modal run src/train.py` |
| Train (local) | `python src/train.py data/cats` |
| Train options | `modal run src/train.py -- --epochs 20 --batch-size 64` |
| Evaluate | `python src/eval.py` |
| **Quality gate** | `bash scripts/quality-gate.sh` (ruff format + ruff check) |
| **Pre-commit** | `pre-commit install && pre-commit run --all-files` |
| Verify | `bash .agents/skills/git-workflow/quality-gate.sh` |
| Tests | `pytest tests/ -v` |
| Lint | `ruff check . && ruff format --check .` |
| Format | `ruff format .` |
| **CI Monitor** | `gh run list && gh run view <id>` |

## Code Style

- **PEP 8** - Python standard
- **Line length**: 88 chars
- **Linting**: `ruff check . --fix` (Ruff replaces flake8, isort, black)
- **Formatting**: `ruff format .`
- **Type hints**: Required for new code
- **500 LOC max per file**

## What NOT to Do

- Never hardcode tokens or secrets
- Never commit `.env` files
- Never skip CI checks before merging
- Never use `--force` with git push to main
- Never commit without running quality gate locally: `bash scripts/quality-gate.sh`
- Never merge if local quality gate passes but CI fails (report config mismatch)

## Agent Skills

See [agents-docs/skills.md](agents-docs/skills.md) for full details.

| Skill | When to Use |
|-------|-------------|
| `analysis-swarm` | Complex decisions, multi-perspective analysis |
| `cli-usage` | Training, evaluation, dataset |
| `testing-workflow` | Running tests, verification |
| `code-quality` | Linting, formatting |
| `gh-actions` | CI/CD, workflows |
| `git-workflow` | Branches, commits, PRs |
| `goap` | Planning, ADR, project goals |
| `security` | Secrets, credentials |
| `model-training` | GPU training, Modal |
| `web-search-researcher` | Web research, documentation lookup |

## CI/CD

- **Workflows**: `.github/workflows/ci.yml`, `train.yml`, `deploy.yml`
- **Trigger**: push + PR to `main` + `workflow_dispatch`
- **Jobs**: lint → test → type-check → build-frontend
- **Never merge if CI fails**

### Quality Gate Parity (ADR-014)

The local quality gate runs the **same checks** as CI:

```bash
bash scripts/quality-gate.sh
```

This ensures: **what passes locally passes in CI**.

**Configuration Files**:
- `ruff.toml` - Ruff config (single source of truth for linting + formatting)
- `pyproject.toml` - Mypy config

See [agents-docs/ci-cd.md](agents-docs/ci-cd.md) for fix workflows and CLI commands.

## Security

- **Never hardcode tokens** - Use env vars
- **Never commit `.env`** - Already gitignored
- **Modal tokens**: Configured globally via `modal token set`

See [agents-docs/security.md](agents-docs/security.md) for details.

## File Structure

```
tiny-cats-model/
├── src/              # train.py, eval.py, model.py, dataset.py
├── tests/            # test_dataset.py, test_model.py, test_train.py
├── data/cats/        # dataset (gitignored)
├── .agents/skills/   # agent automation
├── .github/workflows/
├── plans/            # GOAP, ADR documents
├── agents-docs/      # Extended agent documentation
├── AGENTS.md         # this file
└── modal.yml         # GPU training config
```

## Modal GPU Training

Modal tokens configured globally via `modal token set`.

```bash
modal run src/train.py
modal run src/train.py -- --epochs 20 --batch-size 64
python src/train.py data/cats  # local CPU
```

See [agents-docs/training.md](agents-docs/training.md) for full options.

## Notes

- Run `bash scripts/quality-gate.sh` before every commit (runs all CI checks locally)
- Dataset `data/cats/` and `cats_model.pt` are gitignored
- Max workflow timeout: 10 minutes
- Always use type hints for new functions
- Ruff line-length: 88 chars (matches CI)

## Learnings & Patterns

See [agents-docs/learnings.md](agents-docs/learnings.md) for self-learning loop, key learnings, and reusable patterns.

## Automation Scripts

| Script | Purpose |
|--------|---------|
| `python scripts/update-learnings.py` | Auto-update learnings.md |
| `python scripts/adr-scaffold.py "Title"` | Create new ADR |
| `python scripts/update-goap.py` | Update GOAP action items |
| `bash scripts/quality-gate.sh` | Run format, lint, test |

Install hooks: `bash scripts/install-hooks.sh`
See [scripts/README.md](scripts/README.md) for details.
