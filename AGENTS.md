# Project: tiny-cats-model

This file follows the [AGENTS.md](https://agents.md/) standard for AI agent instructions.
AI agents (Claude Code, Codex, OpenCode, Copilot, etc.) **must read this file first** before taking any action in this repository.

---

## Install & Setup

```bash
pip install -r requirements.txt
bash data/download.sh
```

Dataset will be prepared under `data/cats/`.

---

## Training

```bash
# Local CPU/GPU training
python src/train.py data/cats

# GPU training via Modal
export MODAL_TOKEN_ID=<your_token_id>
export MODAL_TOKEN_SECRET=<your_token_secret>
modal run src/train.py
```

Model checkpoint is saved to `cats_model.pt`.

---

## Evaluation

```bash
python src/eval.py
```

---

## Code Style

- Python code follows **PEP 8**
- Use `ruff` for linting: `ruff check .`
- Use `black` for formatting: `black .`
- Max line length: **88** (black default)
- Type hints are encouraged but not enforced

---

## Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run lint check
flake8 .
```

All tests must pass before committing.

---

## CI

- GitHub Actions workflow: `.github/workflows/train.yml`
- CI runs on every push and pull_request to `main`
- CI steps: lint → test → (optional) training proof
- Do **not** merge if CI fails

---

## CLI

Use the following standard commands:

| Task | Command |
|------|---------|
| Install deps | `pip install -r requirements.txt` |
| Download data | `bash data/download.sh` |
| Train model | `python src/train.py data/cats` |
| Evaluate | `python src/eval.py` |
| Run tests | `pytest tests/` |
| Lint | `ruff check . && flake8 .` |
| Format | `black .` |

---

## Agent Skills

Reusable agent skills are located in `.agents/skills/`. Available skills:

- `.agents/skills/testing-workflow/` — verifying CI, tests, and training integration
- `.agents/skills/gh-actions/` — interacting with CI/CD automation
- `.agents/skills/cli-usage/` — invoking training and evaluation via CLI

---

## Secrets & Security

- **Never** hardcode tokens or credentials in code
- Use environment variables: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`
- Store GitHub secrets via: **Settings → Secrets and variables → Actions**
- Never commit `.env` files or credential files

---

## File Structure

```
tiny-cats-model/
├── data/
│   ├── cats/              # dataset (gitignored)
│   └── download.sh        # dataset download/prepare script
├── src/
│   ├── train.py           # training entrypoint
│   ├── eval.py            # evaluation script
│   ├── dataset.py         # DataLoader factory
│   └── model.py           # model definition
├── tests/
│   └── test_dataset.py
├── .agents/skills/
├── .github/workflows/train.yml
├── AGENTS.md             # this file
├── modal.yml
├── requirements.txt
└── README.md
```

---

## Notes for Agents

- Always run `pytest tests/` after any code change
- Never modify `modal.yml` token fields with real values
- Dataset folder `data/cats/` is gitignored — agents should not try to commit its contents
- The model checkpoint `cats_model.pt` is also gitignored
