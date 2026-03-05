---
name: token-reducer
description: Token-aware wrappers around build, test, lint, and format commands to prevent context window overflow. Centralizes command execution through scripts that truncate noisy logs while preserving actionable error information.
---

# Token Reducer Skill

## Overview

This skill provides token-aware wrappers around build, test, lint, and format commands to prevent context window overflow in Claude Code and similar agents. It centralizes command execution through scripts that truncate noisy logs while preserving actionable error information.

## Files

- `../token_safe_exec.sh` - Bash wrapper for any build/test command
- `../smart_lint.py` - Python wrapper for lint/format commands with error-focused filtering

## Usage

### Build / Test

Instead of running raw commands like:
- `cargo test`
- `npm test`
- `pytest`
- `go test ./...`

Always run:

```bash
bash .agents/skills/token_safe_exec.sh "<your_command_here>"
```

Examples:

```bash
bash .agents/skills/token_safe_exec.sh "cargo test --all --quiet"
bash .agents/skills/token_safe_exec.sh "npm test"
bash .agents/skills/token_safe_exec.sh "pytest -q"
```

### Lint / Format

Instead of:

- `eslint .`
- `flake8 .`
- `ruff check .`
- `black .`

Use:

```bash
python .agents/skills/smart_lint.py "<your_command_here>"
```

Examples:

```bash
python .agents/skills/smart_lint.py "eslint . --max-warnings=0"
python .agents/skills/smart_lint.py "ruff check ."
python .agents/skills/smart_lint.py "flake8 ."
```

## Agent Instructions

Agents working in this repository must follow these rules:

1. Never call raw `cargo test`, `npm test`, `pytest`, `go test`, or similar commands directly.
2. Never call raw lint/format commands directly (e.g., `eslint .`, `ruff check .`, `flake8 .`, `black .`).
3. Always route:
   - Build/test commands through `bash .agents/skills/token_safe_exec.sh "<command>"`.
   - Lint/format commands through `python .agents/skills/smart_lint.py "<command>"`.
4. If more context from a test or lint run is needed, the agent may adjust the filtering logic in these scripts instead of bypassing them.