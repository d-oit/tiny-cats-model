---
name: code-quality
description: Use for linting, formatting, and type checking with ruff and mypy.
---

# Skill: code-quality

This skill covers all code quality tools for this project.

## Linting & Formatting (ruff)

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check . --fix

# Fix specific rules
ruff check . --fix --select F,E

# Show rule explanations
ruff rule E501

# Format code
ruff format .

# Check without modifying
ruff format --check .
```

## Type Checking (mypy)

```bash
# Run mypy
mypy .

# Ignore missing imports
mypy . --ignore-missing-imports

# Exclude directories
mypy . --exclude tests/

# Strict mode
mypy . --strict
```

## Full Quality Check

```bash
# Run all checks in order
ruff format --check .
ruff check .
mypy . --ignore-missing-imports

# Or use the quality gate (includes all checks + pytest + skill verification)
bash scripts/quality-gate.sh
```

## Common Issues

| Issue | Fix |
|-------|-----|
| E501 line too long | `ruff format .` |
| F401 imported but unused | `ruff check . --fix` |
| F811 redefinition | Remove duplicate imports |
| I001 import order | `ruff check . --fix` |
| mypy: no type hints | Add type hints to functions |

## Configuration Files

- `ruff.toml` — ruff format + lint config
- `pyproject.toml` — mypy and project metadata

## Best Practices

1. **Run locally before push** — Always verify locally
2. **Use quality gate** — `bash scripts/quality-gate.sh` runs everything
3. **Fix incrementally** — Don't ignore warnings
4. **Type hints required** — All functions must have type hints
5. **Line length 88** — Ruff default, matches PEP 8
