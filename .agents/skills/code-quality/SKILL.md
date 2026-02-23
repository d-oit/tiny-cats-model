---
name: code-quality
description: Use for linting, formatting, and type checking with ruff, black, flake8, mypy.
---

# Skill: code-quality

This skill covers all code quality tools for Python projects.

## Linting (ruff)

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check . --fix

# Fix specific rules
ruff check . --fix --select F,E

# Show rule explanations
ruff rule E501
```

## Formatting (black)

```bash
# Format code
black .

# Check without modifying
black --check .

# Set line length
black --line-length=88 .

# Exclude directories
black --exclude "tests/" .
```

## Additional Linting (flake8)

```bash
# Run flake8
flake8 . --max-line-length=88

# Ignore specific rules
flake8 . --extend-ignore=E203,W503

# Exclude files
flake8 . --exclude=__pycache__,.git
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

## Import Sorting (isort)

```bash
# Check import order
isort --check-only .

# Fix imports
isort .
```

## Full Quality Check

```bash
# Run all checks in order
ruff check . --fix
black .
flake8 . --max-line-length=88
mypy . --ignore-missing-imports
isort --check-only .
```

## Common Issues

| Issue | Fix |
|-------|-----|
| E501 line too long | Wrap lines or use `black .` |
| F401 imported but unused | `ruff check . --fix` |
| F811 redefinition | Remove duplicate imports |
| I001 isort order | `isort .` |
| mypy: no type hints | Add type hints to functions |

## Configuration Files

- `pyproject.toml` - ruff, black, isort settings
- `.ruff.toml` - ruff-specific config
- `mypy.ini` - mypy configuration
- `.flake8` - flake8 settings

## Best Practices

1. **Run locally before push** - Always verify locally
2. **Use pre-commit hooks** - Add to `.pre-commit-config.yaml`
3. **Fix incrementally** - Don't ignore warnings
4. **Type hints encouraged** - Improves IDE support
5. **Line length 88** - Black default, matches PEP 8
