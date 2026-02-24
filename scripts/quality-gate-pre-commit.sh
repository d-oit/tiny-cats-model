#!/usr/bin/env bash
# scripts/quality-gate-pre-commit.sh
#
# Strict pre-commit quality gate that blocks commits on failure.
# Runs format, lint, and test checks.
#
# Usage (in .git/hooks/pre-commit):
#   bash scripts/quality-gate-pre-commit.sh
#
# Or add to Makefile:
#   pre-commit: quality-gate-pre-commit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

# Only run on relevant file changes
CHANGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(py|pyi)$' || true)

if [[ -z "$CHANGED_FILES" ]]; then
    # No Python files staged, skip checks
    exit 0
fi

echo "Running quality gate on staged Python files..."

# Run quality gate in strict mode (exit on first failure)
bash "$SCRIPT_DIR/quality-gate.sh" --strict
