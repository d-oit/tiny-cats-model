#!/usr/bin/env bash
# .agents/skills/git-workflow/quality-gate.sh
# Quality gate script to run before git commit.
# Usage: bash .agents/skills/git-workflow/quality-gate.sh
# Or use as pre-commit hook: see "Pre-commit Hook" section below.
# Exit code 0 = all checks passed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../"; pwd)"
cd "${ROOT}"

PASS=0
FAIL=0
ERRORS=()

run_check() {
    local name="$1"
    shift
    echo "==> Running: ${name}"
    if "$@"; then
        echo "    PASS: ${name}"
        PASS=$((PASS+1))
    else
        echo "    FAIL: ${name}"
        FAIL=$((FAIL+1))
        ERRORS+=("${name}")
    fi
    echo ""
}

echo "========================================"
echo " tiny-cats-model: Quality Gate"
echo "========================================"
echo ""

# --- Lint ---
run_check "ruff lint" python -m ruff check . --exclude node_modules,.agents,.qwen
run_check "flake8 lint" python -m flake8 . --exclude node_modules,.git,.agents,.qwen --max-line-length=88 --extend-ignore=E203,W503,E402,E501
run_check "black format check" python -m black --check .

# --- Type Check ---
run_check "mypy type check" python -m mypy . --exclude node_modules,.agents,.qwen --ignore-missing-imports

# --- Tests ---
run_check "pytest unit tests" python -m pytest tests/ -v --tb=short

# --- Summary ---
echo "========================================"
echo " Results: ${PASS} passed, ${FAIL} failed"
if [[ ${FAIL} -gt 0 ]]; then
    echo " Failed checks:"
    for err in "${ERRORS[@]}"; do
        echo "   - ${err}"
    done
    echo "========================================"
    echo ""
    echo "Fix failures before committing."
    exit 1
else
    echo " All checks passed!"
    echo "========================================"
    exit 0
fi
