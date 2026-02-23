#!/usr/bin/env bash
# .agents/skills/testing-workflow/verify.sh
# Run the full CI verification suite locally.
# Usage: bash .agents/skills/testing-workflow/verify.sh
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

echo "======================================"
echo " tiny-cats-model: Verification Suite"
echo "======================================"
echo ""

# --- Lint ---
run_check "ruff lint" ruff check .
run_check "flake8 lint" flake8 . --max-line-length=88 --extend-ignore=E203,W503
run_check "black format check" black --check .

# --- Tests ---
run_check "pytest unit tests" pytest tests/ -v --tb=short

# --- Summary ---
echo "======================================"
echo " Results: ${PASS} passed, ${FAIL} failed"
if [[ ${FAIL} -gt 0 ]]; then
    echo " Failed checks:"
    for err in "${ERRORS[@]}"; do
        echo "   - ${err}"
    done
    echo "======================================"
    exit 1
else
    echo " All checks passed!"
    echo "======================================"
    exit 0
fi
