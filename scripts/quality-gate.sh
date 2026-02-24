#!/usr/bin/env bash
# scripts/quality-gate.sh
# Full quality gate: format, lint, test.
#
# Usage:
#   bash scripts/quality-gate.sh [--strict]
#
# Options:
#   --strict  Exit on first failure (default: run all checks)
#
# Returns:
#   0 if all checks pass, 1 otherwise

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track failures
FAILURES=0
STRICT=false

# Parse arguments
if [[ "${1:-}" == "--strict" ]]; then
    STRICT=true
    set -e  # Exit on first failure
fi

# Helper functions
log_info() {
    echo -e "${BLUE}▶ $1${NC}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}"
}

section_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
}

# Change to root directory
cd "$ROOT_DIR"

section_header "Quality Gate"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Format Check (Ruff Format - Black-compatible)
# ─────────────────────────────────────────────────────────────────────────────
log_info "Checking code formatting (ruff format)..."

if python -m ruff format --check . 2>/dev/null; then
    log_success "Format check passed"
else
    log_error "Format check failed"
    echo "   Run 'ruff format .' to fix formatting issues"
    FAILURES=$((FAILURES + 1))
    if [[ "$STRICT" == true ]]; then
        exit 1
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Lint (Ruff - replaces flake8 + isort + pydocstyle)
# ─────────────────────────────────────────────────────────────────────────────
log_info "Running linter (ruff)..."

if RUFF_OUTPUT=$(python -m ruff check . 2>&1); then
    log_success "Ruff check passed"
else
    log_error "Ruff check failed"
    echo "$RUFF_OUTPUT" | head -20
    if echo "$RUFF_OUTPUT" | grep -q "error:"; then
        echo "   Run 'ruff check . --fix' to fix auto-fixable issues"
    fi
    FAILURES=$((FAILURES + 1))
    if [[ "$STRICT" == true ]]; then
        exit 1
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5. Type Check (Mypy)
# ─────────────────────────────────────────────────────────────────────────────
log_info "Running type checker (mypy)..."

if MYPY_OUTPUT=$(python -m mypy . --ignore-missing-imports 2>&1); then
    log_success "Type check passed"
else
    log_error "Type check failed"
    echo "$MYPY_OUTPUT" | head -20
    echo "   Fix type errors or add '# type: ignore' with comment"
    FAILURES=$((FAILURES + 1))
    if [[ "$STRICT" == true ]]; then
        exit 1
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. Tests (Pytest)
# ─────────────────────────────────────────────────────────────────────────────
log_info "Running tests (pytest)..."

if PYTEST_OUTPUT=$(python -m pytest tests/ -v --tb=short 2>&1); then
    log_success "All tests passed"
else
    log_error "Tests failed"
    echo "$PYTEST_OUTPUT" | tail -30
    echo "   Fix failing tests before committing"
    FAILURES=$((FAILURES + 1))
    if [[ "$STRICT" == true ]]; then
        exit 1
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
section_header "Summary"

if [[ $FAILURES -eq 0 ]]; then
    log_success "All quality checks passed!"
    echo ""
    echo "Ready to commit. ✓"
    exit 0
else
    log_error "$FAILURES check(s) failed"
    echo ""
    echo "Fix the issues above before committing."
    echo ""
    echo "Quick fixes:"
    echo "  ruff format .                              # Fix formatting"
    echo "  ruff check . --fix                        # Fix lint issues"
    echo "  pytest tests/ -v                          # Run tests"
    echo ""
    exit 1
fi
