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
# 3. GitHub Actions YAML Validation
# ─────────────────────────────────────────────────────────────────────────────
log_info "Validating GitHub Actions workflows (actionlint)..."

WORKFLOW_DIR="$ROOT_DIR/.github/workflows"
if [[ -d "$WORKFLOW_DIR" ]]; then
    # Check if actionlint is available
    if command -v actionlint &> /dev/null; then
        if ACTIONLINT_OUTPUT=$(actionlint "$WORKFLOW_DIR"/*.yml 2>&1); then
            log_success "Workflow validation passed (actionlint)"
        else
            log_error "Workflow validation failed (actionlint)"
            echo "$ACTIONLINT_OUTPUT" | head -30
            echo "   Install: npm install -g actionlint"
            echo "   Fix workflow syntax/semantics in .github/workflows/"
            FAILURES=$((FAILURES + 1))
            if [[ "$STRICT" == true ]]; then
                exit 1
            fi
        fi
    else
        log_warning "actionlint not installed, skipping workflow validation"
        echo "   Install: npm install -g actionlint"
    fi
else
    log_warning "No workflows directory found, skipping workflow validation"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. YAML Linting (yamllint)
# ─────────────────────────────────────────────────────────────────────────────
log_info "Linting YAML files (yamllint)..."

# Check if yamllint is available
if command -v yamllint &> /dev/null; then
    # Lint GitHub Actions workflows
    if [[ -d "$WORKFLOW_DIR" ]]; then
        if YAMLLINT_OUTPUT=$(yamllint "$WORKFLOW_DIR" 2>&1); then
            log_success "YAML lint passed (yamllint)"
        else
            log_error "YAML lint failed (yamllint)"
            echo "$YAMLLINT_OUTPUT" | head -20
            echo "   Install: pip install yamllint"
            echo "   Config: .yamllint"
            FAILURES=$((FAILURES + 1))
            if [[ "$STRICT" == true ]]; then
                exit 1
            fi
        fi
    fi
else
    log_warning "yamllint not installed, skipping YAML linting"
    echo "   Install: pip install yamllint"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5. Agent Skills Validation
# ─────────────────────────────────────────────────────────────────────────────
log_info "Validating agent skills (agentskills.io specification)..."

SKILLS_VALIDATION_SCRIPT="$SCRIPT_DIR/validate-skills.py"
if [[ -f "$SKILLS_VALIDATION_SCRIPT" ]]; then
    if SKILLS_OUTPUT=$(python "$SKILLS_VALIDATION_SCRIPT" 2>&1); then
        echo "$SKILLS_OUTPUT" | grep -E "^✓|^▶" | head -15
        log_success "Skills validation passed"
    else
        log_error "Skills validation failed"
        echo "$SKILLS_OUTPUT" | head -30
        echo "   See: https://agentskills.io/specification#validation"
        echo "   Fix SKILL.md frontmatter in .agents/skills/"
        FAILURES=$((FAILURES + 1))
        if [[ "$STRICT" == true ]]; then
            exit 1
        fi
    fi
else
    log_warning "Skills validation script not found, skipping skills validation"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. Type Check (Mypy)
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
# 7. Tests (Pytest)
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
