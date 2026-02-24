#!/usr/bin/env bash
# scripts/install-hooks.sh
# Install git hooks for documentation automation.
#
# Usage:
#   bash scripts/install-hooks.sh
#
# This script:
# 1. Creates .git/hooks directory if needed
# 2. Installs pre-commit hook for doc checks
# 3. Installs commit-msg hook for learning capture
# 4. Makes hooks executable

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
HOOKS_DIR="$ROOT_DIR/.git/hooks"

echo "==> Installing git hooks..."

# Ensure hooks directory exists
mkdir -p "$HOOKS_DIR"

# Install pre-commit hook
cat > "$HOOKS_DIR/pre-commit" << 'EOF'
#!/usr/bin/env bash
# Pre-commit hook - checks if docs need updating

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

python3 "$ROOT_DIR/scripts/pre-commit-docs.py"
exit $?
EOF

chmod +x "$HOOKS_DIR/pre-commit"
echo "   ✓ Installed pre-commit hook"

# Install commit-msg hook
cat > "$HOOKS_DIR/commit-msg" << 'EOF'
#!/usr/bin/env bash
# Commit-msg hook - captures learnings from commits

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

python3 "$ROOT_DIR/scripts/commit-msg-hook.py" "$1"
exit $?
EOF

chmod +x "$HOOKS_DIR/commit-msg"
echo "   ✓ Installed commit-msg hook"

# Install quality-gate pre-commit hook (optional, more strict)
cat > "$HOOKS_DIR/pre-commit-quality" << 'EOF'
#!/usr/bin/env bash
# Strict pre-commit hook - runs full quality gate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

bash "$ROOT_DIR/scripts/quality-gate-pre-commit.sh"
exit $?
EOF

chmod +x "$HOOKS_DIR/pre-commit-quality"
echo "   ✓ Installed pre-commit-quality hook (optional)"
echo "     To enable: mv $HOOKS_DIR/pre-commit $HOOKS_DIR/pre-commit-docs"
echo "                mv $HOOKS_DIR/pre-commit-quality $HOOKS_DIR/pre-commit"

# Make all scripts executable
chmod +x "$SCRIPT_DIR"/*.py

echo ""
echo "==> Git hooks installed successfully!"
echo ""
echo "Installed hooks:"
echo "  - pre-commit: Checks if documentation needs updating"
echo "  - commit-msg: Analyzes commits for learning opportunities"
echo ""
echo "Available scripts:"
echo "  - python scripts/update-learnings.py  → Auto-update learnings.md"
echo "  - python scripts/adr-scaffold.py      → Create new ADR"
echo "  - python scripts/update-goap.py       → Update GOAP action items"
echo ""
