# Scripts

Automation scripts for documentation and development workflows.

## Installation

```bash
# Install git hooks
bash scripts/install-hooks.sh

# Enable strict quality gate (blocks commits on failure)
mv .git/hooks/pre-commit .git/hooks/pre-commit-docs
mv .git/hooks/pre-commit-quality .git/hooks/pre-commit
```

## Available Scripts

### Quality Gate

| Script | Purpose | Usage |
|--------|---------|-------|
| `quality-gate.sh` | Full quality check (format, lint, test) | `bash scripts/quality-gate.sh [--strict]` |
| `quality-gate-pre-commit.sh` | Pre-commit quality gate | Used by git hook |

### Documentation Automation

| Script | Purpose | Usage |
|--------|---------|-------|
| `update-learnings.py` | Auto-update learnings.md from commits | `python scripts/update-learnings.py [--dry-run]` |
| `adr-scaffold.py` | Create new Architecture Decision Records | `python scripts/adr-scaffold.py "Title"` |
| `update-goap.py` | Update GOAP.md with action items | `python scripts/update-goap.py [--dry-run]` |

### Git Hooks (installed via install-hooks.sh)

| Hook | Trigger | Purpose |
|------|---------|---------|
| `pre-commit` | Before commit | Warn if docs need updating |
| `commit-msg` | After commit message | Suggest learning captures |

## Git Hooks

### Pre-commit Hook

Checks staged files and warns if documentation updates might be needed:

- CI workflow changes → suggests ci-cd.md or ADR update
- Modal config changes → suggests training.md update
- Security changes → suggests security.md update
- Agent skill changes → suggests skills.md update

**Does not block commits** - only provides warnings.

### Commit-msg Hook

Analyzes commit messages and suggests documentation updates:

- CI/CD fixes → suggests learnings.md documentation
- Modal/training changes → suggests training.md update
- Bug fixes with error handling → suggests pattern documentation
- New features → suggests AGENTS.md update

**Does not block commits** - only provides suggestions.

## Manual Workflows

### Before Committing (Recommended)

```bash
# Run full quality gate
bash scripts/quality-gate.sh

# Or use Makefile
make lint
make test
```

### After a significant fix

```bash
# 1. Analyze what should be documented
python scripts/update-learnings.py --dry-run

# 2. If CI/CD fix, consider ADR
python scripts/adr-scaffold.py "Fix CI timeout issue"

# 3. Update GOAP with any new action items
python scripts/update-goap.py --dry-run

# 4. Apply updates
python scripts/update-learnings.py
python scripts/update-goap.py
```

### Creating an ADR

```bash
# Create scaffold
python scripts/adr-scaffold.py "Add new feature X"

# Edit the file
vim plans/ADR-008-add-new-feature-x.md

# Commit with reference
git add plans/ADR-008-*
git commit -m "docs: Add ADR-008 for feature X"
```

## CI/CD Integration

Add to your CI workflow to ensure docs stay current:

```yaml
- name: Check documentation
  run: |
    python scripts/update-learnings.py --dry-run
    # Review output in logs
```

## Best Practices

1. **Run update-learnings.py** after fixing non-trivial bugs
2. **Create ADRs** for architectural decisions
3. **Update GOAP.md** when new action items emerge
4. **Review hook output** - it provides helpful suggestions
5. **Don't ignore warnings** - they help maintain documentation quality
