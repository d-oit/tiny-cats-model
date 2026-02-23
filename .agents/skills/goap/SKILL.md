---
name: goap
description: Use for GOAP (Goal-Oriented Action Planning) and ADR (Architecture Decision Records) management. Handles project planning, issue tracking, and architectural decisions.
---

# Skill: goap

This skill manages project planning using GOAP and ADR documents in the `plans/` folder.

## When to Use

- Planning new features or fixes
- Tracking project goals and actions
- Documenting architectural decisions
- Managing issues that need multiple commits
- Creating new ADRs

## GOAP Structure

Location: `plans/GOAP.md`

```markdown
# GOAP: <Project Name>

## Goal
<One sentence describing the project goal>

## Objectives
1. <Objective 1>
2. <Objective 2>

## Actions
- [ ] <Action item 1>
- [x] <Completed action>

## Priorities
1. <Priority 1> (high/medium/low)
2. <Priority 2>
```

## ADR Structure

Location: `plans/ADR-XXX-<title>.md`

```markdown
# ADR-XXX: <Title>

## Status
Proposed | Accepted | Deprecated | Replaced

## Context
<Describe the problem or situation>

## Decision
<Describe the chosen solution>

## Consequences
- **Positive**: <Benefit>
- **Negative**: <Drawback>

## Alternatives Considered
1. <Alternative 1> - rejected because...
2. <Alternative 2>

## Related
- ADR-XXX: <Related ADR>
- GOAP.md
```

## Commands

### List Plans

```bash
ls -la plans/
cat plans/GOAP.md
```

### Create New Action Item

```bash
# Add to GOAP.md
# Edit plans/GOAP.md and add:
- [ ] <New action item> (priority: high/medium/low)
```

### Create New ADR

```bash
# Find next ADR number
ls plans/ADR-*.md | tail -1

# Create plans/ADR-XXX-<title>.md
```

### Update ADR Status

```markdown
## Status
Accepted  # Change from Proposed
```

### Mark Action Complete

```markdown
## Actions
- [x] <Completed action>
```

## Integration with CI Fix Workflow

1. **Identify issue** → Check `plans/GOAP.md` for existing action
2. **Create action** → Add new item if not found
3. **Create ADR** → For significant architectural decisions
4. **Fix issue** → Use appropriate skill
5. **Update plans** → Mark action complete

## Best Practices

1. **Always update GOAP** when adding new work items
2. **Create ADR** for any architectural decision
3. **Reference ADRs** in commit messages when relevant
4. **Keep GOAP updated** - stale actions should be removed
5. **Use priorities** - high, medium, low

## Example Workflow

```bash
# 1. Check existing plans
cat plans/GOAP.md

# 2. Add new action for CI fix
# Edit GOAP.md, add:
# - [ ] Fix flake8 E501 errors in src/ (priority: high)

# 3. If architectural decision needed, create ADR
# Create plans/ADR-007-lint-config.md

# 4. Fix the issue using appropriate skill
# @skill code-quality

# 5. Update plans
# Mark action complete in GOAP.md
```

## File Conventions

| Pattern | Description |
|---------|-------------|
| `GOAP.md` | Main planning document |
| `ADR-XXX-*.md` | Architecture decision records (XXX = 3-digit number) |
| Actions | `- [ ]` = pending, `- [x]` = complete |
| Priority | `high`, `medium`, `low` |
