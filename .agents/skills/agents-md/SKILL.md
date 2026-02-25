---
name: agents-md
description: Create, update, and maintain AGENTS.md and agents-docs/ for best practices. Handles documentation structure, skill creation, and reference management.
---

# Skill: agents-md

Manage AGENTS.md and related documentation following project conventions.

## AGENTS.md Structure

Keep AGENTS.md under 120 lines. Include:

- Quick Commands section
- Code Style (format, lint, type check)
- Testing commands
- Agent Skills summary table
- CI/CD summary
- Security summary
- Modal Training summary
- References to agents-docs/

## agents-docs/ Structure

Extended docs in separate files:

```
agents-docs/
├── training.md    # GPU, Modal, options
├── ci-cd.md       # Fix workflow, quality gate
├── security.md    # Secrets, tokens
├── skills.md      # Agent specialization
└── learnings.md   # Patterns, ADRs
```

## Creating New Skills

Follow `.agents/skills/<skill-name>/SKILL.md` structure:

```yaml
---
name: <skill-name>
description: <1-line description>
---

# Skill: <skill-name>

<detailed content under 250 LOC>
```

Requirements:
- YAML frontmatter required
- Under 250 LOC
- Include setup, usage, examples
- Match trigger words to skill purpose

## Updating AGENTS.md

When adding new features:

1. Update AGENTS.md with quick reference
2. Add detailed docs to agents-docs/
3. Create/update skill if needed
4. Run quality gate: `bash scripts/quality-gate.sh`

## Quality Gate

All skills must pass validation:

```bash
bash scripts/quality-gate.sh
```

Checks:
- Ruff format/lint
- YAML frontmatter in skills
- Type checking
- Tests pass

## Best Practices

- Core file < 120 lines
- Extended docs in agents-docs/
- Skills under 250 LOC
- Always use YAML frontmatter
- Match triggers to purpose
- Single responsibility per skill
