# ADR-001: Agent Skill Structure

## Status
Accepted

## Context
The project needs a standardized structure for agent skills that:
- Is compatible with opencode and other AI agents
- Provides clear triggers for skill activation
- Remains under 250 LOC per skill
- Follows 2026 AGENTS.md best practices

## Decision
We will use the YAML frontmatter pattern for skills:
```yaml
---
name: <skill-name>
description: <description>
triggers:
  - "<trigger phrase 1>"
  - "<trigger phrase 2>"
---
```

## Consequences
- **Positive**: Skills auto-activate based on user prompts
- **Positive**: Clear documentation structure
- **Positive**: Easy to extend with new skills
- **Negative**: Requires frontmatter parsing

## Alternatives Considered
1. JSON configuration - rejected, less readable
2. Directory naming convention - rejected, less flexible
3. Comments-based - rejected, less standard

## Related
- ADR-002: CI Workflow Optimization
- ADR-003: AGENTS.md Structure
