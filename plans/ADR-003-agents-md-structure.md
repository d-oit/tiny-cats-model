# ADR-003: AGENTS.md Structure

## Status
Accepted

## Context
AGENTS.md must provide AI agents with essential project context while:
- Remaining concise (< 200 lines)
- Including all critical sections from 2026 best practices
- Linking to skills for detailed workflows
- Following Markdown best practices

## Decision
Structure AGENTS.md with these sections:
1. **Project Header** - Name, quick description
2. **Quick Reference** - Command table (highest value for agents)
3. **Code Style** - Linting, formatting, type hints
4. **Agent Skills** - List available skills
5. **Security** - Secrets, credentials handling
6. **CI/CD** - Workflow info, status checks
7. **File Structure** - Directory layout

## Consequences
- **Positive**: Agents get essential context quickly
- **Positive**: Easy to maintain and update
- **Positive**: Matches 2026 best practices
- **Negative**: Less detail than separate docs (mitigated by skills)

## Implementation Notes
- Use tables for commands (efficient token usage)
- Keep line length under 88 chars
- Use backticks for all commands
- Include gh CLI commands where relevant

## Alternatives Considered
1. Comprehensive single doc - rejected, too long
2. Multiple markdown files - rejected,分散
3. JSON config - rejected, not human-readable

## Related
- ADR-001: Agent Skill Structure
- ADR-002: CI Workflow Optimization
