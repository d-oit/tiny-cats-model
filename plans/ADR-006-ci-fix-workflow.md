# ADR-006: Complete CI/CD Fix Workflow

## Status
Accepted

## Context
When CI fails after a commit, agents must follow a rigorous, repeatable workflow that:
- Never skips any checks
- Uses proper issue tracking via GOAP/ADR
- Spawns specialist agents with appropriate skills
- Loops until all checks pass
- Researches 2026 best practices for fixes

## Decision
We adopt the following atomic workflow:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CI/CD Fix Workflow (Atomic)                      │
├─────────────────────────────────────────────────────────────────────┤
│  1. git commit → git push                                          │
│  2. gh run list → get run-id                                       │
│  3. gh run view <id> → identify failures                           │
│  4. FOR EACH failure:                                               │
│     a. Analyze error type → determine skill needed                 │
│     b. Spawn specialist agent with @skill                         │
│     c. Agent fixes → commits → pushes                              │
│     d. Repeat from step 2 until all pass                          │
│  5. NEVER skip: each fix must go through full cycle               │
└─────────────────────────────────────────────────────────────────────┘
```

### Issue Tracking with GOAP/ADR

For complex issues:
1. Create/update GOAP.md with new action item
2. Create ADR if architectural decision needed
3. Reference in commit message

### Specialist Agent Mapping

| Failure Type | Skill | Agent Task |
|--------------|-------|------------|
| Lint error | `code-quality` | Fix style issues |
| Test failure | `testing-workflow` | Debug and fix tests |
| Type error | `code-quality` | Add type hints |
| CI config | `gh-actions` | Fix workflow YAML |
| Model/training | `model-training` | Fix training code |
| Security | `security` | Fix vulnerability |
| New feature | Multiple | Spawn coordinator |

### 2026 Best Practices Integration

Before implementing fixes:
1. Use `websearch` for latest solutions
2. Use `codesearch` for API patterns
3. Document findings in ADR if significant
4. Apply minimal, correct fix

### Loop Mechanism

```bash
while ! gh run view <run-id> --json conclusion | jq -e '.conclusion == "SUCCESS"'; do
    # Get failures
    gh run view <run-id> --log | grep "^ERR"
    
    # Spawn specialist agent for each failure
    # ... fix ...
    
    # Re-check
done
```

## Consequences
- **Positive**: Reproducible, thorough fixes
- **Positive**: Clear audit trail via GOAP/ADR
- **Positive**: Right specialist for each issue
- **Positive**: 2026 best practices applied
- **Negative**: More commits for complex fixes

## Alternatives Considered
1. Single agent handles all - rejected, lacks specialization
2. Skip CI failures - rejected, violates safety
3. Manual fixes only - rejected, not scalable

## Related
- ADR-001: Agent Skill Structure
- ADR-002: CI Workflow Optimization
- GOAP.md: Project plan

## References
- 2026 AI Agent best practices: single responsibility, verification loops, trace-level observability
- GitHub Actions: concurrency controls, idempotent reruns
