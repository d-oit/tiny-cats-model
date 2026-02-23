# ADR-002: CI Workflow Optimization

## Status
Accepted

## Context
GitHub Actions workflows need to be optimized for:
- Faster execution through concurrency control
- Better caching strategies
- Timeout limits to prevent runaway jobs
- Modern action versions

## Decision
We will implement:
1. **Concurrency control** - Cancel in-progress runs on new pushes
2. **Enhanced caching** - Use setup-python cache feature
3. **Timeout limits** - Add 10-minute timeout to all jobs
4. **Parallel jobs** - Keep lint, test, type-check independent

## Consequences
- **Positive**: Reduces CI costs by canceling stale runs
- **Positive**: Faster feedback with parallel execution
- **Positive**: Prevents hung jobs consuming minutes
- **Negative**: Slightly more complex workflow YAML

## Implementation
```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    timeout-minutes: 10
    # ... steps with cache: pip
```

## Alternatives Considered
1. Keep workflows as-is - rejected, not optimized
2. Use only caching - rejected, missing concurrency
3. Use external CI service - rejected, adds complexity

## Related
- ADR-001: Agent Skill Structure
- ADR-003: AGENTS.md Structure
