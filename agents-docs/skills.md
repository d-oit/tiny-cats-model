# Agent Skills

Load skills using the `@skill` command when the task matches:

| Skill | When to Use | Triggers |
|-------|-------------|----------|
| `ci-monitor` | CI monitoring, failure coordination | "CI failed", "GitHub Actions", "monitor", "orchestrate" |
| `cli-usage` | Training, evaluation, dataset | "train", "evaluate", "download data" |
| `testing-workflow` | Running tests, verification | "test", "verify", "CI" |
| `code-quality` | Linting, formatting | "lint", "format", "style" |
| `gh-actions` | CI/CD, workflows | "CI", "GitHub Actions", "workflow" |
| `git-workflow` | Branches, commits, PRs | "commit", "branch", "PR", "quality gate", "pre-commit" |
| `goap` | Planning, ADR, project goals | "plan", "GOAP", "ADR", "action item", "priority" |
| `security` | Secrets, credentials | "secret", "token", "credential" |
| `model-training` | GPU training, Modal | "GPU", "Modal", "hyperparameter" |
| `web-search-researcher` | Modern information, docs | "2026", "best practices", "latest", "current" |

## Specialist Agent Selection

| Failure Type | Use Skill |
|--------------|-----------|
| CI run failed | `@skill ci-monitor` (orchestrator) |
| Lint error | `@skill code-quality` |
| Test failure | `@skill testing-workflow` |
| Type error | `@skill code-quality` |
| CI/workflow config | `@skill gh-actions` |
| Model/training | `@skill model-training` |
| Security | `@skill security` |

## CI Monitor Workflow (2026)

```
1. git commit → git push
2. @skill ci-monitor → gh run list → get run-id
3. gh run view <id> → identify failures
4. FOR EACH failure:
   a. Analyze error type → determine specialist needed
   b. Spawn specialist agent with @task
   c. Agent fixes → commits → pushes
   d. gh run view <new-id> → verify
   e. Repeat until all pass
5. Update GOAP.md with completed items
6. NEVER skip: each fix must go through full cycle
```

## 2026 Best Practices

Before fixing issues:
1. Run `websearch` for latest solutions on similar issues
2. Run `codesearch` for API/pattern examples
3. Document significant decisions in `plans/ADR-*.md`
4. Update `plans/GOAP.md` with action items
5. Use `@skill ci-monitor` for CI orchestration
