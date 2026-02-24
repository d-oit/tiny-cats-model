# Agent Skills

Load skills using the `@skill` command when the task matches:

| Skill | When to Use | Triggers |
|-------|-------------|----------|
| `cli-usage` | Training, evaluation, dataset | "train", "evaluate", "download data" |
| `testing-workflow` | Running tests, verification | "test", "verify", "CI" |
| `code-quality` | Linting, formatting | "lint", "format", "style" |
| `gh-actions` | CI/CD, workflows | "CI", "GitHub Actions", "workflow" |
| `git-workflow` | Branches, commits, PRs | "commit", "branch", "PR", "quality gate", "pre-commit" |
| `goap` | Planning, ADR, project goals | "plan", "GOAP", "ADR", "action item", "priority" |
| `security` | Secrets, credentials | "secret", "token", "credential" |
| `model-training` | GPU training, Modal | "GPU", "Modal", "hyperparameter" |

## Specialist Agent Selection

| Failure Type | Use Skill |
|--------------|-----------|
| Lint error | `@skill code-quality` |
| Test failure | `@skill testing-workflow` |
| Type error | `@skill code-quality` |
| CI/workflow | `@skill gh-actions` |
| Model/training | `@skill model-training` |
| Security | `@skill security` |

## 2026 Best Practices

Before fixing issues:
1. Run `websearch` for latest solutions on similar issues
2. Run `codesearch` for API/pattern examples
3. Document significant decisions in `plans/ADR-*.md`
4. Update `plans/GOAP.md` with action items
