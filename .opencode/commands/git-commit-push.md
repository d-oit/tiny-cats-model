---
description: Perform atomic git commit and push following project best practices
agent: atomic-git
model: mistral/devstral-2512
---

Perform an atomic git commit and push operation following the project's best practices:

1. Check current git status
2. Stage all appropriate changes
3. Run quality gate checks (ruff, flake8, black, mypy, pytest)
4. Create a conventional commit message based on the changes
5. Push to the remote repository
6. Verify the operation was successful

Follow these rules:
- Use conventional commit format (feat:, fix:, docs:, refactor:, test:, chore:)
- NEVER force push to main/master branches
- NEVER amend commits that have already been pushed
- Always run quality gate before committing
- If quality checks fail, report the issues and stop

Report the results of each step and the final outcome.

If CI failures occur:
1. Use GOAP agent to orchestrate specialist agents with appropriate skills
2. GOAP agent coordinates handoff between specialists
3. Incorporate 2026 best practices via web research
4. Repeat quality gate and commit process until all checks pass