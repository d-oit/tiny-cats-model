# Project Learnings & Patterns

## Self-Learning Loop

After each task, capture learnings here. Use this for continuous improvement.

```
1. Task completed → What worked?
2. Issues encountered → What failed?
3. Fix applied → How was it resolved?
4. Pattern extracted → What's the reusable lesson?
5. Update docs → Add to ADR or this file
```

---

## Key Learnings

### Modal GPU Training (ADR-007)

**Problem**: Training times out on CPU; Modal setup was broken.

**Root Causes**:
- Hardcoded paths didn't match `modal.yml` configuration
- Missing volume setup for data and outputs
- Data not accessible in Modal container
- No proper image with dependencies

**Solution**:
```bash
# Modal tokens configured globally
modal token set

# Run training
modal run src/train.py
modal run src/train.py -- --epochs 20 --batch-size 64
```

**Pattern**: Always use Modal volumes for persistent storage. Download data inside container or pre-upload to volume.

---

### CI/CD Fix Workflow (ADR-006)

**Problem**: CI failures required ad-hoc fixes without systematic approach.

**Solution**: Atomic commit-to-fix loop:

```
1. git commit → git push
2. gh run list → get run-id
3. gh run view <id> → identify failures
4. FOR EACH failure:
   a. Analyze error type → determine skill needed
   b. Spawn specialist agent with @skill
   c. Agent fixes → commits → pushes
   d. Repeat from step 2 until all pass
5. NEVER skip: each fix must go through full cycle
```

**Specialist Mapping**:
| Failure Type | Skill |
|--------------|-------|
| Lint error | `@skill code-quality` |
| Test failure | `@skill testing-workflow` |
| Type error | `@skill code-quality` |
| CI config | `@skill gh-actions` |
| Model/training | `@skill model-training` |
| Security | `@skill security` |

**Pattern**: Always use specialist agents. Never skip the full cycle.

---

### CI Issues Fixed (GOAP.md)

| Issue | Files | Fix |
|-------|-------|-----|
| F401 unused imports | src/dataset.py, src/model.py, tests/test_dataset.py | Remove unused imports |
| F541 f-strings | src/eval.py, src/export_onnx.py | Remove f-prefix from non-f-strings |
| Mypy return type | src/train.py:76 | Change to `tuple[float, float]` |
| Mypy attr-defined | src/train.py:117, src/eval.py:56 | Add `# type: ignore` |
| Missing requirements-dev.txt | New file | Created with dev dependencies |
| Makefile line-length | Makefile | Changed 120 to 88 |

**Pattern**: Run full lint/test/type-check locally before push.

---

### Agent Skills Structure (ADR-001)

**Problem**: Agents lacked clear specialization and triggers.

**Solution**: 8 skills with explicit triggers:

| Skill | LOC | Triggers |
|-------|-----|----------|
| cli-usage | ~115 | "train", "evaluate", "download data" |
| testing-workflow | ~59 | "test", "verify", "CI" |
| code-quality | ~115 | "lint", "format", "style" |
| gh-actions | ~84 | "CI", "GitHub Actions", "workflow" |
| git-workflow | ~127 | "commit", "branch", "PR" |
| goap | ~150 | "plan", "GOAP", "ADR" |
| security | ~124 | "secret", "token", "credential" |
| model-training | ~121 | "GPU", "Modal", "hyperparameter" |

**Pattern**: Skills under 250 LOC. Explicit trigger words. Single responsibility.

---

### AGENTS.md Structure (ADR-003)

**Problem**: AGENTS.md too long, mixed concerns.

**Solution**: Split into core (100 lines) + extended docs:

```
AGENTS.md (100 lines)
├── Quick Commands
├── Code Style
├── What NOT to Do
├── Agent Skills (summary)
├── CI/CD (summary)
├── Security (summary)
├── File Structure
├── Modal Training (summary)
└── References to agents-docs/

agents-docs/
├── skills.md
├── ci-cd.md
├── training.md
└── security.md
```

**Pattern**: Core file < 120 lines. Extended docs in separate folder.

---

## Reusable Patterns

### 1. Security First
- Never hardcode tokens
- Never commit `.env` files
- Use global config (`modal token set`) or GitHub Secrets
- Gitignore sensitive paths

### 2. Type Safety
- Type hints required for all new code
- Run `mypy . --ignore-missing-imports` locally
- Use `# type: ignore` sparingly with comments

### 3. Code Quality
- Line length: 88 chars (black default)
- Run `ruff check . --fix && black .` before commit
- Tests required for new features

### 4. CI/CD Discipline
- Never merge if CI fails
- Use specialist agents for fixes
- Document decisions in ADRs
- Update GOAP.md with action items

### 5. Modal Best Practices
- Configure tokens globally: `modal token set`
- Use volumes for persistent storage
- Set appropriate timeouts (1 hour for training)
- Download data inside container or pre-upload

### 6. Agent Workflow
- Load skill with `@skill <name>`
- Match trigger words to skill
- Use websearch/codesearch for 2026 best practices
- Document findings in ADR if significant

---

## 2026 Best Practices Applied

1. **Single Responsibility**: Each skill has one purpose
2. **Verification Loops**: CI fix loop until success
3. **Trace-Level Observability**: GOAP/ADR audit trail
4. **Concurrency Controls**: GitHub Actions parallelization
5. **Modern Caching**: pip cache in CI workflows
6. **Timeout Limits**: 10 min CI, 1 hour training

---

## Open Questions

- [ ] How to handle large dataset uploads to Modal volumes?
- [ ] Should we add ONNX export to CI pipeline?
- [ ] What's the optimal batch size for T4 GPU?

---

## References

- [GOAP.md](../plans/GOAP.md) - Project plan and action items
- [ADR-001](../plans/ADR-001-agent-skill-structure.md) - Skill structure
- [ADR-006](../plans/ADR-006-ci-fix-workflow.md) - CI fix workflow
- [ADR-007](../plans/ADR-007-modal-training-fix.md) - Modal training
