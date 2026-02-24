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

### feat: add ci-monitor skill and update GOAP progress

**Date**: 2026-02-24 17:33:07 +0000
**Type**: feature
**Areas**: documentation

**Pattern**: [To be documented - what reusable lesson applies?]

**Related**: [Link to ADR if applicable]



### feat: modernize code quality setup for 2026 (ADR-016)

**Date**: 2026-02-24 17:23:45 +0000
**Type**: feature
**Areas**: source, ci-cd, documentation

**Pattern**: [To be documented - what reusable lesson applies?]

**Related**: [Link to ADR if applicable]



### fix: resolve frontend TypeScript build errors

**Date**: 2026-02-24 17:11:25 +0000
**Type**: bugfix
**Areas**: source

**Pattern**: [To be documented - what reusable lesson applies?]

**Related**: [Link to ADR if applicable]



### fix: resolve flake8 E402 and E501 errors for CI

**Date**: 2026-02-24 16:54:52 +0000
**Type**: bugfix
**Areas**: tests, source

**Pattern**: [To be documented - what reusable lesson applies?]

**Related**: [Link to ADR if applicable]


### feat: implement missing GOAP tasks for Phase 4, 5, and 6

**Date**: 2026-02-24 16:06:24 +0000
**Type**: feature
**Areas**: source, documentation, ci-cd

**Pattern**: [To be documented - what reusable lesson applies?]

**Related**: [Link to ADR if applicable]



### feat: implement 2026 GitHub Actions best practices (ADR-013)

**Date**: 2026-02-24 15:09:15 +0000
**Type**: feature
**Areas**: documentation, ci-cd

**Pattern**: [To be documented - what reusable lesson applies?]

**Related**: [Link to ADR if applicable]



### fix: resolve flake8 E402 import order errors

**Date**: 2026-02-24 14:54:26 +0000
**Type**: bugfix
**Areas**: source, documentation, tests

**Pattern**: [To be documented - what reusable lesson applies?]

**Related**: [Link to ADR if applicable]


### fix: align quality gate with CI pipeline (ADR-014)

**Date**: 2026-02-24
**Type**: bugfix
**Areas**: ci-cd, developer-experience, tooling

**Problem**: Local quality gate (`scripts/quality-gate.sh`) passed but CI failed with flake8 errors.

**Root Causes**:
- Local quality gate did not run flake8 at all
- Black configured for 120 chars locally, CI expected 88 chars
- CI flake8 config inline in workflow, not in `.flake8` file
- No single source of truth for linting rules

**Solution**:
1. Added flake8 check to `scripts/quality-gate.sh` with same config as CI
2. Created `.flake8` file as single source of truth
3. Updated `pyproject.toml` black/isort to use 88 chars (not 120)
4. Updated CI to read from `.flake8` file instead of inline flags

**Pattern**: **Local-CI Parity Principle** - The local quality gate must run the same checks with the same configuration as CI.

```bash
# ✅ CORRECT: Local quality gate matches CI
scripts/quality-gate.sh
├── black --check . (88 chars)
├── isort --check-only . (88 chars)
├── ruff check .
├── flake8 . (reads .flake8)
├── mypy . --ignore-missing-imports
└── pytest tests/ -v

.github/workflows/ci.yml
├── black --check . (88 chars)
├── ruff check .
├── flake8 . (reads .flake8)
├── mypy . --ignore-missing-imports
└── pytest tests/ -v --cov=.
```

**Prevention**:
- Always add new CI checks to local quality gate first
- Use config files (`.flake8`, `pyproject.toml`) not inline flags
- Run `bash scripts/quality-gate.sh` before every commit
- Install pre-commit hook: `bash scripts/install-hooks.sh`

**Related**: ADR-014 (Quality Gate CI Alignment)


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
| E402 import order | src/eval.py, tests/test_dataset.py | Move `sys.path.insert()` before imports |
| E501 line length | Multiple files | Auto-fix with ruff + black |

**Pattern**: Run full lint/test/type-check locally before push.

---

### Flake8 Linting Fixes (ADR-012)

**Problem**: CI failing on flake8 with 66 errors (E501 line too long, E402 import order).

**Root Causes**:
- `sys.path.insert()` statements placed after imports in test/eval files
- Long lines exceeding 88 character limit in scripts and source files
- Mismatch between local ruff config (ignores E501) and CI flake8 config

**Solution**:
1. **Import order fix**: Move `sys.path.insert()` before all imports:
   ```python
   # ✅ Correct
   import sys
   from pathlib import Path
   
   sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
   
   import torch
   from model import cats_model
   ```

2. **CI workflow already has ignores**: `.github/workflows/ci.yml`:
   ```yaml
   - name: Run flake8
     run: flake8 . --max-line-length=88 --extend-ignore=E203,W503,E402,E501
   ```

3. **Auto-fix first**: Always run `ruff check . --fix && black .` before manual fixes

**Pattern**: 
- Put path manipulation before imports in test/eval scripts
- CI flake8 ignores E402/E501, but fix locally for cleaner code
- Use ruff + black for 90% of fixes, manual edit for import order

**Prevention**:
- Add pre-commit hook: `flake8 . --max-line-length=88 --extend-ignore=E203,W503,E402,E501`
- Run quality gate before every push: `bash scripts/quality-gate.sh`

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
- **Quality gate**: `bash scripts/quality-gate.sh` before every push

### 3b. Linting Fix Pattern (ADR-012)
When CI fails on linting:
1. Run `ruff check . --fix && black .` (auto-fix 90%)
2. Manual fix remaining E402 (import order) errors
3. Verify locally: `flake8 .` (reads .flake8 config)
4. Commit → push → monitor CI with `gh run watch <id>`

### 3c. Quality Gate Parity Pattern (ADR-014)
**Principle**: Local quality gate must match CI exactly.

**Checklist for New CI Checks**:
1. Add check to `scripts/quality-gate.sh`
2. Create/update config file (`.flake8`, `pyproject.toml`)
3. Update CI to read from config file
4. Test locally: `bash scripts/quality-gate.sh`
5. Verify CI passes with same checks

**Configuration Alignment**:
- Black line-length: 88 (in `pyproject.toml`)
- Flake8 config: `.flake8` file (not inline in CI)
- isort profile: black (in `pyproject.toml`)
- mypy: `--ignore-missing-imports` (both local and CI)

**Pre-commit Hook**:
```bash
# Install to run quality gate automatically before each commit
bash scripts/install-hooks.sh
```

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

### 7. GitHub Actions Best Practices (ADR-013)

**Concurrency Strategy (2026)**:
```yaml
# ✅ CORRECT: Use PR number, not branch
concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}

# ❌ WRONG: Causes cancelled runs to show as failures
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

**Why This Matters**:
- GitHub UI shows cancelled runs as "failing" (bug)
- PR status shows "3/12 checks failing" even when all pass
- Using PR number groups runs correctly
- Only cancel PR runs, not main branch builds

**Branch Protection** (Required for 2026):
- Enable on `main` branch
- Require status checks: Lint, Test, Type Check
- Require PR reviews before merge
- Include administrators

**Workflow Dispatch**: Add manual trigger for debugging:
```yaml
workflow_dispatch:
  inputs:
    debug_enabled:
      description: 'Run with debug logging'
      required: false
      default: 'false'
```

### 8. Quality Gate Parity (ADR-014)

**Golden Rule**: What passes locally must pass in CI.

**Common Mismatches to Avoid**:
| Issue | Local | CI | Fix |
|-------|-------|----|-----|
| Missing flake8 | ❌ Not run | ✅ Runs | Add to quality-gate.sh |
| Line length | 120 chars | 88 chars | Align black config |
| Config flags | Inline | Config file | Use `.flake8` file |
| Coverage | Not checked | Checked | Add --cov to pytest |

**Single Source of Truth**:
- `.flake8` for flake8 configuration
- `pyproject.toml` for black, isort, mypy
- CI reads from config files, not inline flags

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
