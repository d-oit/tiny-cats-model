# GOAP: Agent Infrastructure Enhancement Plan

## Goal
Enhance the tiny-cats-model project with comprehensive agent skills and 2026 best practices for AI-assisted development.

## Objectives

1. **Improve Agent Context** - Create AGENTS.md following 2026 standards
2. **Expand Skill Coverage** - Add new skills for git, code quality, security, model training
3. **Optimize CI/CD** - Update GitHub Actions with concurrency, caching, modern patterns
4. **Document Decisions** - Maintain ADRs for all architectural choices

## Actions

### Phase 1: Planning & Research
- [x] Analyze existing skills structure
- [x] Research 2026 best practices (GitHub Actions, AGENTS.md, gh CLI)
- [x] Identify gaps in current skill coverage

### Phase 2: Skill Development
- [x] Create git-workflow skill
- [x] Create code-quality skill  
- [x] Create security skill
- [x] Create model-training skill
- [x] Update existing skills (gh-actions, testing-workflow, cli-usage)

### Phase 3: Documentation
- [x] Update AGENTS.md with 2026 best practices
- [x] Create ADR documents for key decisions
- [x] Document skill usage patterns

### Phase 4: CI Optimization
- [x] Add concurrency controls to workflows
- [x] Enhance caching strategies
- [x] Add timeout limits
- [x] Implement job parallelization

### Phase 5: CI Issues Fix
- [x] Analyze CI failures via gh CLI
- [x] Fix lint errors (F401, F541)
- [x] Fix mypy type errors
- [x] Create requirements-dev.txt
- [x] Update pyproject.toml with ruff/mypy config
- [x] Fix Makefile line-length
- [x] Verify all tests pass

## Priorities
1. AGENTS.md update (high) - core context for all agents ✅
2. New skills (high) - expand agent capabilities ✅
3. CI optimization (medium) - improve developer experience ✅
4. Documentation (medium) - maintain knowledge ✅
5. CI issues fix (high) - all CI must pass ✅

## Timeline
- Phase 1: Complete
- Phase 2: Complete
- Phase 3: Complete
- Phase 4: Complete
- Phase 5: Complete

## Success Metrics
- All skills under 250 LOC ✅
- AGENTS.md < 200 lines ✅
- CI workflow runtime < 5 minutes ✅
- All tests passing ✅
- ruff check passes ✅

## CI Issues Found & Fixed

### Issue 1: Lint Errors (Flake8/Ruff)
| File | Issue | Fix Applied |
|------|-------|--------------|
| src/dataset.py | F401 unused imports | Removed `os`, `Optional` |
| src/model.py | F401 unused imports | Removed `Optional` |
| src/eval.py | F541 f-strings | Removed f-prefix from print |
| src/export_onnx.py | F401, F541 | Removed unused imports, fixed f-strings |
| tests/test_dataset.py | F401 unused | Removed `tempfile` |

### Issue 2: Mypy Type Errors
| File | Issue | Fix Applied |
|------|-------|--------------|
| src/train.py:76 | Return type mismatch | Changed to `tuple[float, float]` |
| src/train.py:117 | attr-defined | Added `# type: ignore` |
| src/eval.py:56 | attr-defined | Added `# type: ignore` |
| tests/test_train.py | Missing exports | Rewrote tests to use actual API |

### Issue 3: Missing requirements-dev.txt
- Created `requirements-dev.txt` with dev dependencies
- Updated CI workflow to use correct file

### Issue 4: Configuration Mismatch
- Fixed pyproject.toml with ruff/mypy config
- Fixed Makefile line-length from 120 to 88
- Added E501 to ruff ignore (black handles line length)

## Deliverables Created

### Skills (8 total)
| Skill | LOC | Purpose |
|-------|-----|---------|
| cli-usage | ~115 | Training, evaluation, dataset commands |
| testing-workflow | ~59 | Run tests, lint, verification |
| gh-actions | ~84 | CI/CD status, triggers, debugging |
| git-workflow | ~127 | Branch management, commits, PRs |
| code-quality | ~115 | Linting, formatting, type checking |
| security | ~124 | Secrets, credentials, safe practices |
| model-training | ~121 | Training, hyperparameters, Modal |
| goap | ~150 | GOAP/ADR planning and management |

### Documentation
- `AGENTS.md` - Updated with 2026 best practices
- `plans/GOAP.md` - Project plan
- `plans/ADR-001-agent-skill-structure.md` - Skill architecture decision
- `plans/ADR-002-ci-workflow-optimization.md` - CI optimization decision
- `plans/ADR-003-agents-md-structure.md` - AGENTS.md structure decision
- `plans/ADR-004-frontend-cat-model-update.md` - Frontend update proposal
- `plans/ADR-005-ci-pipeline-fixes.md` - CI pipeline fixes
- `plans/ADR-006-ci-fix-workflow.md` - Complete CI fix workflow

---

## Phase 6: Complete CI/CD Fix Workflow

### Atomic Commit-to-Fix Loop

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

### References
- ADR-006: Complete CI/CD Fix Workflow

### CI/CD
- `.github/workflows/ci.yml` - Optimized with concurrency, timeouts, modern caching
- `.github/workflows/train.yml` - Optimized with concurrency, timeouts
- `requirements-dev.txt` - Created with dev dependencies
- `pyproject.toml` - Updated with ruff and mypy configuration

### Files Modified to Fix CI
- `src/dataset.py` - Removed unused imports
- `src/model.py` - Removed unused imports
- `src/eval.py` - Fixed f-strings, added type ignore
- `src/export_onnx.py` - Removed unused imports, fixed f-strings
- `src/train.py` - Fixed return type, added type ignore
- `tests/test_dataset.py` - Removed unused imports
- `tests/test_train.py` - Rewrote to use actual API
- `Makefile` - Fixed line-length to 88
