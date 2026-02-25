# ADR-029: Production Deployment GOAP Plan 2026

**Date:** 2026-02-25
**Status:** Proposed
**Authors:** AI Agent (GOAP System)
**Related:** GOAP.md Phase 10, ADR-026 (HuggingFace Publishing), ADR-027 (MLflow), ADR-028 (Validation Gates)

## Context

### Current State

The tiny-cats-model project has completed core development:
- **Training**: TinyDiT trained for 200k steps with EMA (checkpoints/tinydit_final.pt, 129MB)
- **Evaluation**: 104 samples generated across 13 breeds (ADR-019)
- **Export**: ONNX models exported and quantized (11MB)
- **Frontend**: React + TypeScript app with in-browser inference
- **Utilities**: upload_to_hub.py, validate_model.py created
- **Documentation**: ADRs 026-028 created for publishing, tracking, validation

### Problem Statement

Production deployment requires coordinated execution of multiple interdependent tasks:
1. Code validation against Modal runtime
2. Atomic git commits following best practices
3. Branch management and GitHub push
4. CI/CD monitoring and issue resolution
5. Progress tracking across multiple workstreams

Without a structured planning approach:
- Tasks may be executed out of order (breaking dependencies)
- Issues may not be tracked systematically
- Progress visibility is limited
- Rollback becomes difficult

## Decision

Implement a **GOAP (Goal-Oriented Action Planning)** system for production deployment with:

### 1. World State Representation

```python
@dataclass
class DeploymentWorldState:
    """Current state of the deployment world."""
    
    # Code state
    code_validated_modal: bool = False
    code_validated_ruff: bool = False
    code_validated_typescript: bool = False
    
    # Git state
    branch_created: bool = False
    branch_name: str = ""
    commits_made: int = 0
    commits_pushed: bool = False
    
    # CI/CD state
    ci_pipeline_running: bool = False
    ci_pipeline_passed: bool = False
    ci_issues_resolved: bool = True
    
    # Deployment state
    validation_passed: bool = False
    model_uploaded_hub: bool = False
    mlflow_integrated: bool = False
    
    # Documentation state
    goap_updated: bool = False
    adrs_complete: bool = False
    
    # Timestamps
    started_at: str = ""
    last_action: str = ""
```

### 2. Goal Hierarchy

```
GOAL: Production Deployment Complete
├── SUBGOAL: Code Validation
│   ├── Validate Modal CLI training
│   ├── Validate Ruff linting
│   └── Validate TypeScript build
├── SUBGOAL: Git Branch Management
│   ├── Create feature branch
│   ├── Stage atomic commits
│   └── Push to GitHub
├── SUBGOAL: CI/CD Monitoring
│   ├── Monitor pipeline status
│   ├── Detect failures
│   └── Resolve issues
├── SUBGOAL: Model Deployment
│   ├── Run validation gates
│   ├── Upload to HuggingFace
│   └── Integrate MLflow
└── SUBGOAL: Documentation
    ├── Update GOAP.md
    ├── Complete ADRs
    └── Update progress
```

### 3. Actions with Preconditions and Effects

| Action ID | Action | Preconditions | Effects | Cost |
|-----------|--------|---------------|---------|------|
| A01 | validate_modal_cli | None | code_validated_modal=True | 5 |
| A02 | validate_ruff | None | code_validated_ruff=True | 3 |
| A03 | validate_typescript | None | code_validated_typescript=True | 4 |
| A04 | create_branch | code_validated_modal, code_validated_ruff | branch_created=True, branch_name="feature/production-deployment-2026" | 2 |
| A05 | commit_code_changes | branch_created | commits_made+=1 | 2 |
| A06 | commit_documentation | branch_created | commits_made+=1 | 2 |
| A07 | commit_utility_additions | branch_created | commits_made+=1 | 2 |
| A08 | push_to_github | commits_made>0 | commits_pushed=True, ci_pipeline_running=True | 3 |
| A09 | monitor_ci_status | ci_pipeline_running | ci_status_known | 1 |
| A10 | resolve_ci_failure | ci_pipeline_running, NOT ci_pipeline_passed | ci_issues_resolved=True | 10 |
| A11 | run_validation_gates | code_validated_modal | validation_passed=True | 5 |
| A12 | upload_to_huggingface | validation_passed | model_uploaded_hub=True | 5 |
| A13 | integrate_mlflow | code_validated_modal | mlflow_integrated=True | 8 |
| A14 | update_goap_md | Any | goap_updated=True | 2 |
| A15 | complete_adrs | Any | adrs_complete=True | 3 |

### 4. Planning Algorithm

**A* Search with Heuristic:**
- **g(n)**: Sum of action costs from start state
- **h(n)**: Estimated remaining actions to goal
  - Count of false world state variables × average cost (4)
- **f(n)**: g(n) + h(n)

**Priority Ordering:**
1. Validation actions (unblock all others)
2. Branch creation (enables commits)
3. Commits (enable push)
4. Push (enables CI)
5. CI monitoring (ensures quality)
6. Deployment actions (final goals)
7. Documentation (parallel track)

### 5. Action Execution Flow

```
START
  │
  ▼
┌─────────────────────────────────┐
│  Phase 1: Validation            │
│  ├─ A01: validate_modal_cli     │
│  ├─ A02: validate_ruff          │
│  └─ A03: validate_typescript    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Phase 2: Branch Management     │
│  └─ A04: create_branch          │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Phase 3: Atomic Commits        │
│  ├─ A05: commit_code_changes    │
│  ├─ A06: commit_documentation   │
│  └─ A07: commit_utility_adds    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Phase 4: Push & CI             │
│  ├─ A08: push_to_github         │
│  ├─ A09: monitor_ci_status      │
│  └─ A10: resolve_ci_failure?    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Phase 5: Deployment            │
│  ├─ A11: run_validation_gates   │
│  ├─ A12: upload_to_huggingface  │
│  └─ A13: integrate_mlflow       │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Phase 6: Documentation         │
│  ├─ A14: update_goap_md         │
│  └─ A15: complete_adrs          │
└──────────────┬──────────────────┘
               │
               ▼
             GOAL
```

### 6. Integration with Existing Agent Skills

| Skill | GOAP Actions | Usage |
|-------|--------------|-------|
| `code-quality` | A02 | Ruff linting, formatting |
| `model-training` | A01, A13 | Modal CLI validation, MLflow |
| `git-workflow` | A04-A08 | Branch, commit, push operations |
| `ci-monitor` | A09-A10 | CI monitoring, failure resolution |
| `testing-workflow` | A03 | TypeScript validation |
| `goap` | All | Planning system coordination |
| `agents-md` | A15 | ADR documentation |

### 7. Commit Structure (Atomic Commits)

```
Commit 1: feat: Add model validation gates (src/validate_model.py)
  - src/validate_model.py
  - validation_report.json

Commit 2: feat: Add HuggingFace upload utility (src/upload_to_hub.py)
  - src/upload_to_hub.py
  - requirements-modal.txt (huggingface_hub, safetensors)

Commit 3: docs: Add ADRs 026-028 for publishing, tracking, validation
  - plans/ADR-026-huggingface-model-publishing-implementation.md
  - plans/ADR-027-experiment-tracking-mlflow.md
  - plans/ADR-028-model-validation-gates.md

Commit 4: docs: Update GOAP.md with Phase 10 production deployment
  - plans/GOAP.md

Commit 5: feat: Add Modal monitoring utility
  - src/modal_monitor.py
```

### 8. Timeline Estimate

| Phase | Actions | Estimated Time | Dependencies |
|-------|---------|----------------|--------------|
| Phase 1: Validation | A01-A03 | 15 minutes | None |
| Phase 2: Branch | A04 | 2 minutes | Phase 1 |
| Phase 3: Commits | A05-A07 | 10 minutes | Phase 2 |
| Phase 4: Push & CI | A08-A10 | 30 minutes | Phase 3 |
| Phase 5: Deployment | A11-A13 | 20 minutes | Phase 4 |
| Phase 6: Documentation | A14-A15 | 10 minutes | Parallel |
| **Total** | **15 actions** | **~87 minutes** | |

**Buffer for CI failures:** +30-60 minutes

**Total Estimated Duration:** 2-2.5 hours

## Consequences

### Positive
- **Structured Execution**: Clear action ordering prevents dependency issues
- **Visibility**: World state provides real-time progress tracking
- **Rollback**: Atomic commits enable easy rollback
- **Issue Resolution**: GOAP replanning handles CI failures gracefully
- **Documentation**: ADR captures decision rationale

### Negative
- **Overhead**: Planning system adds initial setup time
- **Complexity**: World state tracking requires maintenance
- **Rigidity**: May not handle unexpected issues without replanning

### Neutral
- **Skill Integration**: Leverages existing agent skills
- **Extensible**: New actions can be added as needed
- **Reusable**: Pattern applicable to future deployments

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| All validations pass | 100% | A01-A03 complete |
| Atomic commits | 5 commits | Git log shows structured commits |
| CI pipeline passes | 100% | GitHub Actions green |
| Model uploaded | Yes | HuggingFace repo exists |
| Documentation complete | ADRs 026-029 | plans/ folder updated |
| Total duration | <3 hours | Start to finish time |

## Implementation Notes

### World State Persistence
Store world state in `plans/deployment_state.json`:
```json
{
  "branch_name": "feature/production-deployment-2026",
  "commits_made": 5,
  "validation_passed": true,
  "ci_passed": true,
  "model_uploaded": false,
  "last_action": "A08: push_to_github",
  "updated_at": "2026-02-25T12:00:00Z"
}
```

### Replanning on Failure
If CI fails (A10 required):
1. Update world state: `ci_pipeline_passed=False`
2. Invoke `ci-monitor` skill for diagnosis
3. Apply fix
4. Create fix commit (A05)
5. Push again (A08)
6. Resume monitoring (A09)

### Progress Reporting
After each action, update:
1. `plans/deployment_state.json`
2. Console output with action ID and result
3. Git commit message (for commit actions)

## References

- GOAP.md: Main goal tracking document
- ADR-026: HuggingFace Publishing Implementation
- ADR-027: MLflow Experiment Tracking
- ADR-028: Model Validation Gates
- Agent Skills: .agents/skills/
