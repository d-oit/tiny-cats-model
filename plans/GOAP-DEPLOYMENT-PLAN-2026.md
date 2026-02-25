# GOAP Production Deployment Plan 2026

**Generated:** 2026-02-25
**Branch:** `feature/production-deployment-2026`
**Related ADR:** ADR-029

---

## Executive Summary

This document provides the complete GOAP (Goal-Oriented Action Planning) system for production deployment of tiny-cats-model to Modal with HuggingFace Hub publishing.

**Goal:** Production-ready deployment with validated code, atomic commits, CI monitoring, and model publishing.

**Estimated Duration:** 2-2.5 hours (including CI buffer)

---

## 1. World State Representation

### 1.1 State Schema

```json
{
  "deployment_id": "deploy-2026-02-25-001",
  "started_at": null,
  "completed_at": null,
  "current_phase": null,
  "last_action": null,
  
  "code_validation": {
    "modal_cli_validated": false,
    "ruff_validated": false,
    "typescript_validated": false
  },
  
  "git_state": {
    "branch_created": false,
    "branch_name": "",
    "commits": [],
    "pushed": false
  },
  
  "ci_state": {
    "pipeline_running": false,
    "pipeline_passed": false,
    "failures_detected": [],
    "failures_resolved": []
  },
  
  "deployment_state": {
    "validation_gates_passed": false,
    "model_uploaded_hub": false,
    "mlflow_integrated": false
  },
  
  "documentation_state": {
    "goap_updated": false,
    "adrs_complete": false
  }
}
```

### 1.2 State Transitions

```
INITIAL STATE:
  All flags = false
  No commits
  No branch

GOAL STATE:
  code_validation.* = true
  git_state.branch_name = "feature/production-deployment-2026"
  git_state.commits.length >= 5
  git_state.pushed = true
  ci_state.pipeline_passed = true
  deployment_state.validation_gates_passed = true
  documentation_state.* = true
```

---

## 2. Goal Hierarchy

### 2.1 Primary Goal

```
🎯 PRODUCTION_DEPLOYMENT_COMPLETE
   Status: PENDING
   Priority: CRITICAL
```

### 2.2 Subgoals

```
📋 SUBGOAL_1: CODE_VALIDATION
   Status: PENDING
   Priority: HIGH
   Actions: [A01, A02, A03]
   
📋 SUBGOAL_2: GIT_BRANCH_MANAGEMENT
   Status: PENDING
   Priority: HIGH
   Actions: [A04]
   Dependencies: [SUBGOAL_1]
   
📋 SUBGOAL_3: ATOMIC_COMMITS
   Status: PENDING
   Priority: HIGH
   Actions: [A05, A06, A07]
   Dependencies: [SUBGOAL_2]
   
📋 SUBGOAL_4: CI_MONITORING
   Status: PENDING
   Priority: CRITICAL
   Actions: [A08, A09, A10]
   Dependencies: [SUBGOAL_3]
   
📋 SUBGOAL_5: MODEL_DEPLOYMENT
   Status: PENDING
   Priority: HIGH
   Actions: [A11, A12, A13]
   Dependencies: [SUBGOAL_4]
   
📋 SUBGOAL_6: DOCUMENTATION
   Status: PENDING
   Priority: MEDIUM
   Actions: [A14, A15]
   Dependencies: []  # Parallel track
```

---

## 3. Action Catalog

### 3.1 Validation Actions

| ID | Action | Description | Preconditions | Effects | Cost | Skill |
|----|--------|-------------|---------------|---------|------|-------|
| A01 | `validate_modal_cli` | Test Modal CLI training with new utilities | None | `modal_cli_validated=true` | 5 | model-training |
| A02 | `validate_ruff` | Run Ruff linting on all Python code | None | `ruff_validated=true` | 3 | code-quality |
| A03 | `validate_typescript` | Run TypeScript build check | None | `typescript_validated=true` | 4 | testing-workflow |

### 3.2 Git Actions

| ID | Action | Description | Preconditions | Effects | Cost | Skill |
|----|--------|-------------|---------------|---------|------|-------|
| A04 | `create_branch` | Create feature/production-deployment-2026 | A01, A02, A03 | `branch_created=true`, `branch_name="feature/production-deployment-2026"` | 2 | git-workflow |
| A05 | `commit_code_changes` | Commit src/validate_model.py, src/upload_to_hub.py | A04 | `commits+=["code"]` | 2 | git-workflow |
| A06 | `commit_documentation` | Commit ADRs 026-028, GOAP.md update | A04 | `commits+=["docs"]` | 2 | git-workflow |
| A07 | `commit_utility_adds` | Commit src/modal_monitor.py | A04 | `commits+=["utility"]` | 2 | git-workflow |

### 3.3 CI/CD Actions

| ID | Action | Description | Preconditions | Effects | Cost | Skill |
|----|--------|-------------|---------------|---------|------|-------|
| A08 | `push_to_github` | Push branch with all commits | A05, A06, A07 | `pushed=true`, `pipeline_running=true` | 3 | git-workflow |
| A09 | `monitor_ci_status` | Monitor GitHub Actions pipeline | A08 | `ci_status_known` | 1 | ci-monitor |
| A10 | `resolve_ci_failure` | Fix CI failures using GOAP replanning | A09, `pipeline_passed=false` | `failures_resolved+=1` | 10 | ci-monitor + analysis |

### 3.4 Deployment Actions

| ID | Action | Description | Preconditions | Effects | Cost | Skill |
|----|--------|-------------|---------------|---------|------|-------|
| A11 | `run_validation_gates` | Execute validate_model.py on checkpoint | A01 | `validation_gates_passed=true` | 5 | model-training |
| A12 | `upload_to_huggingface` | Upload model to HuggingFace Hub | A11 | `model_uploaded_hub=true` | 5 | model-training |
| A13 | `integrate_mlflow` | Add MLflow tracking to training scripts | A01 | `mlflow_integrated=true` | 8 | model-training |

### 3.5 Documentation Actions

| ID | Action | Description | Preconditions | Effects | Cost | Skill |
|----|--------|-------------|---------------|---------|------|-------|
| A14 | `update_goap_md` | Update GOAP.md with Phase 10 progress | None | `goap_updated=true` | 2 | agents-md |
| A15 | `complete_adrs` | Ensure ADRs 026-029 are complete | None | `adrs_complete=true` | 3 | agents-md |

---

## 4. Priority Ordering

### 4.1 Execution Priority Queue

```
P0 (CRITICAL - Must Complete First):
  A01 → A02 → A03  (Validation unblocks everything)

P1 (HIGH - Core Deployment):
  A04 → A05 → A06 → A07 → A08  (Git workflow)
  A09 → [A10 if needed]  (CI monitoring)

P2 (HIGH - Deployment Goals):
  A11 → A12  (Model deployment)
  A13  (MLflow integration)

P3 (MEDIUM - Documentation - Parallel):
  A14, A15  (Can run anytime)
```

### 4.2 Dependency Graph

```
                    ┌─────────────┐
                    │    START    │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │   A01   │      │   A14   │      │   A15   │
    │ validate│      │ update  │      │complete │
    │ modal   │      │ goap.md │      │  ADRs   │
    └────┬────┘      └────┬────┘      └────┬────┘
         │                │                │
         ▼                │                │
    ┌─────────┐           │                │
    │   A02   │           │                │
    │ validate│           │                │
    │  ruff   │           │                │
    └────┬────┘           │                │
         │                │                │
         ▼                │                │
    ┌─────────┐           │                │
    │   A03   │           │                │
    │validate │           │                │
    │   ts    │           │                │
    └────┬────┘           │                │
         │                │                │
         └────────────────┼────────────────┘
                          │
                          ▼
                    ┌─────────┐
                    │   A04   │
                    │ create  │
                    │ branch  │
                    └────┬────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │   A05   │    │   A06   │    │   A07   │
    │ commit  │    │ commit  │    │ commit  │
    │  code   │    │  docs   │    │ utility │
    └────┬────┘    └────┬────┘    └────┬────┘
         │              │              │
         └──────────────┼──────────────┘
                        │
                        ▼
                  ┌─────────┐
                  │   A08   │
                  │  push   │
                  └────┬────┘
                       │
                       ▼
                  ┌─────────┐
                  │   A09   │
                  │ monitor │
                  └────┬────┘
                       │
              ┌────────┴────────┐
              │                 │
         pass │            fail │
              │                 │
              ▼                 ▼
        ┌─────────┐       ┌─────────┐
        │   A11   │       │   A10   │
        │validate │       │ resolve │
        │  gates  │       │ failure │
        └────┬────┘       └────┬────┘
             │                 │
             │                 └──────┐
             │                        │
             ▼                        │
        ┌─────────┐                   │
        │   A12   │                   │
        │ upload  │◄──────────────────┘
        │   to    │     (retry loop)
        │   hub   │
        └────┬────┘
             │
             ▼
        ┌─────────┐
        │   A13   │
        │mlflow   │
        │integrate│
        └────┬────┘
             │
             ▼
        ┌─────────┐
        │   END   │
        │ SUCCESS │
        └─────────┘
```

---

## 5. Estimated Timeline

### 5.1 Phase Breakdown

| Phase | Actions | Duration | Start | End | Status |
|-------|---------|----------|-------|-----|--------|
| **Phase 1: Validation** | A01, A02, A03 | 15 min | T+0 | T+15 | ⏳ PENDING |
| **Phase 2: Branch** | A04 | 2 min | T+15 | T+17 | ⏳ PENDING |
| **Phase 3: Commits** | A05, A06, A07 | 10 min | T+17 | T+27 | ⏳ PENDING |
| **Phase 4: Push & CI** | A08, A09, [A10] | 30-60 min | T+27 | T+57-87 | ⏳ PENDING |
| **Phase 5: Deployment** | A11, A12, A13 | 20 min | T+57-87 | T+77-107 | ⏳ PENDING |
| **Phase 6: Documentation** | A14, A15 | 10 min | T+0 | T+107 | ⏳ PENDING |

### 5.2 Gantt Chart

```
Time (minutes) →
0         15        30        45        60        75        90        105
│─────────│─────────│─────────│─────────│─────────│─────────│─────────│
│█████████│         │         │         │         │         │         │ Phase 1
│         │██       │         │         │         │         │         │ Phase 2
│         │  ████████         │         │         │         │         │ Phase 3
│         │          █████████████████  │         │         │         │ Phase 4
│         │                         │█████████████         │         │ Phase 5
│█████████████████████████████████████████████████████████████████████│ Phase 6
```

### 5.3 Critical Path

```
A01 → A02 → A03 → A04 → A05 → A06 → A07 → A08 → A09 → A11 → A12 → A13
Total: 5 + 3 + 4 + 2 + 2 + 2 + 2 + 3 + 1 + 5 + 5 + 8 = 42 cost units
Estimated time: ~87 minutes (with CI buffer)
```

---

## 6. Agent Skill Integration

### 6.1 Skill Mapping

| Skill | Actions | Invocation |
|-------|---------|------------|
| `model-training` | A01, A11, A12, A13 | `skill: "model-training"` |
| `code-quality` | A02 | `skill: "code-quality"` |
| `testing-workflow` | A03 | `skill: "testing-workflow"` |
| `git-workflow` | A04, A05, A06, A07, A08 | `skill: "git-workflow"` |
| `ci-monitor` | A09, A10 | `skill: "ci-monitor"` |
| `agents-md` | A14, A15 | `skill: "agents-md"` |

### 6.2 Skill Coordination Pattern

```python
# Pseudocode for skill coordination
def execute_deployment_plan():
    world_state = init_world_state()
    
    # Phase 1: Validation (parallel where possible)
    invoke_skill("model-training")  # A01
    invoke_skill("code-quality")    # A02
    invoke_skill("testing-workflow") # A03
    
    # Phase 2: Branch
    invoke_skill("git-workflow")    # A04
    
    # Phase 3: Commits
    invoke_skill("git-workflow")    # A05, A06, A07
    
    # Phase 4: Push & CI
    invoke_skill("git-workflow")    # A08
    invoke_skill("ci-monitor")      # A09
    
    # Handle CI failures
    if not world_state.ci_passed:
        invoke_skill("ci-monitor")  # A10
        goto Phase 4
    
    # Phase 5: Deployment
    invoke_skill("model-training")  # A11, A12, A13
    
    # Phase 6: Documentation (parallel)
    invoke_skill("agents-md")       # A14, A15
```

---

## 7. Progress Tracking

### 7.1 State File

Location: `plans/deployment_state.json`

```json
{
  "deployment_id": "deploy-2026-02-25-001",
  "started_at": "2026-02-25T10:00:00Z",
  "completed_at": null,
  "current_phase": "Phase 1: Validation",
  "last_action": null,
  "actions_completed": [],
  "actions_pending": ["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11", "A12", "A13", "A14", "A15"],
  "world_state": {
    "code_validation": {
      "modal_cli_validated": false,
      "ruff_validated": false,
      "typescript_validated": false
    },
    "git_state": {
      "branch_created": false,
      "branch_name": "",
      "commits": [],
      "pushed": false
    },
    "ci_state": {
      "pipeline_running": false,
      "pipeline_passed": false,
      "failures_detected": [],
      "failures_resolved": []
    },
    "deployment_state": {
      "validation_gates_passed": false,
      "model_uploaded_hub": false,
      "mlflow_integrated": false
    },
    "documentation_state": {
      "goap_updated": false,
      "adrs_complete": false
    }
  }
}
```

### 7.2 Progress Commands

```bash
# Check current state
cat plans/deployment_state.json | jq '.current_phase'

# Mark action complete
jq '.actions_completed += ["A01"] | .actions_pending -= ["A01"]' plans/deployment_state.json

# Update world state
jq '.world_state.code_validation.modal_cli_validated = true' plans/deployment_state.json
```

### 7.3 Progress Report Template

```markdown
## Deployment Progress Report

**Deployment ID:** deploy-2026-02-25-001
**Started:** 2026-02-25T10:00:00Z
**Current Phase:** Phase X
**Progress:** X/15 actions complete

### Completed Actions
- [x] A01: validate_modal_cli (T+5)
- [x] A02: validate_ruff (T+8)

### In Progress
- [~] A03: validate_typescript

### Pending Actions
- [ ] A04: create_branch
- [ ] A05-A07: commits
- [ ] A08-A10: CI
- [ ] A11-A13: deployment
- [ ] A14-A15: documentation

### Issues
None

### Next Action
A03: validate_typescript
```

---

## 8. Contingency Planning

### 8.1 CI Failure Handling

```
IF CI fails (A09 detects failure):
  1. Invoke ci-monitor skill for diagnosis
  2. Identify failure type:
     - Linting failure → Invoke code-quality skill, fix, A05 (new commit)
     - Test failure → Invoke testing-workflow skill, fix, A05 (new commit)
     - Build failure → Invoke testing-workflow skill, fix, A05 (new commit)
  3. Create fix commit (A05)
  4. Push again (A08)
  5. Resume monitoring (A09)
  6. IF failures > 3: Escalate to human review
```

### 8.2 Validation Gate Failure

```
IF validation gates fail (A11):
  1. Review validation_report.json
  2. Identify failure type:
     - Model size exceeded → Optimize model, retrain
     - Accuracy below threshold → Retrain with more data
     - NaN/Inf weights → Debug training, retrain
  3. Fix root cause
  4. Retrain model
  5. Re-run validation (A11)
```

### 8.3 HuggingFace Upload Failure

```
IF upload fails (A12):
  1. Check HF_TOKEN validity
  2. Check network connectivity
  3. Check repo permissions
  4. Retry with exponential backoff
  5. IF still failing: Create issue, escalate
```

---

## 9. Success Criteria

### 9.1 Definition of Done

- [ ] All 15 actions completed
- [ ] World state shows all flags = true
- [ ] Git branch exists with 5+ atomic commits
- [ ] CI pipeline passed (green checkmarks)
- [ ] Model uploaded to HuggingFace Hub
- [ ] Validation report shows PASSED
- [ ] ADRs 026-029 complete and linked
- [ ] GOAP.md updated with Phase 10 status

### 9.2 Quality Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| Validation pass rate | 100% | - |
| CI pass rate | 100% | - |
| Commit atomicity | 5 commits | - |
| Model size | <100MB | - |
| Documentation completeness | 100% | - |

---

## 10. References

- **ADR-029:** Production Deployment GOAP Plan 2026
- **ADR-026:** HuggingFace Model Publishing Implementation
- **ADR-027:** Experiment Tracking with MLflow
- **ADR-028:** Model Validation Gates
- **GOAP.md:** Main goal tracking document
- **Agent Skills:** .agents/skills/

---

## Appendix A: Action Checklists

### A01: validate_modal_cli Checklist
- [ ] Install modal CLI if needed
- [ ] Run `modal token set` if not authenticated
- [ ] Execute test training run: `modal run src/train.py -- --epochs 1`
- [ ] Verify training completes without errors
- [ ] Check checkpoint saved to volume
- [ ] Update deployment_state.json

### A02: validate_ruff Checklist
- [ ] Run `ruff check . --fix`
- [ ] Run `ruff format . --check`
- [ ] Fix any remaining issues
- [ ] Verify no errors in output
- [ ] Update deployment_state.json

### A03: validate_typescript Checklist
- [ ] Navigate to frontend/
- [ ] Run `npm run build`
- [ ] Verify no TypeScript errors
- [ ] Check build output in dist/
- [ ] Update deployment_state.json

### A04: create_branch Checklist
- [ ] Ensure working tree is clean
- [ ] Run `git checkout -b feature/production-deployment-2026`
- [ ] Verify branch created
- [ ] Update deployment_state.json

### A05-A07: Commits Checklist
- [ ] Stage files for each commit
- [ ] Write descriptive commit messages
- [ ] Verify commit in git log
- [ ] Update deployment_state.json

### A08: push_to_github Checklist
- [ ] Run `git push -u origin feature/production-deployment-2026`
- [ ] Verify push successful
- [ ] Note CI pipeline URL
- [ ] Update deployment_state.json

### A09: monitor_ci_status Checklist
- [ ] Open GitHub Actions tab
- [ ] Monitor pipeline progress
- [ ] Wait for completion
- [ ] Record pass/fail status
- [ ] Update deployment_state.json

### A10: resolve_ci_failure Checklist
- [ ] Identify failing job
- [ ] Read error logs
- [ ] Determine root cause
- [ ] Apply fix
- [ ] Commit fix
- [ ] Push and re-monitor
- [ ] Update deployment_state.json

### A11: run_validation_gates Checklist
- [ ] Run `python src/validate_model.py checkpoints/tinydit_final.pt --check-all --verbose`
- [ ] Review validation_report.json
- [ ] Verify all critical checks passed
- [ ] Update deployment_state.json

### A12: upload_to_huggingface Checklist
- [ ] Ensure HF_TOKEN is set
- [ ] Run `python src/upload_to_hub.py checkpoints/tinydit_final.pt --upload --generate-samples`
- [ ] Verify upload successful
- [ ] Check HuggingFace repo page
- [ ] Update deployment_state.json

### A13: integrate_mlflow Checklist
- [ ] Add mlflow to requirements-modal.txt (already done)
- [ ] Create src/experiment_tracker.py
- [ ] Integrate into train.py
- [ ] Integrate into train_dit.py
- [ ] Test tracking locally
- [ ] Update deployment_state.json

### A14: update_goap_md Checklist
- [ ] Open plans/GOAP.md
- [ ] Update Phase 10 status
- [ ] Mark completed items
- [ ] Add timeline results
- [ ] Update deployment_state.json

### A15: complete_adrs Checklist
- [ ] Verify ADR-026 complete
- [ ] Verify ADR-027 complete
- [ ] Verify ADR-028 complete
- [ ] Verify ADR-029 complete
- [ ] Check cross-references
- [ ] Update deployment_state.json
