# GOAP Production Deployment - Quick Reference Card

**Deployment ID:** deploy-2026-02-25-001  
**Branch:** `feature/production-deployment-2026`  
**Status:** READY TO START  

---

## 🎯 Goal

Production deployment of tiny-cats-model to Modal with HuggingFace Hub publishing.

---

## 📊 Current State

| Category | Status |
|----------|--------|
| Code Validation | ⏳ Not Started |
| Git Branch | ⏳ Not Started |
| CI/CD | ⏳ Not Started |
| Deployment | ⏳ Not Started |
| Documentation | ✅ ADRs 026-029 Ready |

---

## 🎬 Action Sequence (15 Actions)

### Phase 1: Validation (15 min)
```
A01 → A02 → A03
```
- **A01** `validate_modal_cli` - Test Modal CLI training
- **A02** `validate_ruff` - Run Ruff linting
- **A03** `validate_typescript` - Run TypeScript build

### Phase 2: Branch (2 min)
```
A04
```
- **A04** `create_branch` - Create `feature/production-deployment-2026`

### Phase 3: Commits (10 min)
```
A05 → A06 → A07
```
- **A05** `commit_code_changes` - validate_model.py, upload_to_hub.py
- **A06** `commit_documentation` - ADRs 026-029, GOAP.md
- **A07** `commit_utility_adds` - modal_monitor.py

### Phase 4: Push & CI (30-60 min)
```
A08 → A09 → [A10 if needed]
```
- **A08** `push_to_github` - Push with proper commit messages
- **A09** `monitor_ci_status` - Track CI/CD pipeline
- **A10** `resolve_ci_failure` - Fix issues (if any)

### Phase 5: Deployment (20 min)
```
A11 → A12 → A13
```
- **A11** `run_validation_gates` - Execute validate_model.py
- **A12** `upload_to_huggingface` - Upload to HuggingFace Hub
- **A13** `integrate_mlflow` - Add MLflow tracking

### Phase 6: Documentation (10 min, parallel)
```
A14, A15
```
- **A14** `update_goap_md` - Update GOAP.md Phase 10
- **A15** `complete_adrs` - Ensure ADRs complete

---

## 🔧 Skill Invocations

| Action | Skill Command |
|--------|---------------|
| A01, A11-A13 | `skill: "model-training"` |
| A02 | `skill: "code-quality"` |
| A03 | `skill: "testing-workflow"` |
| A04-A08 | `skill: "git-workflow"` |
| A09-A10 | `skill: "ci-monitor"` |
| A14-A15 | `skill: "agents-md"` |

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `plans/GOAP-DEPLOYMENT-PLAN-2026.md` | Complete GOAP plan |
| `plans/ADR-029-production-deployment-goap-2026.md` | ADR for deployment strategy |
| `plans/deployment_state.json` | State tracking file |
| `plans/GOAP.md` | Main goal tracking (updated) |
| `src/validate_model.py` | Model validation gates |
| `src/upload_to_hub.py` | HuggingFace upload utility |

---

## ✅ Success Criteria

- [ ] All 15 actions completed
- [ ] Branch `feature/production-deployment-2026` exists
- [ ] 5+ atomic commits pushed
- [ ] CI pipeline passed (green)
- [ ] Model uploaded to HuggingFace
- [ ] Validation report shows PASSED
- [ ] ADRs 026-029 complete

---

## 🚀 Quick Start

```bash
# 1. Check current state
cat plans/deployment_state.json | jq '.current_phase'

# 2. Start Phase 1: Validation
# Invoke skills in sequence:
# - model-training (A01)
# - code-quality (A02)
# - testing-workflow (A03)

# 3. Update state after each action
jq '.actions_completed += ["A01"] | .actions_pending -= ["A01"]' \
   plans/deployment_state.json > tmp && mv tmp plans/deployment_state.json
```

---

## 📞 Escalation

| Issue | Action |
|-------|--------|
| CI fails 3+ times | Human review required |
| Validation gates fail | Retrain model |
| HuggingFace upload fails | Check HF_TOKEN, network |
| Git conflicts | Rebase on main, resolve |

---

## 📈 Progress Tracking

Update `plans/deployment_state.json` after each action:

```json
{
  "current_phase": "Phase X",
  "last_action": "A##",
  "actions_completed": ["A01", "A02", ...],
  "world_state": { ... }
}
```

---

## 🔗 References

- **Full Plan:** `plans/GOAP-DEPLOYMENT-PLAN-2026.md`
- **ADR:** `plans/ADR-029-production-deployment-goap-2026.md`
- **State:** `plans/deployment_state.json`
- **GOAP:** `plans/GOAP.md` (Phase 10)

---

**Generated:** 2026-02-25  
**Version:** 1.0  
**Status:** READY
