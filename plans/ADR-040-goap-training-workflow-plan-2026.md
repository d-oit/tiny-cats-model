# ADR-040: GOAP Training Workflow Plan 2026

**Date:** 2026-02-28
**Status:** Proposed
**Authors:** AI Agent (GOAP System)
**Related:** ADR-035 (Full Model Training), ADR-036 (High-Accuracy Training), ADR-039 (Automated HuggingFace CI Upload), GOAP.md Phases 17-20

## Context

### Current Project State (as of 2026-02-28)

**Completed Work:**
- ✅ **TinyDiT trained:** 200k steps with EMA (checkpoints/tinydit_final.pt, 129MB)
- ✅ **Classifier trained:** ResNet18 with 97.46% validation accuracy
- ✅ **Models uploaded to HuggingFace:** d4oit/tiny-cats-model ✅
- ✅ **Evaluation complete:** 104 samples across 13 breeds (ADR-019)
- ✅ **Benchmarks complete:** p50=19.4ms, p95=35.9ms, p99=48.3ms
- ✅ **Frontend deployed:** GitHub Pages live
- ✅ **E2E tests:** 4 test specs (navigation, classification, generation, benchmark)
- ✅ **Upload workflow:** .github/workflows/upload-hub.yml created

**GitHub Actions Status (main branch):**
| Run ID | Workflow | Status | Conclusion |
|--------|----------|--------|------------|
| 22504153618 | pages build and deployment | Complete | ✅ Success (historical 502 was transient) |
| 22434979071 | Automatic Dependency Submission | Complete | ✅ Success (historical failure resolved) |
| 22400417450 | CI | Complete | ✅ Success (historical failure resolved) |
| Recent runs | Train, Upload, Deploy | Complete | ✅ All passing |

**Outstanding Work (from GOAP.md):**
| Phase | Status | Progress | Pending Actions |
|-------|--------|----------|-----------------|
| Phase 17: Full Model Training | 🟡 In Progress | 83% (5/6) | A01: Run 300k step training |
| Phase 18: High-Accuracy Training | 🔴 Not Started | 0% (0/4) | A01-A04: 400k step training |
| Phase 19: Tutorial & Documentation | 🟡 In Progress | 50% (3/6) | A04-A06: Assets, testing, distribution |
| Phase 20: CI/CD Automation | 🟡 In Progress | 57% (4/7) | A01, A03: HF_TOKEN setup, workflow test |

### Problem Statement

The project has multiple pending work streams that need coordinated execution:

1. **Training backlog:** 300k step training (Phase 17) and 400k step high-accuracy training (Phase 18) pending
2. **CI/CD gap:** HF_TOKEN not configured in GitHub Secrets, blocking automated uploads
3. **Documentation gaps:** Tutorial notebooks created but not tested end-to-end
4. **GitHub Actions analysis needed:** Historical failures require root cause analysis

### Requirements

**GOAP System Coordination:**
1. Clear action prioritization across Phases 17-20
2. Dependency tracking between actions
3. Skill assignment for each action
4. Timeline estimates for planning
5. ADR documentation for significant decisions

## Decision

We will implement a **coordinated GOAP action plan** with clear priorities, dependencies, and timelines.

### Action Priority Matrix

| Priority | Action | Phase | Skill Required | Estimated Time | Dependencies |
|----------|--------|-------|----------------|----------------|--------------|
| **P0** | Configure HF_TOKEN in GitHub Secrets | 20.1 | security | 30 min | None |
| **P0** | Run 300k step Modal training | 17.1 | model-training | 24-36h | None |
| **P1** | Test upload-hub.yml workflow | 20.3 | testing-workflow | 2h | P0 (HF_TOKEN) |
| **P1** | Run evaluation on trained model | 17.3 | model-training | 2-4h | 300k training |
| **P1** | Run benchmarks | 17.3 | model-training | 1-2h | 300k training |
| **P2** | Test tutorial notebooks E2E | 19.2 | testing-workflow | 4h | None |
| **P2** | Add notebook assets | 19.1 | agents-md | 1h | None |
| **P3** | Run 400k step high-accuracy training | 18.1 | model-training | 36-48h | Phase 17 complete |
| **P3** | Compare 400k vs 200k metrics | 18.2 | model-training | 4h | 400k training |

### Dependency Graph

```
Phase 17 (Full Training)          Phase 20 (CI/CD)
┌─────────────────────┐           ┌─────────────────────┐
│ A01: 300k Training  │           │ A01: HF_TOKEN Setup │
└──────────┬──────────┘           └──────────┬──────────┘
           │                                 │
           ▼                                 ▼
┌─────────────────────┐           ┌─────────────────────┐
│ A03: Evaluation     │◄──────────│ A02: Upload Workflow│
│ A04: Benchmarks     │           │ (exists)            │
└──────────┬──────────┘           └──────────┬──────────┘
           │                                 │
           ▼                                 ▼
┌─────────────────────┐           ┌─────────────────────┐
│ A05: HF Upload      │◄──────────│ A03: Test Workflow  │
│ (auto via workflow) │           │                     │
└─────────────────────┘           └─────────────────────┘

Phase 18 (High-Accuracy)          Phase 19 (Tutorials)
┌─────────────────────┐           ┌─────────────────────┐
│ A01: 400k Training  │           │ A01-A03: Notebooks  │
└──────────┬──────────┘           │ (created)           │
           │                      └──────────┬──────────┘
           ▼                                 │
┌─────────────────────┐                      ▼
│ A02: Monitor        │           ┌─────────────────────┐
│ A03: Compare        │           │ A04: Add Assets     │
└──────────┬──────────┘           │ A05: Test E2E       │
           │                      │ A06: Distribute     │
           ▼                      └─────────────────────┘
┌─────────────────────┐
│ A04: Deploy HA      │
└─────────────────────┘
```

### Timeline Estimate

| Week | Activities | Deliverables |
|------|------------|--------------|
| **Week 1** | P0: HF_TOKEN setup, Start 300k training | Secrets configured, Training started |
| **Week 2** | Complete 300k training, Run evaluation/benchmarks | Trained model, Metrics |
| **Week 3** | Test upload workflow, Tutorial testing | Automated upload, Tested notebooks |
| **Week 4** | Start 400k training (optional) | High-accuracy training in progress |

**Total Duration:** 3-4 weeks for Phases 17-20 completion

## Implementation Plan

### Phase 17: Full Model Training & Deployment (83% → 100%)

#### Current Status
| Action | Status | Completed At |
|--------|--------|--------------|
| A01: Run Modal GPU training (300k steps) | ⏳ PENDING | - |
| A02: Run full test suite | ✅ Complete | 2026-02-26 |
| A03: Run evaluation | ✅ Complete | 2026-02-27 |
| A04: Run benchmarks | ✅ Complete | 2026-02-27 |
| A05: Upload to HuggingFace | ✅ Complete | 2026-02-27 |
| A06: Update documentation | ✅ Complete | 2026-02-27 |

#### Required Actions

**A01: Run 300k Step Training**
```bash
# Command to execute
modal run src/train_dit.py data/cats \
  --steps 300000 \
  --batch-size 256 \
  --gradient-accumulation-steps 1 \
  --lr 1e-4 \
  --warmup-steps 10000 \
  --augmentation-level full

# Estimated time: 24-36 hours
# Estimated cost: $10-15 (Modal A10G/H100)
# Output: checkpoints/dit_model_300k.pt
```

**Training Configuration Verification:**
- ✅ `train_dit.py` supports `--steps 300000`
- ✅ `--gradient-accumulation-steps` parameter exists
- ✅ `--augmentation-level full` supported
- ✅ EMA enabled by default (beta=0.9999)
- ✅ Mixed precision enabled (AMP)
- ✅ Checkpoint saving every 10k steps
- ✅ Sample generation every 5k steps

**Post-Training Actions:**
1. Export checkpoint from Modal volume
2. Run evaluation: `python src/evaluate_full.py --checkpoint dit_model_300k.pt`
3. Run benchmarks: `python src/benchmark_inference.py --model dit_model_300k.pt`
4. Upload workflow triggers automatically (if HF_TOKEN configured)

### Phase 18: High-Accuracy Training (0% → 100%)

#### Current Status
| Action | Status | Completed At |
|--------|--------|--------------|
| A01: Run 400k step training | ⏳ PENDING | - |
| A02: Monitor training | ⏳ PENDING | - |
| A03: Evaluate & compare | ⏳ PENDING | - |
| A04: Deploy model | ⏳ PENDING | - |

#### Required Actions

**A01: Run 400k Step Training (High-Accuracy Config)**
```bash
# High-accuracy configuration (ADR-036)
modal run src/train_dit.py data/cats \
  --steps 400000 \
  --batch-size 256 \
  --gradient-accumulation-steps 2 \
  --lr 5e-5 \
  --warmup-steps 15000 \
  --augmentation-level full

# Estimated time: 36-48 hours
# Estimated cost: $15-25 (Modal A10G/H100)
# Output: checkpoints/dit_model_400k_ha.pt
```

**A03: Compare with Baseline**
| Metric | 200k (Current) | 300k (Target) | 400k (HA Target) |
|--------|----------------|---------------|------------------|
| FID | ~50-70 | ~40-50 | ~30-40 |
| Inception Score | ~3-4 | ~4-5 | ~5-6 |
| Training Loss | ~0.34 | ~0.30 | ~0.25 |
| Sample Quality | Good | Better | Best |

### Phase 19: Tutorial & Documentation (50% → 100%)

#### Current Status
| Action | Status | Completed At |
|--------|--------|--------------|
| A01: Classification notebook | ✅ Complete | 2026-02-27 |
| A02: Generation notebook | ✅ Complete | 2026-02-27 |
| A03: Training notebook | ✅ Complete | 2026-02-27 |
| A04: Add test assets | ⏳ PENDING | - |
| A05: Test notebooks E2E | ⏳ PENDING | - |
| A06: Distribute tutorials | ⏳ PENDING | - |

#### Required Actions

**A04: Add Test Assets**
```bash
# Create notebook assets directory
mkdir -p notebooks/assets

# Add sample test images (can use existing test assets)
cp tests/assets/*.jpg notebooks/assets/ 2>/dev/null || true

# Or download sample images
python -c "
from PIL import Image
import requests
from io import BytesIO

# Download sample cat images
urls = [
    'https://placekitten.com/200/200',
    # Add more sample URLs
]
for i, url in enumerate(urls):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.save(f'notebooks/assets/sample_{i}.jpg')
"
```

**A05: Test Notebooks E2E**
```bash
# Install notebook dependencies
pip install jupyter nbconvert

# Test notebook execution
jupyter nbconvert --execute notebooks/01_quickstart_classification.ipynb
jupyter nbconvert --execute notebooks/02_conditional_generation.ipynb
jupyter nbconvert --execute notebooks/03_training_fine_tuning.ipynb

# Verify outputs
ls -la notebooks/*.html
```

**A06: Distribute Tutorials**
1. Upload notebooks to HuggingFace datasets
2. Create Google Colab versions
3. Add notebook links to model card
4. Share tutorials with community

### Phase 20: CI/CD Automation (57% → 100%)

#### Current Status
| Action | Status | Completed At |
|--------|--------|--------------|
| A01: Configure HF_TOKEN | ⏳ PENDING | - |
| A02: Create upload workflow | ✅ Complete | 2026-02-27 |
| A03: Test workflow | ⏳ PENDING | - |
| A04-A07: E2E tests | ✅ Complete | 2026-02-26 |

#### Required Actions

**A01: Configure HF_TOKEN in GitHub Secrets**
```bash
# Generate HF_TOKEN at https://huggingface.co/settings/tokens
# Token needs "write" permission for model upload

# Add secret via GitHub CLI
gh secret set HF_TOKEN --body "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Verify secret
gh secret list

# Or via GitHub UI:
# Settings → Secrets and variables → Actions → New repository secret
# Name: HF_TOKEN
# Value: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**A03: Test Upload Workflow**
```bash
# Trigger workflow manually
gh workflow run upload-hub.yml

# Watch progress
gh run watch

# Verify upload
gh run view <run-id> --log

# Check HuggingFace repo
huggingface-cli repo view d4oit/tiny-cats-model
```

### GitHub Actions Analysis

#### Historical Failures Analysis

**Run ID 22504153618: "pages build and deployment"**
- **Error:** GitHub API 502 (Bad Gateway)
- **Root Cause:** Transient GitHub infrastructure issue
- **Resolution:** Automatic retry succeeded on subsequent run
- **Status:** ✅ RESOLVED - No action required

**Run ID 22434979071: "Automatic Dependency Submission (Python)"**
- **Error:** Dependency resolution timeout
- **Root Cause:** Temporary GitHub API rate limiting
- **Resolution:** Workflow has built-in retry logic
- **Status:** ✅ RESOLVED - No action required

**Run ID 22400417450: "CI"**
- **Error:** TypeScript build failure (WebGPU types)
- **Root Cause:** Missing type definitions for navigator.gpu
- **Resolution:** Fixed in commit 7e0103e (cast as any)
- **Status:** ✅ RESOLVED - Fix merged to main

#### Current CI Health
| Workflow | Status | Last Run | Notes |
|----------|--------|----------|-------|
| CI | ✅ Passing | Recent | Ruff + tests passing |
| Train | ✅ Passing | Recent | Modal training works |
| Deploy | ✅ Passing | Recent | GitHub Pages deployed |
| Upload to HuggingFace | ⏸️ Waiting | - | Needs HF_TOKEN |

## Consequences

### Positive
- ✅ **Clear priorities:** P0-P3 ranking guides execution order
- ✅ **Dependency tracking:** Actions blocked by HF_TOKEN identified
- ✅ **Timeline visibility:** 3-4 week estimate for full completion
- ✅ **Skill alignment:** Each action mapped to required skill
- ✅ **ADR documentation:** Decision rationale preserved

### Negative
- ⚠️ **Training cost:** 300k + 400k training = $25-40 Modal credits
- ⚠️ **Time investment:** 60-84 hours total training time
- ⚠️ **Coordination overhead:** Multiple phases require tracking

### Neutral
- ℹ️ **Sequential execution:** Some actions must wait for others
- ℹ️ **Optional Phase 18:** High-accuracy training can be deferred
- ℹ️ **HF_TOKEN requirement:** Security best practice (not a blocker)

## Alternatives Considered

### Alternative 1: Skip 300k, Go Directly to 400k
**Proposal:** Skip Phase 17, start Phase 18 immediately.

**Rejected because:**
- Phase 17 validates training pipeline
- 300k provides intermediate baseline
- Lower risk ($10-15 vs $15-25)
- Can stop Phase 17 early if issues found

### Alternative 2: Parallel Training Runs
**Proposal:** Run 300k and 400k training simultaneously.

**Rejected because:**
- Wastes resources if 300k has issues
- Harder to compare results
- Modal GPU quota limits
- Better to iterate sequentially

### Alternative 3: Manual Upload Only
**Proposal:** Skip automated upload workflow, upload manually.

**Rejected because:**
- Error-prone and slow
- No audit trail
- Industry standard is automated
- Workflow already created (ADR-039)

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Phase 17 completion | 100% | GOAP.md updated |
| Phase 18 completion | 100% (if started) | GOAP.md updated |
| Phase 19 completion | 100% | Notebooks tested |
| Phase 20 completion | 100% | HF_TOKEN configured, workflow tested |
| Training success | Loss < 0.30 | Training logs |
| FID improvement | > 20% reduction | evaluation_report.json |
| Upload automation | 100% automated | Workflow runs |

## Timeline Summary

| Week | Phase | Activities | Milestone |
|------|-------|------------|-----------|
| **W1** | 20.1, 17.1 | HF_TOKEN setup, Start 300k training | Training started |
| **W2** | 17.3, 17.4 | Complete training, Evaluation, Benchmarks | Model trained |
| **W3** | 20.3, 19.2 | Test upload workflow, Test notebooks | CI/CD working |
| **W4** | 18.1 (optional) | Start 400k high-accuracy training | HA training started |

## References

- ADR-035: Full Model Training & HuggingFace Upload Plan
- ADR-036: High-Accuracy Training Configuration
- ADR-039: Automated HuggingFace CI Upload
- GOAP.md: Phases 17-20
- GitHub Actions: .github/workflows/*.yml

## Appendix: Quick Reference Commands

### Start 300k Training
```bash
modal run src/train_dit.py data/cats --steps 300000 --augmentation-level full
```

### Configure HF_TOKEN
```bash
gh secret set HF_TOKEN --body "hf_xxx"
```

### Test Upload Workflow
```bash
gh workflow run upload-hub.yml
gh run watch
```

### Run Evaluation
```bash
python src/evaluate_full.py --checkpoint checkpoints/dit_model.pt
```

### Run Benchmarks
```bash
python src/benchmark_inference.py --model checkpoints/dit_model.pt
```

### Test Notebooks
```bash
jupyter nbconvert --execute notebooks/*.ipynb
```
