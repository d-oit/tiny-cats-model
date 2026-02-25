# Modal Integration Improvement Plan

**Created:** 2026-02-25
**Status:** Proposed
**Related ADRs:** ADR-022, ADR-023, ADR-024, ADR-025

## Overview

This plan implements Modal.com best practices for the tiny-cats-model project to improve:
- Container build times (10-100x faster with uv_pip_install)
- Training reliability (automatic retries with exponential backoff)
- Storage management (organized checkpoints, cleanup policies)
- Cold start latency (30-60s → 10-20s with optimizations)

## Implementation Tasks

### Phase 1: Container Image Optimization (ADR-022)

**Priority:** HIGH
**Estimated Effort:** 2 hours
**Dependencies:** None

#### Task 1.1: Create requirements-modal.txt
- [ ] Create `requirements-modal.txt` with pinned versions
- [ ] Include: torch, torchvision, pillow, tqdm, modal
- [ ] Use exact versions (==) for reproducibility

**Acceptance Criteria:**
- File exists at `requirements-modal.txt`
- All versions pinned with ==
- Compatible with Python 3.12

#### Task 1.2: Update src/train.py Image Configuration
- [ ] Replace `pip_install` with `uv_pip_install`
- [ ] Pin exact package versions
- [ ] Add environment variables (HF_XET_HIGH_PERFORMANCE)
- [ ] Replace `add_local_dir` with specific `add_local_file` calls

**Acceptance Criteria:**
- Image builds successfully
- Training runs without dependency errors
- Build time < 2 minutes (was 5-10 min)

#### Task 1.3: Update src/train_dit.py Image Configuration
- [ ] Same changes as Task 1.2
- [ ] Ensure DiT-specific dependencies included

**Acceptance Criteria:**
- Image builds successfully
- DiT training runs without errors

---

### Phase 2: GPU Configuration and Retry Strategy (ADR-023)

**Priority:** HIGH
**Estimated Effort:** 2 hours
**Dependencies:** Phase 1 complete

#### Task 2.1: Add Retry Configuration to train.py
- [ ] Import `from modal import Retries`
- [ ] Configure retries: max_retries=3, backoff_coefficient=2.0
- [ ] Set initial_delay=10.0, max_delay=300.0
- [ ] Add to @app.function decorator

**Acceptance Criteria:**
- Retries configured in decorator
- Transient failures automatically retried
- Training completes after recoverable errors

#### Task 2.2: Add Retry Configuration to train_dit.py
- [ ] Same as Task 2.1
- [ ] Use max_retries=2 for long-running jobs
- [ ] Use longer delays (initial_delay=30.0)

**Acceptance Criteria:**
- Retries configured for DiT training
- Long jobs have appropriate retry settings

#### Task 2.3: Update modal.yml Documentation
- [ ] Document GPU selection matrix
- [ ] Document retry configuration
- [ ] Add cost estimates per GPU type

**Acceptance Criteria:**
- modal.yml updated with GPU best practices
- Retry configuration documented

---

### Phase 3: Volume and Storage Improvements (ADR-024)

**Priority:** MEDIUM
**Estimated Effort:** 3 hours
**Dependencies:** Phase 1 complete

#### Task 3.1: Create Volume Utility Module
- [ ] Create `src/volume_utils.py`
- [ ] Implement `cleanup_old_checkpoints()` function
- [ ] Implement `ensure_directory_exists()` function
- [ ] Implement `get_checkpoint_metadata()` function

**Acceptance Criteria:**
- Module created with all functions
- Functions tested locally
- Type hints included

#### Task 3.2: Add Volume Commits to train.py
- [ ] Add `output_volume.commit()` after checkpoint saves
- [ ] Add `data_volume.commit()` after dataset download
- [ ] Add cleanup call at end of training

**Acceptance Criteria:**
- Checkpoints persisted across runs
- Old checkpoints cleaned up (keep last 5)
- Dataset cached in volume

#### Task 3.3: Implement Dated Checkpoint Directories
- [ ] Create directory structure: `/outputs/checkpoints/classifier/YYYY-MM-DD/`
- [ ] Update output paths to use dated directories
- [ ] Add training metadata JSON

**Acceptance Criteria:**
- Checkpoints organized by date
- Metadata includes timestamp, config, metrics

#### Task 3.4: Add Dataset Caching
- [ ] Check for cached dataset before download
- [ ] Copy downloaded dataset to volume
- [ ] Commit volume after download

**Acceptance Criteria:**
- Dataset not redownloaded on subsequent runs
- Training starts faster with cached data

---

### Phase 4: Cold Start Optimization (ADR-025)

**Priority:** MEDIUM
**Estimated Effort:** 3 hours
**Dependencies:** Phase 1 complete

#### Task 4.1: Create TrainingContainer Class in train.py
- [ ] Create `TrainingContainer` class
- [ ] Add `@enter()` method for initialization
- [ ] Move path setup to @enter()
- [ ] Add CUDA warm-up in @enter()

**Acceptance Criteria:**
- Container class created
- @enter() runs once on container start
- CUDA initialized before training

#### Task 4.2: Add CUDA Warm-up
- [ ] Allocate dummy tensor in @enter()
- [ ] Run small conv operation
- [ ] Clear warm-up tensors

**Acceptance Criteria:**
- First training step not delayed by CUDA init
- Cold start time reduced

#### Task 4.3: Update train_dit.py with Container Pattern
- [ ] Same pattern as train.py
- [ ] Pre-load DiT modules in @enter()

**Acceptance Criteria:**
- DiT training uses container pattern
- Cold start optimized

---

### Phase 5: Documentation Updates

**Priority:** MEDIUM
**Estimated Effort:** 2 hours
**Dependencies:** All phases complete

#### Task 5.1: Update model-training Skill
- [ ] Update `.agents/skills/model-training/SKILL.md`
- [ ] Add new Modal best practices
- [ ] Update command examples
- [ ] Add troubleshooting section

**Acceptance Criteria:**
- Skill documentation current
- Examples match new patterns

#### Task 5.2: Update AGENTS.md
- [ ] Add Modal best practices section
- [ ] Update training commands
- [ ] Link to new ADRs

**Acceptance Criteria:**
- AGENTS.md reflects new patterns
- Quick reference up to date

#### Task 5.3: Update ADR-020
- [ ] Reference new ADRs (022-025)
- [ ] Update image configuration examples
- [ ] Add retry configuration examples

**Acceptance Criteria:**
- ADR-020 links to new ADRs
- Examples consistent with implementation

---

## Task Dependencies

```
Phase 1 (Image Optimization)
├── Task 1.1: requirements-modal.txt
├── Task 1.2: train.py image ─────────┐
└── Task 1.3: train_dit.py image ─────┤
                                      │
Phase 2 (GPU & Retry)                 │
├── Task 2.1: train.py retries ───────┤
├── Task 2.2: train_dit.py retries ───┤
└── Task 2.3: modal.yml update        │
                                      │
Phase 3 (Volumes)                     │
├── Task 3.1: volume_utils.py ────────┤
├── Task 3.2: train.py commits ───────┤
├── Task 3.3: dated directories ──────┤
└── Task 3.4: dataset caching ────────┤
                                      │
Phase 4 (Cold Start)                  │
├── Task 4.1: TrainingContainer ──────┤
├── Task 4.2: CUDA warm-up ───────────┤
└── Task 4.3: train_dit.py container ─┤
                                      │
Phase 5 (Documentation)               │
├── Task 5.1: model-training skill ───┼── All phases
├── Task 5.2: AGENTS.md ──────────────┼── must be
└── Task 5.3: ADR-020 update ─────────┘── complete
```

## Priority Ordering

1. **Phase 1** (HIGH): Foundation for all other improvements
2. **Phase 2** (HIGH): Critical for training reliability
3. **Phase 3** (MEDIUM): Important for storage management
4. **Phase 4** (MEDIUM): Improves developer experience
5. **Phase 5** (MEDIUM): Ensures knowledge transfer

## Testing Plan

### Unit Tests
- [ ] Test volume utility functions
- [ ] Test retry behavior with mock failures
- [ ] Test checkpoint cleanup logic

### Integration Tests
- [ ] Run classifier training with new image
- [ ] Run DiT training with new image
- [ ] Verify checkpoint persistence
- [ ] Measure cold start times

### Acceptance Tests
- [ ] Build time < 2 minutes
- [ ] Cold start < 20 seconds
- [ ] Warm start < 2 seconds
- [ ] Retries recover from transient failures
- [ ] Checkpoints organized by date

## Rollback Plan

If issues occur:
1. Revert individual phases via git
2. Old training scripts remain functional
3. No breaking changes to CLI interface

## Success Metrics

| Metric | Before | Target | After |
|--------|--------|--------|-------|
| Image build time | 5-10 min | <2 min | TBD |
| Cold start | 30-60s | <20s | TBD |
| Warm start | N/A | <2s | TBD |
| Transient failure recovery | Manual | Automatic | TBD |
| Checkpoint organization | Flat | Dated | TBD |

## Files to Modify

| File | Changes | Phase |
|------|---------|-------|
| `requirements-modal.txt` | Create new | 1.1 |
| `src/train.py` | Image, retries, volumes, container | 1.2, 2.1, 3.2, 4.1 |
| `src/train_dit.py` | Image, retries, volumes, container | 1.3, 2.2, 3.2, 4.3 |
| `src/volume_utils.py` | Create new | 3.1 |
| `modal.yml` | Update documentation | 2.3, 5.3 |
| `.agents/skills/model-training/SKILL.md` | Update | 5.1 |
| `AGENTS.md` | Update | 5.2 |
| `plans/ADR-020.md` | Update references | 5.3 |

## Related Issues

- ADR-022: Container Image Optimization
- ADR-023: GPU Resource Configuration and Retry Strategy
- ADR-024: Volume and Storage Best Practices
- ADR-025: Cold Start Optimization
- ADR-020: Modal CLI-First Training Strategy (existing)
- ADR-010: Modal Training Improvements (existing)
- ADR-017: TinyDiT Training Infrastructure (existing)
