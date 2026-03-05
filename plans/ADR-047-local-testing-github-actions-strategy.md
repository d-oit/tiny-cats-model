# ADR-047: Local Testing vs GitHub Actions Training Strategy

**Date:** 2026-03-02
**Status:** Accepted
**Authors:** AI Agent
**Related:** ADR-044, ADR-036, scripts/train_dit_high_accuracy.sh

## Context

### Training Options Analysis

| Method | Best For | Max Steps | Duration | Cost | Reliability |
|--------|----------|-----------|----------|------|-------------|
| GitHub Actions | Production 400k | 400k+ | 24-36h | ~$20-30 | ⭐⭐⭐⭐⭐ |
| Local Modal (--medium) | Medium runs | 50k | 2-3h | ~$5-8 | ⭐⭐⭐ |
| Local Modal (--local) | Quick tests | 4k | 5-10m | ~$0.50 | ⭐⭐⭐⭐ |
| nohup modal run | ❌ Not recommended | - | - | - | ⭐ |

### Decision Matrix

**Use GitHub Actions when:**
- Running 400k steps (production training)
- Need reliability for long runs (>1 hour)
- Want automatic artifact upload
- Need CI/CD integration

**Use Local Modal (--local) when:**
- Testing configuration changes
- Verifying dataset loads correctly
- Quick smoke tests (< 10 minutes)
- Debugging import issues

**Use Local Modal (--medium) when:**
- Running medium experiments (10k-50k steps)
- Testing hyperparameter changes
- Cost-sensitive development

**Never use nohup modal run because:**
- SIGHUP termination risk (ADR-044)
- No checkpoint recovery on failure
- Silent failures
- Poor observability

## Decision

### Standardized Training Workflow

```bash
# Step 1: Always test locally first
bash scripts/train_dit_high_accuracy.sh --local

# Step 2: If local test passes, run full training via GitHub Actions
gh workflow run train.yml
```

### Script Modes

**train_dit_high_accuracy.sh** now supports:

```bash
# Mode 1: Quick test (default recommendation)
bash scripts/train_dit_high_accuracy.sh --local
# - 4000 steps
# - ~5-10 minutes
# - Verifies setup works

# Mode 2: Medium run
bash scripts/train_dit_high_accuracy.sh --medium
# - 50000 steps
# - ~2-3 hours
# - Interactive confirmation required

# Mode 3: Production (shows guidance only)
bash scripts/train_dit_high_accuracy.sh
# - Shows GitHub Actions command
# - Explains why GA is better
```

### Prerequisites Check

The script now verifies:
1. ✅ Modal authentication (`modal token info`)
2. ✅ Dataset availability (`data/cats`)
3. ✅ Image count (diagnostic)

## Implementation

### Script Updates

Added to `scripts/train_dit_high_accuracy.sh`:
- `check_prerequisites()` function
- `--local` flag for 4000-step test
- `--medium` flag for 50000-step run
- Interactive confirmation for medium runs
- Clear guidance for production runs

### Workflow Integration

GitHub Actions workflow defaults (train.yml):
- `steps`: 400000
- `lr`: 5e-5
- `gradient_accumulation_steps`: 2
- `batch_size`: 256

## Consequences

### Positive
- ✅ Clear separation between testing and production
- ✅ Prevents accidental long local runs
- ✅ Prerequisites validation catches issues early
- ✅ Consistent with ADR-044 (no nohup)

### Negative
- ⚠️ Requires GitHub Actions for full training
- ⚠️ Two-step process (local test → GA)

### Neutral
- ℹ️ Local testing still available for debugging
- ℹ️ Medium runs possible with explicit confirmation

## Verification Checklist

Before running 400k training:

- [ ] Local test passed (`--local`)
- [ ] Modal authentication verified
- [ ] Dataset present (data/cats)
- [ ] GitHub Actions workflow file valid
- [ ] Secrets configured (MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, HF_TOKEN)

## References

- ADR-044: Modal training 400k termination fix (no nohup)
- ADR-036: High-accuracy training configuration
- scripts/train_dit_high_accuracy.sh
- .github/workflows/train.yml
