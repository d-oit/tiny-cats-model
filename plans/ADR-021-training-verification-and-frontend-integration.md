# ADR-021: Training Verification & Frontend Integration

## Status
Implemented

## Date
2026-02-25

## Context

After training the TinyDiT model with Modal GPU, we need to ensure:
1. The trained model can be exported to ONNX correctly
2. The ONNX model works in the frontend browser environment
3. The generation pipeline produces valid images

## Decision

### 1. Post-Training Verification Pipeline

After each Modal training run, execute verification:

```bash
# Step 1: Verify checkpoint exists and is valid
python src/verify_checkpoint.py --checkpoint checkpoints/tinydit_final.pt

# Step 2: Export to ONNX with verification
python src/export_dit_onnx.py --checkpoint checkpoints/tinydit_final.pt \
    --verify --test

# Step 3: Run frontend E2E tests
npx playwright test tests/e2e/generation.spec.ts
```

### 2. Checkpoint Verification Script

Create `src/verify_checkpoint.py`:
- Load checkpoint and validate structure
- Verify model can be instantiated
- Test inference with dummy input
- Report model parameters and configuration

### 3. Frontend E2E Tests

Create browser-based tests:
- Model loading test
- Generation with different breeds
- Progress callback test
- Download functionality test

### 4. Integration with Modal Training

Add verification step to `train_dit.py`:

```python
def train_dit_modal(...):
    # ... training code ...
    
    # Verify model after training
    logger.info("Verifying trained model...")
    verify_checkpoint(output)
    export_and_test_onnx(output)
    
    return {"status": "completed", "verified": True}
```

## Consequences

- **Positive**: Catches issues before frontend deployment
- **Positive**: Automated verification in CI/CD
- **Negative**: Additional time for verification (~5 min)
- **Negative**: Requires Playwright for browser tests

## Alternatives Considered

1. Manual verification - rejected (not scalable)
2. Only Python ONNX test - rejected (doesn't verify browser)
3. Skip verification - rejected (risk of broken frontend)

## References

- GOAP.md - Production Training & Publishing
- ADR-019: TinyDiT Sample Evaluation Results
- ADR-020: Modal CLI-First Training Strategy
