# ADR-033: Codebase Analysis - Missing Implementations

## Status
Accepted

## Context
Analysis swarm (RYAN, FLASH, SOCRATES) identified critical gaps in the tiny-cats-model implementation. This ADR documents findings and planned remediation.

## Analysis Summary

### Critical Gaps (Blocking)

| Gap | Evidence | Impact |
|-----|----------|--------|
| **MLflow NOT integrated** | `deployment_state.json:54` shows `mlflow_integrated: false` | Experiment tracking unavailable |
| **Classifier ONNX export missing** | No `export_classifier_onnx.py` | Frontend classify page won't work |
| **HuggingFace models not uploaded** | GOAP pending items | Frontend can't load models |
| **Generator ONNX not quantized** | 132MB vs 11MB classifier | Slow browser loading |

### Important Gaps

| Gap | Evidence | Impact |
|-----|----------|--------|
| **No automated HF upload in CI** | train.yml manual trigger only | Manual publishing required |
| **E2E tests incomplete** | Only navigation tested | No inference/generation verification |
| **WebGPU fallback missing** | inference.worker.ts | Safari/older browsers fail |

### Minor Gaps (Edge Cases)

- No offline/fallback if HuggingFace models unavailable
- No file size validation on image upload
- No retry mechanism for model loading
- Python 3.12 not tested (Modal uses 3.12)
- No dataset integrity validation

## Decision

1. **Immediate priorities**:
   - Add mlflow to requirements.txt
   - Create classifier ONNX export script
   - Fix constants.ts typo
   - Quantize generator ONNX model

2. **Short-term priorities**:
   - Add E2E tests for inference/generation
   - Add WebGPU fallback to WASM
   - Add automated HF upload in CI

3. **Long-term priorities**:
   - MLflow tracking server setup
   - Model versioning/changelog

## Changes Required

### requirements.txt
```
mlflow>=2.19.0
```

### New Files Needed
- `src/export_classifier_onnx.py` - Export classifier to ONNX

### Files to Fix
- `frontend/src/constants.ts` - Fix `d4oit` → `d-oit`
- `src/optimize_onnx.py` - Apply to generator model
- `frontend/src/engine/inference.worker.ts` - Add WASM fallback
- `.github/workflows/train.yml` - Add auto-upload on success

## Consequences
- **Positive**: Complete end-to-end pipeline
- **Positive**: Better browser compatibility
- **Negative**: Additional maintenance overhead
- **Negative**: CI workflow complexity increases

## Related
- ADR-027: Experiment Tracking with MLflow
- ADR-026: HuggingFace Model Publishing
- GOAP.md Phase 14-15
