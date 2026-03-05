# ML Deployment Best Practices Research (2026-03-05)

## Summary
Research conducted to identify actionable improvements for the tiny-cats-model deployment pipeline.

## Key Findings

### 1. HuggingFace Hub Publishing
- **Recommendation:** Add SafeTensors format for faster loading and better security
- **Enhancement:** Include `pipeline_tag: text-to-image` for generator discoverability
- **Version Control:** Use git tags for model versions (v1.0.0, v1.1.0)

### 2. ONNX Optimization
- **Current:** INT8 Dynamic quantization (11MB classifier, 33MB generator)
- **Recommendation:** Add FP16 variant for WebGPU inference
- **Target:** Classifier latency <100ms (WASM), Generator latency <2s (WebGPU)

### 3. MLflow Production Setup
- **Current:** Local file-based tracking
- **Recommendation:** Deploy remote tracking server on Modal
- **Enhancement:** Add model registry integration with automatic versioning

### 4. Modal GPU Cost Optimization
- **Current:** Early stopping saves 60-80% cost
- **Enhancement:** Add adaptive patience early stopping
- **Strategy:** Use T4 for dev ($3-5), A10G for production ($5-8)

### 5. WebGPU vs WASM Fallback
- **WebGPU:** 5-10x faster, Chrome 113+/Edge 113+
- **WASM:** Universal baseline, 150ms classifier latency
- **Strategy:** WebGPU-first with WASM fallback

## Recommended New ADRs

| Priority | ADR | Topic |
|----------|-----|-------|
| High | ADR-056 | WebGPU-First Inference Strategy |
| High | ADR-057 | SafeTensors Export Format |
| Medium | ADR-058 | Remote MLflow Tracking Server |
| Medium | ADR-059 | Adaptive Early Stopping |
| Low | ADR-060 | Model Versioning Strategy |

## Implementation Status

| Feature | Current | Recommended | Effort |
|---------|---------|-------------|--------|
| SafeTensors | Not implemented | Add to export scripts | 2-4 hours |
| WebGPU FP16 | Not implemented | Add FP16 export variant | 4-8 hours |
| MLflow Remote | Local only | Modal deployment | 4-8 hours |
| Adaptive Early Stopping | Fixed patience | Dynamic patience | 2-4 hours |

## Sources
- HuggingFace Hub documentation
- ONNX Runtime Web documentation
- MLflow documentation
- Modal Labs documentation

---
*Research conducted by background agent for ADR-055 analysis*