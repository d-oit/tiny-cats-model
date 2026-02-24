# Benchmark Results

> **Note**: This is a template report. Run the benchmark page in your browser to generate actual results for your specific hardware.

## Overview

This document contains benchmark results for frontend inference latency as specified in GOAP.md Phase 5.

**GOAP Success Metric**: <2s for full generation in browser

## Model Specifications

| Model | Size | Format | Location |
|-------|------|--------|----------|
| Classifier | ~11 MB | ONNX | `/tiny-cats-model/models/cats.onnx` |
| Generator | ~126 MB | ONNX | `/tiny-cats-model/models/generator.onnx` |

## Benchmark Configuration

### Classification
- **Image Sizes Tested**: 128x128, 224x224, 256x256
- **Runs per Configuration**: 10
- **Metrics**: mean, std, min, max, p50, p95, p99

### Generation
- **Sampling Steps**: 10, 25, 50, 100
- **CFG Scales**: 1.0, 1.5, 2.0, 3.0
- **Image Size**: 128x128
- **Runs per Configuration**: 3 full generations

## How to Run Benchmarks

1. Navigate to the benchmark page: `/benchmark`
2. Click "Run Benchmark"
3. Wait for the benchmark to complete (~2-5 minutes)
4. Review results in the table
5. Download the report as Markdown

## Expected Performance

### Classification Latency (Estimated)

| Image Size | Expected Mean | Expected P95 |
|------------|---------------|--------------|
| 128x128 | 50-150 ms | 100-250 ms |
| 224x224 | 150-400 ms | 300-600 ms |
| 256x256 | 200-500 ms | 400-800 ms |

### Generation Latency (Estimated)

| Steps | CFG Scale | Expected Total Time | Meets Goal |
|-------|-----------|---------------------|------------|
| 10 | 1.0 | 500-1500 ms | Yes |
| 10 | 1.5 | 800-2000 ms | Yes |
| 10 | 2.0 | 1000-2500 ms | Variable |
| 10 | 3.0 | 1200-3000 ms | Variable |
| 25 | 1.0 | 1500-3500 ms | Variable |
| 25 | 1.5 | 2000-4500 ms | No |
| 50 | 1.0 | 3000-7000 ms | No |
| 50 | 1.5 | 4000-9000 ms | No |
| 100 | 1.0 | 6000-14000 ms | No |
| 100 | 1.5 | 8000-18000 ms | No |

> **Note**: Actual performance varies significantly based on:
> - CPU cores and clock speed
> - Browser JavaScript engine
> - WASM support and optimization
> - System memory and cache
> - Background processes

## Performance Recommendations

### For Faster Classification
1. Use smaller image sizes (128x128) when quality permits
2. Ensure WASM is enabled in browser settings
3. Close other browser tabs to free up CPU resources

### For Faster Generation
1. **Use fewer sampling steps**: 10-25 steps provides good quality/speed tradeoff
2. **Lower CFG scale**: CFG 1.0-1.5 is faster than higher values (avoids double inference)
3. **Consider WebGPU**: If available, WebGPU backend is significantly faster than WASM
4. **Use dedicated workers**: Generation runs in web workers to avoid blocking UI

### Hardware Considerations
- **Minimum**: 4 CPU cores recommended
- **Optimal**: 8+ CPU cores for best performance
- **Memory**: 4GB+ RAM recommended for 126MB model

## Optimization Strategies

### Implemented Optimizations
1. **WASM Execution**: Using ONNX Runtime Web with WASM backend
2. **Graph Optimization**: `graphOptimizationLevel: "all"` enabled
3. **Multi-threading**: WASM threads set to `navigator.hardwareConcurrency`
4. **Web Workers**: Inference runs in background threads
5. **Progressive Rendering**: Generation shows intermediate results

### Future Optimizations
1. **WebGPU Backend**: Switch to WebGPU for GPU-accelerated inference
2. **Model Quantization**: Further quantize models for smaller size
3. **Model Pruning**: Remove unused weights for faster inference
4. **Progressive Loading**: Load model layers on-demand
5. **Cache Warming**: Pre-warm WASM module on page load

## Benchmark Page Features

The benchmark page (`/benchmark`) provides:

- **Real-time Progress**: Shows current benchmark status
- **Detailed Tables**: Latency metrics for all configurations
- **GOAP Comparison**: Visual indicator of goal achievement
- **Performance Recommendations**: Context-aware suggestions
- **Fastest Configurations**: Top 5 configurations meeting the goal
- **Export Report**: Download results as Markdown

## Success Criteria Verification

| Criterion | Status |
|-----------|--------|
| Benchmark utility measures latency accurately | Implemented |
| Benchmark page displays results clearly | Implemented |
| Results compared against GOAP metrics | Implemented |
| GOAP.md updated | Completed |

## Related Files

- `/frontend/src/utils/benchmark.ts` - Benchmark utility functions
- `/frontend/src/pages/benchmark/BenchmarkPage.tsx` - Benchmark UI page
- `/frontend/src/engine/inference.worker.ts` - Classification worker
- `/frontend/src/engine/generation.worker.ts` - Generation worker

## References

- GOAP.md: `/plans/GOAP.md`
- ONNX Runtime Web: https://onnxruntime.ai/docs/get-started/with-javascript.html
- Web Workers: https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API

---

*Generated: Template - Run benchmark page for actual results*
