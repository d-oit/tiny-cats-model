# Cats Classifier Models

This directory contains ONNX models for binary cat classification (cat vs other).

## Model Files

| File | Size | Description |
|------|------|-------------|
| `cats.onnx` + `cats.onnx.data` | 43 MB | Full precision FP32 model |
| `cats_quantized.onnx` | 11 MB | Quantized INT8 model (recommended for web) |

## Model Formats

### Original Model (cats.onnx)
- **Format**: ONNX with external data
- **Precision**: FP32 (32-bit floating point)
- **Structure**: Separate `.onnx` file (graph structure) and `.onnx.data` file (weights)
- **Use case**: High-accuracy scenarios where file size is not a concern

### Quantized Model (cats_quantized.onnx)
- **Format**: ONNX (self-contained)
- **Precision**: INT8 (8-bit integer quantization)
- **Structure**: Single file with quantized weights
- **Use case**: Web deployment where file size and inference speed are critical

## Input/Output Specification

### Input
- **Name**: `input`
- **Shape**: `[batch_size, 3, 224, 224]`
- **Type**: Float32 tensor
- **Preprocessing**:
  1. Resize image to 224x224 pixels
  2. Convert to RGB (3 channels)
  3. Normalize using ImageNet statistics:
     - Mean: `[0.485, 0.456, 0.406]`
     - Std: `[0.229, 0.224, 0.225]`
  4. Scale to range `[-1, 1]`

### Output
- **Name**: `output`
- **Shape**: `[batch_size, 2]`
- **Type**: Float32 tensor (logits)
- **Classes**:
  - Index 0: `cat` (positive class)
  - Index 1: `other` (negative class - primarily dogs)

## How to Use in the Frontend

### Using ONNX Runtime Web

```typescript
import * as ort from 'onnxruntime-web';

// Initialize the session
const session = await ort.InferenceSession.create('/models/cats_quantized.onnx', {
  executionProviders: ['wasm']
});

// Preprocess the image
function preprocessImage(image: HTMLImageElement): ort.Tensor {
  const canvas = document.createElement('canvas');
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(image, 0, 0, 224, 224);

  const imageData = ctx.getImageData(0, 0, 224, 224);
  const data = imageData.data;

  // Normalize and rearrange to CHW format
  const tensorData = new Float32Array(3 * 224 * 224);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  for (let i = 0; i < 224 * 224; i++) {
    for (let c = 0; c < 3; c++) {
      const pixelValue = data[i * 4 + c] / 255.0;
      tensorData[c * 224 * 224 + i] = (pixelValue - mean[c]) / std[c];
    }
  }

  return new ort.Tensor('float32', tensorData, [1, 3, 224, 224]);
}

// Run inference
async function classify(image: HTMLImageElement): Promise<{ cat: number; other: number }> {
  const inputTensor = preprocessImage(image);
  const feeds = { input: inputTensor };
  const results = await session.run(feeds);
  const output = results.output.data as Float32Array;

  // Apply softmax to get probabilities
  const expCat = Math.exp(output[0]);
  const expOther = Math.exp(output[1]);
  const sum = expCat + expOther;

  return {
    cat: expCat / sum,
    other: expOther / sum
  };
}
```

### Using a Web Worker

For better UI responsiveness, run inference in a web worker:

```typescript
// worker.ts
import * as ort from 'onnxruntime-web';

let session: ort.InferenceSession | null = null;

self.onmessage = async (event) => {
  const { type, data } = event.data;

  if (type === 'init') {
    session = await ort.InferenceSession.create(data.modelPath, {
      executionProviders: ['wasm']
    });
    self.postMessage({ type: 'ready' });
  } else if (type === 'infer' && session) {
    const tensor = new ort.Tensor('float32', data.input, [1, 3, 224, 224]);
    const results = await session.run({ input: tensor });
    self.postMessage({ type: 'result', data: results.output.data });
  }
};
```

## Performance Characteristics

### File Size Comparison
| Model | Download Size | Load Time (3G) | Load Time (WiFi) |
|-------|--------------|----------------|------------------|
| Original | 43 MB | ~12 seconds | ~1 second |
| Quantized | 11 MB | ~3 seconds | ~0.3 seconds |

### Inference Performance
| Model | WASM (ms) | WebGL (ms) | WebGPU (ms) |
|-------|-----------|------------|-------------|
| Original | ~500-800 | ~200-400 | ~50-100 |
| Quantized | ~150-300 | ~100-200 | ~30-50 |

*Note: Performance varies by device and browser. Times shown are approximate for a single inference on mid-range hardware.*

### Memory Usage
| Model | Peak Memory |
|-------|-------------|
| Original | ~150 MB |
| Quantized | ~50 MB |

## Recommendations

1. **Use the quantized model** (`cats_quantized.onnx`) for web deployment
   - 4x smaller file size
   - 2-3x faster inference
   - 3x lower memory usage
   - Minimal accuracy loss for most use cases

2. **Enable Web Workers** for inference to prevent UI blocking

3. **Use WebGL or WebGPU** execution providers if available for better performance

4. **Cache the model** using service workers for offline support

## Model Metadata

See `models.json` for complete model metadata including paths, sizes, and class information.

## License

Models are provided under the same license as the parent project. See the root `LICENSE` file for details.

## References

- [ONNX Runtime Web Documentation](https://onnxruntime.ai/docs/get-started/with-javascript-web.html)
- [Oxford IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [ONNX Model Zoo](https://github.com/onnx/models)
