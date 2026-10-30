# Cats Classifier Models

The classifier predicts **cat** vs **not-cat**. ImageFolder folders map as
`cat/` → output index 0 (`cat`) and `other/` → output index 1 (`not_cat`).
The generator is a separate model with 12 breed conditions plus `Other` at
index 12 and an unconditional/null index 13 for classifier-free guidance.

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
  1. Convert to RGB.
  2. Bilinearly resize while preserving aspect ratio so the shorter side is 256.
  3. Centre crop to 224x224 pixels.
  4. Scale pixels to [0, 1], then normalize using ImageNet statistics:
     - Mean: `[0.485, 0.456, 0.406]`
     - Std: `[0.229, 0.224, 0.225]`
  Normalized values are not restricted to `[-1, 1]`.

### Output
- **Name**: `output`
- **Shape**: `[batch_size, 2]`
- **Type**: Float32 tensor containing logits (the ONNX graph does not apply softmax)
- **Classes**:
  - Index 0: `cat` (positive class)
  - Index 1: `not_cat` (`other/` folder; negative examples are primarily dogs)
## Frontend Runtime

The production frontend does not create ONNX sessions from arbitrary URLs.
`frontend/src/generated/modelManifest.json` is generated from the bundled
artifacts and records each artifact's SHA-256, byte size, model contract and
optional immutable Hugging Face commit SHA. `modelAssets.ts` verifies size and
SHA-256 with Web Crypto before either worker creates a session. Remote fetches
are allowed only for a full 40-character Hub commit SHA; the bundled file is
the fallback only when it matches the same digest and size.

The classify page uses `decodeImageFile()` and `preprocessImageSource()` from
`frontend/src/utils/imageUtils.ts`. The generator and classifier workers own
their ONNX sessions and release them on teardown. Use these application APIs
instead of bypassing the manifest with direct `fetch()` or
`InferenceSession.create()` calls.

## Model Promotion and Rollback

1. Place the release-ready classifier and generator artifacts in
   `frontend/public/models/`; the manifest generator reads their bytes and
   inspects ONNX graph metadata.
2. Upload those exact files to the Hugging Face repository and record the
   resulting immutable commit SHA. Never use a branch name or tag as a
   revision.
3. Generate and review the frontend contract and manifest:

   ```bash
   python scripts/generate_frontend_metadata.py --revision <40-character-commit-sha>
   python scripts/generate_frontend_metadata.py --revision <40-character-commit-sha> --check
   ```

4. Commit the model files and both generated JSON files together. The generated
   SHA-256 values must match the uploaded bytes; the frontend rejects mismatched
   remote or bundled artifacts before inference.
5. To roll back, restore the previous known-good model files and matching
   `modelManifest.json` revision as one release, then run the generator with the
   previous commit SHA and `--check`. Do not edit a SHA by hand.

When no immutable Hub revision is configured, the frontend performs no remote
model request and uses only digest-verified bundled artifacts.

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
