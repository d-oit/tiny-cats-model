# ADR-034: HuggingFace Model Loading Strategy

## Status
Accepted

## Context
The tiny-cats-model frontend currently serves models from `frontend/public/models/`, which are bundled with the frontend deployment. This approach has several limitations:

1. **Large bundle size**: Models (especially generator at 132MB) increase frontend bundle significantly
2. **No version control**: Model updates require full frontend redeployment
3. **No CDN caching**: Models served from GitHub Pages without CDN optimization
4. **Manual synchronization**: Models must be manually copied to frontend directory

The HuggingFace Hub (`d4oit/tiny-cats-model`) provides a better distribution mechanism with:
- Built-in CDN caching via CloudFront
- Version control with Git LFS
- Automatic model updates without frontend redeployment
- Smaller frontend bundle (models loaded on-demand)

## Decision

**Load models directly from HuggingFace Hub CDN** with local fallback.

### Model URLs
```typescript
const HF_BASE_URL = "https://huggingface.co/d4oit/tiny-cats-model/resolve/main/";

const MODEL_CONFIGS = {
  cats: {
    modelPath: HF_BASE_URL + "cats_quantized.onnx",
    // ... other config
  },
};

const GENERATOR_CONFIG = {
  modelPath: HF_BASE_URL + "generator_quantized.onnx",
  // ... other config
};
```

### Fallback Strategy
If HF Hub loading fails (network issues, CORS, etc.), fall back to local models:
```typescript
async function loadModel(path: string, modelType: ModelType) {
  try {
    // Try HF Hub first
    await loadFromHFHub(path);
  } catch (error) {
    console.warn("HF Hub loading failed, falling back to local model");
    // Fall back to local model
    await loadFromLocal(path.replace(HF_BASE_URL, "/models/"));
  }
}
```

## Benefits

### Positive
1. **Version Control**: Models tracked in HF Hub with commit history
2. **CDN Caching**: HF Hub uses CloudFront CDN for fast global delivery
3. **Smaller Bundle**: Frontend bundle reduced by ~150MB (both models)
4. **Independent Updates**: Models can be updated without frontend redeployment
5. **Automatic Caching**: Browser caches models separately from frontend assets
6. **Bandwidth**: HF Hub absorbs bandwidth costs for model downloads

### Negative
1. **External Dependency**: Requires HF Hub availability (mitigated by fallback)
2. **CORS Configuration**: Must ensure HF Hub allows CORS (it does by default)
3. **Initial Load**: First-time users download from HF instead of bundled assets
4. **Network Dependency**: Requires internet for initial model load (PWA can cache)

## Technical Details

### HuggingFace Hub URL Format
```
https://huggingface.co/{repo_id}/resolve/{revision}/{path}
```

For our models:
- **Classifier**: `https://huggingface.co/d4oit/tiny-cats-model/resolve/main/cats_quantized.onnx`
- **Generator**: `https://huggingface.co/d4oit/tiny-cats-model/resolve/main/generator_quantized.onnx`
- **Generator (full)**: `https://huggingface.co/d4oit/tiny-cats-model/resolve/main/generator.onnx`

### CORS Handling
HuggingFace Hub sets appropriate CORS headers by default:
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, HEAD, OPTIONS
```

### Caching Strategy
1. **Browser Cache**: Models cached by browser (Cache-Control from HF Hub)
2. **Service Worker**: Optional PWA caching for offline support
3. **IndexedDB**: Store models in IndexedDB for faster subsequent loads

## Implementation

### Frontend Changes

#### constants.ts
```typescript
export const HF_HUB_BASE_URL = 
  "https://huggingface.co/d4oit/tiny-cats-model/resolve/main/";

export const MODEL_CONFIGS: Record<ModelType, ModelConfig> = {
  cats: {
    modelPath: HF_HUB_BASE_URL + "cats_quantized.onnx",
    localFallback: "/models/cats_quantized.onnx",
    // ...
  },
};

export const GENERATOR_CONFIG: GeneratorConfig = {
  modelPath: HF_HUB_BASE_URL + "generator_quantized.onnx",
  localFallback: "/models/generator_quantized.onnx",
  // ...
};
```

#### inference.worker.ts
```typescript
async function loadModel(path: string, modelType: ModelType) {
  const config = MODEL_CONFIGS[modelType];
  const primaryPath = config.modelPath;
  const fallbackPath = config.localFallback;

  try {
    await loadFromPath(primaryPath);
    console.log(`Loaded model from HF Hub: ${primaryPath}`);
  } catch (error) {
    console.warn(`HF Hub load failed, using fallback: ${fallbackPath}`);
    await loadFromPath(fallbackPath);
  }
}
```

### GitHub Actions Changes (train.yml)

Add automated upload after training:
```yaml
- name: Upload to Hugging Face Hub
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
  run: |
    pip install huggingface_hub
    python -c "
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj='cats_quantized.onnx',
        path_in_repo='cats_quantized.onnx',
        repo_id='d4oit/tiny-cats-model',
        repo_type='model',
        commit_message='Upload classifier from GitHub Actions'
    )
    "
```

## Consequences

### Short-term
- Frontend bundle size reduced by ~150MB
- First-time load may be slower (downloading from HF Hub)
- Subsequent loads faster (browser cache)
- Requires HF_TOKEN secret in GitHub Actions

### Long-term
- Easier model updates (no frontend redeployment)
- Better separation of concerns (models vs. frontend code)
- Potential for model versioning (v1, v2, etc.)
- PWA support for offline usage

## Migration Plan

1. **Phase 1**: Update constants.ts with HF Hub URLs + fallback
2. **Phase 2**: Update train.yml for automated upload
3. **Phase 3**: Upload current models to HF Hub
4. **Phase 4**: Test HF Hub loading in production
5. **Phase 5**: Remove large models from frontend/public/models/ (optional)

## Related

- ADR-026: HuggingFace Model Publishing Implementation
- ADR-033: Codebase Analysis - Missing Implementations
- GOAP.md Phase 16: HuggingFace Hub Integration

## References

- HuggingFace Hub Documentation: https://huggingface.co/docs/hub/
- HuggingFace CDN: https://huggingface.co/docs/hub/security-cdn
- ONNX Runtime Web: https://onnxruntime.ai/docs/get-started/with-web.html
