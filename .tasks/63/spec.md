# Specification for Issue #63: Fix Modal ONNX Download Failure

## Issue Summary
The CI workflow fails to download ONNX models from Modal volumes because:
1. Training scripts do not automatically export to ONNX after training.
2. Checkpoints are saved in dated subdirectories, but the workflow expects them in the volume root.
3. Quantization is not applied automatically.

## Requirements
1. **Integrated ONNX Export**: Training scripts must trigger ONNX export and quantization upon successful completion.
2. **Standardized Output Paths**: Final models must be copied to the volume root (`/outputs/`) to ensure predictable download paths for CI.
3. **Workflow Update**: `.github/workflows/train.yml` must be updated to match the standardized output paths and include quantized models.

## Plan

### 1. Update `src/train.py` (Classifier)
- Add `export_onnx.py` and `optimize_onnx.py` to the Modal `image` definition.
- In `train_on_gpu`, after `train()` completes:
  - Export the best checkpoint to `/outputs/cats_classifier.onnx`.
  - Quantize it to `/outputs/cats_quantized.onnx`.
  - Copy the best `.pt` model to `/outputs/best_cats_model.pt`.
  - Ensure `volume_outputs.commit()` is called.

### 2. Update `src/train_dit.py` (DiT)
- Add `export_dit_onnx.py` and `optimize_onnx.py` to the Modal `image` definition.
- In `train_dit_on_gpu`, after `train_dit_local()` completes:
  - Export the best checkpoint to `/outputs/generator.onnx`.
  - Quantize it to `/outputs/generator_quantized.onnx`.
  - Copy the best `.pt` model to `/outputs/tinydit_final.pt`.
  - Ensure `volume_outputs.commit()` is called.

### 3. Update `.github/workflows/train.yml`
- Update `modal volume get` commands to use the standardized paths in `/outputs/`.
- Ensure both full and quantized ONNX models are downloaded and uploaded to Hugging Face.

### 4. Verification
- Run `ruff` and `pytest` locally.
- Validate that the changes align with the project's architectural decisions (ADR-022 to ADR-025).
