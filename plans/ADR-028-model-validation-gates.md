# ADR-028: Model Validation Gates

**Date:** 2026-02-25
**Status:** Proposed
**Authors:** AI Agent
**Related:** GOAP.md Phase 9, ADR-026 (HuggingFace Publishing), ADR-017 (TinyDiT Training)

## Context

### Current State

Models are saved after training without automated quality checks:
- No minimum accuracy threshold enforcement
- No generated sample quality assessment
- No ONNX export validation
- No regression tests against previous versions

### Problem Statement

Without validation gates:
1. Degraded models may be deployed
2. No automated quality enforcement
3. Manual inspection required for every model
4. Risk of publishing broken models
5. No regression detection

## Decision

Implement automated model validation gates with:

### 1. Validation Thresholds

**Classifier Thresholds:**
```json
{
  "min_val_accuracy": 0.85,
  "min_train_accuracy": 0.80,
  "max_model_size_mb": 100.0,
  "check_nan_weights": true,
  "check_inf_weights": true
}
```

**DiT Thresholds:**
```json
{
  "max_final_loss": 0.5,
  "max_fid_score": 50.0,
  "max_model_size_mb": 150.0,
  "min_sample_quality": 0.9
}
```

### 2. Validation Checks

**Critical Checks (Block Deployment):**
- Model file exists
- No NaN weights
- No Inf weights
- Model size within limit
- Validation accuracy above threshold (classifier)
- Training loss below threshold (DiT)
- Generated samples valid (no NaN/Inf)

**Warning Checks (Log Only):**
- ONNX output consistency
- FID score estimate
- Inference time benchmarks

### 3. Validation Workflow

```python
from validate_model import validate_model, ValidationThresholds

# Load thresholds
thresholds = ValidationThresholds(
    min_val_accuracy=0.85,
    max_final_loss=0.5,
    max_model_size_mb=100.0,
)

# Run validation
report = validate_model(
    model_path="checkpoints/dit_model.pt",
    thresholds=thresholds,
    check_all=True,
)

# Check result
if not report.passed:
    print(f"Validation FAILED: {report.errors}")
    exit(1)
else:
    print(f"Validation PASSED: {report.to_json()}")
```

### 4. Integration with Training

**In Training Script:**
```python
# After training completes
if validate_after_training:
    report = validate_model(output_path, thresholds)
    
    if not report.passed:
        logger.error(f"Model validation failed: {report.errors}")
        # Option 1: Don't save model
        # Option 2: Save with warning flag
        # Option 3: Upload to quarantine repo
    else:
        logger.info(f"Model validation passed: {report.to_json()}")
        
        # Proceed with upload
        if upload_to_hub:
            upload_to_huggingface(output_path)
```

### 5. CI/CD Integration

**GitHub Actions:**
```yaml
- name: Validate Model
  run: |
    python src/validate_model.py checkpoints/dit_model.pt \
      --thresholds config/validation.json \
      --check-all \
      --output validation_report.json \
      --verbose

- name: Upload Validation Report
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: validation-report
    path: validation_report.json
```

### 6. Validation Report

**JSON Output:**
```json
{
  "model_path": "checkpoints/dit_model.pt",
  "timestamp": "2026-02-25T12:00:00",
  "passed": true,
  "total_checks": 8,
  "passed_checks": 8,
  "failed_checks": 0,
  "warnings": [],
  "errors": [],
  "results": [
    {
      "name": "Model File Exists",
      "passed": true,
      "value": "checkpoints/dit_model.pt",
      "message": "Model file found"
    },
    {
      "name": "NaN Weights Check",
      "passed": true,
      "value": "No NaN",
      "message": "No NaN weights detected"
    },
    {
      "name": "Model Size",
      "passed": true,
      "value": "89.5 MB",
      "threshold": "<=100.0 MB"
    }
  ]
}
```

### 7. Sample Quality Check

**For Generative Models:**
```python
def check_sample_quality(model_path, max_fid_score=50.0):
    """Generate samples and check quality."""
    # Load model
    model = load_model(model_path)
    
    # Generate samples
    samples = generate_samples(model, num_samples=8)
    
    # Check for valid outputs
    has_nan = torch.isnan(samples).any()
    has_inf = torch.isinf(samples).any()
    
    # Simple statistics
    mean_val = samples.mean().item()
    std_val = samples.std().item()
    
    # Valid if no NaN/Inf and reasonable statistics
    valid = (
        not has_nan and 
        not has_inf and 
        abs(mean_val) < 3 and 
        0.1 < std_val < 3.0
    )
    
    return ValidationResult(
        name="Sample Quality",
        passed=valid,
        message=f"Generated samples {'valid' if valid else 'invalid'}"
    )
```

### 8. ONNX Consistency Check

```python
def check_onnx_consistency(model_path, onnx_path, tolerance=1e-4):
    """Check ONNX output matches PyTorch."""
    # Run both models on same input
    pytorch_output = run_pytorch(model_path, dummy_input)
    onnx_output = run_onnx(onnx_path, dummy_input)
    
    # Compare
    max_diff = abs(pytorch_output - onnx_output).max()
    passed = max_diff <= tolerance
    
    return ValidationResult(
        name="ONNX Consistency",
        passed=passed,
        value=f"max_diff={max_diff:.6f}",
        threshold=f"<={tolerance}"
    )
```

## Implementation Plan

### Phase 1: Core Validation (Week 1)
1. Create `src/validate_model.py`
2. Define validation thresholds
3. Implement critical checks
4. Add JSON report generation

### Phase 2: Training Integration (Week 2)
1. Add validation call post-training
2. Add validation to Modal training scripts
3. Add validation to GitHub Actions

### Phase 3: Advanced Checks (Week 3)
1. Implement FID score estimation
2. Add inference time benchmarking
3. Add regression testing vs previous model

## Consequences

### Positive
- **Quality Assurance**: Automated quality gates
- **Risk Reduction**: Prevent broken model deployment
- **Consistency**: Same checks for every model
- **Documentation**: Validation report as artifact
- **CI/CD**: Automated validation in pipeline

### Negative
- **Time Overhead**: Validation adds time to training
- **False Positives**: May reject borderline models
- **Maintenance**: Thresholds need tuning

### Neutral
- **Configurable**: Thresholds can be adjusted per use case
- **Extensible**: New checks can be added easily

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Validation time | <5 min | Time per validation |
| False positive rate | <5% | Valid models rejected |
| False negative rate | 0% | Broken models passing |
| Check coverage | >90% | Critical checks automated |

## Dependencies

- `torch>=2.0.0`
- `pillow>=10.0.0` (for sample generation)
- `onnxruntime` (optional, for ONNX validation)
- `pytorch-fid` (optional, for FID score)

## References

- ADR-026: HuggingFace Model Publishing
- ADR-017: TinyDiT Training Infrastructure
- Validation Best Practices: https://mlflow.org/docs/latest/model-registry.html
