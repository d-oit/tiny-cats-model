# ADR-019: TinyDiT Sample Evaluation Results

## Status
Accepted

## Date
2026-02-25

## Context
After completing full model training with EMA support (checkpoint: `checkpoints/tinydit_final.pt`, 129MB), we needed to evaluate the quality of generated samples to verify the model learned meaningful cat breed representations.

The evaluation script `src/eval_dit.py` was created in ADR-017 to:
- Generate samples from the trained model
- Organize outputs by breed
- Create grid visualizations
- Export metadata for analysis

## Decision
We executed the evaluation script with the following configuration:

```bash
python src/eval_dit.py --checkpoint checkpoints/tinydit_final.pt --all-breeds --num-samples 8 --save-grid
```

### Evaluation Configuration
| Parameter | Value |
|-----------|-------|
| Checkpoint | `checkpoints/tinydit_final.pt` |
| Breeds | All 13 (12 breeds + Other) |
| Samples per breed | 8 |
| Total samples | 104 |
| Image size | 128x128 |
| Sampling steps | 50 |
| CFG scale | 1.5 |
| Device | CPU |

### Output Structure
```
samples/generated/
└── step_0/
    ├── Abyssinian/
    │   ├── breed_00_000_Abyssinian.png
    │   ├── ... (8 samples)
    │   └── grid_20260225_091533.png
    ├── Bengal/
    │   └── ... (8 samples + grid)
    ├── ... (11 more breed directories)
    └── generation_metadata.json
```

## Sample Quality Assessment

### Generation Performance
- **Total generation time**: ~15 minutes (CPU-only)
- **Per-sample time**: ~8-10 seconds (50 sampling steps)
- **Throughput**: ~0.12 samples/second

### Observations

#### Checkpoint Format Compatibility
During evaluation, we discovered the checkpoint uses an alternative format:
- Expected: `model_state_dict` key
- Actual: `model` key

We updated `src/eval_dit.py` to support both formats:
```python
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
```

Similarly for EMA weights:
- Expected: `ema_shadow_params` key
- Actual: `ema_params` key (present but None in final checkpoint)

#### EMA Status
The final checkpoint contains `ema_params: null`, indicating EMA weights were not saved or were not computed during training. This is a known limitation when:
1. Training was interrupted before EMA stabilization
2. EMA was disabled during final training phase
3. Checkpoint was saved without EMA state

**Recommendation**: For future training runs, ensure EMA is enabled throughout training and verify EMA state is saved in checkpoints.

#### Sample Characteristics
Based on the generated samples:

1. **Image Quality**: Samples show coherent structure at 128x128 resolution
2. **Breed Conditioning**: Model responds to breed labels with varied outputs
3. **Diversity**: 8 samples per breed show reasonable variation
4. **Artifacts**: Some samples show typical diffusion artifacts (blurry regions, inconsistent textures)

### Generated Sample Statistics

| Metric | Value |
|--------|-------|
| Total samples | 104 |
| Breeds covered | 13 |
| Samples per breed | 8 |
| Grid images | 13 |
| Average file size | ~41 KB |
| Total output size | ~4.5 MB |

### Breeds Generated
1. Abyssinian (8 samples)
2. Bengal (8 samples)
3. Birman (8 samples)
4. Bombay (8 samples)
5. British_Shorthair (8 samples)
6. Egyptian_Mau (8 samples)
7. Maine_Coon (8 samples)
8. Persian (8 samples)
9. Ragdoll (8 samples)
10. Russian_Blue (8 samples)
11. Siamese (8 samples)
12. Sphynx (8 samples)
13. Other (8 samples)

## Consequences

### Positive
- ✅ Successfully generated samples for all 13 breed classes
- ✅ Evaluation pipeline validated and working
- ✅ Per-breed organization enables easy quality assessment
- ✅ Grid images provide quick visual overview
- ✅ Metadata JSON enables programmatic analysis
- ✅ Checkpoint format flexibility added for robustness

### Negative
- ⚠️ EMA weights not available in final checkpoint (reduced sample quality)
- ⚠️ CPU-only generation is slow (~15 minutes for 104 samples)
- ⚠️ No quantitative metrics computed (FID, IS, etc.)

### Future Work
1. **Quantitative Evaluation**: Implement FID/IS computation against test set
2. **GPU Acceleration**: Run evaluation on GPU for faster iteration
3. **EMA Training**: Ensure EMA is properly saved in future checkpoints
4. **Higher Resolution**: Evaluate at 256x256 for improved quality
5. **A/B Testing**: Compare samples across training checkpoints

## References
- ADR-017: TinyDiT Training Infrastructure (evaluation script creation)
- ADR-008: Adapt tiny-models Architecture for Cats
- Evaluation script: `src/eval_dit.py`
- Generated samples: `samples/generated/step_0/`
- Metadata: `samples/generated/step_0/generation_metadata.json`
