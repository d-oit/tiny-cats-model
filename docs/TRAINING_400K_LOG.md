# 400k Step High-Accuracy Training Log

**Started:** 2026-02-27 21:11 UTC  
**Status:** 🔄 Running on Modal GPU  
**App ID:** `ap-LGvySigBxcudIJnVZXUvlm`

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Steps** | 400,000 |
| **Batch Size** | 256 |
| **Gradient Accumulation** | 2 (effective batch: 512) |
| **Learning Rate** | 5e-5 |
| **Warmup Steps** | 15,000 |
| **Augmentation** | Full (rotation, color, affine) |
| **Image Size** | 128x128 |
| **Mixed Precision** | Enabled (AMP) |
| **Gradient Clip** | 1.0 (default) |

---

## Expected Results (ADR-036)

| Metric | Current (200k) | Expected (400k) |
|--------|----------------|-----------------|
| **FID** | ~50-70 | ~30-40 |
| **Inception Score** | ~3-4 | ~5-6 |
| **Precision** | ~0.6 | ~0.75 |
| **Recall** | ~0.4 | ~0.55 |
| **Training Loss** | ~0.34 | ~0.25 |

---

## Estimated Duration & Cost

| Metric | Estimate |
|--------|----------|
| **Training Time** | ~36-48 hours |
| **Modal Cost** | ~$15-25 (H100/A10G) |
| **Checkpoint Size** | ~130MB (final) |
| **EMA Checkpoint** | ~130MB (final) |

---

## Training Command

```bash
modal run src/train_dit.py \
  --data-dir data/cats \
  --steps 400000 \
  --batch-size 256 \
  --gradient-accumulation-steps 2 \
  --lr 5e-5 \
  --warmup-steps 15000 \
  --augmentation-level full \
  --image-size 128 \
  --output /outputs/checkpoints/tinydit_400k.pt \
  --ema-output /outputs/checkpoints/tinydit_400k_ema.pt \
  --mixed-precision
```

---

## Monitoring

### Check Training Status

```bash
# View Modal apps
modal app list

# Watch training logs (in Modal dashboard)
# https://modal.com/dashboard
```

### Expected Log Output

```
Starting TinyDiT training with flow matching
Configuration: steps=400000, batch_size=256, lr=5e-05
Using device: cuda
GPU: NVIDIA H100/A10G
Model: TinyDiT | Image size: 128 | Parameters: 22,145,536
EMA initialized with beta=0.9999
Step 100/400,000 | Loss: 0.8234 | LR: 5.00e-05 | Speed: 2.1 steps/s
Step 200/400,000 | Loss: 0.7456 | LR: 5.00e-05 | Speed: 2.2 steps/s
...
Step 400,000/400,000 | Loss: 0.25xx | LR: 5.00e-07 | Speed: 2.0 steps/s
Training complete. Final loss: 0.25xx
Saved checkpoint to /outputs/checkpoints/tinydit_400k.pt
Saved EMA checkpoint to /outputs/checkpoints/tinydit_400k_ema.pt
```

---

## Post-Training Steps

### 1. Download Checkpoints

```bash
# Download from Modal volume
modal volume get tiny-cats-checkpoints checkpoints/tinydit_400k.pt
modal volume get tiny-cats-checkpoints checkpoints/tinydit_400k_ema.pt
```

### 2. Evaluate Model

```bash
# Generate samples
python src/eval_dit.py \
  --checkpoint checkpoints/tinydit_400k_ema.pt \
  --all-breeds \
  --num-samples 16 \
  --output-dir samples/400k_steps

# Compute FID, IS, Precision/Recall
python src/evaluate_full.py \
  --checkpoint checkpoints/tinydit_400k_ema.pt \
  --generate-samples \
  --num-samples 500 \
  --compute-fid \
  --real-dir data/cats/test \
  --fake-dir samples/400k_steps/evaluation \
  --report-path evaluation_400k.json
```

### 3. Run Benchmarks

```bash
python src/benchmark_inference.py \
  --model checkpoints/tinydit_400k_ema.pt \
  --device cuda \
  --num-warmup 10 \
  --num-runs 100 \
  --benchmark-throughput \
  --batch-sizes 1,4,8,16 \
  --report-path benchmark_400k.json
```

### 4. Export to ONNX

```bash
python src/export_dit_onnx.py \
  --checkpoint checkpoints/tinydit_400k_ema.pt \
  --output frontend/public/models/generator_400k.onnx \
  --quantize
```

### 5. Upload to HuggingFace

```bash
python src/upload_to_huggingface.py \
  --generator checkpoints/tinydit_400k_ema.pt \
  --onnx-generator frontend/public/models/generator_400k.onnx \
  --evaluation-report evaluation_400k.json \
  --benchmark-report benchmark_400k.json \
  --samples-dir samples/400k_steps \
  --repo-id d4oit/tiny-cats-model \
  --commit-message "Upload high-accuracy model (400k steps)"
```

---

## Success Criteria

- [ ] Training completes all 400,000 steps
- [ ] Final loss < 0.30
- [ ] FID < 40 (20%+ improvement)
- [ ] Inception Score > 5
- [ ] Checkpoints saved successfully
- [ ] ONNX export works
- [ ] HuggingFace upload successful

---

## Troubleshooting

### Issue: Training fails mid-way
**Solution:** Resume from latest checkpoint
```bash
modal run src/train_dit.py \
  --data-dir data/cats \
  --steps 400000 \
  --resume /outputs/checkpoints/tinydit_400k_step_200000.pt
```

### Issue: Out of memory
**Solution:** Reduce batch size or increase gradient accumulation
```bash
--batch-size 128 --gradient-accumulation-steps 4  # Still effective batch 512
```

### Issue: Slow training
**Solution:** Check GPU type in Modal dashboard (should be H100/A10G)

---

## Timeline

| Time | Milestone |
|------|-----------|
| T+0h | Training started |
| T+12h | ~100k steps (25%) |
| T+24h | ~200k steps (50%) |
| T+36h | ~300k steps (75%) |
| T+48h | Training complete (100%) |
| T+50h | Evaluation complete |
| T+52h | HuggingFace upload |

---

**Last Updated:** 2026-02-27 21:11 UTC  
**Next Update:** Check in 12 hours for 25% progress
