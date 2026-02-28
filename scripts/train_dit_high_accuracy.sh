#!/bin/bash
# High-Accuracy TinyDiT Training Script (400k steps)
# ADR-036: High-Accuracy Training Configuration
# 
# Usage: bash scripts/train_dit_high_accuracy.sh
#
# This script runs the full 400k step training on Modal GPU with:
# - Gradient accumulation (effective batch 512)
# - Lower learning rate for finer convergence
# - Full augmentation level

set -e

echo "============================================"
echo "TinyDiT High-Accuracy Training (400k steps)"
echo "============================================"
echo "Date: $(date)"
echo ""

# Verify Modal authentication
echo "Verifying Modal authentication..."
modal token info

# Run 400k step training
echo ""
echo "Starting Modal GPU training..."
echo "Configuration:"
echo "  - Steps: 400,000"
echo "  - Batch Size: 256"
echo "  - Gradient Accumulation: 2 (effective batch 512)"
echo "  - Learning Rate: 5e-5"
echo "  - Warmup Steps: 15,000"
echo "  - Augmentation: full"
echo ""

modal run src/train_dit.py \
    --data-dir data/cats \
    --steps 400000 \
    --batch-size 256 \
    --gradient-accumulation-steps 2 \
    --lr 5e-5 \
    --warmup-steps 15000 \
    --augmentation-level full

echo ""
echo "============================================"
echo "Training Complete!"
echo "Date: $(date)"
echo "Checkpoints saved to: /outputs/checkpoints/dit/"
echo "============================================"
