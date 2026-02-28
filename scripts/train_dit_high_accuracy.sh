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
#
# Note: For very long training (24h+), use GitHub Actions instead:
#   gh workflow run train.yml -f steps=400000

set -e

echo "============================================"
echo "TinyDiT High-Accuracy Training (400k steps)"
echo "============================================"
echo "Date: $(date)"
echo ""

# Verify Modal authentication
echo "Verifying Modal authentication..."
modal token info

# Run 400k step training with nohup to survive CLI timeout
echo ""
echo "Starting Modal GPU training (background)..."
echo "Configuration:"
echo "  - Steps: 400,000"
echo "  - Batch Size: 256"
echo "  - Gradient Accumulation: 2 (effective batch 512)"
echo "  - Learning Rate: 5e-5"
echo "  - Warmup Steps: 15,000"
echo "  - Augmentation: full"
echo ""
echo "View progress at: https://modal.com/dashboard"
echo ""

# Run with nohup to prevent CLI timeout from killing the process
nohup modal run src/train_dit.py \
    --data-dir data/cats \
    --steps 400000 \
    --batch-size 256 \
    --gradient-accumulation-steps 2 \
    --lr 5e-5 \
    --warmup-steps 15000 \
    --augmentation-level full \
    > modal_training.log 2>&1 &

echo "Training started in background (PID: $!)"
echo "Logs will be saved to: modal_training.log"
echo "View app at: https://modal.com/dashboard"
