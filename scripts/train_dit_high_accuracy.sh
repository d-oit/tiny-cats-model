#!/bin/bash
# High-Accuracy TinyDiT Training Script (400k steps)
# ADR-036: High-Accuracy Training Configuration
# ADR-044: Uses GitHub Actions for reliable long-running training
#
# Usage: 
#   Option A (RECOMMENDED): GitHub Actions
#     gh workflow run train.yml -f steps=400000 -f batch_size=256
#
#   Option B: Modal CLI (for shorter runs only, <1 hour)
#     modal run src/train_dit.py --data-dir data/cats --steps 400000
#
# Note: For 400k steps, use GitHub Actions - nohup is unreliable (ADR-044)

set -e

echo "============================================"
echo "TinyDiT High-Accuracy Training (400k steps)"
echo "============================================"
echo "Date: $(date)"
echo ""

# Check if this is a short run or long run
if [ "$1" == "--local" ]; then
    echo "Running locally (short training)..."
    modal run src/train_dit.py \
        --data-dir data/cats \
        --steps 4000 \
        --batch-size 256 \
        --gradient-accumulation-steps 2 \
        --lr 5e-5 \
        --warmup-steps 15000 \
        --augmentation-level full
else
    echo "For 400k step training, use GitHub Actions:"
    echo ""
    echo "  gh workflow run train.yml -f steps=400000 -f batch_size=256"
    echo ""
    echo "This is more reliable than nohup/modal run for long training."
    echo "See ADR-044 for details."
    echo ""
    echo "To test locally with a short run:"
    echo "  $0 --local"
fi
