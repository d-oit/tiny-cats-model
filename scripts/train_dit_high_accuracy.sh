#!/bin/bash
# High-Accuracy TinyDiT Training Script (400k steps)
# ADR-036: High-Accuracy Training Configuration
# ADR-044: Uses GitHub Actions for reliable long-running training
#
# Usage: 
#   Option A (RECOMMENDED for 400k): GitHub Actions
#     gh workflow run train.yml
#     # Defaults: steps=400000, lr=5e-5, gradient_accumulation_steps=2
#
#   Option B (TESTING ONLY): Local Modal run
#     bash scripts/train_dit_high_accuracy.sh --local
#     # Runs 4000 steps for quick verification
#
#   Option C (Manual): Modal CLI for medium runs (10k-50k steps)
#     modal run src/train_dit.py --data-dir data/cats --steps 10000
#
# Note: For 400k steps, ALWAYS use GitHub Actions - nohup is unreliable (ADR-044)
#       Local runs should NOT exceed 1 hour to avoid timeout issues

set -e

echo "============================================"
echo "TinyDiT High-Accuracy Training"
echo "============================================"
echo "Date: $(date)"
echo ""

# Function to verify prerequisites
check_prerequisites() {
    local errors=0
    
    # Check Modal auth
    if ! modal token info > /dev/null 2>&1; then
        echo "❌ Modal authentication failed"
        echo "   Run: modal token new"
        errors=$((errors + 1))
    else
        echo "✅ Modal authenticated"
    fi
    
    # Check data directory
    if [ ! -d "data/cats" ]; then
        echo "❌ Dataset not found at data/cats"
        echo "   Run: bash data/download.sh"
        errors=$((errors + 1))
    else
        local image_count=$(find data/cats -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
        echo "✅ Dataset found: $image_count images"
    fi
    
    if [ $errors -gt 0 ]; then
        echo ""
        echo "Please fix the above issues before training"
        exit 1
    fi
}

# Check if this is a short test run or long run
if [ "$1" == "--local" ]; then
    echo "Mode: LOCAL TEST RUN (4000 steps)"
    echo "Purpose: Verify training setup works before GitHub Actions"
    echo "Duration: ~5-10 minutes"
    echo ""
    
    check_prerequisites
    
    echo "Starting local test training..."
    modal run src/train_dit.py \
        --data-dir data/cats \
        --steps 4000 \
        --batch-size 256 \
        --gradient-accumulation-steps 2 \
        --lr 5e-5 \
        --warmup-steps 15000 \
        --augmentation-level full
    
    echo ""
    echo "✅ Local test completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Verify checkpoint was created in Modal volume"
    echo "  2. Run full 400k training via GitHub Actions:"
    echo "     gh workflow run train.yml"
    
elif [ "$1" == "--medium" ]; then
    echo "Mode: MEDIUM RUN (50000 steps)"
    echo "⚠️  Warning: This will take ~2-3 hours and cost ~$5-8"
    echo ""
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        exit 0
    fi
    
    check_prerequisites
    
    modal run src/train_dit.py \
        --data-dir data/cats \
        --steps 50000 \
        --batch-size 256 \
        --gradient-accumulation-steps 2 \
        --lr 5e-5 \
        --warmup-steps 15000 \
        --augmentation-level full
        
else
    echo "Mode: PRODUCTION (400k steps)"
    echo ""
    echo "⚠️  IMPORTANT: Use GitHub Actions for 400k training"
    echo ""
    echo "Why GitHub Actions?"
    echo "  - More reliable for long runs (24-36 hours)"
    echo "  - Better timeout handling"
    echo "  - Automatic artifact upload"
    echo "  - Prevents local machine issues"
    echo ""
    echo "Command:"
    echo "  gh workflow run train.yml"
    echo ""
    echo "The workflow now defaults to 400k high-accuracy:"
    echo "  - steps: 400000"
    echo "  - lr: 5e-5"
    echo "  - gradient_accumulation_steps: 2"
    echo ""
    echo "For testing only:"
    echo "  bash $0 --local    # 4000 steps, ~5-10 min"
    echo "  bash $0 --medium   # 50000 steps, ~2-3 hours"
    echo ""
    echo "See ADR-044 for details on why nohup is unreliable"
fi
