#!/usr/bin/env bash
# Reproducible generation-quality measurement for the DiT pool checkpoint.
#
# This is the exact protocol behind docs/EVAL-436K-CHECKPOINT-2026-10-02.md:
# 988 generated samples (13 breeds x 76, 30 steps, CFG 1.5, seed 42) vs the
# full data/cats/cat reference set, scored with FID / Inception Score /
# Precision-Recall, then printed against the 436k baseline so the delta is
# visible at a glance.
#
# Usage:
#   bash scripts/eval_checkpoint.sh [checkpoint] [output_dir]
#
# Defaults: checkpoints/pool/dit_model_ema.pt (pulled from the Modal volume
# if missing locally) and eval_out/samples-<checkpoint-name>.
# Takes ~2h on CPU (generation ~100min, Inception featurization ~20min).
set -euo pipefail

CKPT="${1:-checkpoints/pool/dit_model_ema.pt}"
OUT_DIR="${2:-eval_out/samples-$(basename "${CKPT%.pt}")}"
REPORT="${OUT_DIR}/report.json"

# Fetch the pool checkpoint from the Modal volume when not present locally.
if [ ! -f "$CKPT" ]; then
  echo "Checkpoint $CKPT not found; pulling from Modal volume 'dit-outputs'..."
  mkdir -p "$(dirname "$CKPT")"
  modal volume get dit-outputs "checkpoints/pool/$(basename "$CKPT")" "$CKPT"
fi

mkdir -p "$OUT_DIR"

python src/evaluate_full.py \
  --checkpoint "$CKPT" \
  --generate-samples \
  --all-breeds \
  --num-samples 1000 \
  --num-steps 30 \
  --batch-size 16 \
  --device cpu \
  --compute-fid \
  --real-dir data/cats/cat \
  --fake-dir "$OUT_DIR" \
  --compute-is \
  --compute-precision-recall \
  --report-path "$REPORT" \
  --output-dir "$OUT_DIR"

# Print the score next to the 436k baseline (see docs/EVAL-436K-*.md).
python - "$REPORT" <<'PYEOF'
import json
import sys

BASELINE = {
    "step": 436000,
    "fid": 263.31,
    "inception_score": 1.60,
    "precision": 0.50,
    "recall": 0.48,
}

with open(sys.argv[1]) as fh:
    report = json.load(fh)

step = report.get("metadata", {}).get("step", "unknown")
current = {
    "fid": report["fid"],
    "inception_score": report["inception_score"]["mean"],
    "precision": report["precision"],
    "recall": report["recall"],
}

print(f"\nCheckpoint step: {step}")
print(f"{'metric':<18}{'baseline 436k':>15}{'current':>12}{'delta':>12}")
print("-" * 57)
for key, better_when in [("fid", "lower"), ("inception_score", "higher"),
                         ("precision", "higher"), ("recall", "higher")]:
    base, cur = BASELINE[key], current[key]
    delta = cur - base
    if abs(delta) < 0.005:  # identical once rounded to the printed precision
        arrow = "same"
    elif (delta < 0) if better_when == "lower" else (delta > 0):
        arrow = "improved"
    else:
        arrow = "worse"
    print(f"{key:<18}{base:>15.2f}{cur:>12.2f}{delta:>+12.2f}  {arrow}")
PYEOF

echo ""
echo "Report written to $REPORT"
