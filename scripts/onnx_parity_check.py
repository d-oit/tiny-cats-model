"""scripts/onnx_parity_check.py

Verify that the sampling-step default chosen from the PyTorch sweep also holds
for the quantized ONNX generator the demos actually run (the sweep measured
`checkpoints/pool/dit_model_ema.pt` in PyTorch; both demos execute the ONNX
graph, so the direction needs confirming on that runtime).

Reuses `app_gradio.generate_cat` — the same guided-Euler sampler the Space
serves — by injecting a local ONNX session into its cache, so there is exactly
one implementation of the sampling loop under test.

Usage:
    python scripts/onnx_parity_check.py \
        --onnx artifacts/generator/model_quantized.onnx \
        --step-settings 50,100 \
        --real-dir data/cats/cat \
        --out-dir artifacts/onnx-parity

Inception Score is always reported; FID only when `--real-dir` exists (the
dataset is gitignored, so CI must fetch it first).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ONNX sampling-step parity check",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--onnx", type=str, required=True, help="Path to the quantized ONNX generator"
    )
    parser.add_argument(
        "--step-settings",
        type=str,
        default="50,100",
        help="Comma-separated sampling-step settings to compare",
    )
    parser.add_argument("--cfg-scale", type=float, default=1.5, help="CFG scale")
    parser.add_argument(
        "--samples-per-breed",
        type=int,
        default=13,
        help="Samples per breed per setting",
    )
    parser.add_argument(
        "--real-dir",
        type=str,
        default="data/cats/cat",
        help="Real images for FID (skipped when missing)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/onnx-parity",
        help="Where samples and the report are written",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    return parser.parse_args()


def main() -> int:
    """Run the parity check and print a comparison table."""
    args = parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        print(f"ONNX model not found: {onnx_path}")
        return 1

    import onnxruntime as ort
    import torch

    import app_gradio
    import evaluate_full

    # Reuse the demo's sampler with a local session instead of a Hub download.
    app_gradio.sessions["generator"] = ort.InferenceSession(str(onnx_path))

    device = torch.device("cpu")
    real_images = []
    real_dir = Path(args.real_dir) if args.real_dir else None
    if real_dir and real_dir.exists():
        real_images = evaluate_full.load_images_from_directory(real_dir, 128)
    else:
        print(f"No real images at {real_dir or '(unset)'}; reporting IS only.")

    settings = [int(value) for value in args.step_settings.split(",") if value]
    if len(settings) < 2:
        print("Need at least two step settings to compare.")
        return 1

    breeds = list(app_gradio.BREED_NAMES)
    results: list[dict[str, float | int]] = []

    for steps in settings:
        out_dir = Path(args.out_dir) / f"steps{steps}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(args.seed)

        count = 0
        for breed in breeds:
            slug = breed.replace(" ", "_")
            for index in range(args.samples_per_breed):
                image = app_gradio.generate_cat(
                    breed, cfg_scale=args.cfg_scale, steps=steps
                )
                if image is None:
                    print(f"Generation failed for {breed} @ {steps} steps.")
                    return 1
                image.save(out_dir / f"{slug}_{index:03d}.png")
                count += 1

        fake_images = evaluate_full.load_images_from_directory(out_dir, 128)
        splits = max(1, min(10, count // 10))
        is_mean, is_std = evaluate_full.compute_inception_score(
            fake_images, device, splits=splits
        )
        entry: dict[str, float | int] = {
            "steps": steps,
            "samples": count,
            "inception_score_mean": is_mean,
            "inception_score_std": is_std,
        }
        if real_images:
            entry["fid"] = evaluate_full.compute_fid(real_images, fake_images, device)
        results.append(entry)
        print(
            f"steps={steps}: IS={is_mean:.2f}+/-{is_std:.2f}"
            + (f" FID={entry['fid']:.2f}" if "fid" in entry else "")
        )

    baseline = results[0]
    print(f"\n{'steps':>7}{'FID':>10}{'IS':>9}{'dFID':>10}{'dIS':>8}")
    print("-" * 44)
    for entry in results:
        fid_str = f"{entry['fid']:>10.2f}" if "fid" in entry else f"{'n/a':>10}"
        d_fid = entry.get("fid", baseline.get("fid", 0.0)) - baseline.get("fid", 0.0)
        d_is = float(entry["inception_score_mean"]) - float(
            baseline["inception_score_mean"]
        )
        print(
            f"{entry['steps']:>7}{fid_str}"
            f"{float(entry['inception_score_mean']):>9.2f}{d_fid:>10.2f}{d_is:>+8.2f}"
        )

    report = Path(args.out_dir) / "report.json"
    report.write_text(
        json.dumps(
            {
                "onnx": str(onnx_path),
                "cfg_scale": args.cfg_scale,
                "samples_per_breed": args.samples_per_breed,
                "seed": args.seed,
                "real_dir": str(real_dir) if real_dir else None,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nReport: {report}")

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as handle:
            handle.write("## ONNX sampling-step parity\n\n")
            handle.write("| steps | FID | Inception Score |\n|---|---|---|\n")
            for entry in results:
                fid = f"{entry['fid']:.2f}" if "fid" in entry else "n/a"
                handle.write(
                    f"| {entry['steps']} | {fid} | "
                    f"{float(entry['inception_score_mean']):.2f} |\n"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
