#!/usr/bin/env python3
"""scripts/tune_dit_configs.py

Compare DiT training configurations on one common held-out metric.

Why this exists
---------------
`train_dit_local()` scores each run with the validation loss of *that run's*
timestep distribution, so two arms differing in `--timestep-sampling` cannot be
compared by it. This harness trains each arm, rebuilds the arm's final
checkpoint, and re-scores every arm on the same held-out split under a fixed
uniform timestep distribution, so the numbers are directly comparable.

Usage
-----
    # Default A/B: uniform vs logit-normal timesteps
    python scripts/tune_dit_configs.py --data-dir data/cats --steps 250

    # Learning-rate / warmup sweep
    python scripts/tune_dit_configs.py --data-dir data/cats --steps 250 \
        --arms label=lr5e-4,warmup_steps=20,lr=5e-4 \
        --arms label=lr1e-3,warmup_steps=20,lr=1e-3 \
        --arms label=lr1e-3-w100,warmup_steps=100,lr=1e-3

    # Machine-readable results
    python scripts/tune_dit_configs.py --data-dir data/cats --json-out tune.json

Caveats
-------
A few hundred CPU steps at batch 8 is a smoke comparison, not a substitute for a
full GPU run: differences that only emerge after tens of thousands of optimizer
steps will not show up here, and a small delta may be noise. Use it to catch
regressions and to pick which configuration is worth paying GPU time for.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from dataset import create_train_val_dataloaders  # noqa: E402
from dit_validation import evaluate_flow_loss  # noqa: E402
from train_dit import train_dit_local  # noqa: E402
from validate_model import build_model_from_checkpoint  # noqa: E402

# Fixed evaluation distribution: every arm is scored on the same timesteps.
REFERENCE_SAMPLING = "uniform"

DEFAULT_ARMS = [
    "label=uniform,timestep_sampling=uniform",
    "label=logit-normal,timestep_sampling=logit_normal",
]


def _coerce(value: str) -> Any:
    """Best-effort conversion of a CLI ``key=value`` string."""
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


def parse_arm(spec: str) -> dict[str, Any]:
    """Parse ``key=value,key=value`` into an arm definition.

    Args:
        spec: Comma-separated key=value pairs.

    Returns:
        Arm definition; ``label`` is always present.

    Raises:
        ValueError: If the spec is empty or has no label.
    """
    arm: dict[str, Any] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, value = part.partition("=")
        arm[key.strip()] = _coerce(value.strip())

    if "label" not in arm:
        raise ValueError(f"Arm spec needs a label: {spec!r}")
    return arm


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare DiT training configurations on held-out loss",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset root")
    parser.add_argument("--steps", type=int, default=250, help="Steps per arm")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        help="Base LR warmup (arms can override per-arm)",
    )
    parser.add_argument("--val-split", type=float, default=0.1, help="Held-out split")
    parser.add_argument(
        "--val-batches",
        type=int,
        default=16,
        help="Held-out batches scored per arm (lower = noisier)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for all arms")
    parser.add_argument(
        "--arms",
        action="append",
        default=None,
        help="Arm as key=value,... (repeatable; defaults to a timestep A/B)",
    )
    parser.add_argument("--json-out", type=str, default=None, help="Write results JSON")
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Do not delete each arm's ~500MB checkpoint after scoring",
    )
    return parser.parse_args(argv)


def _quiet_logger(label: str, log_path: Path) -> logging.Logger:
    """Per-arm logger writing to a file instead of the console."""
    logger = logging.getLogger(f"tune.{label}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(log_path))
    logger.propagate = False
    return logger


def run_arm(
    arm: dict[str, Any],
    args: argparse.Namespace,
    work_dir: Path,
) -> dict[str, Any]:
    """Train one arm and score it on the shared held-out metric.

    Args:
        arm: Arm definition including ``label``.
        args: Parsed CLI arguments.
        work_dir: Scratch directory for checkpoints and per-arm logs.

    Returns:
        Result dict with the label, held-out loss, and wall-clock time.
    """
    overrides = {key: value for key, value in arm.items() if key != "label"}
    label = str(arm["label"])
    slug = label.replace(" ", "_")
    output = work_dir / f"{slug}_model.pt"

    params: dict[str, Any] = {
        "data_dir": args.data_dir,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "output": str(output),
        "ema_output": str(work_dir / f"{slug}_ema.pt"),
        "num_workers": 0,
        # One checkpoint at the end keeps per-arm IO to a single pair and lets
        # the harness own the comparison metric instead of the in-run one.
        "save_interval": args.steps,
        "sample_interval": args.steps * 10,
        "val_split": 0.0,
        "early_stopping_patience": 0,
        "seed": args.seed,
        "no_hub_push": True,
        "logger": _quiet_logger(slug, work_dir / f"{slug}.log"),
    }
    params.update(overrides)

    started = time.time()
    train_dit_local(**params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    model, _ = build_model_from_checkpoint(checkpoint)
    model = model.to(device).eval()

    # The held-out split is rebuilt per arm from the arm's own resolution with
    # the same seed, so every arm sees an identical validation set.
    _train_loader, val_loader = create_train_val_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=getattr(model, "image_size", 128),
        num_workers=0,
        val_split=args.val_split,
        seed=args.seed,
    )
    if val_loader is None:
        raise RuntimeError("Held-out split is empty; raise --val-split")

    loss = evaluate_flow_loss(
        model,
        val_loader,
        device,
        num_batches=args.val_batches,
        seed=args.seed,
        timestep_sampling=REFERENCE_SAMPLING,
    )

    if not args.keep_checkpoints:
        for path in (output, Path(params["ema_output"])):
            path.unlink(missing_ok=True)

    return {
        "label": label,
        "overrides": overrides,
        "held_out_loss": loss,
        "ok": loss == loss,  # False when the loss is NaN
        "seconds": round(time.time() - started, 1),
    }


def main(argv: list[str] | None = None) -> int:
    """Run every arm and print a comparison table."""
    args = parse_args(argv)
    arm_specs = args.arms or DEFAULT_ARMS
    arms = [parse_arm(spec) for spec in arm_specs]

    print("=" * 68)
    print("DiT configuration comparison")
    print("=" * 68)
    print(f"Data:     {args.data_dir}")
    print(f"Budget:   {args.steps} steps x batch {args.batch_size} per arm")
    print(f"Metric:   held-out flow-matching MSE under {REFERENCE_SAMPLING} timesteps")
    print(f"Scored:   {args.val_batches} held-out batches, seed {args.seed}")
    print("=" * 68)

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="dit_tune_") as tmp:
        work_dir = Path(tmp)
        for arm in arms:
            print(f"\n▶ {arm['label']} ...", flush=True)
            result = run_arm(arm, args, work_dir)
            results.append(result)
            status = "ok" if result["ok"] else "NON-FINITE"
            print(
                f"  held-out loss: {result['held_out_loss']:.6f} "
                f"({status}, {result['seconds']}s)"
            )

    print("\n" + "=" * 68)
    print(f"{'arm':<22} {'held-out loss':>14} {'vs best':>10}")
    print("-" * 68)
    finite = [r for r in results if r["ok"]]
    best = min((r["held_out_loss"] for r in finite), default=float("nan"))
    ordered = sorted(
        results, key=lambda r: r["held_out_loss"] if r["ok"] else float("inf")
    )
    for result in ordered:
        delta = result["held_out_loss"] - best if result["ok"] else float("nan")
        loss_text = f"{result['held_out_loss']:.6f}" if result["ok"] else "NaN"
        print(f"{result['label']:<22} {loss_text:>14} {delta:>+10.6f}")
    print("=" * 68)

    if len(finite) > 1:
        spread = max(r["held_out_loss"] for r in finite) - min(
            r["held_out_loss"] for r in finite
        )
        print(
            f"Spread across arms: {spread:.6f}. A few hundred CPU steps cannot "
            "resolve small differences —\ntreat the ranking as a smoke check and "
            "confirm on GPU before changing defaults."
        )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "steps": args.steps,
                    "batch_size": args.batch_size,
                    "val_batches": args.val_batches,
                    "seed": args.seed,
                    "reference_sampling": REFERENCE_SAMPLING,
                    "results": results,
                },
                indent=2,
            )
        )
        print(f"Results written to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
