#!/usr/bin/env python3
"""scripts/tune_dit_modal.py

GPU A/B for DiT training configurations on Modal.

Each arm trains in its own container, so the run costs the same GPU-hours as a
sequential sweep but finishes in one arm's wall-clock time. Every arm is then
scored on the same held-out split under a fixed uniform timestep distribution,
which keeps arms that differ in ``--timestep-sampling`` comparable (the in-run
metric follows each arm's own distribution).

Reuses the training app's image and dataset volume from ``src/train_dit.py`` so
the environment can never drift from real training.

Usage
-----
    # Default GPU A/B: timesteps at lr 5e-4, plus a 10x-higher LR arm
    modal run scripts/tune_dit_modal.py --steps 1600 --batch-size 32

    # Custom arms (semicolon-separated key=value specs)
    modal run scripts/tune_dit_modal.py --steps 1600 \\
      --arms "label=a,lr=5e-4;label=b,lr=1e-3,timestep_sampling=logit_normal"

    # Sequential (single container) instead of parallel
    modal run scripts/tune_dit_modal.py --steps 1600 --processes 1

Cost reference
--------------
Measured T4 throughput is ~1.16 steps/s at batch 32 (128x128, AMP), so 1,600
steps is ~23 minutes of GPU time per arm plus container startup and one
checkpoint save. Three arms in parallel is therefore roughly one GPU-hour on T4.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"

# This module is imported both locally and *inside the container* (Modal hydrates
# the entrypoint module there to locate the function), so every import below must
# resolve in both places: ../src locally, /app in the container. Anything not
# baked into the image — e.g. this directory's tuning harness — must be imported
# lazily from the local entrypoint instead.
for _candidate in (SCRIPTS_DIR, SRC_DIR, Path("/app")):
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

# Reuse the training app so the image, volume, and GPU preference are identical
# to a real run (ADR-057: never assume the container differs from training).
from train_dit import app, image, volume_data  # noqa: E402

REFERENCE_SAMPLING = "uniform"

DEFAULT_ARMS = (
    "label=uniform-lr5e-4,timestep_sampling=uniform,lr=5e-4,warmup_steps=100;"
    "label=logitnorm-lr5e-4,timestep_sampling=logit_normal,lr=5e-4,warmup_steps=100;"
    "label=uniform-lr1e-3,timestep_sampling=uniform,lr=1e-3,warmup_steps=100"
)


@app.function(
    image=image,
    volumes={"/data": volume_data},
    gpu=["T4", "L4"],  # T4 preferred for cost, L4 as the fallback
    timeout=5400,
)
def tune_arm(
    arm: dict[str, Any],
    steps: int,
    batch_size: int,
    val_split: float,
    val_batches: int,
    seed: int,
    data_dir: str = "/data/cats",
) -> dict[str, Any]:
    """Train one arm, then score its final checkpoint on the held-out split.

    Args:
        arm: Arm definition including ``label`` and per-arm overrides.
        steps: Training steps for the arm.
        batch_size: Batch size.
        val_split: Held-out fraction.
        val_batches: Held-out batches scored.
        seed: Seed shared by every arm.
        data_dir: Dataset root inside the container.

    Returns:
        Result dict with label, overrides, held-out loss, and elapsed seconds.
    """
    sys.path.insert(0, "/app")
    os.chdir("/app")

    import logging
    import time

    import torch

    from dataset import create_train_val_dataloaders
    from dit import TinyDiT, load_state_dict_checked
    from dit_validation import evaluate_flow_loss
    from train_dit import train_dit_local

    overrides = {key: value for key, value in arm.items() if key != "label"}
    label = str(arm["label"])
    work_dir = Path("/tmp/tuning")
    work_dir.mkdir(parents=True, exist_ok=True)
    output = work_dir / f"{label}_model.pt"
    ema_output = work_dir / f"{label}_ema.pt"

    logger = logging.getLogger(f"tune.{label}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(f"%(asctime)s | {label} | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    params: dict[str, Any] = {
        "data_dir": data_dir,
        "steps": steps,
        "batch_size": batch_size,
        "output": str(output),
        "ema_output": str(ema_output),
        "num_workers": 0,
        # One checkpoint at the end: the harness owns the comparison metric, so
        # the in-run validation window would only add GPU time.
        "save_interval": steps,
        "sample_interval": steps * 10,
        "val_split": 0.0,
        "early_stopping_patience": 0,
        "seed": seed,
        "no_hub_push": True,
        "logger": logger,
    }
    params.update(overrides)

    started = time.time()
    train_dit_local(**params)

    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    raw_config = checkpoint.get("config")
    config = raw_config if isinstance(raw_config, dict) else {}
    model = TinyDiT(
        image_size=config.get("image_size", 128),
        patch_size=config.get("patch_size", 16),
        embed_dim=config.get("embed_dim", 384),
        depth=config.get("depth", 12),
        num_heads=config.get("num_heads", 6),
        num_classes=config.get("num_classes") or 13,
    )
    load_state_dict_checked(model, checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Scoring on {device} under {REFERENCE_SAMPLING} timesteps")
    model = model.to(device).eval()

    _train_loader, val_loader = create_train_val_dataloaders(
        data_dir,
        batch_size=batch_size,
        image_size=model.image_size,
        num_workers=0,
        val_split=val_split,
        seed=seed,
    )
    if val_loader is None:
        raise RuntimeError("Held-out split is empty; raise val_split")

    loss = evaluate_flow_loss(
        model,
        val_loader,
        device,
        num_batches=val_batches,
        seed=seed,
        timestep_sampling=REFERENCE_SAMPLING,
    )

    # Ephemeral /tmp only: no volume commits, no tuning artifacts left behind.
    output.unlink(missing_ok=True)
    ema_output.unlink(missing_ok=True)

    return {
        "label": label,
        "overrides": overrides,
        "held_out_loss": loss,
        "ok": loss == loss,
        "seconds": round(time.time() - started, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


@app.local_entrypoint()
def tune_main(
    steps: int = 1600,
    batch_size: int = 32,
    val_split: float = 0.05,
    val_batches: int = 16,
    seed: int = 42,
    arms: str = DEFAULT_ARMS,
    processes: int = 3,
) -> None:
    """Launch every arm and print a comparison table.

    Args:
        steps: Training steps per arm.
        batch_size: Batch size per arm.
        val_split: Held-out fraction used for scoring.
        val_batches: Held-out batches scored per arm.
        seed: Seed shared by all arms.
        arms: Semicolon-separated ``key=value,...`` arm specs.
        processes: Concurrent containers (1 = sequential in one container).
    """
    from tune_dit_configs import parse_arm

    arm_specs = [spec.strip() for spec in arms.split(";") if spec.strip()]
    parsed = [parse_arm(spec) for spec in arm_specs]

    print("=" * 68)
    print("DiT GPU configuration comparison (Modal)")
    print("=" * 68)
    print(f"Budget:   {steps} steps x batch {batch_size} per arm")
    print(f"Metric:   held-out flow-matching MSE under {REFERENCE_SAMPLING} timesteps")
    print(f"Scored:   {val_batches} held-out batches, seed {seed}")
    print(f"Arms:     {len(parsed)} | concurrency: {processes}")
    print("=" * 68)

    results: list[dict[str, Any]] = []
    if processes > 1:
        handles = [
            tune_arm.spawn(arm, steps, batch_size, val_split, val_batches, seed)
            for arm in parsed
        ]
        results.extend(handle.get() for handle in handles)
    else:
        results.extend(
            tune_arm.remote(arm, steps, batch_size, val_split, val_batches, seed)
            for arm in parsed
        )

    print("\n" + "=" * 68)
    print(f"{'arm':<24} {'held-out loss':>14} {'secs':>7} {'gpu':>5}")
    print("-" * 68)
    finite = [r for r in results if r["ok"]]
    best = min((r["held_out_loss"] for r in finite), default=float("nan"))
    ordered = sorted(
        results, key=lambda r: r["held_out_loss"] if r["ok"] else float("inf")
    )
    for result in ordered:
        loss_text = f"{result['held_out_loss']:.6f}" if result["ok"] else "NaN"
        print(
            f"{result['label']:<24} {loss_text:>14} "
            f"{result['seconds']:>7.0f} {result['gpu']:>5}"
        )
    print("=" * 68)
    if len(finite) > 1:
        spread = max(r["held_out_loss"] for r in finite) - best
        print(f"Spread across arms: {spread:.6f}")
    print("Compare against the CPU smoke ranking before changing defaults.")
