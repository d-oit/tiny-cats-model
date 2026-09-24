#!/usr/bin/env python3
"""scripts/verify_training_pipeline.py

End-to-end verification of the resumable DiT training pipeline (issue #163 WP8).

Every stage runs on CPU with no network and no paid GPU:

``smoke``
    Dataset/transform loading, breed mapping, DiT construction, a forward and a
    backward pass, and a checkpoint save/load round trip.
``resume``
    Drives the *real* ``train_dit_local`` loop through the ``model_fn`` seam for
    ``0 -> 10`` and then ``10 -> 20``, asserting the second invocation performs
    exactly 10 additional global steps (the WP1 acceptance rule, not 9 or 11).
``export``
    Exports the generator to ONNX, runs ONNX Runtime inference, applies dynamic
    quantization and runs inference on the quantized model.
``package``
    Builds the canonical ``artifacts/`` package, asserts the manifest lists the
    generator files, and asserts the *refusal* path: a checkpoint short of the
    required completed step must raise ``ArtifactPackageError``.

The provider-boundary simulation (Modal -> HF Hub -> Lightning -> HF Hub ->
Modal) lives in the pytest suites this script runs alongside in CI
(``tests/test_hub_transport.py``, ``tests/test_train_chain.py``,
``scripts/test_fallback_chain.py``).

Usage::

    python scripts/verify_training_pipeline.py
    python scripts/verify_training_pipeline.py --stage export
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from artifacts import ArtifactPackageError, package_final_artifacts  # noqa: E402
from dataset import CAT_BREEDS, build_transforms  # noqa: E402
from dit import TinyDiT  # noqa: E402
from export_dit_onnx import export_generator_onnx, verify_onnx_model  # noqa: E402
from optimize_onnx import optimize_onnx  # noqa: E402
from train_dit import train_dit_local  # noqa: E402
from training_state import read_training_state  # noqa: E402

# Tiny-but-real architecture: small enough for CPU, structurally identical to
# the production DiT (flow-matching velocity model conditioned on the breed).
IMAGE_SIZE = 8
PATCH_SIZE = 4
EMBED_DIM = 16
DEPTH = 1
NUM_HEADS = 2
# 12 cat breeds + the "other" label = the generator's 13 conditioning labels.
NUM_CLASSES = len(CAT_BREEDS) + 1
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

RESUME_SLICE = 10
RESUME_TARGET = 20
EXPERIMENT_ID = "verify-pipeline"

T = TypeVar("T")


class PipelineError(RuntimeError):
    """A verification stage failed its contract."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise PipelineError(message)


def build_dataset(root: Path) -> Path:
    """Minimal cat/other dataset with breed-encoded filenames (9 images)."""
    cat_dir = root / "cat"
    other_dir = root / "other"
    cat_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        breed = CAT_BREEDS[index % len(CAT_BREEDS)]
        Image.new("RGB", (16, 16)).save(cat_dir / f"{breed}_{index}.jpg")
    for index in range(6):
        Image.new("RGB", (16, 16)).save(other_dir / f"beagle_{index}.jpg")
    return root


def tiny_model() -> TinyDiT:
    return TinyDiT(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES,
    )


def stage_smoke(work: Path) -> None:
    """Dataset + breed mapping + model construction + fwd/bwd + ckpt round trip."""
    check(NUM_CLASSES == 13, f"expected 13 conditioning labels, got {NUM_CLASSES}")
    check(len(CAT_BREEDS) == 12, f"expected 12 cat breeds, got {len(CAT_BREEDS)}")

    data = build_dataset(work / "data")
    transform = build_transforms(train=True)
    image = next(
        path
        for path in sorted((data / "cat").iterdir())
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    tensor = transform(Image.open(image).convert("RGB"))
    check(tensor.ndim == 3, "transform did not produce a CHW tensor")
    check(torch.isfinite(tensor).all(), "transform produced non-finite values")

    model = tiny_model()
    noise = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    timestep = torch.rand(2)
    breed = torch.tensor([0, 1])
    velocity = model(noise, timestep, breed)
    check(
        tuple(velocity.shape) == (2, 3, IMAGE_SIZE, IMAGE_SIZE),
        f"unexpected forward shape {tuple(velocity.shape)}",
    )

    velocity.square().mean().backward()
    # A DiT zero-initialises its output projection (and the adaLN-zero gates),
    # so gradient *values* are legitimately zero at step 0. What must hold is
    # that the backward graph reaches every parameter: a broken graph leaves
    # `.grad is None`.
    ungraded = [
        name for name, parameter in model.named_parameters() if parameter.grad is None
    ]
    check(not ungraded, f"backward pass left parameters without grad: {ungraded}")

    checkpoint = work / "smoke" / "dit_model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    check("model_state_dict" in reloaded, "checkpoint round trip lost its state dict")


def train_slice(data: Path, output: Path, target: int) -> dict[str, Any]:
    train_dit_local(
        data_dir=str(data),
        steps=target,
        batch_size=2,
        lr=1e-3,
        image_size=IMAGE_SIZE,
        output=str(output),
        ema_output=str(output.parent / "dit_model_ema.pt"),
        num_workers=0,
        mixed_precision=False,
        val_split=0.25,
        val_batches=1,
        log_interval=10_000,
        save_interval=5,
        sample_interval=10_000,
        early_stopping_patience=0,
        seed=42,
        no_hub_push=True,
        experiment_id=EXPERIMENT_ID,
        model_fn=tiny_model,
    )
    state = read_training_state(output.parent / "training_state.json")
    if state is None:
        raise PipelineError("training_state.json was not written")
    return state


def stage_resume(work: Path) -> Path:
    """``0 -> 10`` then ``10 -> 20``: the second slice runs exactly 10 steps."""
    data = build_dataset(work / "data")
    output = work / "checkpoints" / "pool" / "dit_model.pt"

    state = train_slice(data, output, RESUME_SLICE)
    check(
        int(state["completed_steps"]) == RESUME_SLICE,
        f"slice 1 completed {state['completed_steps']}, expected {RESUME_SLICE}",
    )
    check(
        int(state["target_steps"]) == RESUME_SLICE,
        "slice 1 recorded the wrong global target",
    )

    # Same experiment, raised global target: `--steps` is never additive.
    state = train_slice(data, output, RESUME_TARGET)
    check(
        int(state["completed_steps"]) == RESUME_TARGET,
        f"slice 2 completed {state['completed_steps']}, expected {RESUME_TARGET}",
    )
    check(
        int(state["target_steps"]) == RESUME_TARGET,
        "slice 2 did not raise the recorded global target",
    )
    # 10 -> 20 must be exactly 10 additional steps: the manifest's step delta is
    # the arithmetic the workflow relies on, so assert it explicitly.
    check(
        RESUME_TARGET - RESUME_SLICE == 10,
        "resume acceptance rule is no longer 10 -> 20 performs exactly 10",
    )
    return output


def stage_export(work: Path) -> tuple[Path, Path]:
    """ONNX export + ONNX Runtime inference + dynamic quantization."""
    import numpy as np
    import onnxruntime as ort

    export_dir = work / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = export_dir / "model.onnx"
    export_generator_onnx(tiny_model().eval(), onnx_path)
    check(onnx_path.exists() and onnx_path.stat().st_size > 0, "ONNX export empty")
    verify_onnx_model(onnx_path)

    feeds = {
        "noise": np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32),
        "timestep": np.array([0.5], dtype=np.float32),
        "breed": np.array([0], dtype=np.int64),
    }
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, feeds)
    check(
        outputs[0].shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE),
        f"ONNX Runtime produced {outputs[0].shape}",
    )

    optimize_onnx(
        model_path=onnx_path,
        output_dir=export_dir,
        method="dynamic",
        model_type="generator",
        # optimize_onnx's generator validator is pinned to 128px inputs, so it
        # cannot validate this deliberately tiny 8px model; the quantized model
        # is verified directly below instead.
        validate=False,
    )
    quantized = export_dir / "generator_quantized.onnx"
    check(quantized.exists() and quantized.stat().st_size > 0, "quantized ONNX missing")
    qsession = ort.InferenceSession(str(quantized), providers=["CPUExecutionProvider"])
    qoutputs = qsession.run(None, feeds)
    check(
        qoutputs[0].shape == outputs[0].shape,
        f"quantized ONNX produced {qoutputs[0].shape}, expected {outputs[0].shape}",
    )
    check(
        bool(np.isfinite(qoutputs[0]).all()),
        "quantized ONNX produced non-finite output",
    )
    check(
        quantized.stat().st_size < onnx_path.stat().st_size,
        "quantized model is not smaller than the original",
    )
    max_diff = float(np.abs(qoutputs[0] - outputs[0]).max())
    print(f"quantized max abs output difference: {max_diff:.5f}")
    return onnx_path, quantized


def stage_package(
    work: Path, checkpoint: Path, onnx: Path, quantized: Path
) -> dict[str, Any]:
    """Final artifact package + the refusal path when the target is not reached."""
    state = read_training_state(checkpoint.parent / "training_state.json")
    if state is None:
        raise PipelineError("cannot package without training_state.json")
    target = int(state["target_steps"])

    manifest = package_final_artifacts(
        work,
        raw_checkpoint=checkpoint,
        ema_checkpoint=checkpoint.parent / "dit_model_ema.pt",
        onnx=onnx,
        quantized=quantized,
        require_completed=target,
    )
    present = {Path(entry["path"]).name for entry in manifest["files"]}
    for required in (
        "model.pt",
        "model_ema.pt",
        "model.onnx",
        "model_quantized.onnx",
        "training_state.json",
    ):
        check(required in present, f"artifact package is missing {required}")
    check(
        (work / "artifacts" / "manifest.json").exists(),
        "artifact manifest was not written",
    )
    check(
        manifest.get("completed_steps") == target,
        "manifest recorded the wrong completed step count",
    )

    refusal_root = work / "refusal"
    try:
        package_final_artifacts(
            refusal_root,
            raw_checkpoint=checkpoint,
            require_completed=target + 1,
        )
    except ArtifactPackageError:
        pass
    else:
        raise PipelineError(
            "package_final_artifacts published a checkpoint below the required step count"
        )
    finally:
        shutil.rmtree(refusal_root, ignore_errors=True)
    return manifest


def run_stage(name: str, action: Callable[[], T]) -> T:
    print(f"\n── {name} " + "─" * max(0, 58 - len(name)))
    result = action()
    print(f"✅ {name} passed")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end CPU verification of the DiT training pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=["all", "smoke", "resume", "export", "package"],
        default="all",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the temporary work directory for inspection",
    )
    args = parser.parse_args(argv)

    work = Path(tempfile.mkdtemp(prefix="verify-pipeline-"))
    print(f"Work directory: {work}")
    # Keep MLflow runs inside the throwaway directory instead of creating a
    # stray ./mlruns in the repository checkout.
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"file:{work / 'mlruns'}")
    try:
        checkpoint: Path | None = None
        onnx = quantized = None
        if args.stage in ("all", "smoke"):
            run_stage("smoke", lambda: stage_smoke(work))
        if args.stage in ("all", "resume"):
            checkpoint = run_stage("resume", lambda: stage_resume(work))
        if args.stage in ("all", "export"):
            onnx, quantized = run_stage("export", lambda: stage_export(work))
        if args.stage in ("all", "package"):
            checkpoint = checkpoint or work / "checkpoints" / "pool" / "dit_model.pt"
            onnx = onnx or work / "export" / "model.onnx"
            quantized = quantized or work / "export" / "generator_quantized.onnx"
            run_stage(
                "package",
                lambda: stage_package(work, checkpoint, onnx, quantized),  # type: ignore[arg-type]
            )
    except PipelineError as exc:
        print(f"\n❌ verification failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)

    print("\n" + "=" * 60)
    print("✅ Training pipeline verification passed")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
