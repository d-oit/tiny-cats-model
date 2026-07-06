"""src/export_onnx.py

Export the trained cats classifier to ONNX format for web inference.

Usage:
    python src/export_onnx.py
    python src/export_onnx.py --checkpoint PATH --output PATH
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import cats_model


def export_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    opset_version: int = 17,
    num_classes: int | None = None,
    backbone: str = "resnet18",
) -> None:
    """Export a trained cats classifier to ONNX format.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        output_path: Path where the ONNX model will be saved.
        opset_version: ONNX opset version to use.
        num_classes: Number of output classes (auto-detected if None).
        backbone: Model backbone architecture.
    """
    checkpoint_path = (
        Path(checkpoint_path)
        if isinstance(checkpoint_path, str)
        else checkpoint_path
        if isinstance(checkpoint_path, str)
        else checkpoint_path
    )

    output_path = (
        Path(output_path)
        if isinstance(output_path, str)
        else output_path
        if isinstance(output_path, str)
        else output_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint and extract state_dict
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    elif "model" in state:
        state_dict = state["model"]
    else:
        state_dict = state

    # Auto-detect num_classes from the final classifier layer
    if num_classes is None:
        num_classes = 2  # default fallback
        for key in reversed(list(state_dict.keys())):
            if key.endswith(".weight") and (
                "fc" in key or "classifier" in key or "head" in key
            ):
                num_classes = state_dict[key].shape[0]
                break

    model = cats_model(num_classes=num_classes, backbone=backbone, pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"Exported ONNX model to {output_path}")
    print(
        "  Input shape: [1, 3, 224, 224] (batch=1, channels=3, height=224, width=224)"
    )
    print("  Output shape: [1, 2] (batch=1, num_classes=2)")
    print("  Classes: ['cat', 'not_cat']")
    print(f"  Opset version: {opset_version}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export cats classifier to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="cats_model.pt",
        help="Path to the trained checkpoint (default: cats_model.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="frontend/public/models/cats.onnx",
        help="Output path for the ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
    )
