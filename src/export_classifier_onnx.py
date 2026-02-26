"""src/export_classifier_onnx.py

Export the ResNet classifier model to ONNX format for web inference.

This script exports the trained cats classifier to ONNX format,
following the same pattern as export_dit_onnx.py.

Features:
- Export classifier checkpoint to ONNX
- Verification and test options
- Support for dynamic batch sizes
- Compatible with frontend classify page

Usage:
    python src/export_classifier_onnx.py
    python src/export_classifier_onnx.py --checkpoint PATH --output PATH
    python src/export_classifier_onnx.py --verify --test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import cats_model


def create_dummy_checkpoint(
    output_path: str | Path,
    num_classes: int = 2,
    backbone: str = "resnet18",
) -> None:
    """Create a dummy checkpoint for testing export.

    Args:
        output_path: Path to save the dummy checkpoint
        num_classes: Number of output classes
        backbone: Backbone architecture to use
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = cats_model(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=False,
    )
    model.eval()

    checkpoint = {
        "model": model.state_dict(),
        "epoch": 0,
        "config": {
            "num_classes": num_classes,
            "backbone": backbone,
            "image_size": 224,
        },
    }

    torch.save(checkpoint, output_path)
    print(f"Created dummy checkpoint at {output_path}")


def load_model(
    checkpoint_path: str | Path,
    num_classes: int = 2,
    backbone: str = "resnet18",
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load a classifier model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        num_classes: Number of output classes
        backbone: Backbone architecture to use
        device: Device to load the model on

    Returns:
        Loaded classifier model in eval mode
    """
    if device is None:
        device = torch.device("cpu")

    checkpoint_path = Path(checkpoint_path)

    # Create model
    model = cats_model(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=False,
    )

    # Load checkpoint if it exists
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Creating model with random weights for export testing")

    model.eval()
    return model.to(device)


def export_classifier_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    opset_version: int = 17,
    image_size: int = 224,
) -> None:
    """Export the classifier model to ONNX format.

    Args:
        model: Classifier model to export
        output_path: Path where the ONNX model will be saved
        opset_version: ONNX opset version to use
        image_size: Input image size (default: 224)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Create dummy inputs for tracing
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, image_size, image_size)

    input_names = ["input"]
    output_names = ["output"]

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )

    print(f"Exported ONNX model to {output_path}")
    print("  Input shape:")
    print(f"    input: [batch_size, 3, {image_size}, {image_size}]")
    print("  Output shape:")
    print("    output: [batch_size, 2] (cat, not_cat)")
    print(f"  Opset version: {opset_version}")


def verify_onnx_model(model_path: str | Path) -> None:
    """Verify the exported ONNX model.

    Args:
        model_path: Path to the ONNX model
    """
    try:
        import onnx

        model_path = Path(model_path)
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)

        print(f"\nONNX Model Verification: {model_path}")
        print(f"  IR Version: {model.ir_version}")
        print(f"  Opset Version: {model.opset_import[0].version}")

        print("  Inputs:")
        for inp in model.graph.input:
            shape = [
                d.dim_value if d.dim_value else d.dim_param
                for d in inp.type.tensor_type.shape.dim
            ]
            print(f"    {inp.name}: {shape}")

        print("  Outputs:")
        for out in model.graph.output:
            shape = [
                d.dim_value if d.dim_value else d.dim_param
                for d in out.type.tensor_type.shape.dim
            ]
            print(f"    {out.name}: {shape}")

        # Count parameters
        param_count = sum(
            onnx.numpy_helper.to_array(init).size for init in model.graph.initializer
        )
        print(f"  Parameters: {param_count:,}")
        print(f"  Model size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

    except ImportError:
        print("ONNX not installed. Install with: pip install onnx")
    except Exception as e:
        print(f"Verification error: {e}")


def test_onnx_inference(model_path: str | Path) -> None:
    """Test inference with the exported ONNX model.

    Args:
        model_path: Path to the ONNX model
    """
    try:
        import numpy as np
        import onnxruntime as ort

        model_path = Path(model_path)
        session = ort.InferenceSession(str(model_path))

        # Get input/output info
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        print(f"\nONNX Runtime Test: {model_path}")
        print(f"  Execution Providers: {session.get_providers()}")

        print("  Inputs:")
        for inp in inputs:
            print(f"    {inp.name}: {inp.shape} ({inp.type})")

        print("  Outputs:")
        for out in outputs:
            print(f"    {out.name}: {out.shape} ({out.type})")

        # Run a test inference
        batch_size = 1
        image_size = 224

        # Create random input (simulating normalized image)
        input_data = np.random.randn(batch_size, 3, image_size, image_size).astype(
            np.float32
        )

        result = session.run(None, {"input": input_data})

        print("\n  Test Inference:")
        print(f"    Input shape: {input_data.shape}")
        print(f"    Output shape: {result[0].shape}")
        print(f"    Output range: [{result[0].min():.4f}, {result[0].max():.4f}]")
        print(f"    Output mean: {result[0].mean():.4f}, std: {result[0].std():.4f}")

        # Apply softmax to get probabilities
        logits = result[0]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        print("\n  Probabilities:")
        print(f"    Cat: {probabilities[0, 0] * 100:.2f}%")
        print(f"    Not Cat: {probabilities[0, 1] * 100:.2f}%")

    except ImportError:
        print("ONNX Runtime not installed. Install with: pip install onnxruntime")
    except Exception as e:
        print(f"Inference test error: {e}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export cats classifier to ONNX format"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/classifier/best_cats_model.pt",
        help="Path to the trained checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="frontend/public/models/cats_classifier.onnx",
        help="Output path for the ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size for the model (default: 224)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of output classes (default: 2)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        help="Backbone architecture (default: resnet18)",
    )
    parser.add_argument(
        "--create-dummy",
        action="store_true",
        help="Create a dummy checkpoint if none exists",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the exported ONNX model",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test inference with the exported model",
    )
    return parser.parse_args()


def main() -> None:
    """Main export function."""
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    # Create dummy checkpoint if needed
    if not checkpoint_path.exists() and args.create_dummy:
        print("Creating dummy checkpoint for testing...")
        create_dummy_checkpoint(
            checkpoint_path,
            num_classes=args.num_classes,
            backbone=args.backbone,
        )

    # Load model
    device = torch.device("cpu")
    model = load_model(
        checkpoint_path,
        num_classes=args.num_classes,
        backbone=args.backbone,
        device=device,
    )

    # Export to ONNX
    export_classifier_onnx(
        model=model,
        output_path=output_path,
        opset_version=args.opset,
        image_size=args.image_size,
    )

    # Verify if requested
    if args.verify:
        verify_onnx_model(output_path)

    # Test inference if requested
    if args.test:
        test_onnx_inference(output_path)

    print("\nExport complete!")
    print(f"Model saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
