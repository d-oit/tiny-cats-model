"""src/test_onnx_inference.py

Test ONNX inference for the cats classifier model.

This script validates numerical consistency between PyTorch and ONNX
inference, and reports latency and accuracy metrics.

Usage:
    python src/test_onnx_inference.py
    python src/test_onnx_inference.py --onnx-model PATH --checkpoint PATH
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError("onnxruntime is required. Install with: pip install onnxruntime>=1.15.0") from e

from model import cats_model

# ImageNet mean/std for normalization (matches training preprocessing)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE = 224


def create_sample_image(size: tuple[int, int] = (224, 224)) -> Image.Image:
    """Create a sample RGB image for testing.

    Args:
        size: Output image size (width, height).

    Returns:
        PIL Image with random pixel values.
    """
    # Create a simple test pattern (gradient)
    img_array = np.zeros((*size, 3), dtype=np.uint8)
    for i in range(size[1]):
        for j in range(size[0]):
            img_array[i, j, 0] = int(255 * i / size[1])  # Red gradient
            img_array[i, j, 1] = int(255 * j / size[0])  # Green gradient
            img_array[i, j, 2] = 128  # Constant blue

    return Image.fromarray(img_array, mode="RGB")


def preprocess_image(
    image: Image.Image,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> torch.Tensor:
    """Preprocess image for model inference.

    Args:
        image: PIL Image to preprocess.
        image_size: Target size for resizing.

    Returns:
        Preprocessed tensor of shape (1, 3, H, W).
    """
    transform = T.Compose(
        [
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform(image).unsqueeze(0)


def run_pytorch_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 10,
) -> dict[str, Any]:
    """Run PyTorch inference and measure latency.

    Args:
        model: PyTorch model in eval mode.
        input_tensor: Input tensor of shape (batch, 3, H, W).
        num_runs: Number of inference runs for timing.

    Returns:
        Dictionary with output tensor and latency statistics.
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        _ = model(input_tensor)

    # Timed runs
    latencies = []
    output = None
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            output = model(input_tensor)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

    return {
        "output": output,
        "latency_mean_ms": np.mean(latencies),
        "latency_std_ms": np.std(latencies),
        "latency_min_ms": np.min(latencies),
        "latency_max_ms": np.max(latencies),
    }


def run_onnx_inference(
    session: ort.InferenceSession,
    input_array: np.ndarray,
    num_runs: int = 10,
) -> dict[str, Any]:
    """Run ONNX inference and measure latency.

    Args:
        session: ONNX Runtime inference session.
        input_array: Input numpy array of shape (batch, 3, H, W).
        num_runs: Number of inference runs for timing.

    Returns:
        Dictionary with output array and latency statistics.
    """
    input_name = session.get_inputs()[0].name

    # Warmup
    _ = session.run(None, {input_name: input_array})

    # Timed runs
    latencies = []
    output = None
    for _ in range(num_runs):
        start = time.perf_counter()
        outputs = session.run(None, {input_name: input_array})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
        output = outputs[0]

    return {
        "output": output,
        "latency_mean_ms": np.mean(latencies),
        "latency_std_ms": np.std(latencies),
        "latency_min_ms": np.min(latencies),
        "latency_max_ms": np.max(latencies),
    }


def compare_outputs(
    pytorch_output: torch.Tensor,
    onnx_output: np.ndarray,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Compare PyTorch and ONNX outputs for numerical consistency.

    Args:
        pytorch_output: PyTorch output tensor.
        onnx_output: ONNX output array.
        tolerance: Maximum allowed absolute difference.

    Returns:
        Dictionary with comparison metrics.
    """
    pytorch_np = pytorch_output.detach().cpu().numpy()

    abs_diff = np.abs(pytorch_np - onnx_output)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    std_diff = np.std(abs_diff)

    passed = max_diff < tolerance

    return {
        "max_difference": max_diff,
        "mean_difference": mean_diff,
        "std_difference": std_diff,
        "tolerance": tolerance,
        "passed": passed,
    }


def test_onnx_inference(
    onnx_model_path: str | Path,
    checkpoint_path: str | Path | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_runs: int = 10,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Run comprehensive ONNX inference tests.

    Args:
        onnx_model_path: Path to the ONNX model file.
        checkpoint_path: Optional path to PyTorch checkpoint for comparison.
        image_size: Input image size for preprocessing.
        num_runs: Number of inference runs for timing.
        tolerance: Maximum allowed numerical difference.

    Returns:
        Dictionary with all test results and metrics.
    """
    onnx_model_path = Path(onnx_model_path)
    if not onnx_model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

    results: dict[str, Any] = {
        "onnx_model_path": str(onnx_model_path),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }

    # Create sample image
    sample_image = create_sample_image((image_size, image_size))
    input_tensor = preprocess_image(sample_image, image_size)
    input_array = input_tensor.numpy()

    results["input_shape"] = list(input_tensor.shape)

    # Load ONNX model
    session = ort.InferenceSession(
        onnx_model_path,
        providers=["CPUExecutionProvider"],
    )
    onnx_inputs = session.get_inputs()[0]
    onnx_outputs = session.get_outputs()[0]

    results["onnx_input_name"] = onnx_inputs.name
    results["onnx_input_shape"] = onnx_inputs.shape
    results["onnx_output_name"] = onnx_outputs.name
    results["onnx_output_shape"] = onnx_outputs.shape

    # Run ONNX inference
    onnx_results = run_onnx_inference(session, input_array, num_runs)
    results["onnx_inference"] = {
        "output_shape": list(onnx_results["output"].shape),
        "latency_mean_ms": float(onnx_results["latency_mean_ms"]),
        "latency_std_ms": float(onnx_results["latency_std_ms"]),
        "latency_min_ms": float(onnx_results["latency_min_ms"]),
        "latency_max_ms": float(onnx_results["latency_max_ms"]),
    }

    # Run PyTorch inference if checkpoint provided
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = cats_model(num_classes=2, backbone="resnet18", pretrained=False)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        pytorch_results = run_pytorch_inference(model, input_tensor, num_runs)
        results["pytorch_inference"] = {
            "output_shape": list(pytorch_results["output"].shape),
            "latency_mean_ms": float(pytorch_results["latency_mean_ms"]),
            "latency_std_ms": float(pytorch_results["latency_std_ms"]),
            "latency_min_ms": float(pytorch_results["latency_min_ms"]),
            "latency_max_ms": float(pytorch_results["latency_max_ms"]),
        }

        # Compare outputs
        comparison = compare_outputs(
            pytorch_results["output"],
            onnx_results["output"],
            tolerance,
        )
        results["comparison"] = {
            "max_difference": float(comparison["max_difference"]),
            "mean_difference": float(comparison["mean_difference"]),
            "std_difference": float(comparison["std_difference"]),
            "tolerance": tolerance,
            "passed": comparison["passed"],
        }

        # Calculate speedup
        if pytorch_results["latency_mean_ms"] > 0:
            speedup = pytorch_results["latency_mean_ms"] / onnx_results["latency_mean_ms"]
            results["speedup"] = {
                "onnx_vs_pytorch": float(speedup),
                "faster": "ONNX" if speedup > 1 else "PyTorch",
            }
    else:
        results["comparison"] = None
        results["speedup"] = None
        results["pytorch_inference"] = None

    return results


def print_results(results: dict[str, Any]) -> None:
    """Print test results in a formatted manner.

    Args:
        results: Dictionary containing test results.
    """
    print("\n" + "=" * 60)
    print("ONNX Inference Test Results")
    print("=" * 60)

    print(f"\nModel: {results['onnx_model_path']}")
    print(f"Input shape: {results['input_shape']}")
    print(f"ONNX input: {results['onnx_input_name']} {results['onnx_input_shape']}")
    print(f"ONNX output: {results['onnx_output_name']} {results['onnx_output_shape']}")

    print("\n--- ONNX Inference ---")
    onnx = results["onnx_inference"]
    print(f"  Output shape: {onnx['output_shape']}")
    print(f"  Latency (mean): {onnx['latency_mean_ms']:.3f} ms")
    print(f"  Latency (std):  {onnx['latency_std_ms']:.3f} ms")
    print(f"  Latency (min):  {onnx['latency_min_ms']:.3f} ms")
    print(f"  Latency (max):  {onnx['latency_max_ms']:.3f} ms")

    if results["pytorch_inference"]:
        print("\n--- PyTorch Inference ---")
        pt = results["pytorch_inference"]
        print(f"  Output shape: {pt['output_shape']}")
        print(f"  Latency (mean): {pt['latency_mean_ms']:.3f} ms")
        print(f"  Latency (std):  {pt['latency_std_ms']:.3f} ms")
        print(f"  Latency (min):  {pt['latency_min_ms']:.3f} ms")
        print(f"  Latency (max):  {pt['latency_max_ms']:.3f} ms")

    if results["comparison"]:
        print("\n--- Numerical Consistency ---")
        comp = results["comparison"]
        status = "PASSED" if comp["passed"] else "FAILED"
        print(f"  Max difference: {comp['max_difference']:.6e}")
        print(f"  Mean difference: {comp['mean_difference']:.6e}")
        print(f"  Std difference: {comp['std_difference']:.6e}")
        print(f"  Tolerance: {comp['tolerance']:.1e}")
        print(f"  Status: [{status}]")

    if results["speedup"]:
        print("\n--- Performance Comparison ---")
        speedup = results["speedup"]
        print(f"  Speedup: {speedup['onnx_vs_pytorch']:.2f}x ({speedup['faster']} is faster)")

    print("\n" + "=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test ONNX inference for cats classifier")
    parser.add_argument(
        "--onnx-model",
        type=str,
        default="frontend/public/models/cats.onnx",
        help="Path to the ONNX model file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to PyTorch checkpoint for comparison (optional)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Input image size (default: 224)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of inference runs for timing (default: 10)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Maximum allowed numerical difference (default: 1e-4)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for ONNX inference testing."""
    args = parse_args()

    try:
        results = test_onnx_inference(
            onnx_model_path=args.onnx_model,
            checkpoint_path=args.checkpoint,
            image_size=args.image_size,
            num_runs=args.num_runs,
            tolerance=args.tolerance,
        )
        print_results(results)

        # Exit with error if comparison failed
        if results["comparison"] and not results["comparison"]["passed"]:
            print("\n[ERROR] Numerical consistency check FAILED!")
            raise SystemExit(1)

        print("\n[SUCCESS] All tests passed!")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nHint: Export the model first with:")
        print("  python src/export_onnx.py --checkpoint PATH_TO_CHECKPOINT")
        raise SystemExit(1) from e
    except ImportError as e:
        print(f"\n[ERROR] {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
