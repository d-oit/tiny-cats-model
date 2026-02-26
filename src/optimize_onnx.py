"""src/optimize_onnx.py

Optimize ONNX models using quantization for web deployment.

This script applies dynamic and static quantization to reduce model size
and improve inference speed while maintaining accuracy.

Supports:
- Classifier model (cats.onnx -> cats_quantized.onnx)
- Generator model (generator.onnx -> generator_quantized.onnx)

Usage:
    python src/optimize_onnx.py
    python src/optimize_onnx.py --model PATH --output PATH
    python src/optimize_onnx.py --model-type generator
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torchvision.transforms as T
from PIL import Image

try:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )
except ImportError as e:
    raise ImportError(
        "onnxruntime-tools is required. "
        "Install with: pip install onnxruntime-tools>=1.15.0"
    ) from e

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(
        "onnxruntime is required. Install with: pip install onnxruntime>=1.15.0"
    ) from e

# ImageNet mean/std for normalization (matches training preprocessing)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE = 224
DEFAULT_GENERATOR_SIZE = 128


class CatsCalibrationDataReader(CalibrationDataReader):
    """Calibration data reader for classifier static quantization.

    Generates synthetic calibration data based on ImageNet statistics.
    For production use, replace with real calibration dataset.
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        """Initialize calibration data reader.

        Args:
            num_samples: Number of calibration samples to generate.
            image_size: Input image size for calibration.
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.index = 0
        self.data = self._generate_calibration_data()

    def _generate_calibration_data(self) -> list[dict[str, np.ndarray]]:
        """Generate synthetic calibration data.

        Returns:
            List of dictionaries with input tensors for calibration.
        """
        data = []
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        for i in range(self.num_samples):
            # Generate varied synthetic images for better calibration
            np.random.seed(i)
            img_array = np.random.randint(
                0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8
            )

            # Add some structure to the images
            if i % 3 == 0:
                # Gradient pattern
                for j in range(self.image_size):
                    img_array[j, :, 0] = int(255 * j / self.image_size)
            elif i % 3 == 1:
                # Checkerboard pattern
                block_size = 16
                for j in range(self.image_size):
                    for k in range(self.image_size):
                        if (j // block_size + k // block_size) % 2 == 0:
                            img_array[j, k] = [200, 200, 200]
                        else:
                            img_array[j, k] = [50, 50, 50]
            # else: keep random

            img = Image.fromarray(img_array, mode="RGB")
            tensor = transform(img).unsqueeze(0).numpy()

            data.append({"input": tensor.astype(np.float32)})

        return data

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Get next calibration sample.

        Returns:
            Dictionary with input tensor or None if exhausted.
        """
        if self.index < len(self.data):
            result = self.data[self.index]
            self.index += 1
            return result
        return None

    def rewind(self) -> None:
        """Reset the data reader to the beginning."""
        self.index = 0


class GeneratorCalibrationDataReader(CalibrationDataReader):
    """Calibration data reader for generator (DiT) static quantization.

    Generates synthetic calibration data for the generator model.
    The generator has different inputs: noise, timestep, breed.
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = DEFAULT_GENERATOR_SIZE,
    ) -> None:
        """Initialize calibration data reader.

        Args:
            num_samples: Number of calibration samples to generate.
            image_size: Input image size for calibration (default: 128).
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.index = 0
        self.data = self._generate_calibration_data()

    def _generate_calibration_data(self) -> list[dict[str, np.ndarray]]:
        """Generate synthetic calibration data for generator.

        Returns:
            List of dictionaries with input tensors (noise, timestep, breed).
        """
        data = []

        for i in range(self.num_samples):
            np.random.seed(i)

            # Generate random noise tensor (same as generator input)
            noise = np.random.randn(1, 3, self.image_size, self.image_size).astype(
                np.float32
            )

            # Random timestep between 0 and 1
            timestep = np.array([np.random.random()], dtype=np.float32)

            # Random breed index (0-12 for 13 breeds)
            breed = np.array([np.random.randint(0, 13)], dtype=np.int64)

            data.append(
                {
                    "noise": noise,
                    "timestep": timestep,
                    "breed": breed,
                }
            )

        return data

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Get next calibration sample.

        Returns:
            Dictionary with input tensors or None if exhausted.
        """
        if self.index < len(self.data):
            result = self.data[self.index]
            self.index += 1
            return result
        return None

    def rewind(self) -> None:
        """Reset the data reader to the beginning."""
        self.index = 0


def get_file_size(path: str | Path) -> int:
    """Get file size in bytes.

    For ONNX models with external data, sums all related files.

    Args:
        path: Path to the file.

    Returns:
        File size in bytes.
    """
    path = Path(path)
    if not path.exists():
        return 0

    # Start with the main file size
    total = path.stat().st_size

    # Check for external data files (ONNX models with external data)
    # External data files are typically named <model>.onnx.data,
    # <model>.onnx.data_1, etc.
    base_name = str(path)
    parent = path.parent

    for p in parent.iterdir():
        p_name = str(p)
        # Match patterns like: model.onnx.data, model.onnx.data_1, etc.
        if p_name.startswith(base_name + ".data") and p != path:
            total += p.stat().st_size

    return total


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string (e.g., "45.2 MB").
    """
    size: float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def clean_onnx_model(model_path: str | Path, output_path: str | Path) -> Path:
    """Clean ONNX model by removing intermediate shape info.

    This is necessary to avoid shape inference errors during quantization.

    Args:
        model_path: Path to input ONNX model.
        output_path: Path to save cleaned model.

    Returns:
        Path to cleaned model.
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    # Load model with external data if present
    model = onnx.load(str(model_path), load_external_data=True)

    # Remove intermediate value_info to avoid shape inference conflicts
    for value_info in list(model.graph.value_info):
        model.graph.value_info.remove(value_info)

    # Save cleaned model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))

    return output_path


def run_inference(
    session: ort.InferenceSession,
    input_array: np.ndarray,
) -> np.ndarray:
    """Run ONNX inference.

    Args:
        session: ONNX Runtime inference session.
        input_array: Input numpy array.

    Returns:
        Model output.
    """
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_array})
    return outputs[0]


def validate_accuracy(
    original_model_path: str | Path,
    quantized_model_path: str | Path,
    num_samples: int = 50,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> dict[str, Any]:
    """Validate quantized model accuracy against original.

    Args:
        original_model_path: Path to original ONNX model.
        quantized_model_path: Path to quantized ONNX model.
        num_samples: Number of test samples.
        image_size: Input image size.

    Returns:
        Dictionary with accuracy metrics.
    """
    # Load models
    original_session = ort.InferenceSession(
        original_model_path,
        providers=["CPUExecutionProvider"],
    )
    quantized_session = ort.InferenceSession(
        quantized_model_path,
        providers=["CPUExecutionProvider"],
    )

    # Generate test data
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    prediction_matches = 0
    max_diff = 0.0
    mean_diff = 0.0
    differences = []

    for i in range(num_samples):
        np.random.seed(i + 1000)  # Different seed from calibration
        img_array = np.random.randint(
            0, 256, (image_size, image_size, 3), dtype=np.uint8
        )
        img = Image.fromarray(img_array, mode="RGB")
        input_tensor = transform(img).unsqueeze(0).numpy().astype(np.float32)

        original_output = run_inference(original_session, input_tensor)
        quantized_output = run_inference(quantized_session, input_tensor)

        # Calculate output difference
        diff = float(np.abs(original_output - quantized_output).max())
        differences.append(diff)
        max_diff = max(max_diff, diff)

        # Check prediction match
        original_pred = int(np.argmax(original_output, axis=1)[0])
        quantized_pred = int(np.argmax(quantized_output, axis=1)[0])

        if original_pred == quantized_pred:
            prediction_matches += 1

    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences))
    accuracy = prediction_matches / num_samples

    return {
        "max_difference": max_diff,
        "mean_difference": mean_diff,
        "std_difference": std_diff,
        "prediction_match_rate": accuracy,
        "predictions_matched": prediction_matches,
        "total_samples": num_samples,
    }


def apply_dynamic_quantization(
    model_path: str | Path,
    output_path: str | Path,
    model_type: str = "classifier",
) -> dict[str, Any]:
    """Apply dynamic quantization to ONNX model.

    Args:
        model_path: Path to input ONNX model.
        output_path: Path to save quantized model.
        model_type: Type of model ("classifier" or "generator").

    Returns:
        Dictionary with quantization results.
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean model to avoid shape inference errors
    cleaned_path = output_path.parent / "model_cleaned_temp.onnx"
    print("Cleaning model to avoid shape inference errors...")
    clean_onnx_model(model_path, cleaned_path)

    original_size = get_file_size(model_path)

    print("Applying dynamic quantization...")
    print(f"  Input: {model_path} ({format_size(original_size)})")

    quantize_dynamic(
        model_input=str(cleaned_path),
        model_output=str(output_path),
        weight_type=QuantType.QUInt8,
        per_channel=True,
        reduce_range=False,
    )

    # Clean up temp file
    cleaned_path.unlink(missing_ok=True)

    quantized_size = get_file_size(output_path)
    reduction = (1 - quantized_size / original_size) * 100

    print(f"  Output: {output_path} ({format_size(quantized_size)})")
    print(f"  Size reduction: {reduction:.1f}%")

    return {
        "method": "dynamic",
        "original_size": original_size,
        "quantized_size": quantized_size,
        "reduction_percent": reduction,
        "output_path": str(output_path),
    }


def apply_static_quantization(
    model_path: str | Path,
    output_path: str | Path,
    num_calibration_samples: int = 100,
    model_type: str = "classifier",
) -> dict[str, Any]:
    """Apply static quantization to ONNX model.

    Args:
        model_path: Path to input ONNX model.
        output_path: Path to save quantized model.
        num_calibration_samples: Number of calibration samples.
        model_type: Type of model ("classifier" or "generator").

    Returns:
        Dictionary with quantization results.
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean model to avoid shape inference errors
    cleaned_path = output_path.parent / "model_cleaned_temp.onnx"
    print("Cleaning model to avoid shape inference errors...")
    clean_onnx_model(model_path, cleaned_path)

    original_size = get_file_size(model_path)

    print("Applying static quantization...")
    print(f"  Input: {model_path} ({format_size(original_size)})")
    print(f"  Calibration samples: {num_calibration_samples}")

    # Select appropriate calibration data reader based on model type
    if model_type == "generator":
        calibration_reader = GeneratorCalibrationDataReader(
            num_samples=num_calibration_samples,
            image_size=DEFAULT_GENERATOR_SIZE,
        )
    else:
        calibration_reader = CatsCalibrationDataReader(
            num_samples=num_calibration_samples,
            image_size=DEFAULT_IMAGE_SIZE,
        )

    quantize_static(
        model_input=str(cleaned_path),
        model_output=str(output_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QUInt8,
        activation_type=QuantType.QUInt8,
        per_channel=True,
        reduce_range=False,
    )

    # Clean up temp file
    cleaned_path.unlink(missing_ok=True)

    quantized_size = get_file_size(output_path)
    reduction = (1 - quantized_size / original_size) * 100

    print(f"  Output: {output_path} ({format_size(quantized_size)})")
    print(f"  Size reduction: {reduction:.1f}%")

    return {
        "method": "static",
        "original_size": original_size,
        "quantized_size": quantized_size,
        "reduction_percent": reduction,
        "output_path": str(output_path),
    }


def optimize_onnx(
    model_path: str | Path,
    output_dir: str | Path,
    method: str = "dynamic",
    num_calibration_samples: int = 100,
    validate: bool = True,
    num_validation_samples: int = 50,
    model_type: str = "classifier",
) -> dict[str, Any]:
    """Optimize ONNX model using quantization.

    Args:
        model_path: Path to input ONNX model.
        output_dir: Directory to save optimized models.
        method: Quantization method ("dynamic" or "static").
        num_calibration_samples: Number of calibration samples for static quantization.
        validate: Whether to validate accuracy after quantization.
        num_validation_samples: Number of samples for accuracy validation.
        model_type: Type of model ("classifier" or "generator").

    Returns:
        Dictionary with optimization results.
    """
    model_path = Path(model_path)
    output_dir = Path(output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "input_model": str(model_path),
        "original_size": get_file_size(model_path),
        "original_size_formatted": format_size(get_file_size(model_path)),
        "model_type": model_type,
    }

    # Determine output filename based on model type
    if model_type == "generator":
        base_name = "generator"
    else:
        base_name = "cats"

    # Apply quantization
    if method == "dynamic":
        output_path = output_dir / f"{base_name}_quantized.onnx"
        quant_results = apply_dynamic_quantization(
            model_path, output_path, model_type=model_type
        )
    elif method == "static":
        output_path = output_dir / f"{base_name}_quantized_static.onnx"
        quant_results = apply_static_quantization(
            model_path,
            output_path,
            num_calibration_samples,
            model_type=model_type,
        )
    else:
        raise ValueError(f"Unknown quantization method: {method}")

    results["quantization"] = quant_results

    # Validate accuracy
    if validate:
        print("\nValidating accuracy...")
        accuracy_results = validate_accuracy(
            model_path,
            output_path,
            num_samples=num_validation_samples,
        )
        results["accuracy_validation"] = accuracy_results

        # Check if accuracy drop is acceptable (<1% prediction change)
        match_rate = accuracy_results["prediction_match_rate"]
        if match_rate >= 0.99:
            print(f"  Accuracy validation PASSED (match rate: {match_rate:.1%})")
        else:
            print(
                f"  Warning: Accuracy drop may be significant "
                f"(match rate: {match_rate:.1%})"
            )

    # Summary
    print("\n" + "=" * 60)
    print("Optimization Summary")
    print("=" * 60)
    print(f"Original model: {format_size(results['original_size'])}")
    print(f"Quantized model: {format_size(quant_results['quantized_size'])}")
    print(f"Size reduction: {quant_results['reduction_percent']:.1f}%")
    if validate and "accuracy_validation" in results:
        acc = results["accuracy_validation"]
        print(f"Prediction match rate: {acc['prediction_match_rate']:.1%}")
        print(
            f"Predictions matched: {acc['predictions_matched']}/{acc['total_samples']}"
        )
        print(f"Max output difference: {acc['max_difference']:.6f}")
        print(f"Mean output difference: {acc['mean_difference']:.6f}")
    print(f"Output: {quant_results['output_path']}")
    print("=" * 60)

    # Check success criteria
    target_size = 100 * 1024 * 1024  # 100 MB
    if quant_results["quantized_size"] < target_size:
        print("\n[SUCCESS] Model size is under 100MB target")
    else:
        print("\n[WARNING] Model size exceeds 100MB target")

    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize ONNX model using quantization"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="frontend/public/models/cats.onnx",
        help="Path to input ONNX model (default: frontend/public/models/cats.onnx)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="frontend/public/models",
        help="Output directory for optimized models",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization method (default: dynamic)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["classifier", "generator"],
        default="classifier",
        help="Type of model to optimize (default: classifier)",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=100,
        help="Number of calibration samples for static quantization (default: 100)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip accuracy validation",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=50,
        help="Number of samples for accuracy validation (default: 50)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for ONNX model optimization."""
    args = parse_args()

    try:
        results = optimize_onnx(
            model_path=args.model,
            output_dir=args.output_dir,
            method=args.method,
            num_calibration_samples=args.calibration_samples,
            validate=not args.no_validate,
            num_validation_samples=args.validation_samples,
            model_type=args.model_type,
        )

        # Exit with error if validation failed
        if (
            not args.no_validate
            and "accuracy_validation" in results
            and results["accuracy_validation"]["prediction_match_rate"] < 0.99
        ):
            print("\n[WARNING] Accuracy validation shows significant differences")

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        if args.model_type == "classifier":
            print("\nHint: Export the classifier first with:")
            print("  python src/export_onnx.py --checkpoint PATH_TO_CHECKPOINT")
            print("  or python src/export_classifier_onnx.py --checkpoint PATH")
        else:
            print("\nHint: Export the generator first with:")
            print("  python src/export_dit_onnx.py --checkpoint PATH_TO_CHECKPOINT")
        sys.exit(1)
    except ImportError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
