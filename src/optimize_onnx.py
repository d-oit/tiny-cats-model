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

Generator validation: when ``--model-type generator`` is selected, the
post-quantization accuracy check feeds the required
``{noise, timestep, breed}`` inputs (the same shape contract used by
``GeneratorCalibrationDataReader``); classifiers continue to feed
``{input}``. The classification-only ``prediction_match_rate`` metric
is reported as ``None`` for generator runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
import torchvision.transforms as T
from PIL import Image, ImageOps
from torchvision.transforms import InterpolationMode

from dataset_manifest import build_dataset_manifest
from model_contract import (
    CLASSIFIER_FOLDER_TO_INDEX,
    CLASSIFIER_IMAGE_SIZE,
    CLASSIFIER_RESIZE_SHORT_SIDE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PREPROCESS_VERSION,
)
from splits import read_split_manifest

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

DEFAULT_IMAGE_SIZE = CLASSIFIER_IMAGE_SIZE
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
        input_name: str = "input",
    ) -> None:
        """Initialize synthetic smoke-test calibration."""
        self.num_samples = num_samples
        self.image_size = image_size
        self.input_name = input_name
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

            data.append({self.input_name: tensor.astype(np.float32)})

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


class QuantizationQualityError(RuntimeError):
    """Raised when required classifier quantization parity is not met."""


class RealImageCalibrationDataReader(CalibrationDataReader):
    """Read deterministically selected, hash-verified real dataset images."""

    def __init__(
        self,
        data_dir: str | Path,
        input_name: str = "input",
        num_samples: int = 100,
        seed: int = 42,
        image_size: int = CLASSIFIER_IMAGE_SIZE,
        split: str = "train",
        split_manifest: str | Path | None = None,
        stratify_by: str = "label",
    ) -> None:
        self.root = Path(data_dir)
        self.input_name = input_name
        self.seed = int(seed)
        self.image_size = int(image_size)
        self.split = split
        self.index = 0
        self.dataset = build_dataset_manifest(self.root, persist_cache=False)
        if self.dataset.get("status") == "missing":
            raise FileNotFoundError(f"Calibration dataset not found: {self.root}")
        entries = self.dataset["entries"]
        self.split_hash: str | None = None
        if split_manifest is not None:
            manifest = read_split_manifest(split_manifest)
            if manifest.get("dataset_hash") != self.dataset.get("dataset_hash"):
                raise ValueError(
                    "Split manifest dataset hash does not match current data"
                )
            mapped = {entry["path"]: entry for entry in manifest["entries"]}
            if set(mapped) != {entry["path"] for entry in entries}:
                raise ValueError("Split manifest entries do not match current dataset")
            for entry in entries:
                split_entry = mapped[entry["path"]]
                if split_entry.get("sha256") not in (None, entry.get("sha256")):
                    raise ValueError(f"Dataset hash mismatch for {entry['path']}")
                entry["split"] = split_entry["split"]
            self.split_hash = manifest["split_hash"]
        else:
            # Stable fallback partition: train=0..79, val=80..89, test=90..99.
            for entry in entries:
                bucket = (
                    int(
                        hashlib.sha256(
                            f"{self.seed}:{entry['path']}".encode()
                        ).hexdigest()[:8],
                        16,
                    )
                    % 100
                )
                entry["split"] = (
                    "train" if bucket < 80 else "val" if bucket < 90 else "test"
                )
            split_identity = [
                {"path": entry["path"], "split": entry["split"]} for entry in entries
            ]
            split_material = json.dumps(
                split_identity, sort_keys=True, separators=(",", ":")
            )
            self.split_hash = hashlib.sha256(
                (
                    f"sha256(seed:path)%100;seed={self.seed};"
                    f"{self.dataset['dataset_hash']};{split_material}"
                ).encode()
            ).hexdigest()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown dataset split: {split!r}")
        available = [entry for entry in entries if entry.get("split") == split]
        if not available:
            raise ValueError(f"No images assigned to calibration split {split!r}")
        if stratify_by not in {"label", "breed"}:
            raise ValueError(f"Unknown stratification field: {stratify_by!r}")
        groups: dict[str, list[dict[str, Any]]] = {}
        for entry in available:
            label = str(entry.get("label", "__unlabeled__"))
            breed = entry.get("breed")
            group_name = f"{label}/{breed}" if breed else label
            groups.setdefault(group_name, []).append(entry)
        rng = random.Random(self.seed)
        for group_entries in groups.values():
            group_entries.sort(key=lambda item: item["path"])
            rng.shuffle(group_entries)
        selected: list[dict[str, Any]] = []
        group_names = sorted(groups)
        target_count = min(num_samples, len(available))
        while len(selected) < target_count:
            changed = False
            for group_name in group_names:
                if groups[group_name]:
                    selected.append(groups[group_name].pop())
                    changed = True
                    if len(selected) == target_count:
                        break
            if not changed:
                break
        self.entries = selected
        self.transform = T.Compose(
            [
                T.Resize(
                    CLASSIFIER_RESIZE_SHORT_SIDE,
                    interpolation=InterpolationMode.BILINEAR,
                ),
                T.CenterCrop(self.image_size),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.data: list[dict[str, np.ndarray]] = []
        self.class_distribution: Counter[str] = Counter()
        self.breed_distribution: Counter[str] = Counter()
        for entry in selected:
            path = self.root / entry["path"]
            if not path.is_file():
                raise FileNotFoundError(f"Dataset image missing: {path}")
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source)
                if "A" in image.getbands():
                    rgba = image.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
                    image = Image.alpha_composite(background, rgba).convert("RGB")
                else:
                    image = image.convert("RGB")
                tensor = self.transform(image).unsqueeze(0).numpy()
            self.data.append({self.input_name: tensor.astype(np.float32)})
            self.class_distribution[str(entry.get("label", "unknown"))] += 1
            if entry.get("breed"):
                self.breed_distribution[str(entry["breed"])] += 1
        fresh_hash = build_dataset_manifest(self.root, persist_cache=False)[
            "dataset_hash"
        ]
        if fresh_hash != self.dataset["dataset_hash"]:
            raise ValueError("Dataset changed while creating calibration reader")
        self.provenance = {
            "dataset_hash": self.dataset.get("dataset_hash"),
            "split_hash": self.split_hash,
            "image_count": len(selected),
            "class_distribution": dict(sorted(self.class_distribution.items())),
            "breed_distribution": dict(sorted(self.breed_distribution.items())),
            "transform_version": PREPROCESS_VERSION,
            "seed": self.seed,
            "split": split,
            "split_rule": (
                "manifest"
                if split_manifest is not None
                else "sha256(seed:path)%100; train<80, val=80..89, test>=90"
            ),
            "selected_paths": [entry["path"] for entry in selected],
        }

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.data):
            return None
        result = self.data[self.index]
        self.index += 1
        return result

    def rewind(self) -> None:
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

        return data  # type: ignore[return-value]

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


def _build_generator_inputs(
    seed: int,
    image_size: int = DEFAULT_GENERATOR_SIZE,
) -> dict[str, np.ndarray]:
    """Build a generator feed dict ``{noise, timestep, breed}``.

    Mirrors the per-sample shape/dtype contract of
    ``GeneratorCalibrationDataReader`` so static-quantization calibration
    inputs and accuracy-validation inputs are interchangeable:

        noise:    (1, 3, image_size, image_size) float32
        timestep: (1,)                          float32 in [0, 1)
        breed:    (1,)                          int64 in [0, 13)

    Args:
        seed: Seed for ``np.random`` (lets callers pin generation).
        image_size: Spatial size of the noise tensor.

    Returns:
        Feed dict compatible with the exported generator ONNX model.
    """
    np.random.seed(seed)
    noise = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    timestep = np.array([np.random.random()], dtype=np.float32)
    breed = np.array([np.random.randint(0, 13)], dtype=np.int64)
    return {"noise": noise, "timestep": timestep, "breed": breed}


def run_inference(
    session: ort.InferenceSession,
    feeds: dict[str, np.ndarray],
) -> np.ndarray:
    """Run ONNX inference with an explicit feed dict.

    Args:
        session: ONNX Runtime inference session.
        feeds: Mapping from input name to numpy array. Must contain every
            entry the model expects (classifier: ``{"input": ...}``;
            generator: ``{"noise", "timestep", "breed"}``).

    Returns:
        First model output.
    """
    outputs = session.run(None, feeds)
    return outputs[0]


def validate_accuracy(
    original_model_path: str | Path,
    quantized_model_path: str | Path,
    num_samples: int = 50,
    image_size: int = DEFAULT_IMAGE_SIZE,
    model_type: str = "classifier",
    validation_data_dir: str | Path | None = None,
    validation_split_manifest: str | Path | None = None,
    validation_split: str = "val",
    validation_seed: int = 42,
    labels: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Compare outputs, optionally measuring classifier parity on real images."""
    if model_type not in {"classifier", "generator"}:
        raise ValueError(
            f"Unknown model_type: {model_type!r}; expected 'classifier' or 'generator'"
        )
    original_session = ort.InferenceSession(
        str(original_model_path), providers=["CPUExecutionProvider"]
    )
    quantized_session = ort.InferenceSession(
        str(quantized_model_path), providers=["CPUExecutionProvider"]
    )
    input_name = original_session.get_inputs()[0].name
    real_reader = None
    if model_type == "classifier" and validation_data_dir is not None:
        real_reader = RealImageCalibrationDataReader(
            validation_data_dir,
            input_name=input_name,
            num_samples=num_samples,
            seed=validation_seed,
            image_size=image_size,
            split=validation_split,
            split_manifest=validation_split_manifest,
        )
    label_names = list(labels or ("cat", "not_cat"))
    differences: list[float] = []
    match = 0
    original_correct = 0
    quantized_correct = 0
    class_counts: Counter[int] = Counter()
    class_true_positive: Counter[int] = Counter()
    class_predicted: Counter[int] = Counter()
    class_correct: Counter[int] = Counter()
    logits_max_error = 0.0
    logits_errors: list[float] = []
    count = len(real_reader.data) if real_reader else num_samples
    if count == 0:
        raise ValueError("Validation set contains no images")
    for i in range(count):
        true_label = None
        if model_type == "generator":
            feeds = _build_generator_inputs(seed=i + 1000, image_size=image_size)
        elif real_reader:
            feeds = real_reader.data[i]
            entry = real_reader.entries[i]
            true_label = entry.get("label_index")
            if true_label is None:
                true_label = CLASSIFIER_FOLDER_TO_INDEX.get(str(entry.get("label")))
            if true_label is None:
                raise ValueError(f"Unknown classifier label in {entry['path']}")
        else:
            np.random.seed(i + 1000)
            transform = T.Compose(
                [T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
            )
            img_array = np.random.randint(
                0, 256, (image_size, image_size, 3), dtype=np.uint8
            )
            input_tensor = transform(Image.fromarray(img_array, mode="RGB"))
            feeds = {input_name: input_tensor.unsqueeze(0).numpy().astype(np.float32)}
        original_output = run_inference(original_session, feeds)
        quantized_output = run_inference(quantized_session, feeds)
        error = np.abs(original_output - quantized_output)
        differences.append(float(error.max()))
        logits_errors.extend(error.astype(np.float64).ravel().tolist())
        logits_max_error = max(logits_max_error, float(error.max()))
        if model_type == "classifier":
            original_pred = int(np.argmax(original_output, axis=1)[0])
            quantized_pred = int(np.argmax(quantized_output, axis=1)[0])
            match += int(original_pred == quantized_pred)
            if true_label is not None:
                class_counts[true_label] += 1
                original_correct += int(original_pred == true_label)
                quantized_correct += int(quantized_pred == true_label)
                class_predicted[quantized_pred] += 1
                class_correct[quantized_pred] += int(quantized_pred == true_label)
                class_true_positive[true_label] += int(quantized_pred == true_label)
    metrics: dict[str, Any] = {
        "model_type": model_type,
        "max_difference": max(differences),
        "mean_difference": float(np.mean(differences)),
        "std_difference": float(np.std(differences)),
        "total_samples": count,
        "prediction_match_rate": match / count if model_type == "classifier" else None,
        "predictions_matched": match if model_type == "classifier" else None,
        "real_images": real_reader is not None,
        "provenance": real_reader.provenance if real_reader else None,
        "logit_error": (
            {"max": logits_max_error, "mean": float(np.mean(logits_errors))}
            if model_type == "classifier"
            else None
        ),
    }
    if real_reader is not None:
        metrics["accuracy"] = {
            "original": original_correct / count,
            "quantized": quantized_correct / count,
            "delta": (quantized_correct - original_correct) / count,
        }
        per_class = {}
        for index, name in enumerate(label_names):
            tp = class_true_positive[index]
            precision = tp / class_predicted[index] if class_predicted[index] else 0.0
            recall = tp / class_counts[index] if class_counts[index] else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            per_class[name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": class_counts[index],
            }
        metrics["per_class_metrics"] = per_class
    return metrics


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
    calibration_data_dir: str | Path | None = None,
    calibration_split: str = "train",
    calibration_seed: int = 42,
    calibration_split_manifest: str | Path | None = None,
    synthetic_calibration: bool = False,
    onnx_input_name: str = "input",
) -> dict[str, Any]:
    """Apply static quantization using real classifier images by default."""
    if model_type not in {"classifier", "generator"}:
        raise ValueError(f"Unknown model_type: {model_type!r}")
    if model_type == "classifier" and calibration_split != "train":
        raise ValueError("Classifier calibration must use the train split")
    if (
        model_type == "classifier"
        and not calibration_data_dir
        and not synthetic_calibration
    ):
        raise ValueError(
            "Static classifier quantization requires --calibration-data-dir; "
            "use --synthetic-calibration only for smoke tests"
        )
    model_path = Path(model_path)
    output_path = Path(output_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_path = output_path.parent / "model_cleaned_temp.onnx"
    clean_onnx_model(model_path, cleaned_path)
    original_size = get_file_size(model_path)
    if model_type == "generator":
        calibration_reader = GeneratorCalibrationDataReader(
            num_samples=num_calibration_samples, image_size=DEFAULT_GENERATOR_SIZE
        )
        provenance = {
            "source": "synthetic_generator",
            "image_count": num_calibration_samples,
            "seed": 0,
        }
    elif calibration_data_dir:
        calibration_reader = RealImageCalibrationDataReader(
            calibration_data_dir,
            input_name=onnx_input_name,
            num_samples=num_calibration_samples,
            seed=calibration_seed,
            image_size=DEFAULT_IMAGE_SIZE,
            split=calibration_split,
            split_manifest=calibration_split_manifest,
        )
        provenance = calibration_reader.provenance
    else:
        print(
            "WARNING: using synthetic classifier calibration data; "
            "this is smoke-test-only and not suitable for release."
        )
        calibration_reader = CatsCalibrationDataReader(
            num_samples=num_calibration_samples,
            image_size=DEFAULT_IMAGE_SIZE,
            input_name=onnx_input_name,
        )
        provenance = {
            "source": "synthetic_classifier",
            "image_count": num_calibration_samples,
            "seed": None,
        }
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
    cleaned_path.unlink(missing_ok=True)
    quantized_size = get_file_size(output_path)
    reduction = (1 - quantized_size / original_size) * 100
    return {
        "method": "static",
        "original_size": original_size,
        "quantized_size": quantized_size,
        "reduction_percent": reduction,
        "output_path": str(output_path),
        "calibration_provenance": provenance,
    }


def optimize_onnx(
    model_path: str | Path,
    output_dir: str | Path,
    method: str = "dynamic",
    num_calibration_samples: int = 100,
    validate: bool = True,
    num_validation_samples: int = 50,
    model_type: str = "classifier",
    calibration_data_dir: str | Path | None = None,
    calibration_split: str = "train",
    calibration_seed: int = 42,
    calibration_split_manifest: str | Path | None = None,
    synthetic_calibration: bool = False,
    onnx_input_name: str = "input",
    validation_data_dir: str | Path | None = None,
    validation_split_manifest: str | Path | None = None,
    validation_split: str = "val",
    validation_seed: int = 42,
    min_agreement: float = 0.99,
    max_accuracy_drop: float = 0.02,
    require_parity: bool = False,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Optimize a model and optionally enforce a real-image release parity gate."""
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not 0 <= min_agreement <= 1 or not 0 <= max_accuracy_drop <= 1:
        raise ValueError("Parity thresholds must be in [0, 1]")
    if require_parity and model_type == "classifier" and not validation_data_dir:
        raise ValueError("A real --validation-data-dir is required by --require-parity")
    if (
        calibration_split_manifest is not None
        and validation_split_manifest is not None
        and Path(calibration_split_manifest).resolve()
        == Path(validation_split_manifest).resolve()
        and calibration_split == validation_split
    ):
        raise ValueError("Calibration and validation splits must be disjoint")
    if validation_data_dir is not None and validation_split == "train":
        raise ValueError("Real validation must use a held-out val or test split")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = "generator" if model_type == "generator" else "cats"
    results: dict[str, Any] = {
        "input_model": str(model_path),
        "original_size": get_file_size(model_path),
        "original_size_formatted": format_size(get_file_size(model_path)),
        "model_type": model_type,
        "method": method,
        "calibration_provenance": None,
        "gate": {"required": require_parity, "passed": None},
    }
    if method == "dynamic":
        output_path = output_dir / f"{base_name}_quantized.onnx"
        quant_results = apply_dynamic_quantization(
            model_path, output_path, model_type=model_type
        )
        results["calibration_provenance"] = {
            "source": "not_required_weight_only",
            "image_count": 0,
        }
    elif method == "static":
        output_path = output_dir / f"{base_name}_quantized_static.onnx"
        onnx_model = onnx.load(str(model_path))
        actual_input = (
            onnx_model.graph.input[0].name
            if onnx_model.graph.input
            else onnx_input_name
        )
        if (
            model_type == "classifier"
            and calibration_data_dir is not None
            and validation_data_dir is not None
        ):
            calibration_preflight = RealImageCalibrationDataReader(
                calibration_data_dir,
                input_name=actual_input,
                num_samples=num_calibration_samples,
                seed=calibration_seed,
                split=calibration_split,
                split_manifest=calibration_split_manifest,
            )
            validation_preflight = RealImageCalibrationDataReader(
                validation_data_dir,
                input_name=actual_input,
                num_samples=num_validation_samples,
                seed=validation_seed,
                split=validation_split,
                split_manifest=validation_split_manifest,
            )
            calibration_paths = cast(
                list[str], calibration_preflight.provenance["selected_paths"]
            )
            validation_paths = cast(
                list[str], validation_preflight.provenance["selected_paths"]
            )
            if set(calibration_paths) & set(validation_paths):
                raise ValueError("Calibration and validation images must be disjoint")
        quant_results = apply_static_quantization(
            model_path,
            output_path,
            num_calibration_samples,
            model_type=model_type,
            calibration_data_dir=calibration_data_dir,
            calibration_split=calibration_split,
            calibration_seed=calibration_seed,
            calibration_split_manifest=calibration_split_manifest,
            synthetic_calibration=synthetic_calibration,
            onnx_input_name=actual_input,
        )
        results["calibration_provenance"] = quant_results.get("calibration_provenance")
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    results["quantization"] = quant_results
    should_validate = validate or require_parity
    if should_validate:
        accuracy_results = validate_accuracy(
            model_path,
            output_path,
            num_samples=num_validation_samples,
            image_size=(
                DEFAULT_GENERATOR_SIZE
                if model_type == "generator"
                else DEFAULT_IMAGE_SIZE
            ),
            model_type=model_type,
            validation_data_dir=validation_data_dir,
            validation_split_manifest=validation_split_manifest,
            validation_split=validation_split,
            validation_seed=validation_seed,
        )
        results["accuracy_validation"] = accuracy_results
        calibration_provenance = results.get("calibration_provenance")
        validation_provenance = accuracy_results.get("provenance")
        if not isinstance(calibration_provenance, dict) or not isinstance(
            validation_provenance, dict
        ):
            raise ValueError("Calibration and validation provenance are required")
        calibration_selected_paths = calibration_provenance.get("selected_paths")
        validation_selected_paths = validation_provenance.get("selected_paths")
        if (
            not isinstance(calibration_selected_paths, list)
            or not all(isinstance(path, str) for path in calibration_selected_paths)
            or not isinstance(validation_selected_paths, list)
            or not all(isinstance(path, str) for path in validation_selected_paths)
        ):
            raise ValueError("Calibration and validation paths must be string lists")
        overlap = set(calibration_selected_paths) & set(validation_selected_paths)
        if overlap:
            raise ValueError(
                f"Calibration and validation datasets overlap ({len(overlap)} images)"
            )
        if model_type == "classifier":
            agreement = accuracy_results.get("prediction_match_rate")
            accuracy = accuracy_results.get("accuracy", {})
            drop = max(
                0.0,
                float(accuracy.get("original", 0.0))
                - float(accuracy.get("quantized", 0.0)),
            )
            has_true_accuracy = not accuracy_results.get("real_images") or (
                "original" in accuracy and "quantized" in accuracy
            )
            parity_passed = (
                agreement is not None
                and agreement >= min_agreement
                and has_true_accuracy
                and (
                    not accuracy_results.get("real_images") or drop <= max_accuracy_drop
                )
                and (not require_parity or accuracy_results.get("real_images") is True)
            )
            results["gate"] = {
                "required": require_parity,
                "passed": bool(parity_passed),
                "min_agreement": min_agreement,
                "actual_agreement": agreement,
                "max_accuracy_drop": max_accuracy_drop,
                "actual_accuracy_drop": drop
                if accuracy_results.get("real_images")
                else None,
                "real_images": bool(accuracy_results.get("real_images")),
            }
    report = (
        Path(report_path)
        if report_path
        else output_dir / f"{base_name}_quantization_report.json"
    )
    results["report_path"] = str(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    if require_parity and model_type == "classifier":
        gate = results["gate"]
        if not gate.get("passed"):
            raise QuantizationQualityError(
                f"Classifier parity gate failed: agreement={gate.get('actual_agreement')}, "
                f"accuracy_drop={gate.get('actual_accuracy_drop')}"
            )
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
    parser.add_argument("--calibration-data-dir")
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--calibration-split-manifest")
    parser.add_argument("--synthetic-calibration", action="store_true")
    parser.add_argument("--validation-data-dir")
    parser.add_argument("--validation-split-manifest")
    parser.add_argument("--validation-split", default="val")
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--min-agreement", type=float, default=0.99)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.02)
    parser.add_argument("--require-parity", action="store_true")
    parser.add_argument(
        "--report",
        nargs="?",
        const="",
        default=None,
        help="Write quantization report (optionally provide output JSON path)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for ONNX model optimization."""
    args = parse_args()

    try:
        optimize_onnx(
            model_path=args.model,
            output_dir=args.output_dir,
            method=args.method,
            num_calibration_samples=args.calibration_samples,
            validate=not args.no_validate,
            num_validation_samples=args.validation_samples,
            model_type=args.model_type,
            calibration_data_dir=args.calibration_data_dir,
            calibration_split=args.calibration_split,
            calibration_seed=args.calibration_seed,
            calibration_split_manifest=args.calibration_split_manifest,
            synthetic_calibration=args.synthetic_calibration,
            validation_data_dir=args.validation_data_dir,
            validation_split_manifest=args.validation_split_manifest,
            validation_split=args.validation_split,
            validation_seed=args.validation_seed,
            min_agreement=args.min_agreement,
            max_accuracy_drop=args.max_accuracy_drop,
            require_parity=args.require_parity,
            report_path=args.report or None,
        )

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
