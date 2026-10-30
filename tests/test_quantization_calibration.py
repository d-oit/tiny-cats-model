"""Real-image quantization calibration and parity gate coverage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
pytest.importorskip("onnxruntime")

import optimize_onnx as optimizer
from splits import build_split_manifest, write_split_manifest


def _dataset(root: Path, count: int = 20) -> Path:
    for label in ("cat", "other"):
        directory = root / label
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            array = np.full((270, 300, 3), (index * 7) % 255, dtype=np.uint8)
            name = f"Abyssinian_{index}.png" if label == "cat" else f"dog_{index}.png"
            Image.fromarray(array).save(directory / name)
    return root


def _manifest(root: Path) -> Path:
    manifest = build_split_manifest(
        root, seed=19, val_split=0.25, test_split=0.15, stratify_by="label"
    )
    return write_split_manifest(root, manifest)


def test_real_reader_is_deterministic_and_emits_canonical_tensor(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "dataset")
    manifest_path = _manifest(root)
    first = optimizer.RealImageCalibrationDataReader(
        root, num_samples=8, seed=11, split_manifest=manifest_path
    )
    second = optimizer.RealImageCalibrationDataReader(
        root, num_samples=8, seed=11, split_manifest=manifest_path
    )
    assert first.provenance["selected_paths"] == second.provenance["selected_paths"]
    assert first.provenance["class_distribution"] == {"cat": 4, "other": 4}
    sample = first.get_next()
    assert sample is not None
    tensor = sample["input"]
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    assert -3.0 <= float(tensor.min()) <= float(tensor.max()) <= 3.0
    assert first.provenance["dataset_hash"]
    assert first.provenance["split_hash"]


def test_calibration_and_validation_manifest_splits_are_disjoint(
    tmp_path: Path,
) -> None:
    root = _dataset(tmp_path / "dataset")
    manifest_path = _manifest(root)
    calibration = optimizer.RealImageCalibrationDataReader(
        root, num_samples=100, split="train", split_manifest=manifest_path
    )
    validation = optimizer.RealImageCalibrationDataReader(
        root, num_samples=100, split="val", split_manifest=manifest_path
    )
    calibration_paths = calibration.provenance["selected_paths"]
    validation_paths = validation.provenance["selected_paths"]
    assert isinstance(calibration_paths, list)
    assert isinstance(validation_paths, list)
    assert set(calibration_paths).isdisjoint(validation_paths)


def test_static_classifier_requires_real_data_or_explicit_synthetic(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires --calibration-data-dir"):
        optimizer.apply_static_quantization("missing.onnx", tmp_path / "out.onnx")
    with pytest.raises(FileNotFoundError):
        optimizer.apply_static_quantization(
            "missing.onnx", tmp_path / "out.onnx", synthetic_calibration=True
        )


def _fake_dynamic(
    model_path: Path, output_path: Path, **_: object
) -> dict[str, object]:
    output_path.write_bytes(b"fake quantized")
    return {
        "method": "dynamic",
        "quantized_size": output_path.stat().st_size,
        "reduction_percent": 10.0,
        "output_path": str(output_path),
    }


def _fake_validation(*_: object, **__: object) -> dict[str, object]:
    return {
        "model_type": "classifier",
        "prediction_match_rate": 0.95,
        "real_images": True,
        "accuracy": {"original": 0.8, "quantized": 0.75, "delta": -0.05},
        "provenance": {"dataset_hash": "abc", "image_count": 8},
    }


def test_optimize_writes_report_and_raises_when_required_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake model")
    monkeypatch.setattr(optimizer, "apply_dynamic_quantization", _fake_dynamic)
    monkeypatch.setattr(optimizer, "validate_accuracy", _fake_validation)
    report = tmp_path / "report.json"
    with pytest.raises(optimizer.QuantizationQualityError):
        optimizer.optimize_onnx(
            model,
            tmp_path / "output",
            require_parity=True,
            validation_data_dir=tmp_path / "dataset",
            report_path=report,
        )
    document = json.loads(report.read_text())
    assert document["accuracy_validation"]["real_images"] is True
    assert document["gate"]["passed"] is False
    assert document["calibration_provenance"]["source"] == "not_required_weight_only"


def test_optimize_passes_parity_gate_with_real_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake model")
    monkeypatch.setattr(optimizer, "apply_dynamic_quantization", _fake_dynamic)
    monkeypatch.setattr(
        optimizer,
        "validate_accuracy",
        lambda *args, **kwargs: {
            **_fake_validation(),
            "prediction_match_rate": 0.995,
            "accuracy": {"original": 0.8, "quantized": 0.79, "delta": -0.01},
        },
    )
    result = optimizer.optimize_onnx(
        model,
        tmp_path / "output",
        require_parity=True,
        validation_data_dir=tmp_path / "dataset",
    )
    assert result["gate"]["passed"] is True
    assert Path(result["report_path"]).is_file()
