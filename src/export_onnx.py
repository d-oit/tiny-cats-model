"""src/export_onnx.py

Export the trained cats classifier to ONNX format for web inference.

Usage:
    python src/export_onnx.py
    python src/export_onnx.py --checkpoint PATH --output PATH
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import cats_model
from model_contract import (
    CLASSIFIER_LABELS,
    ContractError,
    classifier_metadata,
    metadata_from_checkpoint,
    validate_classifier_metadata,
)
from safe_checkpoint import load_torch_checkpoint, sha256_file


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
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    state = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        state_dict = state.get(
            "model_state_dict", state.get("model", state.get("state_dict", state))
        )
    else:
        raise ContractError("classifier checkpoint must be a state dictionary")
    if not isinstance(state_dict, dict):
        raise ContractError("classifier checkpoint model weights must be a dictionary")
    metadata = metadata_from_checkpoint(state)
    inferred_classes = None
    for key in ("fc.1.weight", "classifier.3.1.weight", "classifier.1.weight"):
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            inferred_classes = int(tensor.shape[0])
            break
    config = state.get("config", {})
    config = config if isinstance(config, dict) else {}
    architecture = metadata.get("architecture") if metadata is not None else None
    if isinstance(architecture, dict):
        stored_classes = architecture.get("num_classes")
    else:
        stored_classes = config.get("num_classes")
    if num_classes is None:
        num_classes = int(stored_classes or inferred_classes or 0)
    if (
        not num_classes
        or num_classes != inferred_classes
        or num_classes != len(CLASSIFIER_LABELS)
    ):
        raise ContractError(
            f"classifier checkpoint has {inferred_classes} classes; "
            f"expected {len(CLASSIFIER_LABELS)}"
        )
    if stored_classes is not None and int(stored_classes) != num_classes:
        raise ContractError(
            "checkpoint metadata/config num_classes disagrees with tensors"
        )
    if metadata:
        validate_classifier_metadata(
            metadata,
            expected_num_classes=num_classes,
            expected_backbone=backbone,
            expected_labels=CLASSIFIER_LABELS,
        )

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
    import onnx

    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)
    graph_inputs = {value.name: value for value in graph.graph.input}
    graph_outputs = {value.name: value for value in graph.graph.output}
    if len(graph_inputs) != 1 or len(graph_outputs) != 1:
        raise RuntimeError(
            "exported classifier graph must have one input and one output"
        )
    input_name = next(iter(graph_inputs))
    output_name = next(iter(graph_outputs))
    input_shape = graph_inputs[input_name].type.tensor_type.shape.dim
    output_shape = graph_outputs[output_name].type.tensor_type.shape.dim
    if (
        graph_inputs[input_name].type.tensor_type.elem_type != 1
        or graph_outputs[output_name].type.tensor_type.elem_type != 1
    ):
        raise RuntimeError("exported classifier graph tensors must be float32")
    if (
        len(input_shape) != 4
        or [dim.dim_value for dim in input_shape[1:]] != [3, 224, 224]
        or len(output_shape) != 2
        or output_shape[1].dim_value != num_classes
    ):
        raise RuntimeError("exported classifier graph dimensions violate contract")
    contract = classifier_metadata(
        backbone=backbone,
        num_classes=num_classes,
        opset=opset_version,
        model_sha256=sha256_file(output_path),
    )
    contract["input"]["name"] = input_name
    contract["output"]["name"] = output_name
    Path(str(output_path) + ".json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
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
