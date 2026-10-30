"""src/verify_checkpoint.py

Verify a trained checkpoint is valid and can be used for inference.

Usage:
    python src/verify_checkpoint.py --checkpoint checkpoints/pool/dit_model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import torch

from dit import TinyDiT, count_parameters, load_state_dict_checked, state_dict_diff
from model_contract import GENERATOR_CONDITIONS
from safe_checkpoint import load_torch_checkpoint


def _checkpoint_state(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dictionary")
    for key in ("model_state_dict", "model", "ema_params"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return cast(dict[str, torch.Tensor], value)
    return cast(dict[str, torch.Tensor], checkpoint)


def _architecture(checkpoint: dict[str, object]) -> dict[str, Any]:
    raw_config = checkpoint.get("config", {})
    metadata = checkpoint.get("metadata", {})
    architecture = (
        metadata.get("architecture", {}) if isinstance(metadata, dict) else {}
    )
    config = raw_config if isinstance(raw_config, dict) else {}
    if not isinstance(architecture, dict):
        architecture = {}
    required = (
        "image_size",
        "patch_size",
        "embed_dim",
        "depth",
        "num_heads",
        "num_classes",
    )
    values: dict[str, int] = {}
    errors: list[str] = []
    for key in required:
        if key in config and key in architecture and config[key] != architecture[key]:
            errors.append(
                f"{key} disagrees between checkpoint config and metadata architecture"
            )
        value = config.get(key, architecture.get(key))
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(f"{key} must be a positive integer")
        else:
            values[key] = value
    if len(values) == len(required):
        if values["image_size"] % values["patch_size"]:
            errors.append("image_size must be divisible by patch_size")
        if values["num_classes"] != len(GENERATOR_CONDITIONS):
            errors.append(
                f"num_classes must be {len(GENERATOR_CONDITIONS)} generator conditions"
            )
        if values["embed_dim"] % values["num_heads"]:
            errors.append("embed_dim must be divisible by num_heads")
    return {"values": values, "errors": errors}


def verify_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> dict:
    """Verify strict architecture/tensor compatibility and representative inference."""
    device = device or torch.device("cpu")
    checkpoint_path = Path(checkpoint_path)
    results: dict[str, Any] = {
        "path": str(checkpoint_path),
        "exists": checkpoint_path.exists(),
        "valid": False,
        "error": None,
        "config": None,
        "config_valid": False,
        "parameter_count": 0,
        "parameters": 0,
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "dtype_mismatches": [],
        "inference_tests": [],
        "inference_test": False,
    }
    if not checkpoint_path.exists():
        results["error"] = "File does not exist"
        return results
    try:
        checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
        state_dict = _checkpoint_state(checkpoint)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint must contain a config envelope")
        config_result = _architecture(checkpoint)
        config = config_result["values"]
        results["config"] = config
        results["config_valid"] = not config_result["errors"]
        if config_result["errors"]:
            raise ValueError("; ".join(config_result["errors"]))
        raw_checkpoint_config = checkpoint.get("config", {})
        legacy = (
            raw_checkpoint_config.get("legacy_layout", False)
            if isinstance(raw_checkpoint_config, dict)
            else False
        )
        if "pos_embed" not in state_dict and legacy:
            # Explicit adapter for the known pre-positional-embedding layout.
            state_dict = dict(state_dict)
            state_dict["pos_embed"] = torch.zeros(
                1,
                (config["image_size"] // config["patch_size"]) ** 2,
                config["embed_dim"],
            )
        model = TinyDiT(**config)
        diff = state_dict_diff(model, state_dict)
        results.update(diff)
        load_state_dict_checked(model, state_dict)
        model.to(device).eval()
        count = count_parameters(model)
        results["parameter_count"] = count
        results["parameters"] = count
        for batch_size in (1, 2):
            for breed_index in (0, 12, 13):
                with torch.no_grad():
                    output = model(
                        torch.randn(
                            batch_size,
                            3,
                            config["image_size"],
                            config["image_size"],
                            device=device,
                        ),
                        torch.full((batch_size,), 0.5, device=device),
                        torch.full(
                            (batch_size,), breed_index, dtype=torch.long, device=device
                        ),
                    )
                results["inference_tests"].append(
                    {
                        "batch_size": batch_size,
                        "breed_index": breed_index,
                        "output_shape": list(output.shape),
                        "finite": bool(torch.isfinite(output).all().item()),
                    }
                )
        results["inference_test"] = all(
            test["finite"]
            and test["output_shape"]
            == [test["batch_size"], 3, config["image_size"], config["image_size"]]
            for test in results["inference_tests"]
        )
        results["valid"] = results["inference_test"] and not any(
            results[key]
            for key in (
                "missing_keys",
                "unexpected_keys",
                "shape_mismatches",
                "dtype_mismatches",
            )
        )
    except Exception as exc:
        results["error"] = str(exc)
    return results


def verify_onnx_inference(onnx_path: str | Path) -> dict:
    """Check ONNX graph structure and representative runtime outputs."""

    path = Path(onnx_path)
    results: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "valid": False,
        "error": None,
        "inference_test": False,
        "node_validation": [],
    }
    if not path.exists():
        results["error"] = "File does not exist"
        results["node_validation"].append(results["error"])
        return results
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort

        graph = onnx.load(str(path))
        onnx.checker.check_model(graph)
        session = ort.InferenceSession(str(path))
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        expected: dict[str, tuple[str, list[Any]]] = {
            "noise": ("tensor(float)", ["batch", 3, 128, 128]),
            "timestep": ("tensor(float)", ["batch"]),
            "breed": ("tensor(int64)", ["batch"]),
        }
        for spec in inputs:
            if spec.name not in expected:
                results["node_validation"].append(f"unexpected input name {spec.name}")
                continue
            dtype, dims = expected[spec.name]
            if spec.type != dtype:
                results["node_validation"].append(
                    f"input {spec.name} dtype {spec.type} != {dtype}"
                )
            raw_input_shape = spec.shape
            if not isinstance(raw_input_shape, list):
                results["node_validation"].append(
                    f"input {spec.name} dimensions are not a list"
                )
                continue
            input_shape: list[Any] = cast(Any, raw_input_shape)
            if len(input_shape) != len(dims) or input_shape[1:] != dims[1:]:
                results["node_validation"].append(
                    f"input {spec.name} dimensions {input_shape} incompatible with {dims}"
                )
        if {spec.name for spec in inputs} != set(expected):
            results["node_validation"].append("input names do not match contract")
        if len(outputs) != 1:
            results["node_validation"].append("expected exactly one output")
        else:
            out = outputs[0]
            if out.type != "tensor(float)":
                results["node_validation"].append(
                    f"output dtype {out.type} is not float"
                )
            if len(out.shape) != 4 or out.shape[1:] != [3, 128, 128]:
                results["node_validation"].append(
                    f"output dimensions {out.shape} are incompatible with [batch,3,128,128]"
                )
        runs = []
        for batch_size in (1, 2):
            for breed_index in (0, 12, 13):
                feed = {
                    "noise": np.zeros((batch_size, 3, 128, 128), dtype=np.float32),
                    "timestep": np.full((batch_size,), 0.5, dtype=np.float32),
                    "breed": np.full((batch_size,), breed_index, dtype=np.int64),
                }
                output = session.run(None, feed)[0]
                if output.shape != (batch_size, 3, 128, 128):
                    results["node_validation"].append(
                        f"output shape {output.shape} for batch {batch_size}"
                    )
                if not np.isfinite(output).all():
                    results["node_validation"].append(
                        "output contains non-finite values"
                    )
                runs.append({"batch_size": batch_size, "breed_index": breed_index})
        results["inference_tests"] = runs
        results["inference_test"] = not results["node_validation"]
        results["valid"] = results["inference_test"]
    except Exception as exc:
        results["error"] = str(exc)
        results["node_validation"].append(str(exc))
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Verify trained checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/pool/dit_model.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="frontend/public/models/generator.onnx",
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX verification",
    )
    parser.add_argument(
        "--require-onnx",
        action="store_true",
        help=(
            "Fail (exit 1) when ONNX verification fails instead of warning; "
            "used by train.yml's publication gate (issue #163 WP7/WP8)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main verification function."""
    args = parse_args(argv)

    print("=" * 60)
    print("TinyDiT Checkpoint Verification")
    print("=" * 60)

    checkpoint_results = verify_checkpoint(args.checkpoint)

    if not checkpoint_results["valid"]:
        print("\nCheckpoint verification failed. Exiting.")
        sys.exit(1)

    if not args.skip_onnx:
        onnx_results = verify_onnx_inference(args.onnx)
        if not onnx_results["valid"]:
            if args.require_onnx:
                print("\nONNX verification failed (--require-onnx). Exiting.")
                sys.exit(1)
            print("\nWARNING: ONNX verification failed, but checkpoint is valid")

    print("\n" + "=" * 60)
    print("Verification Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
