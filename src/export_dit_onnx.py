"""src/export_dit_onnx.py

Export the TinyDiT generator model to ONNX format for web inference.

This script exports:
1. The main generator model (velocity prediction)
2. A sampler model that handles ODE integration steps

Usage:
    python src/export_dit_onnx.py
    python src/export_dit_onnx.py --checkpoint PATH --output PATH
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dit import TinyDiT, load_state_dict_checked, tinydit_128
from model_contract import GENERATOR_CONDITIONS, ContractError, generator_metadata
from safe_checkpoint import load_torch_checkpoint, sha256_file


class SamplerWrapper(torch.nn.Module):
    """Wrapper for single-step ODE integration in ONNX.

    This wrapper performs one Euler step of the flow matching ODE:
    x(t+dt) = x(t) + dt * velocity(x(t), t, breed)

    For browser inference, this allows step-by-step generation
    with progress tracking.
    """

    def __init__(
        self,
        model: TinyDiT,
        cfg_scale: float = 1.5,
        use_cfg: bool = True,
    ) -> None:
        """Initialize sampler wrapper.

        Args:
            model: TinyDiT model for velocity prediction
            cfg_scale: Classifier-free guidance scale
            use_cfg: Whether to use classifier-free guidance
        """
        super().__init__()
        self.model = model
        self.cfg_scale = cfg_scale
        self.use_cfg = use_cfg

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        breeds: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one ODE integration step.

        Args:
            x: Current noisy image (B, C, H, W)
            t: Current timestep (B,)
            breeds: Breed indices (B,)

        Returns:
            Next state x(t+dt) (B, C, H, W)
        """
        # Get velocity prediction
        if self.use_cfg and self.cfg_scale > 1.0:
            velocity = self.model.forward_with_cfg(x, t, breeds, self.cfg_scale)
        else:
            velocity = self.model(x, t, breeds)

        # Fixed dt for Euler integration (will be set during export)
        # For flexibility, we return velocity and let frontend handle dt
        return velocity


class CFGSamplerWrapper(torch.nn.Module):
    """Wrapper for CFG sampling with explicit batch handling.

    This wrapper handles CFG by concatenating conditional and
    unconditional inputs for a single forward pass, then splitting.
    This avoids dynamic shape issues with the view operation.
    """

    def __init__(
        self,
        model: TinyDiT,
        cfg_scale: float = 1.5,
    ) -> None:
        """Initialize CFG sampler wrapper.

        Args:
            model: TinyDiT model for velocity prediction
            cfg_scale: Classifier-free guidance scale
        """
        super().__init__()
        self.model = model
        self.cfg_scale = cfg_scale
        self.uncond_class = model.num_classes - 1  # "other" class

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        breeds: torch.Tensor,
    ) -> torch.Tensor:
        """Perform CFG velocity prediction.

        Args:
            x: Current noisy image (B, C, H, W)
            t: Current timestep (B,)
            breeds: Breed indices (B,)

        Returns:
            Velocity with CFG applied (B, C, H, W)
        """
        batch_size = x.shape[0]

        # Create unconditional input (use "other" class)
        uncond_breeds = torch.full_like(breeds, self.uncond_class)

        # Concatenate conditional and unconditional inputs
        x_cat = torch.cat([x, x], dim=0)  # (2B, C, H, W)
        t_cat = torch.cat([t, t], dim=0)  # (2B,)
        breeds_cat = torch.cat([breeds, uncond_breeds], dim=0)  # (2B,)

        # Single forward pass for both
        output = self.model(x_cat, t_cat, breeds_cat)  # (2B, C, H, W)

        # Split and apply CFG
        pred_cond = output[:batch_size]
        pred_uncond = output[batch_size:]

        # CFG: v = v_uncond + cfg_scale * (v_cond - v_uncond)
        velocity = pred_uncond + self.cfg_scale * (pred_cond - pred_uncond)

        return velocity


def create_dummy_checkpoint(
    output_path: str | Path,
    image_size: int = 128,
    num_classes: int = 13,
) -> None:
    """Create a dummy checkpoint for testing export.

    Args:
        output_path: Path to save the dummy checkpoint
        image_size: Image size for the model
        num_classes: Number of breed classes
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = tinydit_128(num_classes=num_classes)
    model.eval()

    # Save with EMA shadow params structure for compatibility
    checkpoint = {
        "model": model.state_dict(),
        "ema_params": None,
        "step": 0,
        "config": {
            "image_size": image_size,
            "num_classes": num_classes,
            "patch_size": 16,
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
        },
    }

    torch.save(checkpoint, output_path)
    print(f"Created dummy checkpoint at {output_path}")


def load_model(
    checkpoint_path: str | Path,
    image_size: int = 128,
    num_classes: int = 13,
    device: torch.device | None = None,
) -> TinyDiT:
    """Load a TinyDiT model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        image_size: Image size for the model
        num_classes: Number of breed classes
        device: Device to load the model on

    Returns:
        Loaded TinyDiT model in eval mode
    """
    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device = torch.device("cpu")
    if num_classes != len(GENERATOR_CONDITIONS):
        raise ContractError(
            f"generator must have {len(GENERATOR_CONDITIONS)} labels, got {num_classes}"
        )
    model = tinydit_128(num_classes=num_classes)
    if checkpoint_path.exists():
        checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
        if not isinstance(checkpoint, dict):
            raise ContractError("generator checkpoint must be a dictionary")
        config = checkpoint.get("config", {})
        if isinstance(config, dict) and "num_classes" in config:
            if int(config["num_classes"]) != len(GENERATOR_CONDITIONS):
                raise ContractError(
                    "generator checkpoint num_classes disagrees with condition labels"
                )
        metadata = checkpoint.get("metadata", {})
        if isinstance(metadata, dict):
            conditions = metadata.get("conditions")
            if conditions is not None:
                names = [
                    item.get("name") for item in conditions if isinstance(item, dict)
                ]
                if tuple(names) != GENERATOR_CONDITIONS:
                    raise ContractError(
                        "generator checkpoint condition labels disagree with contract"
                    )
            architecture = metadata.get("architecture", {})
            if (
                isinstance(architecture, dict)
                and "num_classes" in architecture
                and int(architecture["num_classes"]) != len(GENERATOR_CONDITIONS)
            ):
                raise ContractError(
                    "generator metadata num_classes disagrees with labels"
                )
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif checkpoint.get("ema_params") is not None:
            state_dict = checkpoint["ema_params"]
        else:
            state_dict = checkpoint
        load_state_dict_checked(model, state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Creating model with random weights for export testing")

    model.eval()
    return model.to(device)


def export_generator_onnx(
    model: TinyDiT,
    output_path: str | Path,
    opset_version: int = 17,
    use_cfg: bool = True,
    cfg_scale: float = 1.5,
) -> None:
    """Export the generator model to ONNX format.

    Args:
        model: TinyDiT model to export
        output_path: Path where the ONNX model will be saved
        opset_version: ONNX opset version to use
        use_cfg: Whether to export with classifier-free guidance
        cfg_scale: CFG scale for guidance
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create sampler wrapper - use CFG wrapper for better ONNX compatibility
    if use_cfg and cfg_scale > 1.0:
        sampler = CFGSamplerWrapper(model, cfg_scale=cfg_scale)
    else:
        sampler = SamplerWrapper(model, cfg_scale=cfg_scale, use_cfg=False)
    sampler.eval()

    # Create dummy inputs for tracing
    batch_size = 1
    image_size = model.image_size

    dummy_x = torch.randn(batch_size, 3, image_size, image_size)
    dummy_t = torch.tensor([0.5], dtype=torch.float32)
    dummy_breeds = torch.tensor([0], dtype=torch.int64)

    # Prepare input names
    input_names = ["noise", "timestep", "breed"]
    output_names = ["velocity"]

    # Use dynamo=False for better compatibility with dynamic shapes
    torch.onnx.export(
        sampler,
        (dummy_x, dummy_t, dummy_breeds),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "noise": {0: "batch_size"},
            "timestep": {0: "batch_size"},
            "breed": {0: "batch_size"},
            "velocity": {0: "batch_size"},
        },
        dynamo=False,
    )
    import onnx

    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)
    graph_inputs = {value.name: value for value in graph.graph.input}
    graph_outputs = {value.name: value for value in graph.graph.output}
    if set(graph_inputs) != set(input_names) or set(graph_outputs) != set(output_names):
        raise RuntimeError("exported generator graph names violate contract")
    for name, value in graph_inputs.items():
        dims = value.type.tensor_type.shape.dim
        expected_rank = 4 if name == "noise" else 1
        if len(dims) != expected_rank:
            raise RuntimeError(f"exported generator input {name} has invalid rank")
        expected_type = 7 if name == "breed" else 1
        if value.type.tensor_type.elem_type != expected_type:
            raise RuntimeError(f"exported generator input {name} has invalid dtype")
        if name == "noise" and [dim.dim_value for dim in dims[1:]] != [
            3,
            image_size,
            image_size,
        ]:
            raise RuntimeError("exported generator noise dimensions violate contract")
    if graph_outputs["velocity"].type.tensor_type.elem_type != 1:
        raise RuntimeError("exported generator output must be float32")
    out_dims = graph_outputs["velocity"].type.tensor_type.shape.dim
    if len(out_dims) != 4 or [dim.dim_value for dim in out_dims[1:]] != [
        3,
        image_size,
        image_size,
    ]:
        raise RuntimeError("exported generator output dimensions violate contract")
    metadata = generator_metadata(
        num_classes=model.num_classes,
        opset=opset_version,
        image_size=image_size,
        model_sha256=sha256_file(output_path),
    )
    Path(str(output_path) + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    print(f"Exported ONNX model to {output_path}")
    print("  Input shapes:")
    print(f"    noise: [batch_size, 3, {image_size}, {image_size}]")
    print("    timestep: [batch_size]")
    print("    breed: [batch_size]")
    print("  Output shape:")
    print(f"    velocity: [batch_size, 3, {image_size}, {image_size}]")
    if use_cfg and cfg_scale > 1.0:
        print(f"  CFG scale: {cfg_scale}")
    else:
        print("  CFG: disabled")
    print(f"  Opset version: {opset_version}")


def export_velocity_model_onnx(
    model: TinyDiT,
    output_path: str | Path,
    opset_version: int = 17,
) -> None:
    """Export the raw velocity prediction model to ONNX.

    This exports the base TinyDiT model without CFG or sampling logic.
    Useful for custom sampling implementations.

    Args:
        model: TinyDiT model to export
        output_path: Path where the ONNX model will be saved
        opset_version: ONNX opset version to use
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Create dummy inputs for tracing
    batch_size = 1
    image_size = model.image_size

    dummy_x = torch.randn(batch_size, 3, image_size, image_size)
    dummy_t = torch.tensor([0.5])
    dummy_breeds = torch.tensor([0])

    input_names = ["noise", "timestep", "breed"]
    output_names = ["velocity"]

    torch.onnx.export(
        model,
        (dummy_x, dummy_t, dummy_breeds),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "noise": {0: "batch_size"},
            "timestep": {0: "batch_size"},
            "breed": {0: "batch_size"},
            "velocity": {0: "batch_size"},
        },
    )
    if model.num_classes != len(GENERATOR_CONDITIONS):
        raise ContractError("generator classes disagree with conditioning labels")
    import onnx

    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)
    graph_inputs = {value.name: value for value in graph.graph.input}
    graph_outputs = {value.name: value for value in graph.graph.output}
    if set(graph_inputs) != set(input_names) or set(graph_outputs) != set(output_names):
        raise RuntimeError("exported generator graph names violate contract")
    for name, value in graph_inputs.items():
        if value.type.tensor_type.elem_type != (7 if name == "breed" else 1):
            raise RuntimeError(f"exported generator input {name} has wrong dtype")
        dims = value.type.tensor_type.shape.dim
        if len(dims) != (4 if name == "noise" else 1):
            raise RuntimeError(f"exported generator input {name} has wrong rank")
        if name == "noise" and [dim.dim_value for dim in dims[1:]] != [
            3,
            image_size,
            image_size,
        ]:
            raise RuntimeError("exported generator noise dimensions violate contract")
    output_value = graph_outputs["velocity"]
    output_dims = output_value.type.tensor_type.shape.dim
    if (
        output_value.type.tensor_type.elem_type != 1
        or len(output_dims) != 4
        or [dim.dim_value for dim in output_dims[1:]] != [3, image_size, image_size]
    ):
        raise RuntimeError(
            "exported generator output violates dtype/dimension contract"
        )
    metadata = generator_metadata(
        num_classes=model.num_classes,
        opset=opset_version,
        image_size=image_size,
        model_sha256=sha256_file(output_path),
    )
    Path(str(output_path) + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    print(f"Exported raw velocity model to {output_path}")


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
        image_size = 128

        noise = np.random.randn(batch_size, 3, image_size, image_size).astype(
            np.float32
        )
        timestep = np.array([0.5], dtype=np.float32)
        breed = np.array([0], dtype=np.int64)

        result = session.run(
            None, {"noise": noise, "timestep": timestep, "breed": breed}
        )

        print("\n  Test Inference:")
        print(f"    Input noise shape: {noise.shape}")
        print(f"    Output velocity shape: {result[0].shape}")
        print(f"    Output range: [{result[0].min():.4f}, {result[0].max():.4f}]")
        print(f"    Output mean: {result[0].mean():.4f}, std: {result[0].std():.4f}")

    except ImportError:
        print("ONNX Runtime not installed. Install with: pip install onnxruntime")
    except Exception as e:
        print(f"Inference test error: {e}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export TinyDiT generator to ONNX format"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/pool/dit_model.pt",
        help="Path to the trained checkpoint (default: checkpoints/pool/dit_model.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="frontend/public/models/generator.onnx",
        help="Output path for the ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.5,
        help="Classifier-free guidance scale (default: 1.5)",
    )
    parser.add_argument(
        "--no-cfg",
        action="store_true",
        help="Disable classifier-free guidance",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Image size for the model (default: 128)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=13,
        help="Number of breed classes (default: 13)",
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
            image_size=args.image_size,
            num_classes=args.num_classes,
        )

    # Load model
    device = torch.device("cpu")
    model = load_model(
        checkpoint_path,
        image_size=args.image_size,
        num_classes=args.num_classes,
        device=device,
    )

    # Export to ONNX
    export_generator_onnx(
        model=model,
        output_path=output_path,
        opset_version=args.opset,
        use_cfg=not args.no_cfg,
        cfg_scale=args.cfg_scale,
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
