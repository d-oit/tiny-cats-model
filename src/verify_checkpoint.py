"""src/verify_checkpoint.py

Verify a trained checkpoint is valid and can be used for inference.

Usage:
    python src/verify_checkpoint.py --checkpoint checkpoints/tinydit_final.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from dit import count_parameters, tinydit_128


def verify_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> dict:
    """Verify checkpoint can be loaded and used for inference.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Verification results dict
    """
    if device is None:
        device = torch.device("cpu")

    checkpoint_path = Path(checkpoint_path)
    results = {
        "path": str(checkpoint_path),
        "exists": False,
        "valid": False,
        "error": None,
        "config": None,
        "parameters": 0,
        "inference_test": False,
    }

    print(f"Verifying checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        results["error"] = "File does not exist"
        print(f"ERROR: {results['error']}")
        return results

    results["exists"] = True
    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"Checkpoint size: {file_size_mb:.1f} MB")

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        print("Checkpoint loaded successfully")

        config = checkpoint.get("config", {})
        results["config"] = config
        print(f"Config: {config}")

        model = tinydit_128(num_classes=13)
        state_dict = None

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("Found 'model_state_dict' key")
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
            print("Found 'model' key")
        elif "ema_params" in checkpoint and checkpoint["ema_params"] is not None:
            state_dict = checkpoint["ema_params"]
            print("Found 'ema_params' key")
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        print("Model state loaded")

        num_params = count_parameters(model)
        results["parameters"] = num_params
        print(f"Model parameters: {num_params:,}")

        model.eval()
        model.to(device)

        dummy_input = torch.randn(1, 3, 128, 128, device=device)
        dummy_t = torch.tensor([0.5], device=device)
        dummy_breed = torch.tensor([0], device=device)

        with torch.no_grad():
            output = model(dummy_input, dummy_t, dummy_breed)

        results["inference_test"] = True
        print(f"Inference test passed! Output shape: {output.shape}")

        results["valid"] = True
        print("\n✓ Checkpoint verification PASSED")

    except Exception as e:
        results["error"] = str(e)
        results["valid"] = False
        print(f"\n✗ Checkpoint verification FAILED: {e}")

    return results


def verify_onnx_inference(
    onnx_path: str | Path,
) -> dict:
    """Verify ONNX model can be loaded and run inference.

    Args:
        onnx_path: Path to ONNX model file

    Returns:
        Verification results dict
    """
    onnx_path = Path(onnx_path)
    results = {
        "path": str(onnx_path),
        "exists": False,
        "valid": False,
        "error": None,
        "inference_test": False,
    }

    print(f"\nVerifying ONNX model: {onnx_path}")

    if not onnx_path.exists():
        results["error"] = "File does not exist"
        print(f"ERROR: {results['error']}")
        return results

    results["exists"] = True
    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"ONNX model size: {file_size_mb:.1f} MB")

    try:
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print("ONNX model is valid")

        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))
        print(f"Execution providers: {session.get_providers()}")

        noise = np.random.randn(1, 3, 128, 128).astype(np.float32)
        timestep = np.array([0.5], dtype=np.float32)
        breed = np.array([0], dtype=np.int64)

        result = session.run(
            None, {"noise": noise, "timestep": timestep, "breed": breed}
        )

        results["inference_test"] = True
        print(f"ONNX inference test passed! Output shape: {result[0].shape}")
        results["valid"] = True
        print("\n✓ ONNX verification PASSED")

    except ImportError as e:
        results["error"] = f"Missing dependency: {e}"
        print(f"WARNING: {results['error']}")
    except Exception as e:
        results["error"] = str(e)
        results["valid"] = False
        print(f"\n✗ ONNX verification FAILED: {e}")

    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Verify trained checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/tinydit_final.pt",
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
    return parser.parse_args()


def main() -> None:
    """Main verification function."""
    args = parse_args()

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
            print("\nWARNING: ONNX verification failed, but checkpoint is valid")

    print("\n" + "=" * 60)
    print("Verification Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
