"""src/benchmark_inference.py

Benchmark inference performance for TinyDiT models.

Measures:
- Latency (p50, p95, p99)
- Throughput (images/second)
- Memory usage (peak)
- Model size

Usage:
    python src/benchmark_inference.py --model checkpoints/tinydit_final.pt
    python src/benchmark_inference.py --onnx frontend/public/models/generator.onnx
    python src/benchmark_inference.py --device cuda --batch-size 32

Dependencies:
    # For ONNX benchmarking
    pip install onnxruntime onnxruntime-gpu

    # For memory profiling (optional)
    pip install py3nvml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dit import tinydit_128


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark inference performance for TinyDiT models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to PyTorch model checkpoint",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for benchmarking",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Image size for input tensor",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--benchmark-throughput",
        action="store_true",
        help="Run throughput benchmark with multiple batch sizes",
    )
    parser.add_argument(
        "--throughput-duration",
        type=int,
        default=30,
        help="Duration in seconds for each throughput test",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8,16,32",
        help="Comma-separated batch sizes for throughput test",
    )
    parser.add_argument(
        "--measure-memory",
        action="store_true",
        help="Measure peak memory usage (CUDA only)",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="benchmark_report.json",
        help="Path to save benchmark report",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed benchmark output",
    )
    return parser.parse_args()


def load_pytorch_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Load PyTorch model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model on.

    Returns:
        Tuple of (model, metadata).

    Raises:
        FileNotFoundError: If checkpoint doesn't exist.
        KeyError: If checkpoint format is invalid.
    """
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config
    config = checkpoint.get(
        "config",
        {
            "image_size": 128,
            "patch_size": 16,
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "num_classes": 13,
        },
    )

    # Create model
    num_classes = config.get("num_classes", 13)
    image_size = config.get("image_size", 128)
    model = tinydit_128(num_classes=num_classes).to(device)

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "ema_shadow_params" in checkpoint:
        model.load_state_dict(checkpoint["ema_shadow_params"])
    else:
        raise KeyError(
            f"Checkpoint missing model weights. Keys: {list(checkpoint.keys())}"
        )

    model.eval()

    metadata = {
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", 0.0),
        "config": config,
        "image_size": image_size,
    }

    print(f"Loaded PyTorch model from {checkpoint_path}")
    print(f"  Step: {metadata['step']:,}")
    print(f"  Image size: {image_size}")

    return model, metadata


def load_onnx_model(onnx_path: str) -> Any:
    """Load ONNX model for inference.

    Args:
        onnx_path: Path to ONNX file.

    Returns:
        ONNX Runtime session.

    Raises:
        FileNotFoundError: If ONNX file doesn't exist.
        ImportError: If onnxruntime not installed.
    """
    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "ONNX Runtime not installed. Install with: pip install onnxruntime"
        ) from e

    # Determine execution provider
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(onnx_path, providers=providers)

    print(f"Loaded ONNX model from {onnx_path}")
    print(f"  Providers: {providers}")
    print(f"  Inputs: {[i.name for i in session.get_inputs()]}")

    return session


def create_input_tensor(
    batch_size: int,
    image_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create input tensors for TinyDiT forward pass.

    Args:
        batch_size: Batch size.
        image_size: Image size.
        device: Device to create tensors on.

    Returns:
        Tuple of (x, t, breeds) tensors.
    """
    # Noisy image input
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)

    # Timestep (uniform random)
    t = torch.rand(batch_size, device=device)

    # Breed indices (random)
    breeds = torch.randint(0, 13, (batch_size,), device=device)

    return x, t, breeds


def benchmark_latency(
    model: torch.nn.Module | Any,
    device: torch.device,
    image_size: int = 128,
    num_warmup: int = 10,
    num_runs: int = 100,
    is_onnx: bool = False,
    verbose: bool = False,
) -> dict[str, float]:
    """Benchmark inference latency.

    Args:
        model: Model to benchmark (PyTorch or ONNX session).
        device: Device to run on.
        image_size: Image size.
        num_warmup: Warmup iterations.
        num_runs: Benchmark iterations.
        is_onnx: Whether model is ONNX.
        verbose: Print detailed output.

    Returns:
        Dict with p50, p95, p99 latencies in ms.
    """
    batch_size = 1

    # Create input
    if is_onnx:
        x = torch.randn(1, 3, image_size, image_size).cpu().numpy()
        t = np.random.rand(1).astype(np.float32)
        breeds = np.random.randint(0, 13, size=(1,)).astype(np.int64)

        # Get input names from ONNX model
        input_names = [i.name for i in model.get_inputs()]
    else:
        x, t, breeds = create_input_tensor(batch_size, image_size, device)

    # Warmup
    if verbose:
        print(f"Warming up ({num_warmup} iterations)...")

    for _ in range(num_warmup):
        if is_onnx:
            x = torch.randn(1, 3, image_size, image_size).cpu().numpy()
            t = np.random.rand(1).astype(np.float32)
            breeds = np.random.randint(0, 13, size=(1,)).astype(np.int64)

            input_feed = dict(zip(input_names, [x, t, breeds]))
            _ = model.run(None, input_feed)
        else:
            x, t, breeds = create_input_tensor(batch_size, image_size, device)
            _ = model(x, t, breeds)

    # Benchmark
    if verbose:
        print(f"Benchmarking ({num_runs} iterations)...")

    latencies = []
    for i in range(num_runs):
        if is_onnx:
            x = torch.randn(1, 3, image_size, image_size).cpu().numpy()
            t = np.random.rand(1).astype(np.float32)
            breeds = np.random.randint(0, 13, size=(1,)).astype(np.int64)

            input_feed = dict(zip(input_names, [x, t, breeds]))

            start = time.perf_counter()
            _ = model.run(None, input_feed)
            latencies.append((time.perf_counter() - start) * 1000)
        else:
            x, t, breeds = create_input_tensor(batch_size, image_size, device)

            start = time.perf_counter()
            _ = model(x, t, breeds)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        if verbose and (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{num_runs} iterations")

    # Compute statistics
    latencies_array = np.array(latencies)

    results = {
        "p50": float(np.percentile(latencies_array, 50)),
        "p95": float(np.percentile(latencies_array, 95)),
        "p99": float(np.percentile(latencies_array, 99)),
        "mean": float(np.mean(latencies_array)),
        "std": float(np.std(latencies_array)),
        "min": float(np.min(latencies_array)),
        "max": float(np.max(latencies_array)),
    }

    return results


def benchmark_throughput(
    model: torch.nn.Module | Any,
    device: torch.device,
    image_size: int = 128,
    batch_sizes: list[int] | None = None,
    duration_seconds: int = 30,
    is_onnx: bool = False,
    verbose: bool = False,
) -> dict[int, float]:
    """Benchmark throughput for different batch sizes.

    Args:
        model: Model to benchmark.
        device: Device to run on.
        image_size: Image size.
        batch_sizes: Batch sizes to test.
        duration_seconds: How long to run each test.
        is_onnx: Whether model is ONNX.
        verbose: Print detailed output.

    Returns:
        Dict mapping batch size to images/second.
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]

    results = {}

    for batch_size in batch_sizes:
        if verbose:
            print(f"\nTesting batch size {batch_size}...")

        # Create input
        if is_onnx:
            x = torch.randn(batch_size, 3, image_size, image_size).cpu().numpy()
            t = np.random.rand(batch_size).astype(np.float32)
            breeds = np.random.randint(0, 13, size=(batch_size,)).astype(np.int64)

            input_names = [i.name for i in model.get_inputs()]
        else:
            x, t, breeds = create_input_tensor(batch_size, image_size, device)

        # Run for specified duration
        start = time.perf_counter()
        count = 0

        while time.perf_counter() - start < duration_seconds:
            if is_onnx:
                x = torch.randn(batch_size, 3, image_size, image_size).cpu().numpy()
                t = np.random.rand(batch_size).astype(np.float32)
                breeds = np.random.randint(0, 13, size=(batch_size,)).astype(np.int64)

                input_feed = dict(zip(input_names, [x, t, breeds]))
                _ = model.run(None, input_feed)
            else:
                x, t, breeds = create_input_tensor(batch_size, image_size, device)
                _ = model(x, t, breeds)

                if device.type == "cuda":
                    torch.cuda.synchronize()

            count += batch_size

        elapsed = time.perf_counter() - start
        throughput = count / elapsed
        results[batch_size] = throughput

        if verbose:
            print(f"  Batch size {batch_size}: {throughput:.1f} images/sec")

    return results


def measure_memory_usage(
    model: torch.nn.Module,
    device: torch.device,
    image_size: int = 128,
    batch_size: int = 1,
) -> dict[str, float | str]:
    """Measure peak memory usage.

    Args:
        model: Model to benchmark.
        device: Device to run on.
        image_size: Image size.
        batch_size: Batch size.

    Returns:
        Dict with memory stats in MB.
    """
    if device.type != "cuda":
        return {
            "note": "Memory measurement only available on CUDA",
            "device": str(device),
        }

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Create input and run inference
    x, t, breeds = create_input_tensor(batch_size, image_size, device)

    with torch.no_grad():
        _ = model(x, t, breeds)

    torch.cuda.synchronize()

    # Get memory stats
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB

    # Model size
    model_size_mb = (
        sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    )

    return {
        "peak_memory_mb": peak_memory,
        "total_memory_mb": total_memory,
        "model_size_mb": model_size_mb,
        "memory_utilization": peak_memory / total_memory,
    }


def compute_model_size(model: torch.nn.Module) -> dict[str, float]:
    """Compute model size statistics.

    Args:
        model: PyTorch model.

    Returns:
        Dict with size statistics.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    trainable_size = sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    )

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "total_size_mb": total_size / 1024 / 1024,
        "trainable_size_mb": trainable_size / 1024 / 1024,
    }


def generate_benchmark_report(
    latency: dict[str, float],
    throughput: dict[int, float],
    memory: dict[str, float | str],
    model_info: dict[str, Any],
    output_path: str | Path = "benchmark_report.json",
) -> None:
    """Generate JSON benchmark report.

    Args:
        latency: Latency metrics.
        throughput: Throughput metrics.
        memory: Memory metrics.
        model_info: Model information.
        output_path: Path to save report.
    """
    report = {
        "model_info": model_info,
        "latency_ms": latency,
        "throughput_images_per_sec": throughput,
        "memory": memory,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Benchmark report saved to {output_path}")
    print(f"{'=' * 60}")
    print(f"\nModel: {model_info.get('name', 'Unknown')}")
    print(f"Device: {model_info.get('device', 'Unknown')}")

    print("\nLatency (batch=1):")
    print(
        f"  p50: {latency.get('p50', 'N/A'):.2f} ms"
        if latency.get("p50")
        else "  p50: N/A"
    )
    print(
        f"  p95: {latency.get('p95', 'N/A'):.2f} ms"
        if latency.get("p95")
        else "  p95: N/A"
    )
    print(
        f"  p99: {latency.get('p99', 'N/A'):.2f} ms"
        if latency.get("p99")
        else "  p99: N/A"
    )
    print(
        f"  mean: {latency.get('mean', 'N/A'):.2f} ms"
        if latency.get("mean")
        else "  mean: N/A"
    )

    print("\nThroughput:")
    if throughput:
        for bs, tp in throughput.items():
            print(f"  Batch {bs}: {tp:.1f} img/s")

    if memory.get("peak_memory_mb"):
        print("\nMemory:")
        print(f"  Peak: {memory['peak_memory_mb']:.1f} MB")
        print(f"  Model size: {memory.get('model_size_mb', 'N/A'):.1f} MB")


def benchmark_pytorch_model(
    checkpoint_path: str,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Benchmark PyTorch model.

    Args:
        checkpoint_path: Path to checkpoint.
        device: Device to use.
        args: Command line arguments.
    """
    # Load model
    model, metadata = load_pytorch_model(checkpoint_path, device)
    image_size = metadata.get("image_size", args.image_size)

    # Model info
    model_info = {
        "name": "TinyDiT (PyTorch)",
        "checkpoint": checkpoint_path,
        "device": str(device),
        "image_size": image_size,
        "step": metadata.get("step", 0),
    }

    # Add model size info
    size_info = compute_model_size(model)
    model_info.update(size_info)

    print(f"\nModel parameters: {size_info['total_parameters']:,}")
    print(f"Model size: {size_info['total_size_mb']:.2f} MB")

    # Benchmark latency
    print(f"\n{'=' * 60}")
    print("Benchmarking latency...")
    print(f"{'=' * 60}")

    latency = benchmark_latency(
        model=model,
        device=device,
        image_size=image_size,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        is_onnx=False,
        verbose=args.verbose,
    )

    print(f"Latency p50: {latency['p50']:.2f} ms")
    print(f"Latency p95: {latency['p95']:.2f} ms")
    print(f"Latency p99: {latency['p99']:.2f} ms")

    # Benchmark throughput
    throughput = {}
    if args.benchmark_throughput:
        print(f"\n{'=' * 60}")
        print("Benchmarking throughput...")
        print(f"{'=' * 60}")

        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

        throughput = benchmark_throughput(
            model=model,
            device=device,
            image_size=image_size,
            batch_sizes=batch_sizes,
            duration_seconds=args.throughput_duration,
            is_onnx=False,
            verbose=args.verbose,
        )

    # Measure memory
    memory = {}
    if args.measure_memory and device.type == "cuda":
        print(f"\n{'=' * 60}")
        print("Measuring memory usage...")
        print(f"{'=' * 60}")

        memory = measure_memory_usage(
            model=model,
            device=device,
            image_size=image_size,
            batch_size=args.batch_size,
        )

        if "peak_memory_mb" in memory:
            print(f"Peak memory: {memory['peak_memory_mb']:.1f} MB")
            print(f"Model size: {memory['model_size_mb']:.1f} MB")

    # Generate report
    generate_benchmark_report(
        latency=latency,
        throughput=throughput,
        memory=memory,
        model_info=model_info,
        output_path=args.report_path,
    )


def benchmark_onnx_model(
    onnx_path: str,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Benchmark ONNX model.

    Args:
        onnx_path: Path to ONNX file.
        device: Device to use.
        args: Command line arguments.
    """
    # Load model
    session = load_onnx_model(onnx_path)

    # Model info
    model_info = {
        "name": "TinyDiT (ONNX)",
        "onnx_file": onnx_path,
        "device": str(device),
        "image_size": args.image_size,
    }

    # Benchmark latency
    print(f"\n{'=' * 60}")
    print("Benchmarking latency...")
    print(f"{'=' * 60}")

    latency = benchmark_latency(
        model=session,
        device=device,
        image_size=args.image_size,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        is_onnx=True,
        verbose=args.verbose,
    )

    print(f"Latency p50: {latency['p50']:.2f} ms")
    print(f"Latency p95: {latency['p95']:.2f} ms")
    print(f"Latency p99: {latency['p99']:.2f} ms")

    # Benchmark throughput
    throughput = {}
    if args.benchmark_throughput:
        print(f"\n{'=' * 60}")
        print("Benchmarking throughput...")
        print(f"{'=' * 60}")

        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

        throughput = benchmark_throughput(
            model=session,
            device=device,
            image_size=args.image_size,
            batch_sizes=batch_sizes,
            duration_seconds=args.throughput_duration,
            is_onnx=True,
            verbose=args.verbose,
        )

    # Memory measurement not available for ONNX
    memory: dict[str, float | str] = {
        "note": "Memory measurement not available for ONNX models"
    }

    # Generate report
    generate_benchmark_report(
        latency=latency,
        throughput=throughput,
        memory=memory,
        model_info=model_info,
        output_path=args.report_path,
    )


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Validate inputs
    if not args.model and not args.onnx:
        print("Error: Must specify either --model or --onnx")
        sys.exit(1)

    if args.model and args.onnx:
        print("Error: Cannot specify both --model and --onnx")
        sys.exit(1)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Run appropriate benchmark
    if args.model:
        try:
            benchmark_pytorch_model(args.model, device, args)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except KeyError as e:
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)
    elif args.onnx:
        try:
            benchmark_onnx_model(args.onnx, device, args)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except ImportError as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
