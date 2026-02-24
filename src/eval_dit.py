"""src/eval_dit.py

Evaluation script for TinyDiT generated samples.

Features:
- Generate samples from trained model
- Compute FID-like metrics (if reference dataset available)
- Visualize generated samples
- Compare EMA vs non-EMA model quality
- Batch generation for quality assessment

Usage:
    # Generate samples with default model
    python src/eval_dit.py

    # Evaluate specific checkpoint
    python src/eval_dit.py --checkpoint checkpoints/dit_model_ema.pt

    # Generate with specific breed
    python src/eval_dit.py --breed 0 --num-samples 16

    # Compare EMA vs non-EMA
    python src/eval_dit.py --compare-ema

    # Generate grid of all breeds
    python src/eval_dit.py --all-breeds --output-dir samples/
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from dit import TinyDiT, tinydit_128
from flow_matching import sample

# Cat breed names (Oxford IIIT Pet dataset)
CAT_BREEDS = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx",
    "Other",  # 13th class for unconditional/other
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate TinyDiT generated samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/dit_model_ema.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--ema-checkpoint",
        type=str,
        default=None,
        help="Path to EMA checkpoint for comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="samples/generated",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of samples per breed",
    )
    parser.add_argument(
        "--breed",
        type=int,
        default=None,
        help="Specific breed index (0-12), or None for all",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Image size",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of sampling steps",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--compare-ema",
        action="store_true",
        help="Compare EMA vs non-EMA model",
    )
    parser.add_argument(
        "--all-breeds",
        action="store_true",
        help="Generate samples for all breeds",
    )
    parser.add_argument(
        "--save-grid",
        action="store_true",
        help="Save samples as grid image",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[TinyDiT, dict]:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model on.

    Returns:
        Tuple of (model, checkpoint metadata).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config from checkpoint
    config = checkpoint.get(
        "config",
        {
            "image_size": 128,
            "patch_size": 16,
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
        },
    )

    # Create model
    model = tinydit_128(num_classes=13).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metadata = {
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", 0.0),
        "timestamp": checkpoint.get("timestamp", "unknown"),
        "config": config,
    }

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Step: {metadata['step']:,}")
    print(f"  Loss: {metadata['loss']:.4f}")
    print(f"  Config: {config}")

    return model, metadata


def apply_ema(
    model: TinyDiT,
    checkpoint: dict,
    device: torch.device,
) -> None:
    """Apply EMA weights to model.

    Args:
        model: Model to update.
        checkpoint: Checkpoint with EMA shadow params.
        device: Device.
    """
    if "ema_shadow_params" not in checkpoint:
        print("Warning: No EMA weights found in checkpoint")
        return

    shadow_params = checkpoint["ema_shadow_params"]
    for name, param in model.named_parameters():
        if name in shadow_params:
            param.data.copy_(shadow_params[name].data)

    print("Applied EMA weights to model")


def generate_samples(
    model: TinyDiT,
    breeds: torch.Tensor,
    num_steps: int = 50,
    cfg_scale: float = 1.5,
    image_size: int = 128,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate samples for given breeds.

    Args:
        model: TinyDiT model.
        breeds: Breed indices.
        num_steps: Sampling steps.
        cfg_scale: CFG scale.
        image_size: Image size.
        device: Device.

    Returns:
        Generated images (B, C, H, W).
    """
    return sample(
        model,
        breeds,
        num_steps=num_steps,
        device=device,
        image_size=image_size,
        cfg_scale=cfg_scale,
        progress=True,
    )


def save_images(
    images: torch.Tensor,
    output_dir: str,
    prefix: str = "sample",
    breeds: torch.Tensor | None = None,
) -> list[Path]:
    """Save generated images to disk.

    Args:
        images: Images tensor (B, C, H, W).
        output_dir: Output directory.
        prefix: Filename prefix.
        breeds: Optional breed indices for filenames.

    Returns:
        List of saved file paths.
    """
    try:
        from PIL import Image
    except ImportError:
        print("Error: PIL not available. Install with: pip install Pillow")
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for i, img in enumerate(images):
        # Convert to PIL Image
        img_np = (
            (img.permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5)
            .clip(0, 255)
            .astype("uint8")
        )
        pil_img = Image.fromarray(img_np)

        # Generate filename
        if breeds is not None:
            breed_idx = breeds[i].item()
            breed_name = (
                CAT_BREEDS[breed_idx]
                if breed_idx < len(CAT_BREEDS)
                else f"breed_{breed_idx}"
            )
            filename = f"{prefix}_{i:03d}_{breed_name}.png"
        else:
            filename = f"{prefix}_{i:03d}.png"

        filepath = output_path / filename
        pil_img.save(filepath)
        saved_files.append(filepath)

    return saved_files


def save_grid(
    images: torch.Tensor,
    output_path: str,
    nrow: int | None = None,
    breeds: torch.Tensor | None = None,
) -> Path:
    """Save images as a grid.

    Args:
        images: Images tensor (B, C, H, W).
        output_path: Output file path.
        nrow: Number of images per row.
        breeds: Optional breed indices for labels.

    Returns:
        Path to saved grid image.
    """
    try:
        from PIL import Image
    except ImportError:
        print("Error: PIL not available")
        return Path(output_path)

    # Convert to grid using torchvision if available
    try:
        from torchvision.utils import make_grid

        grid = make_grid(images, nrow=nrow or int(len(images) ** 0.5), padding=2)
        grid_np = (
            (grid.permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5)
            .clip(0, 255)
            .astype("uint8")
        )
        grid_img = Image.fromarray(grid_np)
    except ImportError:
        # Manual grid creation
        B, C, H, W = images.shape
        if nrow is None:
            nrow = int(B**0.5)
        ncol = (B + nrow - 1) // nrow

        # Create blank grid
        grid_np = torch.zeros(C, H * ncol + 2 * (ncol - 1), W * nrow + 2 * (nrow - 1))
        for i, img in enumerate(images):
            row = i // nrow
            col = i % nrow
            grid_np[
                :, row * (H + 2) : row * (H + 2) + H, col * (W + 2) : col * (W + 2) + W
            ] = img

        grid_np = (
            (grid_np.permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5)
            .clip(0, 255)
            .astype("uint8")
        )
        grid_img = Image.fromarray(grid_np)

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(output_file)

    return output_file


def evaluate(
    checkpoint: str,
    output_dir: str,
    num_samples: int = 8,
    breed: int | None = None,
    image_size: int = 128,
    num_steps: int = 50,
    cfg_scale: float = 1.5,
    batch_size: int = 8,
    seed: int = 42,
    device: torch.device | None = None,
    compare_ema: bool = False,
    ema_checkpoint: str | None = None,
    all_breeds: bool = False,
    save_grid_images: bool = True,
) -> dict:
    """Main evaluation function.

    Args:
        checkpoint: Model checkpoint path.
        output_dir: Output directory.
        num_samples: Samples per breed.
        breed: Specific breed index.
        image_size: Image size.
        num_steps: Sampling steps.
        cfg_scale: CFG scale.
        batch_size: Batch size.
        seed: Random seed.
        device: Device.
        compare_ema: Compare EMA vs non-EMA.
        ema_checkpoint: EMA checkpoint path.
        all_breeds: Generate all breeds.
        save_grid_images: Save grid image.

    Returns:
        Evaluation results dict.
    """
    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, metadata = load_model(checkpoint, device)

    # Determine breeds to generate
    if breed is not None:
        breeds_to_generate = [breed]
    elif all_breeds:
        breeds_to_generate = list(range(len(CAT_BREEDS)))
    else:
        breeds_to_generate = list(range(min(8, len(CAT_BREEDS))))

    print(
        f"\nGenerating samples for breeds: {[CAT_BREEDS[i] for i in breeds_to_generate]}"
    )

    # Generate timestamp for output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    step = metadata.get("step", 0)

    results = {
        "checkpoint": checkpoint,
        "step": step,
        "breeds_generated": [],
        "samples_per_breed": num_samples,
        "output_dir": output_dir,
        "config": {
            "num_steps": num_steps,
            "cfg_scale": cfg_scale,
            "image_size": image_size,
        },
    }

    # Generate samples for each breed
    for breed_idx in breeds_to_generate:
        breed_name = CAT_BREEDS[breed_idx]
        print(f"\n{'=' * 40}")
        print(f"Generating {breed_name} samples...")

        # Create breed tensor
        breeds = torch.full((num_samples,), breed_idx, device=device)

        # Generate in batches
        all_images = []
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_breeds = breeds[start:end]

            images = generate_samples(
                model,
                batch_breeds,
                num_steps=num_steps,
                cfg_scale=cfg_scale,
                image_size=image_size,
                device=device,
            )
            all_images.append(images)

        all_images = torch.cat(all_images, dim=0)

        # Save images
        breed_output_dir = Path(output_dir) / f"step_{step}" / breed_name
        saved_files = save_images(
            all_images,
            str(breed_output_dir),
            prefix=f"breed_{breed_idx:02d}",
            breeds=breeds,
        )
        print(f"Saved {len(saved_files)} images to {breed_output_dir}")

        # Save grid
        if save_grid_images and len(all_images) > 1:
            grid_path = breed_output_dir / f"grid_{timestamp}.png"
            save_grid(all_images, str(grid_path), nrow=4, breeds=breeds)
            print(f"Saved grid to {grid_path}")

        results["breeds_generated"].append(
            {
                "breed_idx": breed_idx,
                "breed_name": breed_name,
                "num_samples": len(saved_files),
            }
        )

    # Compare EMA vs non-EMA if requested
    if compare_ema and ema_checkpoint:
        print(f"\n{'=' * 60}")
        print("Comparing EMA vs non-EMA models...")

        # Load EMA model
        ema_model, _ema_metadata = load_model(ema_checkpoint, device)

        # Generate same seed with both models
        set_seed(seed)
        test_breeds = torch.arange(min(4, len(CAT_BREEDS)), device=device)

        # Non-EMA
        non_ema_images = generate_samples(
            model,
            test_breeds,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            image_size=image_size,
            device=device,
        )

        # EMA
        set_seed(seed)  # Same seed
        ema_images = generate_samples(
            ema_model,
            test_breeds,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            image_size=image_size,
            device=device,
        )

        # Save comparison
        compare_dir = Path(output_dir) / f"step_{step}_ema_comparison"
        compare_dir.mkdir(parents=True, exist_ok=True)

        save_images(non_ema_images, str(compare_dir / "non_ema"), prefix="non_ema")
        save_images(ema_images, str(compare_dir / "ema"), prefix="ema")

        # Side-by-side grid
        combined = torch.cat([non_ema_images, ema_images], dim=0)
        grid_path = compare_dir / f"comparison_grid_{timestamp}.png"
        save_grid(combined, str(grid_path), nrow=4)

        print(f"Comparison saved to {compare_dir}")

    # Save metadata
    import json

    metadata_path = Path(output_dir) / f"step_{step}" / "generation_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Evaluation complete!")
    print(f"Total breeds generated: {len(results['breeds_generated'])}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")

    return results


def main():
    """Main entry point."""
    args = parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    )

    results = evaluate(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        breed=args.breed,
        image_size=args.image_size,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
        compare_ema=args.compare_ema,
        ema_checkpoint=args.ema_checkpoint,
        all_breeds=args.all_breeds,
        save_grid_images=args.save_grid,
    )

    print("\nGeneration Summary:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Step: {results['step']:,}")
    print(f"  Breeds: {len(results['breeds_generated'])}")
    for breed_info in results["breeds_generated"]:
        print(f"    - {breed_info['breed_name']}: {breed_info['num_samples']} samples")


if __name__ == "__main__":
    main()
