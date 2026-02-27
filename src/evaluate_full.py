"""src/evaluate_full.py

Comprehensive evaluation of TinyDiT generative model.

Computes:
- FID (Fréchet Inception Distance)
- Inception Score (IS)
- Precision/Recall for generative models
- Per-breed sample quality

Usage:
    python src/evaluate_full.py --checkpoint checkpoints/tinydit_final.pt
    python src/evaluate_full.py --generate-samples --num-samples 1000
    python src/evaluate_full.py --compute-fid --real-dir data/cats/test --fake-dir samples/

Dependencies:
    pip install pytorch-fid  # For accurate FID computation
    pip install torch-fidelity  # Alternative for IS and FID
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dit import tinydit_128
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
        description="Comprehensive evaluation of TinyDiT generative model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--generate-samples",
        action="store_true",
        help="Generate samples for evaluation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--compute-fid",
        action="store_true",
        help="Compute FID score",
    )
    parser.add_argument(
        "--real-dir",
        type=str,
        default=None,
        help="Directory with real images for FID",
    )
    parser.add_argument(
        "--fake-dir",
        type=str,
        default=None,
        help="Directory with generated images for FID",
    )
    parser.add_argument(
        "--compute-is",
        action="store_true",
        help="Compute Inception Score",
    )
    parser.add_argument(
        "--compute-precision-recall",
        action="store_true",
        help="Compute Precision and Recall",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="samples/evaluation",
        help="Output directory for generated samples",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="evaluation_report.json",
        help="Path to save evaluation report",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Image size for generation",
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
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--all-breeds",
        action="store_true",
        help="Generate samples for all breeds",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model on.

    Returns:
        Tuple of (model, checkpoint metadata).

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        KeyError: If checkpoint has invalid format.
    """
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
    num_classes = config.get("num_classes", 13)
    model = tinydit_128(num_classes=num_classes).to(device)

    # Load weights - support both checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "ema_shadow_params" in checkpoint:
        # Apply EMA weights directly
        model.load_state_dict(checkpoint["ema_shadow_params"])
    else:
        raise KeyError(
            f"Checkpoint missing model weights. Keys: {list(checkpoint.keys())}"
        )

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


def load_images_from_directory(
    directory: str | Path,
    image_size: int = 128,
    max_images: int | None = None,
) -> list[Image.Image]:
    """Load images from a directory.

    Args:
        directory: Path to directory containing images.
        image_size: Size to resize images to.
        max_images: Maximum number of images to load.

    Returns:
        List of PIL Images.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Supported image extensions
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_files = [f for f in dir_path.rglob("*") if f.suffix.lower() in extensions]

    if max_images is not None:
        image_files = image_files[:max_images]

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
        ]
    )

    images = []
    for img_path in tqdm(image_files, desc=f"Loading images from {directory}"):
        try:
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")

    print(f"Loaded {len(images)} images from {directory}")
    return images


def images_to_tensor(images: list[Image.Image], device: torch.device) -> torch.Tensor:
    """Convert list of PIL Images to tensor.

    Args:
        images: List of PIL Images.
        device: Device to put tensor on.

    Returns:
        Tensor of shape (N, C, H, W) with values in [-1, 1].
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    tensors = [transform(img).to(device) for img in images]
    return torch.stack(tensors)


def compute_fid(
    real_images: list[Image.Image],
    fake_images: list[Image.Image],
    device: torch.device,
) -> float:
    """Compute Fréchet Inception Distance.

    Uses InceptionV3 features to compute the FID between real and generated
    image distributions. Lower FID indicates better quality.

    Args:
        real_images: List of real images.
        fake_images: List of generated images.
        device: Device to run computation on.

    Returns:
        FID score (lower is better).

    Note:
        For production use, consider installing pytorch-fid:
        pip install pytorch-fid
        Then use: python -m pytorch_fid path/to/real path/to/fake
    """
    try:
        # Try to use torchvision's InceptionV3
        from torchvision.models import Inception_V3_Weights, inception_v3
    except ImportError:
        print("Warning: torchvision inception model not available")
        print("Install with: pip install torchvision")
        return float("inf")

    # Load InceptionV3 model
    weights = Inception_V3_Weights.IMAGENET1K_V1
    inception = inception_v3(weights=weights, transform_input=False).to(device)
    inception.eval()

    # Remove the final classification layer to get features
    inception.fc = torch.nn.Identity()

    # Transform for Inception (299x299, normalized for ImageNet)
    inception_transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    def get_features(images: list[Image.Image]) -> torch.Tensor:
        """Extract Inception features from images."""
        features = []
        batch_size = 32

        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            batch_tensors = torch.stack(
                [inception_transform(img).to(device) for img in batch_images]
            )

            with torch.no_grad():
                feat = inception(batch_tensors)
                features.append(feat)

        return torch.cat(features, dim=0)

    print("Computing Inception features for real images...")
    real_features = get_features(real_images)

    print("Computing Inception features for fake images...")
    fake_features = get_features(fake_images)

    # Compute statistics
    real_mean = real_features.mean(dim=0)
    real_cov = torch.cov(real_features.T)

    fake_mean = fake_features.mean(dim=0)
    fake_cov = torch.cov(fake_features.T)

    # Compute FID
    diff = real_mean - fake_mean

    # Covariance term: trace(real_cov + fake_cov - 2*sqrt(real_cov @ fake_cov))
    # Use eigendecomposition for numerical stability
    try:
        cov_mean = _sqrtm_newton_schulz(real_cov @ fake_cov, device=device)
        cov_term = torch.trace(real_cov + fake_cov - 2 * cov_mean)
    except Exception as e:
        print(f"Warning: Covariance computation failed: {e}")
        cov_term = torch.tensor(0.0, device=device)

    fid = diff.dot(diff) + cov_term

    return float(fid.cpu())


def _sqrtm_newton_schulz(
    matrix: torch.Tensor,
    device: torch.device,
    num_iters: int = 100,
) -> torch.Tensor:
    """Compute matrix square root using Newton-Schulz iteration.

    Args:
        matrix: Input matrix.
        device: Device to run on.
        num_iters: Number of iterations.

    Returns:
        Matrix square root.
    """
    A = matrix.to(device)
    n = A.size(0)

    # Normalize
    norm = A.norm()
    A = A / norm

    # Newton-Schulz iteration
    Y = torch.eye(n, device=device)
    Z = torch.eye(n, device=device)

    for _ in range(num_iters):
        T = 0.5 * (3.0 * torch.eye(n, device=device) - Z @ A)
        Y = Y @ T
        Z = T @ Z

    # Denormalize
    return Y * torch.sqrt(norm)


def compute_inception_score(
    fake_images: list[Image.Image],
    device: torch.device,
    splits: int = 10,
) -> tuple[float, float]:
    """Compute Inception Score.

    The Inception Score measures the quality and diversity of generated images.
    Higher IS indicates better quality and diversity.

    Args:
        fake_images: List of generated images.
        device: Device to run computation on.
        splits: Number of splits for IS calculation.

    Returns:
        Tuple of (IS mean, IS std) - higher is better.
    """
    try:
        from torchvision.models import Inception_V3_Weights, inception_v3
    except ImportError:
        print("Warning: torchvision inception model not available")
        return (0.0, 0.0)

    # Load InceptionV3 model
    weights = Inception_V3_Weights.IMAGENET1K_V1
    inception = inception_v3(weights=weights, transform_input=False).to(device)
    inception.eval()

    # Transform for Inception
    inception_transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Get predictions
    preds = []
    batch_size = 32

    for i in range(0, len(fake_images), batch_size):
        batch_images = fake_images[i : i + batch_size]
        batch_tensors = torch.stack(
            [inception_transform(img).to(device) for img in batch_images]
        )

        with torch.no_grad():
            pred = torch.softmax(inception(batch_tensors), dim=1)
            preds.append(pred.cpu())

    preds = torch.cat(preds, dim=0).numpy()

    # Compute IS
    scores = []
    split_size = len(preds) // splits

    for i in range(splits):
        start = i * split_size
        end = start + split_size if i < splits - 1 else len(preds)
        split_preds = preds[start:end]

        # p(y|x) marginal
        py_x = split_preds

        # p(y) marginal
        py = py_x.mean(axis=0, keepdims=True)

        # KL divergence
        kl_div = py_x * (np.log(py_x + 1e-10) - np.log(py + 1e-10))
        kl_sum = kl_div.sum(axis=1)

        scores.append(np.exp(kl_sum.mean()))

    is_mean = float(np.mean(scores))
    is_std = float(np.std(scores))

    return (is_mean, is_std)


def compute_precision_recall(
    real_images: list[Image.Image],
    fake_images: list[Image.Image],
    device: torch.device,
    k: int = 3,
) -> tuple[float, float]:
    """Compute Precision and Recall for generative models.

    Precision measures how many generated images are close to real images.
    Recall measures how well generated images cover the real distribution.

    Args:
        real_images: Real images.
        fake_images: Generated images.
        device: Device to run computation on.
        k: Number of nearest neighbors.

    Returns:
        Tuple of (precision, recall) - both in [0, 1].
    """
    try:
        from torchvision.models import Inception_V3_Weights, inception_v3
    except ImportError:
        print("Warning: torchvision inception model not available")
        return (0.0, 0.0)

    # Load InceptionV3 model
    weights = Inception_V3_Weights.IMAGENET1K_V1
    inception = inception_v3(weights=weights, transform_input=False).to(device)
    inception.eval()

    # Transform for Inception
    inception_transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    def get_features(images: list[Image.Image]) -> torch.Tensor:
        """Extract Inception features."""
        features = []
        batch_size = 32

        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            batch_tensors = torch.stack(
                [inception_transform(img).to(device) for img in batch_images]
            )

            with torch.no_grad():
                feat = inception(batch_tensors)
                features.append(feat)

        return torch.cat(features, dim=0)

    print("Extracting features for precision/recall...")
    real_features = get_features(real_images)
    fake_features = get_features(fake_images)

    # Compute pairwise distances
    # Precision: for each fake, is it close to any real?
    # Recall: for each real, is it close to any fake?

    # Use simplified approach: compute distances and thresholds
    real_np = real_features.cpu().numpy()
    fake_np = fake_features.cpu().numpy()

    # Compute distance matrix
    from scipy.spatial.distance import cdist

    distances = cdist(fake_np, real_np, metric="euclidean")

    # For each fake image, find k nearest real neighbors
    # Precision = fraction of fakes that have a close real neighbor
    k_distances_real = np.sort(distances, axis=1)[:, k - 1]
    threshold = np.percentile(k_distances_real, 50)  # Median distance

    precision = float(np.mean(k_distances_real < threshold))

    # For recall, compute distances from real to fake
    distances_rt = cdist(real_np, fake_np, metric="euclidean")
    k_distances_fake = np.sort(distances_rt, axis=1)[:, k - 1]

    recall = float(np.mean(k_distances_fake < threshold))

    return (precision, recall)


def generate_samples_per_breed(
    model: torch.nn.Module,
    device: torch.device,
    num_samples_per_breed: int = 8,
    output_dir: str | Path = "samples/evaluation",
    image_size: int = 128,
    num_steps: int = 50,
    cfg_scale: float = 1.5,
    batch_size: int = 8,
    seed: int = 42,
) -> dict[str, list[Image.Image]]:
    """Generate samples for each breed.

    Args:
        model: TinyDiT model.
        device: Device to run on.
        num_samples_per_breed: Samples per breed.
        output_dir: Directory to save samples.
        image_size: Image size.
        num_steps: Sampling steps.
        cfg_scale: CFG scale.
        batch_size: Batch size.
        seed: Random seed.

    Returns:
        Dict mapping breed name to list of images.
    """
    set_seed(seed)
    model.eval()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[Image.Image]] = {}

    for breed_idx, breed_name in enumerate(CAT_BREEDS):
        print(f"\nGenerating {breed_name} samples...")

        # Create breed tensor
        breeds = torch.full((num_samples_per_breed,), breed_idx, device=device)

        # Generate in batches
        all_images = []
        for start in range(0, num_samples_per_breed, batch_size):
            end = min(start + batch_size, num_samples_per_breed)
            batch_breeds = breeds[start:end]

            images = sample(
                model,
                batch_breeds,
                num_steps=num_steps,
                device=device,
                image_size=image_size,
                cfg_scale=cfg_scale,
                progress=True,
            )
            all_images.append(images)

        all_images = torch.cat(all_images, dim=0)

        # Convert to PIL Images
        pil_images = []
        for i, img in enumerate(all_images):
            img_np = (
                (img.permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5)
                .clip(0, 255)
                .astype("uint8")
            )
            pil_img = Image.fromarray(img_np)
            pil_images.append(pil_img)

            # Save individual image
            img_path = output_path / f"{breed_name}_{i:03d}.png"
            pil_img.save(img_path)

        results[breed_name] = pil_images
        print(f"  Saved {len(pil_images)} images for {breed_name}")

    return results


def generate_all_samples(
    model: torch.nn.Module,
    device: torch.device,
    num_samples: int,
    output_dir: str | Path,
    image_size: int = 128,
    num_steps: int = 50,
    cfg_scale: float = 1.5,
    batch_size: int = 8,
    seed: int = 42,
) -> list[Image.Image]:
    """Generate all samples (mixed breeds).

    Args:
        model: TinyDiT model.
        device: Device to run on.
        num_samples: Total number of samples.
        output_dir: Directory to save samples.
        image_size: Image size.
        num_steps: Sampling steps.
        cfg_scale: CFG scale.
        batch_size: Batch size.
        seed: Random seed.

    Returns:
        List of generated PIL Images.
    """
    set_seed(seed)
    model.eval()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_images = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx

        # Random breed assignment
        breeds = torch.randint(0, len(CAT_BREEDS), (current_batch_size,), device=device)

        images = sample(
            model,
            breeds,
            num_steps=num_steps,
            device=device,
            image_size=image_size,
            cfg_scale=cfg_scale,
            progress=True,
        )

        # Convert to PIL Images
        for i, img in enumerate(images):
            img_np = (
                (img.permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5)
                .clip(0, 255)
                .astype("uint8")
            )
            pil_img = Image.fromarray(img_np)
            all_images.append(pil_img)

            # Save individual image
            global_idx = start_idx + i
            img_path = output_path / f"sample_{global_idx:05d}.png"
            pil_img.save(img_path)

        print(f"Generated batch {batch_idx + 1}/{num_batches}")

    return all_images


def generate_evaluation_report(
    fid: float | None,
    inception_score: tuple[float, float] | None,
    precision: float | None,
    recall: float | None,
    output_path: str | Path = "evaluation_report.json",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Generate JSON evaluation report.

    Args:
        fid: FID score.
        inception_score: (mean, std).
        precision: Precision score.
        recall: Recall score.
        output_path: Path to save report.
        metadata: Additional metadata to include.
    """
    report: dict[str, Any] = {}

    if fid is not None:
        report["fid"] = fid

    if inception_score is not None:
        report["inception_score"] = {
            "mean": inception_score[0],
            "std": inception_score[1],
        }

    if precision is not None:
        report["precision"] = precision

    if recall is not None:
        report["recall"] = recall

    if metadata:
        report["metadata"] = metadata

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Evaluation report saved to {output_path}")
    print(f"{'=' * 60}")

    if fid is not None:
        print(f"  FID: {fid:.2f} (lower is better)")
    if inception_score is not None:
        print(
            f"  Inception Score: {inception_score[0]:.2f} (+/- {inception_score[1]:.2f})"
            " (higher is better)"
        )
    if precision is not None:
        print(f"  Precision: {precision:.3f}")
    if recall is not None:
        print(f"  Recall: {recall:.3f}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize results
    fid_result: float | None = None
    is_result: tuple[float, float] | None = None
    pr_result: tuple[float, float] | None = None
    metadata: dict[str, Any] = {}

    # Load model if checkpoint provided
    model: torch.nn.Module | None = None
    if args.checkpoint:
        try:
            model, model_metadata = load_model(args.checkpoint, device)
            metadata.update(model_metadata)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except KeyError as e:
            print(f"Error loading checkpoint: {e}")
            sys.exit(1)

    # Generate samples if requested
    if args.generate_samples and model is not None:
        print(f"\n{'=' * 60}")
        print("Generating samples...")
        print(f"{'=' * 60}")

        if args.all_breeds:
            # Generate per-breed samples
            samples_per_breed = generate_samples_per_breed(
                model=model,
                device=device,
                num_samples_per_breed=args.num_samples // len(CAT_BREEDS),
                output_dir=args.output_dir,
                image_size=args.image_size,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                batch_size=args.batch_size,
                seed=args.seed,
            )

            # Flatten for metrics
            all_generated = []
            for breed_images in samples_per_breed.values():
                all_generated.extend(breed_images)
        else:
            all_generated = generate_all_samples(
                model=model,
                device=device,
                num_samples=args.num_samples,
                output_dir=args.output_dir,
                image_size=args.image_size,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                batch_size=args.batch_size,
                seed=args.seed,
            )

        metadata["num_samples_generated"] = len(all_generated)
        print(f"\nGenerated {len(all_generated)} samples to {args.output_dir}")

    # Compute FID if requested
    if args.compute_fid:
        if not args.real_dir or not args.fake_dir:
            print("Error: --real-dir and --fake-dir required for FID computation")
            sys.exit(1)

        print(f"\n{'=' * 60}")
        print("Computing FID...")
        print(f"{'=' * 60}")

        real_images = load_images_from_directory(args.real_dir, args.image_size)
        fake_images = load_images_from_directory(args.fake_dir, args.image_size)

        if len(real_images) == 0 or len(fake_images) == 0:
            print("Error: No images found in directories")
            sys.exit(1)

        fid_result = compute_fid(real_images, fake_images, device)
        print(f"FID: {fid_result:.2f}")

    # Compute Inception Score if requested
    if args.compute_is:
        if args.fake_dir:
            fake_images = load_images_from_directory(args.fake_dir, args.image_size)
        elif args.generate_samples and "all_generated" in locals():
            fake_images = all_generated
        else:
            print("Error: No generated images available for IS computation")
            sys.exit(1)

        print(f"\n{'=' * 60}")
        print("Computing Inception Score...")
        print(f"{'=' * 60}")

        is_result = compute_inception_score(fake_images, device)
        print(f"Inception Score: {is_result[0]:.2f} (+/- {is_result[1]:.2f})")

    # Compute Precision/Recall if requested
    if args.compute_precision_recall:
        if not args.real_dir or not args.fake_dir:
            print("Error: --real-dir and --fake-dir required for P/R computation")
            sys.exit(1)

        print(f"\n{'=' * 60}")
        print("Computing Precision and Recall...")
        print(f"{'=' * 60}")

        real_images = load_images_from_directory(args.real_dir, args.image_size)
        fake_images = load_images_from_directory(args.fake_dir, args.image_size)

        pr_result = compute_precision_recall(real_images, fake_images, device)
        print(f"Precision: {pr_result[0]:.3f}, Recall: {pr_result[1]:.3f}")

    # Generate report
    generate_evaluation_report(
        fid=fid_result,
        inception_score=is_result,
        precision=pr_result[0] if pr_result else None,
        recall=pr_result[1] if pr_result else None,
        output_path=args.report_path,
        metadata=metadata,
    )


# Import numpy for IS computation
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

# Import scipy for precision/recall
try:
    from scipy.spatial.distance import cdist
except ImportError:
    cdist = None  # type: ignore


if __name__ == "__main__":
    main()
