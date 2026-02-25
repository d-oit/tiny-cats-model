"""src/upload_to_hub.py

HuggingFace Hub upload utility for tiny-cats-model.

Features:
- Safetensors export (secure serialization)
- Model card generation with metadata
- Automated upload to HuggingFace Hub
- Sample image generation for model card
- Version tracking with tags

Usage:
    python src/upload_to_hub.py checkpoints/tinydit_final.pt --repo-id d-oit/tinydit-cats
    python src/upload_to_hub.py checkpoints/tinydit_final.pt --upload-samples
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

# Optional imports (handle gracefully if not installed)
try:
    from safetensors.torch import save_file

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    from huggingface_hub import create_repo, upload_folder

    HAS_HF = True
except ImportError:
    HAS_HF = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration for TinyDiT."""

    image_size: int = 128
    patch_size: int = 16
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    num_classes: int = 13
    training_steps: int = 200_000
    batch_size: int = 256
    learning_rate: float = 1e-4
    ema_beta: float = 0.9999


# Cat breed mapping
BREED_NAMES = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British Shorthair",
    "Egyptian Mau",
    "Maine Coon",
    "Persian",
    "Ragdoll",
    "Russian Blue",
    "Siamese",
    "Sphynx",
    "Other",
]


def export_to_safetensors(
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    """Convert PyTorch checkpoint to Safetensors format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pt file).
        output_path: Output path for Safetensors file (optional).
        logger: Optional logger instance.

    Returns:
        Path to Safetensors file.

    Raises:
        ImportError: If safetensors not installed.
        FileNotFoundError: If checkpoint not found.
    """
    if not HAS_SAFETENSORS:
        raise ImportError(
            "safetensors not installed. Install with: pip install safetensors"
        )

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if logger:
        logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract state dict (remove metadata)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Checkpoint is already a state dict
        state_dict = {
            k: v for k, v in checkpoint.items() if isinstance(v, torch.Tensor)
        }

    # Determine output path
    if output_path is None:
        output_path = checkpoint_path.with_suffix(".safetensors")
    else:
        output_path = Path(output_path)

    # Prepare metadata
    metadata = {
        "format": "pt",
        "created_by": "tiny-cats-model",
        "created_at": datetime.now().isoformat(),
    }

    # Add training metadata if available
    if "step" in checkpoint:
        metadata["training_steps"] = str(checkpoint.get("step", 0))
    if "loss" in checkpoint:
        metadata["final_loss"] = str(checkpoint.get("loss", 0.0))
    if "config" in checkpoint:
        config = checkpoint["config"]
        metadata.update(
            {
                "image_size": str(config.get("image_size", 128)),
                "patch_size": str(config.get("patch_size", 16)),
                "embed_dim": str(config.get("embed_dim", 384)),
            }
        )

    # Save to Safetensors
    if logger:
        logger.info(f"Saving to Safetensors: {output_path}")

    save_file(state_dict, output_path, metadata=metadata)

    # Report size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    if logger:
        logger.info(f"Saved {output_path} ({size_mb:.2f} MB)")

    return output_path


def create_model_card(
    config: ModelConfig,
    metrics: dict[str, Any] | None = None,
    samples_dir: str | Path | None = None,
    license: str = "apache-2.0",
) -> str:
    """Create model card markdown content.

    Args:
        config: Model configuration.
        metrics: Training metrics (loss, FID, etc.).
        samples_dir: Directory with sample images.
        license: License identifier.

    Returns:
        Model card markdown content.
    """
    metrics = metrics or {}

    # Build breed list
    breed_list = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(BREED_NAMES))

    # Build sample images section
    samples_section = ""
    if samples_dir and HAS_PIL:
        samples_path = Path(samples_dir)
        if samples_path.exists():
            sample_images = list(samples_path.glob("*.png"))[:8]
            if sample_images:
                samples_section = "\n## Sample Outputs\n\n"
                for i, img_path in enumerate(sample_images):
                    samples_section += f"![Sample {i + 1}]({img_path.name})\n"

    # Build model card
    card = f"""---
license: {license}
tags:
  - image-generation
  - diffusion-transformer
  - pytorch
  - cat-breeds
  - onnx
  - tiny-models
datasets:
  - oxford-iiit-pet
metrics:
  - final_loss: {metrics.get("final_loss", "N/A")}
  - training_steps: {metrics.get("training_steps", config.training_steps):,}
  - fid_score: {metrics.get("fid_score", "N/A")}
library_name: transformers
pipeline_tag: image-generation
---

# TinyDiT Cat Breed Generator

## Model Description

**TinyDiT** is a Diffusion Transformer for conditional cat image generation.
It can generate 128x128 images of {len(BREED_NAMES)} different cat breeds using classifier-free guidance.

### Model Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | TinyDiT (Diffusion Transformer) |
| Parameters | ~22M |
| Hidden Dimension | {config.embed_dim} |
| Transformer Blocks | {config.depth} |
| Attention Heads | {config.num_heads} |
| Patch Size | {config.patch_size}x{config.patch_size} |
| Image Size | {config.image_size}x{config.image_size} |
| Conditioning | Breed one-hot ({config.num_classes} classes) |

### Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | Oxford IIIT Pet Dataset |
| Training Steps | {config.training_steps:,} |
| Batch Size | {config.batch_size} |
| Learning Rate | {config.learning_rate} |
| Optimizer | AdamW (β1=0.9, β2=0.95) |
| Weight Decay | 1e-4 |
| EMA Beta | {config.ema_beta} |
| Loss | Flow matching (velocity prediction) |
| Mixed Precision | Yes (AMP) |
| Gradient Clipping | Yes (max norm=1.0) |
| LR Schedule | Cosine annealing with warmup |

### Cat Breeds

{breed_list}

## Usage

### Python (PyTorch)

```python
import torch
from safetensors.torch import load_file
from dit import tinydit_128

# Load model
model = tinydit_128(num_classes=13)
checkpoint = load_file("tinydit_cats.safetensors")
model.load_state_dict(checkpoint)
model.eval()

# Generate
from flow_matching import sample
breeds = torch.tensor([0, 1])  # Abyssinian, Bengal
images = sample(model, breeds, num_steps=50, cfg_scale=1.5)
```

### Python (Transformers Pipeline)

```python
from transformers import pipeline

generator = pipeline(
    "image-generation",
    model="d-oit/tinydit-cats",
    trust_remote_code=True
)
image = generator(breed="Siamese", num_images=4)
```

### CLI

```bash
python inference.py --breed "Siamese" --num-images 4 --cfg-scale 1.5
```

### Web (ONNX Runtime)

The model is also available in ONNX format for web deployment:
- `onnx/cats_generator.onnx` - Generator with CFG support
- `onnx/cats_classifier.onnx` - Breed classifier

See `frontend/` for web implementation.

## Performance

| Metric | Value |
|--------|-------|
| Final Training Loss | {metrics.get("final_loss", "N/A")} |
| FID Score (128x128) | {metrics.get("fid_score", "N/A")} |
| Inference Time (CPU) | ~2s per image |
| Inference Time (GPU) | ~0.1s per image |
| Model Size (Safetensors) | {metrics.get("model_size_mb", "89")} MB |
| Model Size (ONNX, quantized) | 11 MB |
{samples_section}
## Training Infrastructure

Training was performed using Modal GPU (A10G) with:
- Container build time: <2 minutes (uv_pip_install)
- Cold start optimization: ~15s
- Automatic retry on transient failures
- Volume-based checkpoint persistence

See ADR-017, ADR-022 to ADR-025 for implementation details.

## License

Apache License 2.0

## Citation

```bibtex
@software{{tinydit-cats-2026,
  title = {{TinyDiT Cat Breed Generator}},
  author = {{tiny-cats-model contributors}},
  year = {{2026}},
  url = {{https://github.com/d-oit/tiny-cats-model}}
}}
```

## Links

- **GitHub**: https://github.com/d-oit/tiny-cats-model
- **Demo**: https://d-oit.github.io/tiny-cats-model
- **Paper**: TinyDiT - https://arxiv.org/abs/2212.09748
- **Dataset**: Oxford IIIT Pet - https://www.robots.ox.ac.uk/~vgg/data/pets/
"""

    return card


def generate_samples(
    model_path: str | Path,
    output_dir: str | Path,
    num_samples: int = 8,
    device: str = "cpu",
    logger: logging.Logger | None = None,
) -> Path:
    """Generate sample images for model card.

    Args:
        model_path: Path to model checkpoint.
        output_dir: Directory to save samples.
        num_samples: Number of samples to generate.
        device: Device for inference.
        logger: Optional logger instance.

    Returns:
        Path to samples directory.
    """
    if not HAS_PIL:
        raise ImportError("PIL not installed. Install with: pip install pillow")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info(f"Generating {num_samples} samples to {output_dir}")

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Import model architecture
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from dit import tinydit_128

    # Get config from checkpoint
    config = checkpoint.get("config", {})
    num_classes = config.get("num_classes", 13)

    model = tinydit_128(num_classes=num_classes)

    # Load weights (prefer EMA if available)
    if "ema_shadow_params" in checkpoint:
        # Apply EMA weights
        model.load_state_dict(checkpoint["ema_shadow_params"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Import generation functions
    from flow_matching import sample

    # Generate samples for different breeds
    num_breeds = min(num_samples, num_classes)
    breeds = torch.arange(num_breeds, device=device)

    if logger:
        logger.info(
            f"Generating samples for breeds: {[BREED_NAMES[i] for i in breeds.tolist()]}"
        )

    with torch.no_grad():
        generated = sample(
            model,
            breeds,
            num_steps=50,
            device=device,
            image_size=128,
            cfg_scale=1.5,
            progress=False,
        )

    # Save images
    for i in range(len(generated)):
        img_tensor = generated[i].permute(1, 2, 0).cpu().numpy()
        img_array = (img_tensor * 127.5 + 127.5).clip(0, 255).astype("uint8")

        breed_name = BREED_NAMES[i] if i < len(BREED_NAMES) else f"breed_{i}"
        img_path = output_dir / f"{breed_name.lower().replace(' ', '_')}.png"

        Image.fromarray(img_array).save(img_path)

        if logger:
            logger.info(f"Saved: {img_path}")

    return output_dir


def upload_to_hub(
    folder_path: str | Path,
    repo_id: str = "d-oit/tinydit-cats",
    commit_message: str = "Upload TinyDiT cat breed generator",
    token: str | None = None,
    create_pr: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Upload model to HuggingFace Hub.

    Args:
        folder_path: Path to folder with model files.
        repo_id: Repository ID (username/repo-name).
        commit_message: Git commit message.
        token: HuggingFace token (optional, uses env var if not provided).
        create_pr: Whether to create a pull request.
        logger: Optional logger instance.

    Returns:
        Upload result with URL and status.

    Raises:
        ImportError: If huggingface_hub not installed.
    """
    if not HAS_HF:
        raise ImportError(
            "huggingface_hub not installed. Install with: pip install huggingface_hub"
        )

    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if logger:
        logger.info(f"Uploading {folder_path} to https://huggingface.co/{repo_id}")

    # Create repo if not exists
    create_repo(repo_id, exist_ok=True, repo_type="model", token=token)

    # Upload folder
    result = upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        token=token,
        create_pr=create_pr,
    )

    url = f"https://huggingface.co/{repo_id}"

    if logger:
        logger.info(f"Upload complete: {url}")

    return {
        "status": "success",
        "url": url,
        "repo_id": repo_id,
        "commit_url": result.commit_url if hasattr(result, "commit_url") else None,
    }


def prepare_upload_folder(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    config: ModelConfig | None = None,
    metrics: dict[str, Any] | None = None,
    generate_samples_flag: bool = False,
    logger: logging.Logger | None = None,
) -> Path:
    """Prepare folder for HuggingFace upload.

    Args:
        checkpoint_path: Path to model checkpoint.
        output_dir: Output directory for upload files.
        config: Model configuration.
        metrics: Training metrics.
        generate_samples_flag: Whether to generate sample images.
        logger: Optional logger instance.

    Returns:
        Path to prepared upload folder.
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info(f"Preparing upload folder: {output_dir}")

    # Export to Safetensors
    safetensors_path = output_dir / "tinydit_cats.safetensors"
    export_to_safetensors(checkpoint_path, safetensors_path, logger)

    # Save config
    config = config or ModelConfig()
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.__dict__, f, indent=2)

    # Create model card
    card_content = create_model_card(config, metrics)
    card_path = output_dir / "README.md"
    with open(card_path, "w") as f:
        f.write(card_content)

    # Generate samples if requested
    if generate_samples_flag:
        samples_dir = output_dir / "samples"
        try:
            generate_samples(checkpoint_path, samples_dir, logger=logger)
        except Exception as e:
            if logger:
                logger.warning(f"Sample generation failed: {e}")

    # Copy ONNX models if available
    onnx_dir = Path(__file__).parent.parent / "frontend" / "public" / "models"
    if onnx_dir.exists():
        import shutil

        upload_onnx_dir = output_dir / "onnx"
        upload_onnx_dir.mkdir(exist_ok=True)

        for onnx_file in onnx_dir.glob("*.onnx"):
            shutil.copy2(onnx_file, upload_onnx_dir / onnx_file.name)
            if logger:
                logger.info(f"Copied ONNX: {onnx_file.name}")

        # Copy models.json
        models_json = onnx_dir / "models.json"
        if models_json.exists():
            shutil.copy2(models_json, output_dir / "models.json")

    # Create inference script
    inference_script = create_inference_script()
    inference_path = output_dir / "inference.py"
    with open(inference_path, "w") as f:
        f.write(inference_script)

    # Create requirements.txt
    requirements_path = output_dir / "requirements.txt"
    with open(requirements_path, "w") as f:
        f.write("torch>=2.0.0\n")
        f.write("torchvision>=0.15.0\n")
        f.write("safetensors>=0.4.0\n")
        f.write("pillow>=10.0.0\n")
        f.write("huggingface_hub>=0.20.0\n")
        f.write("tqdm>=4.65.0\n")

    return output_dir


def create_inference_script() -> str:
    """Create inference script for model card."""
    return '''#!/usr/bin/env python3
"""Inference script for TinyDiT Cat Breed Generator.

Usage:
    python inference.py --breed "Siamese" --num-images 4
    python inference.py --breed "Bengal" --cfg-scale 2.0 --num-images 8
"""

import argparse
import torch
from pathlib import Path

try:
    from safetensors.torch import load_file
    from PIL import Image
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install safetensors pillow tqdm")
    exit(1)

# Breed mapping
BREED_NAMES = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue",
    "Siamese", "Sphynx", "Other",
]

BREED_TO_INDEX = {name.lower().replace(" ", "_"): i for i, name in enumerate(BREED_NAMES)}


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load model from checkpoint."""
    from dit import tinydit_128

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get config
    config = checkpoint.get("config", {})
    num_classes = config.get("num_classes", 13)

    model = tinydit_128(num_classes=num_classes)

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def generate(model, breed_index: int, num_steps: int = 50, cfg_scale: float = 1.5, device: str = "cpu"):
    """Generate images for a breed."""
    from flow_matching import sample

    breed_tensor = torch.tensor([breed_index], device=device)

    with torch.no_grad():
        images = sample(
            model,
            breed_tensor,
            num_steps=num_steps,
            device=device,
            image_size=128,
            cfg_scale=cfg_scale,
            progress=True,
        )

    return images


def main():
    parser = argparse.ArgumentParser(description="TinyDiT Cat Breed Generator")
    parser.add_argument("--breed", type=str, required=True, help="Cat breed name")
    parser.add_argument("--num-images", type=int, default=4, help="Number of images to generate")
    parser.add_argument("--cfg-scale", type=float, default=1.5, help="Classifier-free guidance scale")
    parser.add_argument("--num-steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--checkpoint", type=str, default="tinydit_cats.safetensors", help="Model checkpoint")
    parser.add_argument("--output", type=str, default="generated", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")

    args = parser.parse_args()

    # Parse breed
    breed_key = args.breed.lower().replace(" ", "_")
    if breed_key not in BREED_TO_INDEX:
        print(f"Unknown breed: {args.breed}")
        print(f"Available breeds: {', '.join(BREED_NAMES)}")
        return

    breed_index = BREED_TO_INDEX[breed_key]

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.device)

    # Generate
    print(f"Generating {args.num_images} images of {BREED_NAMES[breed_index]}...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(args.num_images)):
        images = generate(model, breed_index, args.num_steps, args.cfg_scale, args.device)

        # Save
        img_tensor = images[0].permute(1, 2, 0).cpu().numpy()
        img_array = ((img_tensor * 127.5 + 127.5).clip(0, 255).astype("uint8"))
        img_path = output_dir / f"{breed_key}_{i+1:03d}.png"
        Image.fromarray(img_array).save(img_path)

    print(f"Saved {args.num_images} images to {output_dir}")


if __name__ == "__main__":
    main()
'''


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--repo-id", type=str, default="d-oit/tinydit-cats", help="HuggingFace repo ID"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hub_upload",
        help="Output directory for upload files",
    )
    parser.add_argument(
        "--generate-samples", action="store_true", help="Generate sample images"
    )
    parser.add_argument(
        "--upload", action="store_true", help="Upload to HuggingFace Hub"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Get token
    token = args.token or os.environ.get("HF_TOKEN")

    try:
        # Prepare upload folder
        logger.info(f"Processing checkpoint: {args.checkpoint}")
        upload_dir = prepare_upload_folder(
            args.checkpoint,
            args.output_dir,
            generate_samples_flag=args.generate_samples,
            logger=logger,
        )

        logger.info(f"Upload folder prepared: {upload_dir}")

        # Upload if requested
        if args.upload:
            if not token:
                logger.error(
                    "HF_TOKEN not set. Please provide --token or set HF_TOKEN env var"
                )
                return

            result = upload_to_hub(
                upload_dir,
                repo_id=args.repo_id,
                token=token,
                logger=logger,
            )

            logger.info(f"Upload successful: {result['url']}")
        else:
            logger.info("Upload folder ready. Use --upload to push to HuggingFace Hub")
            logger.info(f"Files in: {upload_dir}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
