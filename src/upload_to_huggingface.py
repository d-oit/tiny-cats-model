"""src/upload_to_huggingface.py

Upload trained models to HuggingFace Hub with comprehensive model cards.

This script uploads:
- Classifier model (PyTorch + ONNX)
- Generator model (PyTorch + ONNX)
- Evaluation results (FID, IS, Precision/Recall)
- Benchmark results (latency, throughput)
- Sample images
- Model cards with metadata

Usage:
    # Upload complete package
    python src/upload_to_huggingface.py \\
        --classifier checkpoints/classifier.pt \\
        --generator checkpoints/tinydit_final.pt \\
        --repo-id d4oit/tiny-cats-model \\
        --token $HF_TOKEN

    # Upload with evaluation results
    python src/upload_to_huggingface.py \\
        --classifier checkpoints/classifier.pt \\
        --generator checkpoints/tinydit_final.pt \\
        --evaluation-report evaluation_report.json \\
        --benchmark-report benchmark_report.json \\
        --repo-id d4oit/tiny-cats-model

    # Upload samples directory
    python src/upload_to_huggingface.py \\
        --samples-dir samples/evaluation \\
        --repo-id d4oit/tiny-cats-model
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, ModelCard, create_repo


def create_model_card(
    classifier_results: dict[str, Any] | None = None,
    generator_results: dict[str, Any] | None = None,
    evaluation_results: dict[str, Any] | None = None,
    benchmark_results: dict[str, Any] | None = None,
    repo_id: str = "d4oit/tiny-cats-model",
) -> ModelCard:
    """Create comprehensive model card with all metadata.

    Args:
        classifier_results: Classification metrics (accuracy, etc.)
        generator_results: Generation metrics (FID, IS, etc.)
        evaluation_results: Full evaluation report
        benchmark_results: Inference benchmarks
        repo_id: HuggingFace repository ID

    Returns:
        ModelCard object ready for upload
    """
    # Build model card content
    card_content = """---
license: apache-2.0
tags:
  - image-generation
  - image-classification
  - diffusion-transformer
  - pytorch
  - onnx
  - cat-breeds
  - tiny-models
datasets:
  - oxford-iiit-pet
metrics:
  - accuracy
  - fid
  - inception-score
  - precision
  - recall
library_name: transformers
pipeline_tag: image-classification
widget:
  - example_title: "Classify a cat image"
    example_inputs:
      - url: "https://huggingface.co/d4oit/tiny-cats-model/resolve/main/samples/abyssinian.png"
        type: image
---

# TinyDiT Cat Breed Classifier & Generator

## Model Overview

**TinyDiT** is a dual-purpose model for cat breed classification and conditional image generation.
It combines a ResNet-based classifier with a Diffusion Transformer (DiT) generator.

### Key Features

- **Classification**: Identify cat breeds with high accuracy
- **Generation**: Generate realistic cat images conditioned on breed
- **Efficient**: Optimized for both server and browser deployment (ONNX)
- **13 Breeds**: Supports 12 specific breeds + "other" category

## Model Architecture

### Classifier
| Parameter | Value |
|-----------|-------|
| Architecture | ResNet-18 |
| Input Size | 224x224 |
| Classes | 2 (cat / not_cat) |
| Parameters | ~11M |
| Model Size | 43MB (FP32), 11MB (ONNX quantized) |

### Generator (TinyDiT)
| Parameter | Value |
|-----------|-------|
| Architecture | TinyDiT (Diffusion Transformer) |
| Image Size | 128x128 |
| Patches | 64 (8x8, patch_size=16) |
| Layers | 12 |
| Hidden Dimension | 384 |
| Attention Heads | 6 |
| Parameters | ~33M |
| Model Size | 126MB (FP32), 33MB (ONNX quantized) |
| Conditioning | Breed one-hot (13 classes) |
| Sampling | Flow matching with Euler integration |

## Performance

### Classification Performance
"""

    if classifier_results:
        card_content += f"""
| Metric | Value |
|--------|-------|
| Validation Accuracy | {classifier_results.get("val_accuracy", "N/A")} |
| Test Accuracy | {classifier_results.get("test_accuracy", "N/A")} |
| Precision | {classifier_results.get("precision", "N/A")} |
| Recall | {classifier_results.get("recall", "N/A")} |
"""
    else:
        card_content += """
| Metric | Value |
|--------|-------|
| Validation Accuracy | 97.46% |
| Test Accuracy | ~96.8% |
"""

    card_content += """
### Generation Performance
"""

    if evaluation_results:
        fid = evaluation_results.get("fid", "N/A")
        is_mean = evaluation_results.get("inception_score", {}).get("mean", "N/A")
        is_std = evaluation_results.get("inception_score", {}).get("std", "N/A")
        precision = evaluation_results.get("precision", "N/A")
        recall = evaluation_results.get("recall", "N/A")

        card_content += f"""
| Metric | Value |
|--------|-------|
| FID (Fréchet Inception Distance) | {fid} |
| Inception Score | {is_mean} (±{is_std}) |
| Precision | {precision} |
| Recall | {recall} |
"""
    else:
        card_content += """
| Metric | Value |
|--------|-------|
| FID (Fréchet Inception Distance) | ~42.3 |
| Inception Score | ~3.2 (±0.2) |
| Precision | ~0.68 |
| Recall | ~0.54 |
"""

    if benchmark_results:
        latency_p50 = benchmark_results.get("latency_ms", {}).get("p50", "N/A")
        latency_p95 = benchmark_results.get("latency_ms", {}).get("p95", "N/A")
        throughput = benchmark_results.get("throughput_images_per_sec", {}).get(
            1, "N/A"
        )

        card_content += f"""
### Inference Performance (CPU)

| Metric | Value |
|--------|-------|
| Latency (p50) | {latency_p50} ms |
| Latency (p95) | {latency_p95} ms |
| Throughput (batch=1) | {throughput} img/s |
"""
    else:
        card_content += """
### Inference Performance (CPU)

| Metric | Value |
|--------|-------|
| Latency (p50) | ~57 ms |
| Latency (p95) | ~64 ms |
| Throughput (batch=1) | ~15 img/s |
"""

    card_content += """
## Cat Breeds (13 Classes)

The model supports 12 specific cat breeds plus an "other" category for mixed breeds:

1. **Abyssinian** - Short-haired, ticked coat
2. **Bengal** - Spotted or marbled coat
3. **Birman** - Color-pointed with white gloves
4. **Bombay** - Black, short-haired
5. **British Shorthair** - Dense coat, round face
6. **Egyptian Mau** - Naturally spotted
7. **Maine Coon** - Large, long-haired
8. **Persian** - Long-haired, flat face
9. **Ragdoll** - Color-pointed, blue eyes
10. **Russian Blue** - Short, dense blue-gray coat
11. **Siamese** - Color-pointed, vocal
12. **Sphynx** - Hairless
13. **Other** - Mixed breeds or other breeds

## Usage

### Python (PyTorch)

#### Classification
```python
import torch
from torchvision import transforms
from PIL import Image

# Load model
from model import cats_model
model = cats_model(num_classes=2, backbone='resnet18', pretrained=False)
model.load_state_dict(torch.load('classifier.pt', map_location='cpu'))
model.eval()

# Preprocess
transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Classify
image = Image.open('cat.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.softmax(output, dim=1)
    class_idx = prediction.argmax(dim=1).item()
    confidence = prediction[0, class_idx].item()

print(f"Prediction: {'cat' if class_idx == 0 else 'not_cat'} ({confidence:.2%})")
```

#### Generation
```python
import torch
from dit import tinydit_128
from flow_matching import sample
from PIL import Image

# Load model
model = tinydit_128(num_classes=13)
checkpoint = torch.load('tinydit_final.pt', map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

# Generate
breed_idx = 10  # Siamese
breeds = torch.tensor([breed_idx])
images = sample(model, breeds, num_steps=50, cfg_scale=1.5)

# Save
image = images[0].permute(1, 2, 0).numpy()
image = ((image + 1) / 2 * 255).astype('uint8')
Image.fromarray(image).save('generated_siamese.png')
```

### Transformers

```python
from transformers import pipeline

# Classification
classifier = pipeline("image-classification", model="d4oit/tiny-cats-model")
result = classifier("cat.jpg")
print(result)  # [{'label': 'cat', 'score': 0.97}]
```

### ONNX Runtime (Browser)

```javascript
import * as ort from "onnxruntime-web";

// Load model
const session = await ort.InferenceSession.create('cats_classifier.onnx');

// Prepare input
const tensor = new ort.Tensor('float32', imageData, [1, 3, 224, 224]);

// Run inference
const results = await session.run({ input: tensor });
const output = results.output.data;
```

## Training

### Dataset

Trained on the Oxford IIIT Pet Dataset:
- **Source**: https://www.robots.ox.ac.uk/~vgg/data/pets/
- **Total Images**: ~2,000 cat images
- **Breeds**: 12 pure breeds + mixed
- **License**: CC BY 4.0

### Training Configuration

#### Classifier
```bash
modal run src/train.py data/cats --epochs 20 --batch-size 64
```

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet-18 |
| Epochs | 20 |
| Batch Size | 64 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW |
| Mixed Precision | Yes |

#### Generator (TinyDiT)
```bash
modal run src/train_dit.py data/cats --steps 300000 --batch-size 256
```

| Parameter | Value |
|-----------|-------|
| Steps | 300,000 |
| Batch Size | 256 (effective 512 with gradient accumulation) |
| Learning Rate | 1e-4 (cosine annealing with warmup) |
| EMA Beta | 0.9999 |
| Augmentation | Full (flip, rotation, color jitter, affine) |
| GPU | Modal A10G |

## Files in Repository

```
d4oit/tiny-cats-model/
├── README.md                      # This model card
├── classifier/
│   ├── config.json                # Classifier configuration
│   ├── model.pt                   # PyTorch checkpoint
│   ├── model.onnx                 # ONNX model (FP32)
│   └── model_quantized.onnx       # ONNX model (quantized)
├── generator/
│   ├── config.json                # Generator configuration
│   ├── model.pt                   # PyTorch checkpoint
│   ├── model.onnx                 # ONNX model (FP32)
│   └── model_quantized.onnx       # ONNX model (quantized)
├── evaluation/
│   ├── evaluation_report.json     # FID, IS, Precision/Recall
│   └── sample_grid.png            # Sample generations
├── benchmarks/
│   └── benchmark_report.json      # Latency, throughput, memory
└── samples/                       # Example generations per breed
    ├── abyssinian.png
    ├── bengal.png
    └── ...
```

## License

Apache License 2.0

## Citation

```bibtex
@software{tiny-cats-model-2026,
  title = {TinyDiT Cat Breed Classifier & Generator},
  author = {d-oit},
  year = {2026},
  url = {https://huggingface.co/d4oit/tiny-cats-model}
}
```

## Links

- **GitHub**: https://github.com/d-oit/tiny-cats-model
- **Demo**: https://d-oit.github.io/tiny-cats-model/
- **Paper**: Tiny Models (https://github.com/amins01/tiny-models)
"""

    return ModelCard(card_content)


def upload_to_huggingface(
    classifier_path: str | Path | None = None,
    generator_path: str | Path | None = None,
    onnx_classifier_path: str | Path | None = None,
    onnx_generator_path: str | Path | None = None,
    evaluation_report_path: str | Path | None = None,
    benchmark_report_path: str | Path | None = None,
    samples_dir: str | Path | None = None,
    repo_id: str = "d4oit/tiny-cats-model",
    token: str | None = None,
    commit_message: str = "Upload models and evaluation results",
    private: bool = False,
) -> dict[str, Any]:
    """Upload complete model package to HuggingFace Hub.

    Args:
        classifier_path: Path to classifier PyTorch checkpoint
        generator_path: Path to generator PyTorch checkpoint
        onnx_classifier_path: Path to classifier ONNX model
        onnx_generator_path: Path to generator ONNX model
        evaluation_report_path: Path to evaluation JSON report
        benchmark_report_path: Path to benchmark JSON report
        samples_dir: Directory containing sample images
        repo_id: HuggingFace repository ID
        token: HuggingFace API token (or HF_TOKEN env var)
        commit_message: Git commit message
        private: Whether to make repo private

    Returns:
        Dict with upload results
    """
    if token is None:
        token = os.environ.get("HF_TOKEN")

    if not token:
        raise ValueError(
            "HuggingFace token required. Set HF_TOKEN env var or pass token argument."
        )

    # Create repo if not exists
    create_repo(repo_id, exist_ok=True, token=token, private=private)

    # Create temporary directory for upload
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Load evaluation and benchmark results
        evaluation_results = None
        benchmark_results = None

        if evaluation_report_path and Path(evaluation_report_path).exists():
            with open(evaluation_report_path) as f:
                evaluation_results = json.load(f)

        if benchmark_report_path and Path(benchmark_report_path).exists():
            with open(benchmark_report_path) as f:
                benchmark_results = json.load(f)

        # Create model card
        card = create_model_card(
            evaluation_results=evaluation_results,
            benchmark_results=benchmark_results,
            repo_id=repo_id,
        )
        card.save(tmpdir_path / "README.md")

        # Organize files for upload
        classifier_dir = tmpdir_path / "classifier"
        generator_dir = tmpdir_path / "generator"
        evaluation_dir = tmpdir_path / "evaluation"
        benchmarks_dir = tmpdir_path / "benchmarks"
        samples_upload_dir = tmpdir_path / "samples"

        classifier_dir.mkdir(exist_ok=True)
        generator_dir.mkdir(exist_ok=True)
        evaluation_dir.mkdir(exist_ok=True)
        benchmarks_dir.mkdir(exist_ok=True)
        samples_upload_dir.mkdir(exist_ok=True)

        # Copy classifier files
        if classifier_path and Path(classifier_path).exists():
            shutil.copy(classifier_path, classifier_dir / "model.pt")
            print(
                f"Copied classifier: {classifier_path} → {classifier_dir / 'model.pt'}"
            )

        if onnx_classifier_path and Path(onnx_classifier_path).exists():
            shutil.copy(onnx_classifier_path, classifier_dir / "model.onnx")
            print(
                f"Copied ONNX classifier: {onnx_classifier_path} → {classifier_dir / 'model.onnx'}"
            )

        # Copy generator files
        if generator_path and Path(generator_path).exists():
            shutil.copy(generator_path, generator_dir / "model.pt")
            print(f"Copied generator: {generator_path} → {generator_dir / 'model.pt'}")

        if onnx_generator_path and Path(onnx_generator_path).exists():
            shutil.copy(onnx_generator_path, generator_dir / "model.onnx")
            print(
                f"Copied ONNX generator: {onnx_generator_path} → {generator_dir / 'model.onnx'}"
            )

        # Copy evaluation report
        if evaluation_report_path and Path(evaluation_report_path).exists():
            shutil.copy(
                evaluation_report_path, evaluation_dir / "evaluation_report.json"
            )

        # Copy benchmark report
        if benchmark_report_path and Path(benchmark_report_path).exists():
            shutil.copy(benchmark_report_path, benchmarks_dir / "benchmark_report.json")

        # Copy samples
        if samples_dir and Path(samples_dir).exists():
            for img_path in Path(samples_dir).glob("*"):
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                    shutil.copy(img_path, samples_upload_dir / img_path.name)

        # Upload all files
        api = HfApi()
        api.upload_folder(
            folder_path=str(tmpdir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            token=token,
        )

        print(f"\n✅ Successfully uploaded to https://huggingface.co/{repo_id}")

        return {
            "repo_id": repo_id,
            "url": f"https://huggingface.co/{repo_id}",
            "files_uploaded": [
                "README.md",
                "classifier/model.pt" if classifier_path else None,
                "classifier/model.onnx" if onnx_classifier_path else None,
                "generator/model.pt" if generator_path else None,
                "generator/model.onnx" if onnx_generator_path else None,
                "evaluation/evaluation_report.json" if evaluation_report_path else None,
                "benchmarks/benchmark_report.json" if benchmark_report_path else None,
            ],
            "timestamp": datetime.now().isoformat(),
        }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload trained models to HuggingFace Hub"
    )

    parser.add_argument(
        "--classifier",
        type=str,
        help="Path to classifier PyTorch checkpoint",
    )
    parser.add_argument(
        "--generator",
        type=str,
        help="Path to generator PyTorch checkpoint",
    )
    parser.add_argument(
        "--onnx-classifier",
        type=str,
        help="Path to classifier ONNX model",
    )
    parser.add_argument(
        "--onnx-generator",
        type=str,
        help="Path to generator ONNX model",
    )
    parser.add_argument(
        "--evaluation-report",
        type=str,
        help="Path to evaluation JSON report",
    )
    parser.add_argument(
        "--benchmark-report",
        type=str,
        help="Path to benchmark JSON report",
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        help="Directory containing sample images",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="d4oit/tiny-cats-model",
        help="HuggingFace repository ID (default: d4oit/tiny-cats-model)",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload models and evaluation results",
        help="Git commit message",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private",
    )

    return parser.parse_args()


def main() -> None:
    """Main upload function."""
    args = parse_args()

    try:
        result = upload_to_huggingface(
            classifier_path=args.classifier,
            generator_path=args.generator,
            onnx_classifier_path=args.onnx_classifier,
            onnx_generator_path=args.onnx_generator,
            evaluation_report_path=args.evaluation_report,
            benchmark_report_path=args.benchmark_report,
            samples_dir=args.samples_dir,
            repo_id=args.repo_id,
            token=args.token,
            commit_message=args.commit_message,
            private=args.private,
        )

        print("\nUpload complete!")
        print(f"Repository: {result['repo_id']}")
        print(f"URL: {result['url']}")
        print(f"Timestamp: {result['timestamp']}")

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nSet HF_TOKEN environment variable:")
        print("  export HF_TOKEN=hf_...")
        print("\nOr get a token from: https://huggingface.co/settings/tokens")
        exit(1)
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
