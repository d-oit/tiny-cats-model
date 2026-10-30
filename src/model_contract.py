"""src/model_contract.py

Single source of truth for classifier/generator semantics and preprocessing.

Issues #197, #200, #203 and #204 require training, evaluation, ONNX export,
the Gradio app and the browser worker to agree on:

* what the classifier predicts (binary cat / not-cat, **not** 13 breeds),
* the exact label ordering and generator conditioning indices,
* the preprocessing transform (RGB, resize short side 256, centre crop 224,
  ImageNet normalisation, NCHW float32) and whether the graph emits logits,
* the ONNX input/output names and tensor shapes.

Duplicating those constants is what allowed the demos and the exported graph to
drift.  Everything now derives from :func:`classifier_metadata` /
:func:`generator_metadata`, which are embedded in checkpoints, written beside
ONNX exports and mirrored into ``frontend/src/generated/modelContract.json``.

The module deliberately has no heavy imports (no torch, no PIL) so the
frontend generator, dataset download validation and unit tests can all import
it cheaply.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

#: Version of the metadata schema itself. Bump when fields change meaning.
CONTRACT_VERSION = "1.0.0"

#: Classifier task identity. The classifier is binary; 13 breeds belong to the
#: *generator* (12 cat breeds + "Other").
CLASSIFIER_TASK = "binary_cat_classification"

#: Public classifier labels, in output-index order. ``other`` is the dataset
#: folder name; the public/demo label is ``not_cat`` so the UI can say
#: "Not a Cat" without claiming every negative sample is a dog.
CLASSIFIER_LABELS: tuple[str, ...] = ("cat", "not_cat")

#: Dataset folder -> classifier output index. ImageFolder sorts classes
#: alphabetically (``cat`` before ``other``), which is index 0/1.
CLASSIFIER_FOLDER_TO_INDEX: dict[str, int] = {"cat": 0, "other": 1}

#: Generator conditioning order. Must never be reordered: checkpoints encode
#: these indices in their embedding table.
CAT_BREEDS: tuple[str, ...] = (
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
)
OTHER_CLASS_INDEX = len(CAT_BREEDS)
GENERATOR_CONDITIONS: tuple[str, ...] = (*CAT_BREEDS, "Other")

# ImageNet statistics used by every runtime.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: Canonical classifier preprocessing. Mirrors ``dataset.build_transforms``
#: (evaluation branch) and ``app_gradio._preprocess_classifier``.
CLASSIFIER_IMAGE_SIZE = 224
CLASSIFIER_RESIZE_SHORT_SIDE = 256
PREPROCESS_VERSION = "classifier-preprocess-v1"

#: Canonical generator settings.
GENERATOR_IMAGE_SIZE = 128
GENERATOR_DEFAULT_STEPS = 100
GENERATOR_DEFAULT_CFG_SCALE = 1.5
GENERATOR_MIN_CFG_SCALE = 1.0
GENERATOR_MAX_CFG_SCALE = 3.0


class ContractError(ValueError):
    """Metadata is missing, inconsistent, or disagrees with a checkpoint."""


def classifier_architecture(
    backbone: str = "resnet18", num_classes: int = 2, dropout: float = 0.3
) -> dict[str, Any]:
    """Describe the classifier architecture recorded in metadata."""
    return {
        "name": backbone,
        "num_classes": int(num_classes),
        "dropout": float(dropout),
        "input_size": CLASSIFIER_IMAGE_SIZE,
    }


def classifier_preprocessing() -> dict[str, Any]:
    """Canonical classifier preprocessing contract."""
    return {
        "version": PREPROCESS_VERSION,
        "color_mode": "RGB",
        "channel_order": "RGB",
        "tensor_layout": "NCHW",
        "dtype": "float32",
        "resize": {
            "mode": "short_side",
            "short_side": CLASSIFIER_RESIZE_SHORT_SIDE,
            "interpolation": "bilinear",
        },
        "center_crop": {
            "height": CLASSIFIER_IMAGE_SIZE,
            "width": CLASSIFIER_IMAGE_SIZE,
        },
        "scale": 1.0 / 255.0,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "exif_orientation": "apply_before_resize",
        "alpha": "composite_on_black_then_rgb",
    }


def classifier_metadata(
    *,
    backbone: str = "resnet18",
    num_classes: int = len(CLASSIFIER_LABELS),
    dropout: float = 0.3,
    opset: int | None = None,
    labels: tuple[str, ...] | list[str] = CLASSIFIER_LABELS,
    dataset_version: str | None = None,
    split_manifest_hash: str | None = None,
    model_sha256: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Build the versioned classifier metadata artifact.

    Args:
        backbone: torchvision backbone name.
        num_classes: Output dimension; must equal ``len(labels)``.
        dropout: Dropout probability before the final linear layer.
        opset: ONNX opset for exported graphs.
        labels: Output labels in index order.
        dataset_version: Dataset fingerprint used for training.
        split_manifest_hash: Hash of the persisted split manifest, when used.
        model_sha256: Digest of the checkpoint/ONNX artifact.
        revision: Immutable Hub revision for released artifacts.

    Returns:
        JSON-serialisable metadata dictionary.
    """
    labels = tuple(labels)
    if len(labels) != num_classes:
        raise ContractError(
            f"num_classes={num_classes} but {len(labels)} labels were supplied"
        )
    metadata: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "task": CLASSIFIER_TASK,
        "architecture": classifier_architecture(backbone, num_classes, dropout),
        "input": {
            "name": "input",
            "dtype": "float32",
            "layout": "NCHW",
            "shape": [1, 3, CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE],
            "dynamic_axes": {"0": "batch_size"},
        },
        "output": {
            "name": "output",
            "dtype": "float32",
            "shape": [1, num_classes],
            "semantics": "logits",
            "softmax_applied_in_graph": False,
            "labels": list(labels),
        },
        "preprocessing": classifier_preprocessing(),
        "dataset": {
            "folder_to_label_index": dict(CLASSIFIER_FOLDER_TO_INDEX),
            "dataset_id": "oxford-iiit-pet-cats",
        },
    }
    if opset is not None:
        metadata["opset"] = int(opset)
    if dataset_version is not None:
        metadata["dataset"]["version"] = dataset_version
    if split_manifest_hash is not None:
        metadata["dataset"]["split_manifest_hash"] = split_manifest_hash
    if model_sha256 is not None:
        metadata["model_sha256"] = model_sha256
    if revision is not None:
        metadata["revision"] = revision
    return metadata


def generator_metadata(
    *,
    num_classes: int = len(GENERATOR_CONDITIONS),
    opset: int | None = None,
    image_size: int = GENERATOR_IMAGE_SIZE,
    model_sha256: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Build the versioned generator metadata artifact."""
    if num_classes != len(GENERATOR_CONDITIONS):
        raise ContractError(
            f"generator must have {len(GENERATOR_CONDITIONS)} conditions, "
            f"got {num_classes}"
        )
    metadata: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "task": "breed_conditioned_generation",
        "input": {
            "noise": {
                "dtype": "float32",
                "layout": "NCHW",
                "shape": [1, 3, image_size, image_size],
            },
            "timestep": {"dtype": "float32", "shape": [1]},
            "breed": {"dtype": "int64", "shape": [1]},
        },
        "output": {
            "name": "velocity",
            "dtype": "float32",
            "shape": [1, 3, image_size, image_size],
            "semantics": "flow_velocity",
        },
        "conditions": [
            {"index": index, "name": name}
            for index, name in enumerate(GENERATOR_CONDITIONS)
        ],
        "unconditional_index": len(GENERATOR_CONDITIONS),
        "sampling": {
            "method": "euler",
            "default_steps": GENERATOR_DEFAULT_STEPS,
            "default_cfg_scale": GENERATOR_DEFAULT_CFG_SCALE,
            "min_cfg_scale": GENERATOR_MIN_CFG_SCALE,
            "max_cfg_scale": GENERATOR_MAX_CFG_SCALE,
        },
    }
    if opset is not None:
        metadata["opset"] = int(opset)
    if model_sha256 is not None:
        metadata["model_sha256"] = model_sha256
    if revision is not None:
        metadata["revision"] = revision
    return metadata


def full_contract(
    *, classifier_backbone: str = "resnet18", opset: int = 17
) -> dict[str, Any]:
    """The complete classifier + generator contract written to JSON."""
    return {
        "contract_version": CONTRACT_VERSION,
        "classifier": classifier_metadata(backbone=classifier_backbone, opset=opset),
        "generator": generator_metadata(opset=opset),
    }


def canonical_json(value: Any) -> str:
    """Canonical JSON used for hashes (stable across dict ordering)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def contract_hash(metadata: dict[str, Any]) -> str:
    """SHA-256 of the canonical metadata document."""
    return hashlib.sha256(canonical_json(metadata).encode()).hexdigest()


def write_contract_file(path: str | Path) -> dict[str, Any]:
    """Write :func:`full_contract` atomically and return it."""
    contract = full_contract()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return contract


def _require_keys(document: dict[str, Any], keys: tuple[str, ...], name: str) -> None:
    missing = [key for key in keys if key not in document]
    if missing:
        raise ContractError(f"{name} metadata is missing fields: {missing}")


def validate_classifier_metadata(
    metadata: dict[str, Any] | None,
    *,
    expected_num_classes: int | None = None,
    expected_backbone: str | None = None,
    expected_labels: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Validate classifier metadata and return it unchanged.

    Raises:
        ContractError: When the task, labels, dimensions, names or
            preprocessing disagree with the canonical contract.
    """
    if not isinstance(metadata, dict):
        raise ContractError("classifier metadata is missing")
    _require_keys(
        metadata,
        (
            "contract_version",
            "task",
            "architecture",
            "input",
            "output",
            "preprocessing",
        ),
        "classifier",
    )
    if metadata["task"] != CLASSIFIER_TASK:
        raise ContractError(
            f"unexpected classifier task {metadata['task']!r}; "
            f"expected {CLASSIFIER_TASK!r}"
        )

    architecture = metadata["architecture"]
    num_classes = int(architecture.get("num_classes", -1))
    if num_classes <= 0:
        raise ContractError("classifier metadata has no positive num_classes")
    if expected_num_classes is not None and num_classes != int(expected_num_classes):
        raise ContractError(
            f"metadata num_classes={num_classes} but checkpoint/model has "
            f"{expected_num_classes}"
        )
    if expected_backbone is not None:
        actual = str(architecture.get("name", ""))
        if actual != expected_backbone:
            raise ContractError(
                f"metadata backbone={actual!r} but requested {expected_backbone!r}"
            )

    output = metadata["output"]
    labels = output.get("labels")
    if not isinstance(labels, list) or len(labels) != num_classes:
        raise ContractError(
            f"metadata has {len(labels) if isinstance(labels, list) else 'no'} "
            f"labels for {num_classes} output classes"
        )
    if len(set(labels)) != len(labels):
        raise ContractError(f"classifier labels are not unique: {labels}")
    if expected_labels is not None and list(labels) != list(expected_labels):
        raise ContractError(
            f"classifier labels {labels} do not match expected {list(expected_labels)}"
        )

    output_shape = output.get("shape")
    if not isinstance(output_shape, list) or len(output_shape) != 2:
        raise ContractError("classifier output shape must be [batch, num_classes]")
    if int(output_shape[-1]) != num_classes:
        raise ContractError(
            f"output shape {output_shape} disagrees with num_classes={num_classes}"
        )
    if output.get("name") != "output":
        raise ContractError(
            f"classifier output name must be 'output', got {output.get('name')!r}"
        )

    input_spec = metadata["input"]
    if input_spec.get("name") != "input":
        raise ContractError(
            f"classifier input name must be 'input', got {input_spec.get('name')!r}"
        )
    shape = input_spec.get("shape")
    if not isinstance(shape, list) or len(shape) != 4 or int(shape[1]) != 3:
        raise ContractError(f"classifier input shape must be NCHW RGB, got {shape}")
    if int(shape[2]) != CLASSIFIER_IMAGE_SIZE or int(shape[3]) != CLASSIFIER_IMAGE_SIZE:
        raise ContractError(
            f"classifier input size {shape[2:]} != canonical "
            f"{[CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE]}"
        )

    preprocessing = metadata["preprocessing"]
    if preprocessing.get("color_mode") != "RGB":
        raise ContractError("classifier preprocessing must convert to RGB")
    if preprocessing.get("tensor_layout") != "NCHW":
        raise ContractError("classifier preprocessing must produce NCHW tensors")
    if tuple(preprocessing.get("mean", ())) != IMAGENET_MEAN:
        raise ContractError("classifier preprocessing mean is not ImageNet")
    if tuple(preprocessing.get("std", ())) != IMAGENET_STD:
        raise ContractError("classifier preprocessing std is not ImageNet")
    crop = preprocessing.get("center_crop", {})
    if (
        int(crop.get("height", -1)) != CLASSIFIER_IMAGE_SIZE
        or int(crop.get("width", -1)) != CLASSIFIER_IMAGE_SIZE
    ):
        raise ContractError(
            f"classifier centre crop must be {CLASSIFIER_IMAGE_SIZE}x"
            f"{CLASSIFIER_IMAGE_SIZE}, got {crop}"
        )
    resize = preprocessing.get("resize", {})
    if int(resize.get("short_side", -1)) != CLASSIFIER_RESIZE_SHORT_SIDE:
        raise ContractError(
            f"classifier resize short side must be "
            f"{CLASSIFIER_RESIZE_SHORT_SIDE}, got {resize.get('short_side')}"
        )
    return metadata


def metadata_from_checkpoint(checkpoint: Any) -> dict[str, Any] | None:
    """Extract contract metadata from a checkpoint document.

    Supports both new checkpoints (``metadata``) and older exports where only
    ``config`` was stored. Returns ``None`` when no metadata is present.
    """
    if not isinstance(checkpoint, dict):
        return None
    metadata = checkpoint.get("metadata") or checkpoint.get("model_metadata")
    if isinstance(metadata, dict):
        return metadata
    return None


def classifier_labels(metadata: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Return the classifier label order, falling back to the canonical order."""
    if metadata is None:
        return CLASSIFIER_LABELS
    return tuple(metadata["output"]["labels"])


__all__ = [
    "CAT_BREEDS",
    "CLASSIFIER_FOLDER_TO_INDEX",
    "CLASSIFIER_IMAGE_SIZE",
    "CLASSIFIER_LABELS",
    "CLASSIFIER_RESIZE_SHORT_SIDE",
    "CLASSIFIER_TASK",
    "CONTRACT_VERSION",
    "GENERATOR_CONDITIONS",
    "GENERATOR_IMAGE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "OTHER_CLASS_INDEX",
    "PREPROCESS_VERSION",
    "ContractError",
    "canonical_json",
    "classifier_architecture",
    "classifier_labels",
    "classifier_metadata",
    "classifier_preprocessing",
    "contract_hash",
    "full_contract",
    "generator_metadata",
    "metadata_from_checkpoint",
    "validate_classifier_metadata",
    "write_contract_file",
]
