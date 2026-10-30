"""src/app_gradio.py

Gradio demo for Cat Breed Classification and Generation.
Loads models from HuggingFace Hub and runs inference via ONNX Runtime.
"""

import json
import os
from typing import Any

import gradio as gr
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

from model_contract import (
    CLASSIFIER_LABELS,
    GENERATOR_CONDITIONS,
    validate_classifier_metadata,
)
from preprocessing import classifier_eval_transform

# Constants
REPO_ID = "d4oit/tiny-cats-model"
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
]
GENERATOR_BREED_INDICES = {
    name.replace("_", " "): index
    for index, name in enumerate(GENERATOR_CONDITIONS[:-1])
}
# Cache for model sessions
sessions: dict[str, ort.InferenceSession] = {}


def get_session(model_type):
    """Load or return cached ONNX session."""
    if model_type in sessions:
        return sessions[model_type]

    filename = (
        "classifier/model_quantized.onnx"
        if model_type == "classifier"
        else "generator/model_quantized.onnx"
    )
    print(f"Downloading {model_type} model from {REPO_ID}...")

    try:
        model_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
        session = ort.InferenceSession(model_path)
        sessions[model_type] = session
        return session
    except Exception as e:
        print(f"Error loading {model_type} model: {e}")
        return None


def _preprocess_classifier(image: np.ndarray) -> np.ndarray:
    """Apply the canonical RGB/resize/crop/ImageNet normalization transform."""
    return (
        classifier_eval_transform()(Image.fromarray(image))
        .unsqueeze(0)
        .numpy()
        .astype(np.float32, copy=False)
    )


def classify_cat(image: np.ndarray | None) -> dict[str, float] | str | None:
    """Return binary cat/not-cat probabilities, or a clear inference error."""
    if image is None:
        return None

    session = get_session("classifier")
    if session is None:
        return "Error: Could not load classifier model."
    metadata_reader = getattr(session, "get_modelmeta", None)
    if metadata_reader is not None:
        contract_json = metadata_reader().custom_metadata_map.get("classifier_metadata")
        if contract_json:
            try:
                validate_classifier_metadata(
                    json.loads(contract_json),
                    expected_num_classes=len(CLASSIFIER_LABELS),
                    expected_labels=CLASSIFIER_LABELS,
                )
            except (ValueError, TypeError) as error:
                return f"Error: invalid classifier metadata: {error}"

    inputs = session.get_inputs()
    input_spec = inputs[0]
    input_shape = input_spec.shape
    if len(input_shape) != 4 or input_shape[2] != 224 or input_shape[3] != 224:
        return (
            "Error: classifier ONNX input must be NCHW with 224x224 spatial "
            f"dimensions; got {input_shape}."
        )
    img_np = _preprocess_classifier(image)
    outputs = session.run(None, {input_spec.name: img_np})
    logits = np.asarray(outputs[0][0], dtype=np.float32)
    if logits.ndim != 1 or len(logits) != len(CLASSIFIER_LABELS) or len(logits) != 2:
        count = logits.shape[0] if logits.ndim else 0
        return f"Error: classifier produced {count} outputs; expected 2."
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)
    return {"Cat": float(probs[0]), "Not a cat": float(probs[1])}


def _generator_feed_keys(session: Any) -> tuple[str, str, str]:
    """Map our tensors onto the ONNX graph's declared input names.

    The exporter writes ``noise``/``timestep``/``breed`` (see
    ``src/export_dit_onnx.py``), while this module originally fed
    ``x``/``t``/``breeds`` — which fails at ``session.run`` with the published
    artifact. Deriving the keys from the session keeps the demo working with
    either convention.

    Args:
        session: An ``onnxruntime.InferenceSession`` (duck-typed).

    Returns:
        Tuple of (state, timestep, breed) input names.

    Raises:
        ValueError: If the graph does not declare a recognizable input set.
    """
    names = {spec.name for spec in session.get_inputs()}
    picked = (
        ("noise", next((k for k in ("x", "noise") if k in names), None)),
        ("timestep", next((k for k in ("t", "timestep") if k in names), None)),
        ("breed", next((k for k in ("breeds", "breed") if k in names), None)),
    )
    missing = [label for label, key in picked if key is None]
    if missing:
        raise ValueError(
            f"generator graph is missing expected inputs {missing}; "
            f"declared inputs: {sorted(names)}"
        )
    return (picked[0][1], picked[1][1], picked[2][1])  # type: ignore[return-value]


def generate_cat(breed_name, cfg_scale=1.5, steps=100):
    """Generate cat image for a given breed."""
    session = get_session("generator")
    if session is None:
        return None

    x_key, t_key, breed_key = _generator_feed_keys(session)

    breed_idx = GENERATOR_BREED_INDICES[breed_name]
    breed_tensor = np.array([breed_idx], dtype=np.int64)

    # Simple Euler ODE solver for flow matching
    batch_size = 1
    x = np.random.randn(batch_size, 3, 128, 128).astype(np.float32)
    dt = 1.0 / steps

    # This is a simplified sampler. For production, use the one in flow_matching.py
    # But since we are using ONNX in the demo, we implement it here manually.
    for i in range(steps):
        t = i / steps
        t_tensor = np.array([t], dtype=np.float32)

        if cfg_scale > 1.0:
            uncond_breed = np.array([len(GENERATOR_CONDITIONS)], dtype=np.int64)

            # Predict cond and uncond
            # Note: Inputs order might vary depending on export
            inputs_cond = {x_key: x, t_key: t_tensor, breed_key: breed_tensor}
            inputs_uncond = {x_key: x, t_key: t_tensor, breed_key: uncond_breed}

            v_cond = session.run(None, inputs_cond)[0]
            v_uncond = session.run(None, inputs_uncond)[0]
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            inputs = {x_key: x, t_key: t_tensor, breed_key: breed_tensor}
            v = session.run(None, inputs)[0]

        x = x + v * dt

    # Postprocess
    x = (x[0].transpose(1, 2, 0) * 0.5 + 0.5) * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    return Image.fromarray(x)


# Build Interface
with gr.Blocks(title="Tiny Cats Model Demo") as demo:
    gr.Markdown("Detect cats vs not-cats or generate cats with TinyDiT.")

    with gr.Tab("Classify"):
        with gr.Row():
            input_img = gr.Image(label="Upload a cat image")
            output_label = gr.Label(label="Predictions", num_top_classes=5)
        btn_classify = gr.Button("Classify")
        btn_classify.click(fn=classify_cat, inputs=input_img, outputs=output_label)

        gr.Examples(
            examples=[
                os.path.join("notebooks/assets", f)
                for f in os.listdir("notebooks/assets")
                if f.endswith((".jpg", ".png"))
            ]
            if os.path.exists("notebooks/assets")
            else [],
            inputs=input_img,
        )

    with gr.Tab("Generate"):
        with gr.Row():
            input_breed = gr.Dropdown(
                choices=BREED_NAMES, label="Select Breed", value="Siamese"
            )
            input_cfg = gr.Slider(
                minimum=1.0, maximum=5.0, step=0.1, value=1.5, label="CFG Scale"
            )
            input_steps = gr.Slider(
                minimum=10, maximum=100, step=10, value=100, label="Sampling Steps"
            )
            btn_generate = gr.Button("Generate")
            output_gen = gr.Image(label="Generated Cat")
        btn_generate.click(
            fn=generate_cat,
            inputs=[input_breed, input_cfg, input_steps],
            outputs=output_gen,
        )

if __name__ == "__main__":
    demo.launch()
