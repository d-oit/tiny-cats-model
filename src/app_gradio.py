"""src/app_gradio.py

Gradio demo for Cat Breed Classification and Generation.
Loads models from HuggingFace Hub and runs inference via ONNX Runtime.
"""

import os

import gradio as gr
import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

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
    "Other",
]

# ImageNet stats used to train the classifier (src/dataset.py) and to export
# the ONNX graph (src/export_onnx.py, 224x224 input).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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


def _preprocess_classifier(image: np.ndarray, size: int) -> np.ndarray:
    """Preprocess a raw image for the classifier ONNX graph.

    Matches the training-time validation transform: resize to 256, center-crop
    to 224, ImageNet-normalize (for non-224 models we fall back to a plain
    resize to the graph's expected input size).
    """
    img = Image.fromarray(image).convert("RGB")
    if size == 224:
        img = img.resize((256, 256))
        left = (256 - 224) // 2
        top = (256 - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))
    else:
        img = img.resize((size, size))
    img_np = np.array(img).astype(np.float32) / 255.0
    # Use float32 operands so numpy weak-scalar promotion never upcasts to
    # float64 (ONNX graphs declare float32 inputs).
    img_np = (
        (img_np - np.array(IMAGENET_MEAN, dtype=np.float32))
        / np.array(IMAGENET_STD, dtype=np.float32)
    ).astype(np.float32)
    return img_np.transpose(2, 0, 1)[np.newaxis, ...]


def classify_cat(image):
    """Classify an image with the published classifier model.

    The published classifier ONNX is binary (cat vs other, 2 outputs). Older
    exports were 13-way breed classifiers; both are supported here so the app
    does not hard-depend on the current head size. Returns a {label: prob} dict
    for the Gradio Label widget, or an error string.
    """
    if image is None:
        return None

    session = get_session("classifier")
    if session is None:
        return "Error: Could not load classifier model."

    # Read the graph's expected input resolution (batch dim may be dynamic).
    input_shape = session.get_inputs()[0].shape
    size = int(input_shape[2]) if len(input_shape) >= 3 and input_shape[2] else 224

    img_np = _preprocess_classifier(image, size)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_np})
    logits = outputs[0][0]
    n_out = int(logits.shape[0]) if hasattr(logits, "shape") else len(logits)

    probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=0).numpy()

    if n_out == 2:
        # Binary cat-vs-other: ImageFolder class order is ['cat', 'other'].
        return {"Cat": float(probs[0]), "Other (dog / not a cat)": float(probs[1])}

    if n_out == len(BREED_NAMES):
        return {BREED_NAMES[i]: float(probs[i]) for i in range(n_out)}

    return (
        f"Error: classifier produced {n_out} outputs; "
        f"expected 2 or {len(BREED_NAMES)}. Model and app drift."
    )


def generate_cat(breed_name, cfg_scale=1.5, steps=50):
    """Generate cat image for a given breed."""
    session = get_session("generator")
    if session is None:
        return None

    # Preprocess
    breed_idx = BREED_NAMES.index(breed_name)
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

        # CFG
        if cfg_scale > 1.0:
            uncond_breed = np.array([len(BREED_NAMES) - 1], dtype=np.int64)

            # Predict cond and uncond
            # Note: Inputs order might vary depending on export
            inputs_cond = {"x": x, "t": t_tensor, "breeds": breed_tensor}
            inputs_uncond = {"x": x, "t": t_tensor, "breeds": uncond_breed}

            v_cond = session.run(None, inputs_cond)[0]
            v_uncond = session.run(None, inputs_uncond)[0]
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            inputs = {"x": x, "t": t_tensor, "breeds": breed_tensor}
            v = session.run(None, inputs)[0]

        x = x + v * dt

    # Postprocess
    x = (x[0].transpose(1, 2, 0) * 0.5 + 0.5) * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    return Image.fromarray(x)


# Build Interface
with gr.Blocks(title="Tiny Cats Model Demo") as demo:
    gr.Markdown("# 🐈 Tiny Cats Model Demo")
    gr.Markdown(
        "Classify cat breeds or generate new cats using TinyDiT (Diffusion Transformer)."
    )

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
            with gr.Column():
                input_breed = gr.Dropdown(
                    choices=BREED_NAMES[:-1], label="Select Breed", value="Siamese"
                )
                input_cfg = gr.Slider(
                    minimum=1.0, maximum=5.0, step=0.1, value=1.5, label="CFG Scale"
                )
                input_steps = gr.Slider(
                    minimum=10, maximum=100, step=10, value=50, label="Sampling Steps"
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
