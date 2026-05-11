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


def classify_cat(image):
    """Classify cat breed from image."""
    if image is None:
        return None

    session = get_session("classifier")
    if session is None:
        return "Error: Could not load classifier model."

    # Preprocess
    img = Image.fromarray(image).convert("RGB")
    img = img.resize((128, 128))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np - 0.5) / 0.5  # Normalize to [-1, 1]
    img_np = img_np.transpose(2, 0, 1)[np.newaxis, ...]

    # Inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_np})
    logits = outputs[0][0]

    # Postprocess
    probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=0).numpy()
    results = {BREED_NAMES[i]: float(probs[i]) for i in range(len(BREED_NAMES))}
    return results


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
