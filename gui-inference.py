#!/usr/bin/env python3
import gradio as gr
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from diffusers import (
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel
)
from tqdm.auto import tqdm
import os
import threading
import random
from typing import Any, Union

# --- Global Variables for Model & State ---
# This dictionary will hold the loaded models to avoid reloading them on every run.
MODELS = {
    "vae": None,
    "unet": None,
    "controlnet": None,
    "scheduler": None,
    "device": None,
    "dtype": None
}

# This dictionary will store the paths to all available masks, indexed by feature.
MASK_LIBRARY = {}

# --- Core Inference and Preprocessing Logic ---

def load_models(controlnet_dir, device_str):
    """Loads all necessary models into the global MODELS dictionary."""
    if MODELS["controlnet"] is not None:
        print("Models already loaded.")
        return "Models are already loaded."

    print("Loading models... This may take a moment.")
    device = torch.device(device_str)
    dtype = torch.float16 if device_str == 'cuda' else torch.float32

    base_model_id = 'runwayml/stable-diffusion-v1-5'

    try:
        MODELS["vae"] = AutoencoderKL.from_pretrained(base_model_id, subfolder='vae', torch_dtype=dtype).to(device)
        MODELS["unet"] = UNet2DConditionModel.from_pretrained(base_model_id, subfolder='unet', torch_dtype=dtype).to(device)
        MODELS["controlnet"] = ControlNetModel.from_pretrained(controlnet_dir, torch_dtype=dtype).to(device)
        MODELS["scheduler"] = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler")
        MODELS["device"] = device
        MODELS["dtype"] = dtype

        print("Models loaded successfully.")
        return f"Models loaded successfully on {device}."
    except Exception as e:
        print(f"Error loading models: {e}")
        return f"Error: Could not load models. Check path and libraries. Details: {e}"


@torch.no_grad()
def decode_image(latents):
    """Decodes latents into a 3-channel RGB PIL image."""
    vae = MODELS["vae"]
    latents = latents / vae.config.scaling_factor
    image_tensor = vae.decode(latents, return_dict=False)[0]
    image_tensor = (image_tensor.clamp(-1, 1) + 1) / 2
    image_tensor = image_tensor.cpu().permute(0, 2, 3, 1).numpy()
    image_np = (image_tensor * 255).round().astype(np.uint8)
    return Image.fromarray(image_np[0])


def scan_dataset(data_root_str: str, required_indices: list[int], max_items: int):
    """Scans the dataset folder and populates the MASK_LIBRARY."""
    global MASK_LIBRARY
    MASK_LIBRARY = {index: [] for index in required_indices}

    data_root = Path(data_root_str)
    # Prepare placeholders for the galleries we'll update.
    empty_updates = [gr.update(visible=False) for _ in required_indices]

    if not data_root.is_dir():
        return "Error: Provided path is not a directory.", *empty_updates

    control_base_dir = data_root / "output" / "resnet34"
    if not control_base_dir.is_dir():
        return f"Error: 'output/resnet34' not found in {data_root_str}", *empty_updates

    print(f"Scanning {control_base_dir} for masks...")
    face_folders = [d for d in control_base_dir.iterdir() if d.is_dir()]

    for face_folder in tqdm(face_folders, desc="Indexing masks"):
        for index in required_indices:
            mask_path = face_folder / f"{index}.png"
            if mask_path.exists():
                MASK_LIBRARY[index].append(str(mask_path))

    print(f"Limiting galleries to a maximum of {max_items} random samples.")
    for index in required_indices:
        if len(MASK_LIBRARY[index]) > max_items:
            MASK_LIBRARY[index] = random.sample(MASK_LIBRARY[index], max_items)

    gallery_updates = []
    for index in required_indices:
        gallery_updates.append(gr.update(value=MASK_LIBRARY.get(index, []), visible=True))

    return f"Scan complete. Found and sampled masks for {len(required_indices)} features.", *gallery_updates


def transform_mask(base_image, x_offset, y_offset, scale, aspect_ratio, canvas_size=512):
    """Applies transformations to a single mask image and returns the transformed image and its position."""
    if base_image is None:
        return None, None

    new_width = int(base_image.width * scale * aspect_ratio)
    new_height = int(base_image.height * scale)

    if new_width <= 0 or new_height <= 0:
        return None, None

    resized_mask = base_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    paste_x = (canvas_size - new_width) // 2 + x_offset
    paste_y = (canvas_size - new_height) // 2 + y_offset

    return resized_mask, (paste_x, paste_y)


def _ensure_pil_gray(img_or_path: Union[str, Image.Image, None]) -> Union[Image.Image, None]:
    """Utility: accept a filepath or PIL image and return a grayscale PIL Image."""
    if img_or_path is None:
        return None
    if isinstance(img_or_path, str):
        if os.path.exists(img_or_path):
            return Image.open(img_or_path).convert("L")
        return None
    if isinstance(img_or_path, Image.Image):
        return img_or_path.convert("L")
    return None


def update_and_infer(*args, progress=gr.Progress()):
    """
    This function serves a dual purpose:
    1. It always calculates and returns the updated preview canvas.
    2. If real-time mode is enabled, it also runs inference and returns the generated image.
    """
    # Unpack arguments
    realtime_enabled, seed, num_steps, control_scale = args[0], args[1], args[2], args[3]
    feature_args = args[4:]

    # --- 1. Always update the preview canvas ---
    canvas_size = 512
    num_features = len(feature_args) // 5
    preview_canvas = Image.new("L", (canvas_size, canvas_size), 0)
    
    individual_canvases = [Image.new("L", (canvas_size, canvas_size), 0) for _ in range(num_features)]
    any_mask_selected = False

    for i in range(num_features):
        base_img_path, x, y, scale, aspect = feature_args[i*5], feature_args[i*5+1], feature_args[i*5+2], feature_args[i*5+3], feature_args[i*5+4]
        base_img = _ensure_pil_gray(base_img_path)
        if base_img is not None:
            any_mask_selected = True
            transformed_mask, position = transform_mask(base_img, x, y, scale, aspect, canvas_size)
            if transformed_mask:
                preview_canvas.paste(transformed_mask, position, transformed_mask)
                individual_canvases[i].paste(transformed_mask, position, transformed_mask)

    # --- 2. Conditionally run inference ---
    if not realtime_enabled:
        return preview_canvas, gr.update() # Return canvas, don't update output image

    if MODELS["controlnet"] is None:
        # Don't raise an error, just return blank images to avoid UI crash on startup
        return preview_canvas, None
    
    if not any_mask_selected:
        # Don't run inference if no masks are on the canvas
        return preview_canvas, None

    # --- Run Inference Logic (copied from original run_inference) ---
    required_indices = [2, 3, 4, 5, 10, 11, 12, 17]
    mask_tensors = [transforms.ToTensor()(canvas) for canvas in individual_canvases]
    control_tensor = torch.cat(mask_tensors, dim=0).unsqueeze(0)
    control_tensor = control_tensor.to(device=MODELS["device"], dtype=MODELS["dtype"])

    if seed is not None and seed > 0:
        torch.manual_seed(int(seed))

    latent_shape = (1, MODELS["unet"].config.in_channels, 512 // 8, 512 // 8)
    latents = torch.randn(latent_shape, device=MODELS["device"], dtype=MODELS["dtype"])

    MODELS["scheduler"].set_timesteps(int(num_steps), device=MODELS["device"])
    latents = latents * MODELS["scheduler"].init_noise_sigma

    prompt_embeds = torch.zeros((1, 77, MODELS["unet"].config.cross_attention_dim), device=MODELS["device"], dtype=MODELS["dtype"])

    progress(0, desc="Denoising (real-time)...")
    for i, t in enumerate(MODELS["scheduler"].timesteps):
        latent_model_input = MODELS["scheduler"].scale_model_input(latents, t)
        down_samples, mid_sample = MODELS["controlnet"](
            latent_model_input, t, prompt_embeds, control_tensor, control_scale, return_dict=False
        )
        noise_pred = MODELS["unet"](
            latent_model_input, t, prompt_embeds,
            down_block_additional_residuals=down_samples,
            mid_block_additional_residual=mid_sample,
        ).sample
        latents = MODELS["scheduler"].step(noise_pred, t, latents).prev_sample
        progress((i + 1) / int(num_steps), desc=f"Step {i+1}/{int(num_steps)}")

    generated_image = decode_image(latents)
    return preview_canvas, generated_image


# --- Helpers for Gallery.select ---
def _normalize_gallery_item(item: Any) -> Union[str, None]:
    if item is None: return None
    if isinstance(item, (list, tuple)) and len(item) > 0: item = item[0]
    if isinstance(item, dict):
        for k in ("image", "value", "filepath", "name", "path"):
            v = item.get(k)
            if isinstance(v, str): return v
        return None
    if isinstance(item, str): return item
    return None

def on_gallery_select(gallery_value: list, evt: gr.SelectData):
    idx = getattr(evt, "index", None)
    if idx is None: return None
    try: item = gallery_value[idx]
    except Exception: return None
    return _normalize_gallery_item(item)

# --- Build the Gradio UI ---
def create_gui(args):
    required_mask_indices = [2, 3, 4, 5, 10, 11, 12, 17]
    feature_names = {
        2: "Right Eyebrow", 3: "Left Eyebrow", 4: "Right Eye", 5: "Left Eye",
        10: "Nose", 11: "Lower Lip", 12: "Upper Lip", 17: "Hair"
    }

    threading.Thread(target=load_models, args=(args.controlnet_dir, 'cuda' if torch.cuda.is_available() else 'cpu')).start()

    with gr.Blocks(theme=gr.themes.Soft(), title="Interactive Face Generator") as app:
        gr.Markdown("# 🎨 Interactive Face Generator")
        gr.Markdown("Load a dataset, select facial features, arrange them on the canvas, and generate a new face!")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Load Dataset")
                status_box = gr.Textbox(label="Status", value="Waiting for models to load...", interactive=False)
                dataset_path = gr.Textbox(label="Dataset Root Folder Path", placeholder="/path/to/your/dataset")
                load_btn = gr.Button("Scan Dataset for Masks", variant="secondary")

                gr.Markdown("### 2. Select Features")
                feature_ui_elements = []
                gallery_elements = []

                for index in required_mask_indices:
                    name = feature_names.get(index, f"Feature {index}")
                    with gr.Accordion(name, open=False):
                        gallery = gr.Gallery(label=f"Available {name}s", elem_id=f"gallery_{index}", visible=False, columns=4, height=200)
                        selected_image = gr.Image(type="filepath", label="Selected", interactive=False, height=100)
                        with gr.Row():
                            x_slider = gr.Slider(-200, 200, value=0, step=1, label="X Offset")
                            y_slider = gr.Slider(-200, 200, value=0, step=1, label="Y Offset")
                        with gr.Row():
                            scale_slider = gr.Slider(0.1, 3.0, value=1.0, step=0.05, label="Scale")
                            aspect_slider = gr.Slider(0.2, 2.0, value=1.0, step=0.05, label="Width/Height")
                        clear_btn = gr.Button("Clear")
                        gallery.select(on_gallery_select, inputs=[gallery], outputs=selected_image, show_progress=False)
                        clear_btn.click(lambda: None, None, selected_image)
                        feature_ui_elements.extend([selected_image, x_slider, y_slider, scale_slider, aspect_slider])
                        gallery_elements.append(gallery)

            with gr.Column(scale=2):
                gr.Markdown("### 3. Compose & Generate")
                with gr.Row():
                    preview_canvas = gr.Image(label="Composition Canvas", width=512, height=512, image_mode="L", interactive=False)
                    output_image = gr.Image(label="Generated Face", width=512, height=512, interactive=False)

                with gr.Accordion("Generation Settings", open=True):
                    realtime_checkbox = gr.Checkbox(label="Real-time Inference", value=False)
                    seed = gr.Slider(1, 100000, label="Seed", step=1, value=42)
                    steps = gr.Slider(1, 100, label="Inference Steps", step=1, value=30)
                    scale = gr.Slider(0.1, 2.0, label="ControlNet Scale", step=0.1, value=1.0)
                
                generate_btn = gr.Button("Generate Face", variant="primary")

        # --- Event Handlers ---
        load_btn.click(
            fn=lambda path: scan_dataset(path, required_mask_indices, args.max_gallery_items),
            inputs=[dataset_path],
            outputs=[status_box] + gallery_elements,
            api_name="scan_dataset"
        )
        
        # All UI controls that affect the output will trigger this one function
        all_inputs = [realtime_checkbox, seed, steps, scale] + feature_ui_elements
        for elem in feature_ui_elements:
            elem.change(
                fn=update_and_infer,
                inputs=all_inputs,
                outputs=[preview_canvas, output_image],
                show_progress="minimal"
            )
        
        # Manual generate button still uses the same function
        generate_btn.click(
            fn=update_and_infer,
            inputs=all_inputs,
            outputs=[preview_canvas, output_image],
            api_name="generate_face"
        )

    return app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gradio GUI for face mask ControlNet")
    parser.add_argument('--controlnet_dir', required=True, type=str, help="Path to the trained ControlNet model directory.")
    parser.add_argument('--share', action='store_true', help="Create a public Gradio share link.")
    parser.add_argument('--max_gallery_items', type=int, default=50, help="Maximum number of masks to display per feature gallery.")
    args = parser.parse_args()

    gui = create_gui(args)
    gui.launch(share=args.share)
