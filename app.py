import gradio as gr
from PIL import Image
from pipeline import load_img2img, img2img_generate
import io
import math
import os
import uuid

# Load single pipeline
pipe = load_img2img("runwayml/stable-diffusion-v1-5")

# History storage
history_images = []

# Preset library
def get_presets():
    return {
        "Neon Cat": {"prompt": "A tiny Munchkin cat with short legs on a neon-lit sofa, cyberpunk atmosphere, glowing accents, cinematic lighting.", "style": "Cyberpunk", "strength": 0.75, "guidance_scale": 7.5},
        "Photo-Realistic Cat": {"prompt": "A close-up portrait of a small Munchkin kitten with short legs, sitting on a soft gray sofa in warm natural light, high detail, photo-realistic.", "style": "Photo", "strength": 0.5, "guidance_scale": 8.0},
        "Cartoon Cat": {"prompt": "A stylized cartoon of a Munchkin cat with big round eyes and short legs, playful pose, bold outlines, flat colors.", "style": "Cartoon", "strength": 0.70, "guidance_scale": 8.0},
        "Oil Painting Cat": {"prompt": "A cute Munchkin kitten with short stubby legs, rendered as an oil painting on canvas, soft brush strokes, muted pastel background.", "style": "Oil Painting", "strength": 0.5, "guidance_scale": 8.0},
        "Astronaut": {"prompt": "An astronaut floating above Earth with galaxy background, highly detailed, cinematic.", "style": "Photo", "strength": 0.7, "guidance_scale": 13.4},
    }

# Callback to load preset into inputs
def load_preset(preset_name):
    presets = get_presets()
    p = presets.get(preset_name, {})
    return (
        p.get("prompt", ""),
        p.get("style", "Cyberpunk"),
        p.get("strength", 0.75),
        p.get("guidance_scale", 7.5)
    )

# Utility to tile images into a grid
def make_grid(images, cols):
    rows = math.ceil(len(images) / cols)
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h))
    for idx, img in enumerate(images):
        x = (idx % cols) * w
        y = (idx // cols) * h
        grid.paste(img, (x, y))
    return grid

# Main generation function
# Returns: generated image, history images, logs
def generate(
    image: Image.Image,
    prompt: str,
    style: str,
    width: int,
    height: int,
    n_samples: int,
    strength: float,
    guidance_scale: float,
    scheduler: str,
    steps: int
):
    if image is None or not prompt.strip():
        return None, [], "No input or prompt."
    full_prompt = f"{prompt}, in {style} style"
    results, logs = [], []
    for i in range(n_samples):
        logs.append(f"Generating sample {i+1}/{n_samples}...")
        out = pipe(
            prompt=full_prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=steps
        ).images[0]
        results.append(out)
    final = make_grid(results, cols=min(n_samples, 2)) if n_samples > 1 else results[0]
    history_images.append(final)
    logs.append("Generation complete.")
    return final, history_images, "\n".join(logs)

# UI theme
theme = gr.themes.Monochrome(primary_hue="green")

with gr.Blocks(theme=theme) as demo:
    gr.Markdown("## 🧠 img2img Studio", elem_id="header")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Image")
            prompt = gr.Textbox(label="Prompt", lines=2)
            style = gr.Dropdown(
                choices=["Cyberpunk","Photo","Cartoon","Oil Painting"],
                value="Cyberpunk",
                label="Style"
            )
            width = gr.Slider(minimum=128, maximum=1024, step=128, value=512, label="Width")
            height = gr.Slider(minimum=128, maximum=1024, step=128, value=512, label="Height")
            n_samples = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of samples")
            strength = gr.Slider(minimum=0.1, maximum=1.0, value=0.75, step=0.01, label="Strength")
            guidance_scale = gr.Slider(minimum=1.0, maximum=15.0, value=7.5, step=0.1, label="Guidance Scale")
            with gr.Accordion("Advanced Settings", open=False):
                scheduler = gr.Dropdown(
                    choices=["DDIM","Euler A","PNDM"],
                    value="DDIM",
                    label="Scheduler"
                )
                steps = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Inference steps")
            preset_sel = gr.Dropdown(choices=list(get_presets().keys()), label="Presets")
            load_btn = gr.Button("Load Preset")
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Output Image")
            download_btn = gr.Button("Download Image")
            history_gallery = gr.Gallery(label="History", columns=2, height="auto")
            log_box = gr.Textbox(label="Logs", interactive=False)

    # Load presets into inputs
    load_btn.click(
        fn=load_preset,
        inputs=[preset_sel],
        outputs=[prompt, style, strength, guidance_scale]
    )

    # Generate and update history
    generate_btn.click(
        fn=generate,
        inputs=[
            input_image, prompt, style,
            width, height, n_samples, strength,
            guidance_scale, scheduler, steps
        ],
        outputs=[output_image, history_gallery, log_box]
    )

    # Download functionality
    def make_download(img):
        temp_path = os.path.join(os.getcwd(), f"download_{uuid.uuid4().hex[:8]}.png")
        img.save(temp_path)
        return temp_path

    download_btn.click(
        fn=make_download,
        inputs=[output_image],
        outputs=[gr.File(label="Download")]
    )

if __name__ == "__main__":
    demo.launch()
