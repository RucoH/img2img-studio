import gradio as gr
from PIL import Image
from pipeline import load_img2img
import math
import os
import uuid

# Load single pipeline
pipe = load_img2img("runwayml/stable-diffusion-v1-5")

# History storage
history_images = []

# Directory for local input images
INPUT_DIR = ".gradio/flagged/Input Image"
os.makedirs(INPUT_DIR, exist_ok=True)

# Preset library
def get_presets():
    return {
        "None": {"prompt": "", "style": "None"},
        "Neon Cat": {"prompt": "A tiny Munchkin cat with short legs on a neon-lit sofa, cyberpunk atmosphere, glowing accents, cinematic lighting.", "style": "Cyberpunk"},
        "Photo-Realistic Cat": {"prompt": "A close-up portrait of a small Munchkin kitten with short legs, sitting on a soft gray sofa in warm natural light, high detail, photo-realistic.", "style": "Photo"},
        "Cartoon Cat": {"prompt": "A stylized cartoon of a Munchkin cat with big round eyes and short legs, playful pose, bold outlines, flat colors.", "style": "Cartoon"},
        "Oil Painting Cat": {"prompt": "A cute Munchkin kitten with short stubby legs, rendered as an oil painting on canvas, soft brush strokes, muted pastel background.", "style": "Oil Painting"},
        "Astronaut": {"prompt": "An astronaut floating above Earth with galaxy background, highly detailed, cinematic.", "style": "Photo"},
    }

# List local images
def list_local_images():
    return [f for f in os.listdir(INPUT_DIR)
            if os.path.isfile(os.path.join(INPUT_DIR, f))
            and f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]

# Load a local image by filename
def load_local_image(filename):
    if not isinstance(filename, str) or filename == "None":
        return None
    path = os.path.join(INPUT_DIR, filename)
    try:
        return Image.open(path).convert("RGB")
    except:
        return None

# Load preset into prompt/style fields
def load_preset(preset_name):
    p = get_presets().get(preset_name, {})
    return p.get("prompt", ""), p.get("style", "None")

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
def generate(
    image: Image.Image,
    prompt: str,
    style: str,
    resolution: str,
    n_samples: str,
    scheduler: str,
    steps: int
):
    n = int(n_samples)
    if image is None or not prompt.strip():
        return None, [], "No input or prompt."
    strength, guidance = 0.8, 7.0
    if resolution == "Same as Input":
        width, height = image.size
    else:
        width, height = map(int, resolution.split("x"))
    img = image.resize((width, height))
    full_prompt = prompt if style == "None" else f"{prompt}, in {style} style"
    results, logs = [], []
    for i in range(1, n + 1):
        logs.append(f"Generating {i}/{n}...")
        out = pipe(
            prompt=full_prompt,
            image=img,
            strength=strength,
            guidance_scale=guidance,
            height=height,
            width=width,
            num_inference_steps=steps
        ).images[0]
        results.append(out)
    cols = math.ceil(math.sqrt(n))
    final = make_grid(results, cols=cols) if n > 1 else results[0]
    history_images.append(final)
    logs.append("Generation complete.")
    return final, history_images, "\n".join(logs)

# Build UI
theme = gr.themes.Monochrome(primary_hue="green")
with gr.Blocks(theme=theme, title="img2img Lab") as demo:
    gr.Markdown("## 🎨 img2img Lab", elem_id="header")
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Image")
            local_sel = gr.Dropdown(
                choices=["None"] + list_local_images(),
                value="None",
                label="Local Images",
                allow_custom_value=False
            )
            local_sel.change(load_local_image, inputs=[local_sel], outputs=[input_image])

            prompt = gr.Textbox(label="Prompt", lines=2)
            style = gr.Dropdown(
                choices=["None", "Cyberpunk", "Photo", "Cartoon", "Oil Painting"],
                value="None",
                label="Style",
                allow_custom_value=False
            )
            resolution = gr.Radio(
                choices=["Same as Input", "512x512", "1024x1024"],
                value="Same as Input",
                label="Output Resolution"
            )
            n_samples = gr.Dropdown(
                choices=[str(i) for i in range(1,10)],
                value="1",
                label="Number of samples"
            )
            with gr.Accordion("Advanced Settings", open=False):
                scheduler = gr.Dropdown(
                    choices=["DDIM", "Euler A", "PNDM"],
                    value="DDIM",
                    label="Scheduler"
                )
                steps = gr.Slider(1, 100, value=50, step=1, label="Inference steps")
            preset_sel = gr.Dropdown(
                choices=["None"] + [k for k in get_presets() if k != "None"],
                value="None",
                label="Presets",
                allow_custom_value=False
            )
            load_btn = gr.Button("Load Preset")
            load_btn.click(load_preset, inputs=[preset_sel], outputs=[prompt, style])
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Output Image")
            download_btn = gr.Button("Download Image")
            history_gallery = gr.Gallery(label="History", columns=2, height="auto")
            log_box = gr.Textbox(label="Logs", interactive=False)

    generate_btn.click(
        generate,
        inputs=[input_image, prompt, style, resolution, n_samples, scheduler, steps],
        outputs=[output_image, history_gallery, log_box]
    )

    def make_download(img):
        path = os.path.join(os.getcwd(), f"download_{uuid.uuid4().hex[:8]}.png")
        img.save(path)
        return path
    download_btn.click(make_download, inputs=[output_image], outputs=[gr.File(label="Download")])

if __name__ == "__main__":
    demo.launch(favicon_path=".gradio/favicon.ico")
