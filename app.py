import gradio as gr
from PIL import Image
from pipeline import load_img2img
import math
import os
import uuid

# Load pipeline once
title = "🧠 img2img Generator"
pipe = load_img2img("runwayml/stable-diffusion-v1-5")
try:
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
except:
    pass

# History storage
history_images = []

# Input directory
INPUT_DIR = ".gradio/flagged/Input Image"
os.makedirs(INPUT_DIR, exist_ok=True)

# Helper functions
def list_local_images():
    try:
        return ["None"] + [f for f in os.listdir(INPUT_DIR)
            if os.path.isfile(os.path.join(INPUT_DIR, f))
            and f.lower().endswith((".png",".jpg",".jpeg",".webp"))]
    except:
        return ["None"]

def load_local_image(filename):
    if not filename or filename == "None":
        return None
    path = os.path.join(INPUT_DIR, filename)
    try:
        return Image.open(path).convert("RGB")
    except:
        return None

def make_grid(images, cols):
    rows = math.ceil(len(images) / cols)
    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h))
    for i, img in enumerate(images):
        x = (i % cols) * w
        y = (i // cols) * h
        grid.paste(img, (x, y))
    return grid

# Presets and load_preset
def get_presets():
    return {
        "None": {"prompt": "", "style": "None"},
        "Neon Cat": {"prompt": "A tiny Munchkin cat with short legs on a neon-lit sofa, cyberpunk atmosphere, glowing accents, cinematic lighting.", "style": "Cyberpunk"},
        "Photo-Realistic Cat": {"prompt": "A close-up portrait of a small Munchkin kitten with short legs, sitting on a soft gray sofa in warm natural light, high detail, photo-realistic.", "style": "Photo"},
        "Cartoon Cat": {"prompt": "A stylized cartoon of a Munchkin cat with big round eyes and short legs, playful pose, bold outlines, flat colors.", "style": "Cartoon"},
        "Oil Painting Cat": {"prompt": "A cute Munchkin kitten with short stubby legs, rendered as an oil painting on canvas, soft brush strokes, muted pastel background.", "style": "Oil Painting"},
        "Astronaut": {"prompt": "An astronaut floating above Earth with galaxy background, highly detailed, cinematic.", "style": "Photo"},
    }

def load_preset(preset_name):
    p = get_presets().get(preset_name, {})
    return p.get("prompt", ""), p.get("style", "None")

# Style-specific settings
style_settings = {
    "Cyberpunk": {"strength": 0.4, "guidance": 6.0},
    "Photo": {"strength": 0.8, "guidance": 7.0},
    "Cartoon": {"strength": 0.7, "guidance": 5.0},
    "Oil Painting": {"strength": 0.5, "guidance": 6.0},
    "None": {"strength": 0.8, "guidance": 7.0},
}

# Main generation function
def generate(image, prompt, style, resolution, samples, scheduler, steps):
    if image is None or not prompt.strip():
        return None, history_images, "No input or prompt."
    cfg = style_settings.get(style, {"strength":0.8, "guidance":7.0})
    strength, guidance = cfg["strength"], cfg["guidance"]
    if resolution == "Same as Input":
        w, h = image.size
    else:
        w, h = map(int, resolution.split("x"))
    if w*h > 2048*2048:
        return None, history_images, f"Error: {w}x{h} exceeds 2048x2048"
    img = image.resize((w, h))
    full_prompt = prompt if style=="None" else f"{prompt}, in {style} style"
    n = int(samples)
    results, logs = [], []
    for i in range(n):
        logs.append(f"Gen {i+1}/{n} s={strength},g={guidance}")
        out = pipe(
            prompt=full_prompt,
            image=img,
            strength=strength,
            guidance_scale=guidance,
            height=h,
            width=w,
            num_inference_steps=steps,
            scheduler=scheduler
        ).images[0]
        results.append(out)
    final = make_grid(results, cols=min(n,2)) if n>1 else results[0]
    history_images.append(final)
    logs.append("Done.")
    return final, history_images, "\n".join(logs)

# Define UI theme
theme = gr.themes.Monochrome(primary_hue="green")

# Build UI
with gr.Blocks(title=title, theme=theme) as demo:
    gr.Markdown(f"## {title}")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="Input Image")
            local = gr.Dropdown(list_local_images(), value="None", label="Local Images")
            local.change(load_local_image, local, inp)
            prompt = gr.Textbox(label="Prompt", lines=2)
            style = gr.Dropdown(list(style_settings.keys()), value="None", label="Style")
            presets = gr.Dropdown(list(get_presets().keys()), value="None", label="Presets")
            presets.change(load_preset, presets, [prompt, style])
            resolution = gr.Radio(["Same as Input","512x512","1024x1024","2048x2048"], value="Same as Input", label="Output Resolution")
            samples = gr.Dropdown([str(i) for i in range(1,5)], value="1", label="Samples")
            with gr.Accordion("Advanced Settings", open=False):
                scheduler = gr.Dropdown(["DDIM","Euler A","PNDM"], value="DDIM", label="Scheduler")
                steps = gr.Slider(1,100,value=30,step=1,label="Inference Steps")
            gen = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            out = gr.Image(label="Output")
            gallery = gr.Gallery(label="History", columns=2)
            logs = gr.Textbox(label="Logs")
    gen.click(generate, [inp, prompt, style, resolution, samples, scheduler, steps], [out, gallery, logs])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
