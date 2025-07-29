# app.py

import gradio as gr
from PIL import Image
from pipeline import load_img2img, img2img_generate

# Load the pipeline once at startup
i2i_pipe = load_img2img()

# Generation callback: image, prompt, strength, guidance
def generate(
    image: Image.Image,
    prompt: str,
    strength: float,
    guidance_scale: float
) -> Image.Image | None:
    if image is None or not prompt.strip():
        return None
    return img2img_generate(i2i_pipe, image, prompt, strength, guidance_scale)

# Gradio UI with sliders for strength and guidance
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Textbox(label="Prompt"),
        gr.Slider(0.1, 1.0, value=0.75, step=0.01, label="Strength"),
        gr.Slider(1.0, 15.0, value=7.5, step=0.1, label="Guidance Scale")
    ],
    outputs=gr.Image(type="pil", label="Output Image"),
    title="🧠 img2img Studio",
    description="Generate new images from your photo using Stable Diffusion v1.5 img2img with adjustable parameters.",
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch()