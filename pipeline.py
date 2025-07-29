# pipeline.py

"""
Stable Diffusion v1.5 img2img pipeline:
- Model loading
- Image-to-image transformation with adjustable strength and guidance_scale
"""
import torch
from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

# Load Img2Img pipeline and move to device
def load_img2img(
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
) -> StableDiffusionImg2ImgPipeline:
    """
    Downloads the Img2Img pipeline using the given model_id and moves it to the appropriate device.
    Falls back to runwayml/stable-diffusion-v1-5 if the specified ID fails.
    """
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        )
    except Exception:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    return pipe

# Apply img2img transformation with adjustable parameters and negative prompt
def img2img_generate(
    pipe: StableDiffusionImg2ImgPipeline,
    image: Image.Image,
    prompt: str,
    strength: float = 0.75,
    guidance_scale: float = 7.5
) -> Image.Image:
    """
    Transforms the input image using the text prompt with specified strength and guidance.
    Uses Euler Ancestral scheduler for conservative editing.

    Args:
        pipe: Loaded Img2Img pipeline
        image: Input PIL image
        prompt: Text prompt
        strength: Float (0.0–1.0) degree of transformation
        guidance_scale: Float prompt adherence strength
    """
    # Preprocess image
    img = image.convert("RGB").resize((512, 512))
    # Swap scheduler for conservative edits
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    # Negative prompt to preserve form
    negative_prompt = (
        "deformed, distorted, cartoonish, unrealistic, extra limbs, mutated, blurry"
    )
    # Run generation
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=img,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=50
    )
    return output.images[0]