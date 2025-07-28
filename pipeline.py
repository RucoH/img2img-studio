# 🇹🇷 Gerekli kütüphaneleri içe aktar
# 🇬🇧 Import required libraries
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image


# 🇹🇷 Stable Diffusion modelini yükleyen fonksiyon
# 🇬🇧 Function to load Stable Diffusion model
def load_model(model_id="stabilityai/stable-diffusion-2-1"):
    """
    🇹🇷 HuggingFace üzerinden modeli yükler ve GPU/CPU'ya taşır.
    🇬🇧 Loads the model from HuggingFace and moves it to GPU or CPU.

    Args:
        model_id (str): 🇹🇷 Model ismi / 🇬🇧 HuggingFace model name

    Returns:
        pipeline (StableDiffusionImg2ImgPipeline): 🇹🇷 Yüklenmiş model / 🇬🇧 Loaded pipeline
    """
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    return pipe


# 🇹🇷 Görselden görsel üretim fonksiyonu
# 🇬🇧 Image-to-image generation function
def img2img_generate(pipeline, input_image: Image.Image, prompt: str,
                     strength: float = 0.75, guidance_scale: float = 7.5):
    """
    🇹🇷 Girdi görseli ve prompt'a göre yeni bir görsel üretir.
    🇬🇧 Generates a new image based on input image and prompt.

    Args:
        pipeline: 🇹🇷 Model pipeline / 🇬🇧 Loaded img2img pipeline
        input_image (PIL.Image): 🇹🇷 Başlangıç görseli / 🇬🇧 Input image
        prompt (str): 🇹🇷 Açıklayıcı metin / 🇬🇧 Text prompt
        strength (float): 🇹🇷 Görselden sapma oranı (0.0-1.0)  
                          🇬🇧 How much to transform the original image
        guidance_scale (float): 🇹🇷 Prompt'a ne kadar sadık kalacağı  
                                🇬🇧 Prompt adherence strength

    Returns:
        PIL.Image: 🇹🇷 Üretilen yeni görsel / 🇬🇧 Generated image
    """
    image = input_image.convert("RGB").resize((512, 512))

    output = pipeline(
        prompt=prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale
    )

    return output.images[0]


# ✅ 🇹🇷 Test bloğu: Bu dosya doğrudan çalıştırılırsa model yüklenir
# ✅ 🇬🇧 Test block: Run this file directly to check model loading
if __name__ == "__main__":
    print("⏳ Model yükleniyor / Loading model...")
    pipe = load_model()
    print("✅ Model başarıyla yüklendi! / Model loaded successfully!")
