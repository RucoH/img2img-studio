# pipeline.py

"""
🇹🇷 Stable Diffusion img2img pipeline modülü:
- Model yükleme
- Img2img dönüşüm fonksiyonu
🇬🇧 Stable Diffusion img2img pipeline module:
- Model loading
- Img2img generation function
"""
import os
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

# 🇹🇷 Hugging Face erişimi için token (gerekirse)
# 🇬🇧 HF access token for private models (if needed)
HF_TOKEN = os.getenv("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")

# 🇹🇷 Modeli yükleyen fonksiyon
# 🇬🇧 Function to load the Stable Diffusion img2img model
def load_model(
    model_id: str = "kandinsky-community/kandinsky-2-2-decoder"
) -> StableDiffusionImg2ImgPipeline:
    """
    🇹🇷 Varsayılan model "gsdf/Counterfeit-V2.5" olarak ayarlandı: yüksek detay, herkese açık.
    🇬🇧 Default model set to "gsdf/Counterfeit-V2.5": high-detail, public.

    Arg:
        model_id (str): 🇹🇷 HF model identifier / 🇬🇧 HuggingFace model name

    Returns:
        StableDiffusionImg2ImgPipeline: 🇹🇷 Yüklenmiş pipeline / 🇬🇧 Loaded pipeline
    """
    # 🇹🇷 Kimlik doğrulama token’ı ekle (private modelse)
    # 🇬🇧 Include auth token if the model is private
    kwargs = {}
    if HF_TOKEN:
        kwargs["use_auth_token"] = HF_TOKEN

    # 🇹🇷 Pipeline’i oluştur ve cihaza taşı
    # 🇬🇧 Create the pipeline and move to device
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        **kwargs
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return pipe.to(device)

# 🇹🇷 Görselden görsel üreten fonksiyon
# 🇬🇧 Image-to-image generation function
def img2img_generate(
    pipeline: StableDiffusionImg2ImgPipeline,
    input_image: Image.Image,
    prompt: str,
    strength: float = 0.75,
    guidance_scale: float = 7.5
) -> Image.Image:
    """
    🇹🇷 Girdi görselini ve prompt’u kullanarak dönüşüm yapar.
    🇬🇧 Transforms the input image using the prompt.

    Args:
        pipeline: 🇹🇷 Yüklenmiş pipeline / 🇬🇧 Loaded pipeline
        input_image: 🇹🇷 Başlangıç görseli / 🇬🇧 Input image
        prompt: 🇹🇷 Metin komutu / 🇬🇧 Text prompt
        strength: 🇹🇷 Dönüşüm gücü (0.0–1.0) / 🇬🇧 Transformation strength
        guidance_scale: 🇹🇷 Prompt sadakati (1.0–15.0) / 🇬🇧 Prompt adherence

    Returns:
        Image.Image: 🇹🇷 Üretilen görsel / 🇬🇧 Generated image
    """
    # 🇹🇷 Görseli RGB’ye çevir ve yeniden boyutlandır
    # 🇬🇧 Convert to RGB and resize
    image = input_image.convert("RGB").resize((512, 512))

    # 🇹🇷 Pipeline’i çalıştır
    # 🇬🇧 Run the pipeline
    result = pipeline(
        prompt=prompt,
        image=image,
        strength=strength,
        guidance_scale=guidance_scale
    )
    return result.images[0]

# 🇹🇷 Test bloğu: Bu dosya doğrudan çalıştırıldığında
# 🇬🇧 Test block: Run this file directly to check model loading
if __name__ == "__main__":
    print("⏳ Model yükleniyor / Loading model...")
    pipe = load_model()
    print("✅ Model başarıyla yüklendi! / Model loaded successfully!")
