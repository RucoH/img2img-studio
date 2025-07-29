# pipeline.py

"""
🇹🇷 Kandinsky 2.2 img2img modülü:
- Prior pipeline ile görüntü embed’i oluşturma
- Decoder pipeline ile img2img oluşturma
🇬🇧 Kandinsky 2.2 img2img module:
- Generates image embeddings using the Prior pipeline
- Generates images using the Decoder pipeline
"""
import os
import torch
from diffusers import KandinskyV22PriorPipeline, KandinskyV22Img2ImgPipeline
from PIL import Image

# 🇹🇷 Gated modellere erişim için token ( gerekiyorsa )
# 🇬🇧 HF token for accessing gated models (if needed)
HF_TOKEN = os.getenv("HF_TOKEN")

# 🇹🇷 Tüm pipeline’ları yükler: prior ve decoder
# 🇬🇧 Load both Prior and Decoder pipelines
def load_pipelines(
    prior_id: str = "kandinsky-community/kandinsky-2-2-prior",
    decoder_id: str = "kandinsky-community/kandinsky-2-2-decoder"
) -> tuple[KandinskyV22PriorPipeline, KandinskyV22Img2ImgPipeline]:
    """
    🇹🇷 Prior ve Decoder pipeline’larını indirir ve cihaza taşır.
    🇬🇧 Downloads Prior and Decoder pipelines and moves them to device.
    """
    kwargs = {}
    if HF_TOKEN:
        kwargs["use_auth_token"] = HF_TOKEN

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 🇹🇷 Prior pipeline: prompt + image -> image_embeds, negative_image_embeds
    # 🇬🇧 Prior pipeline: prompt + image -> image_embeds, negative_image_embeds
    prior = KandinskyV22PriorPipeline.from_pretrained(
        prior_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        **kwargs
    )
    prior = prior.to(device)

    # 🇹🇷 Decoder pipeline: image_embeds -> final image
    # 🇬🇧 Decoder pipeline: image_embeds -> final image
    decoder = KandinskyV22Img2ImgPipeline.from_pretrained(
        decoder_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        **kwargs
    )
    decoder = decoder.to(device)

    return prior, decoder

# 🇹🇷 img2img oluştur: öncelikle embed üret, sonra decode et
# 🇬🇧 Image-to-image: first embed, then decode
def img2img_generate(
    prior_pipe: KandinskyV22PriorPipeline,
    decoder_pipe: KandinskyV22Img2ImgPipeline,
    input_image: Image.Image,
    prompt: str,
    strength: float = 0.5,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5
) -> Image.Image:
    """
    🇹🇷 Girdi görseli ve prompt ile dönüşüm yapar.
    🇬🇧 Transforms the input image using the prompt.

    Args:
        prior_pipe: 🇹🇷 Prior pipeline
        decoder_pipe: 🇹🇷 Decoder pipeline
        input_image (Image.Image): 🇹🇷 Başlangıç görseli / 🇬🇧 Input image
        prompt (str): 🇹🇷 Metin komutu / 🇬🇧 Text prompt
        strength (float): 🇹🇷 Ön embed dönüşüm gücü (0.0–1.0) / 🇬🇧 Embedding strength
        num_inference_steps (int): 🇹🇷 Decoder adım sayısı / 🇬🇧 Decoder inference steps
        guidance_scale (float): 🇹🇷 Prompt sadakati (1.0–15.0) / 🇬🇧 Guidance scale

    Returns:
        Image.Image: 🇹🇷 Üretilen görsel / 🇬🇧 Generated image
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 🇹🇷 Prior pipeline embed üretimi
    # 🇬🇧 Generate image embeddings
    prior_output = prior_pipe(
        prompt=prompt,
        image=input_image,
        strength=strength
    )
    image_embeds = prior_output.image_embeds
    negative_embeds = prior_output.negative_image_embeds

    # 🇹🇷 Decoder pipeline ile görsel oluşturma
    # 🇬🇧 Generate final image via decoder pipeline
    result = decoder_pipe(
        image_embeds=image_embeds,
        negative_image_embeds=negative_embeds,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    )

    return result.images[0]

# 🇹🇷 Test bloğu: Bu dosya doğrudan çalıştırıldığında
# 🇬🇧 Test block when run directly
if __name__ == "__main__":
    print("⏳ Loading Kandinsky 2.2 pipelines...")
    prior, decoder = load_pipelines()
    print("✅ Prior and Decoder loaded successfully!")
