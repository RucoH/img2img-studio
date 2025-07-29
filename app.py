# app.py

"""
🇹🇷 Bu uygulama, Kandinsky 2.2 prior ve decoder pipeline'larını kullanarak img2img dönüşümleri gerçekleştirir.
🇬🇧 This app uses Kandinsky 2.2 Prior and Decoder pipelines to perform img2img transformations.
"""
import gradio as gr
from PIL import Image

from pipeline import load_pipelines, img2img_generate

# 🇹🇷 Uygulama açılır açılmaz prior ve decoder pipeline'larını yükle
# 🇬🇧 Load the Prior and Decoder pipelines at startup
prior_pipe, decoder_pipe = load_pipelines()

# 🇹🇷 Gradio için resim üretme fonksiyonu
# 🇬🇧 Gradio image generation function
def generate_image(
    input_image: Image.Image,
    prompt: str,
    strength: float,
    num_inference_steps: int,
    guidance_scale: float
) -> Image.Image | None:
    """
    🇹🇷 Girdi görseli ve prompt ile Kandinsky img2img işlemeyi tetikler.
    🇬🇧 Triggers Kandinsky img2img processing with the input image and prompt.

    Args:
        input_image: 🇹🇷 Kullanıcının yüklediği PIL Image / 🇬🇧 User-uploaded PIL Image
        prompt: 🇹🇷 Dönüşüm için metin komutu / 🇬🇧 Text command for transformation
        strength: 🇹🇷 Embedding gücü (0.0–1.0) / 🇬🇧 Embedding strength
        num_inference_steps: 🇹🇷 Decoder adım sayısı / 🇬🇧 Number of decoder inference steps
        guidance_scale: 🇹🇷 Prompt sadakati seviyesi / 🇬🇧 Level of prompt adherence

    Returns:
        Image.Image | None: 🇹🇷 Üretilen görsel veya None / 🇬🇧 Generated image or None
    """
    if input_image is None or not prompt.strip():
        return None
    try:
        output = img2img_generate(
            prior_pipe,
            decoder_pipe,
            input_image,
            prompt,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        return output
    except Exception as e:
        print(f"Error in img2img_generate: {e}")
        return None

# 🇹🇷 Gradio arayüzünü tanımla
# 🇬🇧 Define the Gradio interface
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(type="pil", label="🎨 Input Image / Girdi Görseli"),
        gr.Textbox(label="📝 Prompt (örn: transform this cat into a wizard)", placeholder="E.g.: transform this car into a hovercraft"),
        gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="🔧 Embedding Strength / Strength"),
        gr.Slider(10, 100, value=25, step=1, label="🔄 Inference Steps / Decoder Steps"),
        gr.Slider(1.0, 15.0, value=7.5, step=0.1, label="🎯 Guidance Scale / Prompt Sadakati")
    ],
    outputs=gr.Image(type="pil", label="🖼️ Output Image / Çıktı Görseli"),
    title="🧠 Kandinsky 2.2 Img2Img Studio",
    description=(
        "🇹🇷 Kandinsky 2.2 prior ve decoder pipeline kullanılarak görselden görsel oluşturur."
        "\n🇬🇧 Generates images from images using Kandinsky 2.2 Prior and Decoder pipelines."
    ),
    allow_flagging="never"
)

if __name__ == "__main__":
    # 🇹🇷 Uygulamayı başlat
    # 🇬🇧 Launch the app
    iface.launch()
