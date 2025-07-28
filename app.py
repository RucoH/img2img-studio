# 🇹🇷 Gerekli kütüphaneleri içe aktar
# 🇬🇧 Import necessary libraries
import gradio as gr
from PIL import Image

# 🇹🇷 pipeline.py içindeki fonksiyonları al
# 🇬🇧 Import functions from pipeline.py
from pipeline import load_model, img2img_generate

# 🇹🇷 Uygulama açılır açılmaz modeli bir kez yükle (startup)
# 🇬🇧 Load the model once at startup
pipe = load_model()


# 🇹🇷 Gradio için görsel üretme fonksiyonu
# 🇬🇧 Gradio image-generation function
def generate_image(input_image, prompt, strength, guidance_scale):
    """
    🇹🇷 Yüklenen görsel ve prompt ile img2img üretimi yapar.
    🇬🇧 Generates a new image from the input image and prompt.

    Args:
        input_image (PIL.Image or None): 🇹🇷 Kullanıcının yüklediği görsel
                                         🇬🇧 User-uploaded image
        prompt (str): 🇹🇷 Kullanıcının girdiği açıklayıcı metin
                      🇬🇧 Text prompt from the user
        strength (float): 🇹🇷 Görselden ne kadar sapılacağını belirler (0.1–1.0)
                          🇬🇧 How much to transform the original image (0.1–1.0)
        guidance_scale (float): 🇹🇷 Prompt uyum gücü (1.0–15.0)
                                🇬🇧 Prompt adherence strength (1.0–15.0)

    Returns:
        PIL.Image or None: 🇹🇷 Üretilen görsel veya None
                           🇬🇧 Generated image or None
    """
    if input_image is None or prompt.strip() == "":
        return None
    # img2img fonksiyonunu kullanarak yeni görsel üret
    return img2img_generate(pipe, input_image, prompt, strength, guidance_scale)


# 🇹🇷 Gradio arayüzünü tanımla
# 🇬🇧 Define the Gradio interface
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(type="pil", label="🎨 Girdi Görseli / Input Image"),
        gr.Textbox(label="📝 Prompt (örnek: fluffy cat with witch hat)"),
        gr.Slider(0.1, 1.0, value=0.3, step=0.01, label="🎚️ Güç / Strength"),
        gr.Slider(1.0, 15.0, value=8.5, step=0.1, label="🎯 Yönlendirme / Guidance Scale")
    ],
    outputs=gr.Image(type="pil", label="🖼️ Çıktı Görseli / Output Image"),
    title="🧠 img2img Studio",
    description=(
        "🇹🇷 Bu uygulama, yüklediğiniz görsel üzerine metin tabanlı değişiklikler ekler.\n"
        "🇬🇧 This app applies text-guided transformations to your uploaded image."
    ),
    allow_flagging="never"
)

# 🇹🇷 Uygulamayı başlat
# 🇬🇧 Launch the application
if __name__ == "__main__":
    demo.launch()
