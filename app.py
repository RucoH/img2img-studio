# 🇹🇷 Kütüphaneleri içe aktar
# 🇬🇧 Import necessary libraries
import gradio as gr
from PIL import Image
from pipeline import load_model, img2img_generate


# 🇹🇷 Modeli baştan yükle (sadece bir kez)
# 🇬🇧 Load the model once when the app starts
pipe = load_model()


# 🇹🇷 Gradio arayüzü için görsel üretme fonksiyonu
# 🇬🇧 Gradio image generation function
def generate_image(input_image, prompt, strength, guidance):
    if input_image is None or prompt.strip() == "":
        return None
    output = img2img_generate(pipe, input_image, prompt, strength, guidance)
    return output


# 🇹🇷 Gradio Arayüzü
# 🇬🇧 Gradio Interface
demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(type="pil", label="🎨 Girdi Görseli / Input Image"),
        gr.Textbox(label="📝 Prompt (örnek: cyberpunk city, glowing lights)"),
        gr.Slider(0.1, 1.0, value=0.75, label="🎚️ Güç / Strength"),
        gr.Slider(1, 15, value=7.5, label="🎯 Yönlendirme / Guidance Scale")
    ],
    outputs=gr.Image(type="pil", label="🖼️ Çıktı Görseli / Output Image"),
    title="🧠 img2img Studio",
    description=(
        "🇹🇷 Girdi görseline göre yeni bir görsel üretir. Prompt girin ve sonucu görün!\n"
        "🇬🇧 Generates a new image from an input image using your text prompt."
    ),
    allow_flagging="never"
)

# 🇹🇷 Uygulamayı başlat
# 🇬🇧 Launch the application
if __name__ == "__main__":
    demo.launch()
