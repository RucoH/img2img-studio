# 🧠 img2img-studio

## 📘 Project Description

This project enables **text-guided image-to-image (img2img)** transformations using Stable Diffusion-based models.  
It provides high-quality, detailed, and customizable image generation with a simple Gradio interface.  
Features include prompt-based control, adjustable strength and guidance scale, GPU/CPU auto-detection, and local image selection.

## ✨ Features

* 🖼️ Image Upload & Live Preview
* 📝 Prompt-Based Image Transformation
* 🎚️ Adjustable Strength & Guidance Scale
* ⚙️ Automatic GPU/CPU Detection
* 🗂️ Local image selection from Gradio’s flagged folder
* 🧩 Simple preset/prompt library

## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/RucoH/img2img-studio.git
cd img2img-studio
```

Create a virtual environment and install dependencies:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** Ensure your `torch` installation is compatible with your GPU if CUDA is available.

## 🚀 Usage

### 🔌 Launch Gradio Interface
```bash
python app.py
```
Open your browser at 👉 `http://127.0.0.1:7860`

## 🗂️ Project Structure

```
img2img-studio/
├── app.py             # Gradio interface and main application flow
├── pipeline.py        # Stable Diffusion img2img pipeline loading and execution
├── utils.py           # Helper functions (grid creation, local image listing)
├── requirements.txt   # Dependencies
├── Dockerfile         # Docker image setup
├── docker-compose.yml # Optional: Docker Compose service
└── .github/workflows  # Optional: CI configuration
```

## 🐳 Docker Usage (Optional)

Build the image:
```bash
docker build -t img2img-studio:latest .
```

Run (CPU):
```bash
docker run --rm -p 7860:7860 img2img-studio:latest
```

Run with NVIDIA GPU:
```bash
docker run --rm -p 7860:7860 --gpus all img2img-studio:latest
```

## 💡 Example Prompts

* **Photo-realistic:**  
  `A close-up portrait, natural lighting, detailed texture, high quality`
* **Cyberpunk:**  
  `Neon-lit city vibe, glowing accents, cinematic lighting, high contrast`
* **Cartoon:**  
  `Cartoon style, bold outlines, flat colors, simple shading`

## 🛠 Troubleshooting

* **CUDA OOM:** Lower resolution, decrease `strength` and `num_inference_steps`, enable attention slicing.
* **Output too different:** Reduce `strength` (0.3–0.5).
* **Prompt ignored:** Increase `guidance_scale` (7–11) and write a more precise prompt.
* **Slow on CPU:** Try smaller or memory-efficient models.

## 📄 License

Distributed under the [MIT License](LICENSE).

## 👤 Author

* GitHub: [@RucoH](https://github.com/RucoH)
* Live Site: [https://rucoh.github.io/](https://rucoh.github.io/)
