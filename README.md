# 🖼️ img2img-studio

## 📘 Project Description

This project allows users to apply **text-guided transformations (img2img)** to an input image to generate new outputs.
Powered by Stable Diffusion-based models, it provides high-quality, detailed, and controllable image editing.

## ✨ Features

* 🖼️ Image Upload & Live Preview
* 📝 Prompt-Based Image Generation
* 🎚️ Control over Strength & Guidance Scale
* ⚙️ Automatic GPU/CPU Detection

## ⚙️ Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 🔌 Launch Gradio Interface

```bash
python app.py
```

Then open your browser at 👉 `http://127.0.0.1:7860`

## 🗂️ Project Structure

```
img2img-studio/
├── app.py            # Main Gradio app
├── pipeline.py       # Image-to-image generation logic
├── utils.py          # Utility functions for image processing
```

---

## 📄 License

Distributed under the [MIT License](LICENSE).

## 👤 Author

* GitHub: [@RucoH](https://github.com/RucoH)
* Live Site: [https://rucoh.github.io/](https://rucoh.github.io/)

---
