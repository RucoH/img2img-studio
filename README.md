# img2img-studio

## Project Description

This project allows users to apply text-guided transformations (img2img) to an input image to generate new images. Using Stable Diffusion-based models, it delivers high-quality, detailed, and precisely controllable transformations.

---

## Features

* 🎨 Image Upload & Preview
* 📝 Text Prompt Support
* 🎚️ Strength & Guidance Scale Controls
* ⚙️ GPU/CPU Auto Detection
* 🌐 Both Local & API Modes (FastAPI Integration)

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Gradio Interface

```bash
python app.py
```

Open your browser at `http://127.0.0.1:7860`.

### FastAPI (API Mode)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Test via Swagger UI at `http://127.0.0.1:8000/docs`.

---

## Project Structure

```
img2img-studio/
├── app.py
├── pipeline.py
├── api.py         # FastAPI endpoint
├── utils.py       # Utility functions
├── requirements.txt
├── README.md
├── inputs/        # Example images
└── outputs/       # Generated images
```

---

## License

MIT License © 2025 Alperen
