# utils.py

"""
Helper utilities for img2img-studio:
- Directory creation/check
- Timestamped image saving
"""
import os
from PIL import Image
from datetime import datetime

# Ensure a directory exists
# 🇹🇷 Klasör oluşturur veya var ise geçer
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# Save image with timestamp
# 🇹🇷 Zaman damgalı dosya adıyla kaydeder
def save_image(
    image: Image.Image,
    output_dir: str,
    prefix: str = "output"
) -> str:
    ensure_dir(output_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{ts}.png"
    path = os.path.join(output_dir, filename)
    image.save(path)
    return path