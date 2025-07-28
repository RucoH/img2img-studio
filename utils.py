# utils.py

"""
🇹🇷 Img2Img Studio için yardımcı fonksiyonlar:
- dosya ve klasör yönetimi
- görsel yükleme/kaydetme
- basit günlük (logging)
🇬🇧 Helper utilities for Img2Img Studio:
- file and directory management
- image loading/saving
- simple logging
"""

import os
from PIL import Image
from datetime import datetime

# 🇹🇷 Klasör var mı kontrol et, yoksa oluştur
# 🇬🇧 Ensure directory exists, create if not
def ensure_dir(path: str):
    """
    🇹🇷 Verilen dizin var mı kontrol eder, yoksa oluşturur.
    🇬🇧 Checks if the given directory exists and creates it if missing.
    """
    os.makedirs(path, exist_ok=True)

# 🇹🇷 Görseli kaydeder, zaman damgası ile isim üretir
# 🇬🇧 Saves image with timestamped filename
def save_image(image: Image.Image, output_dir: str, prefix: str = "output") -> str:
    """
    🇹🇷 PIL görselini output dizinine kaydeder, dosya adında zaman damgası kullanır.
    🇬🇧 Saves a PIL image to the output directory with a timestamp-based filename.

    Args:
        image (Image.Image): 🇹🇷 Kaydedilecek görsel objesi / 🇬🇧 PIL Image to save.
        output_dir (str): 🇹🇷 Görselin kaydedileceği dizin / 🇬🇧 Directory to save the image in.
        prefix (str): 🇹🇷 Dosya adı öneki (varsayılan 'output') / 🇬🇧 Filename prefix (default 'output').

    Returns:
        str: 🇹🇷 Kaydedilen dosya yolu / 🇬🇧 Full path of the saved file.
    """
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    full_path = os.path.join(output_dir, filename)
    image.save(full_path)
    return full_path

# 🇹🇷 Görsel dosyasını açar ve yeniden boyutlandırır
# 🇬🇧 Opens and resizes image from disk
def load_and_resize_image(path: str, size: tuple = (512, 512)) -> Image.Image:
    """
    🇹🇷 Diskten bir görsel açar, RGB'ye çevirir ve belirtilen boyuta yeniden boyutlandırır.
    🇬🇧 Opens an image file, converts to RGB, and resizes to the given size.

    Args:
        path (str): 🇹🇷 Görsel dosya yolu / 🇬🇧 Path to the image file.
        size (tuple): 🇹🇷 (genişlik, yükseklik) / 🇬🇧 (width, height)

    Returns:
        Image.Image: 🇹🇷 Yeniden boyutlandırılmış PIL görseli / 🇬🇧 Resized PIL Image.
    """
    img = Image.open(path).convert("RGB")
    return img.resize(size)

# 🇹🇷 Basit zaman damgalı log mesajı yazdırır
# 🇬🇧 Prints a timestamped log message
def log(message: str):
    """
    🇹🇷 Konsola zaman damgalı bir mesaj yazdırır.
    🇬🇧 Prints a timestamped message to the console.

    Args:
        message (str): 🇹🇷 Loglanacak mesaj / 🇬🇧 Message to log.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")
