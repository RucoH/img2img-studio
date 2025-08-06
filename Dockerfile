# 1) Builder aşaması: pip install
FROM python:3.13-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Downloader aşaması: modeli indir, cache’e yaz
FROM builder AS downloader
RUN python - <<EOF
from diffusers import StableDiffusionImg2ImgPipeline
StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)
EOF

# 3) Run aşaması: kodu kopyala ve başlat
FROM python:3.13-slim AS final
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=downloader /root/.cache/huggingface /root/.cache/huggingface
COPY . .
EXPOSE 7860
CMD ["python", "app.py", "--server-name", "0.0.0.0"]
