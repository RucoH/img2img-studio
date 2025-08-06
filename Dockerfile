# syntax=docker/dockerfile:1.4
FROM python:3.13-slim

WORKDIR /app

# (İhtiyaç varsa) sistem paketleri
RUN apt-get update \
  && apt-get install -y --no-install-recommends git \
  && rm -rf /var/lib/apt/lists/*

# pip timeout/retry ve unbuffered I/O
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=10 \
    PIP_RESUME_RETRIES=10 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

COPY requirements.txt .

# Pip ile bağımlılıkları indir (ilk seferde yavaş ama sonraki build'ler cache'den)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --no-cache-dir \
      --default-timeout=$PIP_DEFAULT_TIMEOUT \
      --retries=$PIP_RETRIES \
      --resume-retries=$PIP_RESUME_RETRIES \
      -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py", "--server-name", "0.0.0.0", "--server-port", "7860"]
