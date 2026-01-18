FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      pkg-config \
      python3-dev \
      curl \
      tesseract-ocr \
      libgl1-mesa-glx \
      libglib2.0-0 \
      libgtk-3-0\
      libx11-6 \
      libxrender1 \
      libxtst6 \
      libsm6 \
      libxext6 \
      ffmpeg \
    && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./

RUN pip install --no-cache-dir uv && \
    uv sync --index-strategy unsafe-best-match

COPY . .

CMD ["uv", "run", "python", "main.py"]