# Use a base image with CUDA 11.8, cuDNN 8, and Ubuntu 22.04, which includes Python 3.10
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app/MuseTalk
ENV PIP_CACHE_DIR=/app/.pip-cache

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10-venv python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libsm6 \
    libxrender1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# --- Install Python Dependencies following the specified order ---

# 1. General dependencies (excluding PyTorch for now)
RUN python3.10 -m pip install \
    fastapi uvicorn[standard] websockets python-multipart ffmpeg-python

# 2. Install target PyTorch version FIRST
# This ensures MMLab packages are built/installed against the correct PyTorch
RUN python3.10 -m pip install \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

RUN python3.10 -m pip install \
    diffusers==0.30.2 \
    accelerate==0.28.0 \
    numpy==1.23.5 \
    tensorflow==2.12.0 \
    tensorboard==2.12.0 \
    opencv-python==4.9.0.80 \
    soundfile==0.12.1 \
    transformers==4.39.2 \
    huggingface_hub==0.30.2 \
    librosa==0.11.0 \
    einops==0.8.1 \
    gradio==5.24.0 \
    gdown \
    requests \
    imageio[ffmpeg] \
    omegaconf \
    ffmpeg-python \
    moviepy

# 3. Install OpenMMLab tools and libraries using mim
RUN python3.10 -m pip install --upgrade setuptools six
RUN python3.10 -m pip install -U openmim
RUN python3.10 -m mim install mmengine
RUN python3.10 -m mim install "mmcv==2.0.1"
RUN python3.10 -m mim install "mmdet==3.1.0"
RUN python3.10 -m mim install "mmpose==1.1.0"

RUN python3.10 -m pip install \
    "huggingface-hub[cli]" \
    scikit-image scipy  decord kornia tqdm \
    pydub PyYAML

# --- Model Downloading and Code Copy ---
COPY download_weights.sh .
RUN chmod +x download_weights.sh

RUN mkdir -p MuseTalk/models/musetalk \
    MuseTalk/models/musetalkV15 \
    MuseTalk/models/syncnet \
    MuseTalk/models/dwpose \
    MuseTalk/models/face-parse-bisent \
    MuseTalk/models/sd-vae \
    MuseTalk/models/whisper

RUN ./download_weights.sh

COPY main.py .
COPY client_test_batch.py .
COPY gradio_frontend.py .
COPY MuseTalk/ MuseTalk/

RUN mkdir temp_files

RUN python3.10 -m pip install -U numpy==1.23.5

RUN python3.10 -m pip install -U opencv-python

EXPOSE 8000
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]