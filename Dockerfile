# ---- STAGE 1: BUILDER ----
# This stage installs all dependencies, builds the environment, and downloads models.
# It includes build tools that will not be in the final image.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

# Set non-interactive environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# Define a persistent cache directory for uv
ENV UV_CACHE_DIR=/app/.uv-cache

WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    ffmpeg libgl1-mesa-glx libsm6 libxrender1 \
    wget curl git build-essential \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Install `uv` by copying the static binary from its official image into a directory in the PATH.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set python3.10 as the default python and pip as pip3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create the cache directory for uv
RUN mkdir -p ${UV_CACHE_DIR}

# Using --mount=type=cache utilizes Docker's build cache for uv, speeding up subsequent builds.
# Using --system installs packages to the global Python environment, standard for containers.
# Install PyTorch first as it's a large, foundational layer.
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    uv pip install --system torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install the remaining general and ML dependencies
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    uv pip install --system \
    fastapi uvicorn[standard] websockets python-multipart ffmpeg-python \
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
    "huggingface-hub[cli]" \
    scikit-image scipy decord kornia tqdm \
    pydub PyYAML

# First, install the OpenMMLab installer tool, `mim`.
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    uv pip install --system openmim

# Now, use `mim` to install the MMLab packages.
# `mim` will automatically find the correct pre-built wheels for the installed PyTorch and CUDA versions,
# avoiding the problematic source compilation.
RUN mim install mmengine "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0"

# Copy the application source code and model download script
COPY . .

# List all files with details in the current directory (/app).
# This will show us if download_weights.sh is present, what its permissions are, and its exact name.
RUN ls -la

# Convert the script from Windows (CRLF) to Unix (LF) line endings.
# The `sed` utility is used to remove the carriage return characters (\r).
# Then make the download script executable and run it.
RUN sed -i 's/\r$//' ./download_weights.sh && \
    chmod +x ./download_weights.sh && \
    ./download_weights.sh

# ---- STAGE 2: FINAL IMAGE ----
# This is the production image. It copies only necessary artifacts from the builder.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install only the required RUNTIME system dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    ffmpeg libgl1-mesa-glx libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Set python3.10 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --set python /usr/bin/python3.10

# Copy the installed Python packages from the builder stage.
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Copy executables installed by pip/mim (like uvicorn, gradio, mim etc.)
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code and downloaded models from the builder stage
COPY --from=builder /app /app

# Create the temporary directory required by your application
RUN mkdir temp_files

# Expose the ports for the FastAPI application and Gradio UI
EXPOSE 8000
EXPOSE 7860

# Define the default command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]