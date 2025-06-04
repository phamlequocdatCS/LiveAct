#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting model downloads within the container..."

# Define the target base directory for models, now correctly under MuseTalk
# This assumes the script is run from /app (WORKDIR in Dockerfile)
MODEL_BASE_DIR="MuseTalk/models" # KEY CHANGE

echo "Ensuring necessary directories exist inside $MODEL_BASE_DIR..."
# These directories are already created by the Dockerfile,
# but mkdir -p is safe and confirms.
mkdir -p "$MODEL_BASE_DIR/musetalk" \
         "$MODEL_BASE_DIR/musetalkV15" \
         "$MODEL_BASE_DIR/syncnet" \
         "$MODEL_BASE_DIR/dwpose" \
         "$MODEL_BASE_DIR/face-parse-bisent" \
         "$MODEL_BASE_DIR/sd-vae" \
         "$MODEL_BASE_DIR/whisper"

# Set HuggingFace mirror endpoint
export HF_ENDPOINT=https://hf-mirror.com

# Download MuseTalk V1.0 weights
# Files like "musetalk/pytorch_model.bin" from repo will go into $MODEL_BASE_DIR/musetalk/pytorch_model.bin
# e.g., MuseTalk/models/musetalk/pytorch_model.bin
echo "Downloading TMElyralab/MuseTalk V1.0 weights..."
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir "$MODEL_BASE_DIR" \
  --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin" \
  --cache-dir "/app/.hf_cache" # Explicit cache dir for consistency

# Download MuseTalk V1.5 weights
# Files like "musetalkV15/unet.pth" from repo will go into $MODEL_BASE_DIR/musetalkV15/unet.pth
# e.g., MuseTalk/models/musetalkV15/unet.pth
echo "Downloading TMElyralab/MuseTalk V1.5 weights..."
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir "$MODEL_BASE_DIR" \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth" \
  --cache-dir "/app/.hf_cache"

# Download SD VAE weights into $MODEL_BASE_DIR/sd-vae
echo "Downloading stabilityai/sd-vae-ft-mse..."
huggingface-cli download stabilityai/sd-vae-ft-mse \
  --local-dir "$MODEL_BASE_DIR/sd-vae" \
  --include "config.json" "diffusion_pytorch_model.bin" \
  --cache-dir "/app/.hf_cache"

# Download Whisper weights into $MODEL_BASE_DIR/whisper
echo "Downloading openai/whisper-tiny..."
huggingface-cli download openai/whisper-tiny \
  --local-dir "$MODEL_BASE_DIR/whisper" \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json" \
  --cache-dir "/app/.hf_cache"

# Download DWPose weights into $MODEL_BASE_DIR/dwpose
echo "Downloading yzd-v/DWPose..."
huggingface-cli download yzd-v/DWPose \
  --local-dir "$MODEL_BASE_DIR/dwpose" \
  --include "dw-ll_ucoco_384.pth" \
  --cache-dir "/app/.hf_cache"

# Download SyncNet weights into $MODEL_BASE_DIR/syncnet
echo "Downloading ByteDance/LatentSync..."
huggingface-cli download ByteDance/LatentSync \
  --local-dir "$MODEL_BASE_DIR/syncnet" \
  --include "latentsync_syncnet.pt" \
  --cache-dir "/app/.hf_cache"

# Download Face Parse Bisent weights into $MODEL_BASE_DIR/face-parse-bisent
echo "Downloading face-parse-bisent 79999_iter.pth using gdown..."
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O "$MODEL_BASE_DIR/face-parse-bisent/79999_iter.pth"

echo "Downloading resnet18-5c106cde.pth using curl..."
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o "$MODEL_BASE_DIR/face-parse-bisent/resnet18-5c106cde.pth"

echo "✅ All weights have been downloaded successfully to $MODEL_BASE_DIR!"