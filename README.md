# LiveAct

Demo for live lipsyncing

## Setup

Tested on Python 3.10.12, RTX3060 12GB, Windows 10

```bash
python -m venv venv
.\venv\Scripts\activate

# General dependencies
pip install fastapi uvicorn[standard] websockets python-multipart opencv-python ffmpeg-python

# MuseTalk dependencies
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# MuseTalk
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
mim install "mmpose==1.1.0"

git clone https://github.com/TMElyralab/MuseTalk
cd MuseTalk
download_weights.bat

pip install -r requirements.txt
```

## Demo

### Default demo

```bash
python -m scripts.inference --inference_config configs\inference\test.yaml --result_dir results\test --unet_model_path models\musetalkV15\unet.pth --unet_config models\musetalkV15\musetalk.json --version v15


```
