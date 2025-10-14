FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache/transformers \
    INSIGHTFACE_HOME=/opt/insightface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Outils
RUN apt-get update && apt-get install -y git git-lfs && git lfs install && rm -rf /var/lib/apt/lists/*

# Librairies système requises par OpenCV (ffmpeg, libgl, glib)
RUN apt-get update && apt-get install -y ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# pip/setuptools à jour
RUN python -m pip install --upgrade pip setuptools wheel

# 1) Base libs
RUN pip install "pillow>=10.0.0" numpy requests boto3 runpod hf_transfer "huggingface_hub>=0.23"

# 2) Diffusers/Transformers/Accelerate/Safetensors
RUN pip install diffusers==0.29.0 transformers==4.44.0 accelerate==0.33.0 safetensors==0.4.3

# 3) ONNX + ONNXRUNTIME GPU (versions compatibles CUDA 12.1)
#    NOTE: 1.18.0 est plus sûr que 1.18.1 sur certaines images CUDA 12.1
RUN pip install onnx==1.16.0 onnxruntime-gpu==1.18.0

# 4) OpenCV + InsightFace (pin versions stables)
RUN pip install opencv-python-headless==4.10.0.84 insightface==0.7.3

# Caches persistants
RUN mkdir -p $HF_HOME $TRANSFORMERS_CACHE $INSIGHTFACE_HOME && chmod -R 777 $HF_HOME $INSIGHTFACE_HOME

# --- Pré-téléchargement des modèles ---
RUN python - <<'PY'
from huggingface_hub import snapshot_download
import os
cache = os.getenv("HF_HOME")

# SDXL base, refiner, VAE
snapshot_download("stabilityai/stable-diffusion-xl-base-1.0", cache_dir=cache)
snapshot_download("stabilityai/stable-diffusion-xl-refiner-1.0", cache_dir=cache)
snapshot_download("madebyollin/sdxl-vae-fp16-fix", cache_dir=cache)

# IP-Adapter FaceID (poids SDXL)
snapshot_download(
    "h94/IP-Adapter-FaceID",
    cache_dir=cache,
    allow_patterns=["ip-adapter-faceid-plusv2_sdxl.bin"]
)

# InsightFace: télécharge les modèles au build (CPU ici)
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)
print("Prewarm InsightFace done")
PY

WORKDIR /app
COPY src/ /app

CMD ["python", "-u", "/app/handler.py"]