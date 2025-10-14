FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache/transformers \
    INSIGHTFACE_HOME=/opt/insightface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Outils de base
RUN apt-get update && apt-get install -y git git-lfs && git lfs install && rm -rf /var/lib/apt/lists/*

# pip à jour
RUN python -m pip install --upgrade pip

# --- Librairies Python (versions compatibles CUDA 12.1) ---
RUN pip install \
    diffusers==0.29.0 \
    transformers==4.44.0 \
    accelerate==0.33.0 \
    safetensors==0.4.3 \
    pillow>=10.0.0 \
    numpy \
    boto3 \
    requests \
    runpod \
    huggingface_hub>=0.23 \
    insightface==0.7.3 \
    onnx==1.16.0 \
    onnxruntime-gpu==1.18.1 \
    opencv-python-headless==4.10.0.84 \
    hf_transfer

# Caches persistants
RUN mkdir -p $HF_HOME $TRANSFORMERS_CACHE $INSIGHTFACE_HOME && chmod -R 777 $HF_HOME $INSIGHTFACE_HOME

# --- Pré-téléchargement des modèles pour éviter les re-downloads au démarrage ---
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

# InsightFace: téléchargement des modèles (CPU au build, GPU au runtime)
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)  # CPU ici pour éviter la dépendance au GPU pendant le build
print("Prewarm InsightFace done")
PY

WORKDIR /app
COPY src/ /app

# (Optionnel mais propre) re-déclarer les env utiles au runtime
ENV HF_HOME=/opt/hf_cache
ENV TRANSFORMERS_CACHE=/opt/hf_cache/transformers
ENV INSIGHTFACE_HOME=/opt/insightface

CMD ["python", "-u", "/app/handler.py"]