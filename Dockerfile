# --- bust cache si besoin ---
    ARG BUILD_NO=8

    # Image PyTorch avec CUDA 11.8 (meilleure compatibilité GPU)
    FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
    
    ENV DEBIAN_FRONTEND=noninteractive \
        PYTHONUNBUFFERED=1 \
        PIP_NO_CACHE_DIR=1 \
        HF_HOME=/opt/hf_cache \
        TRANSFORMERS_CACHE=/opt/hf_cache/transformers \
        INSIGHTFACE_HOME=/opt/insightface \
        HF_HUB_ENABLE_HF_TRANSFER=1 \
        PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    
    # Outils + runtime OpenCV minimum + toolchain
    RUN apt-get update && apt-get install -y \
        git git-lfs ffmpeg libgl1 libglib2.0-0 \
        build-essential cmake ninja-build python3-dev libopenblas-dev liblapack-dev ca-certificates \
     && git lfs install \
     && rm -rf /var/lib/apt/lists/*
    
    RUN python -m pip install --upgrade pip setuptools wheel
    
    # 1) Bases
    RUN pip install numpy==1.26.4 "pillow>=10.0.0" requests boto3 runpod hf_transfer "huggingface_hub>=0.23"
    
    # 2) Diffusion stack
    RUN pip install diffusers==0.31.0 transformers==4.44.0 accelerate==0.33.0 safetensors==0.4.3
    
    # 3) ONNXRuntime CPU (PyTorch déjà inclus dans l'image de base Runpod)
    RUN pip install onnx==1.16.0 onnxruntime==1.18.0
    
    # 4) OpenCV + dépendances binaires
    RUN pip install --only-binary=:all: opencv-python-headless==4.9.0.80 scipy==1.11.4 scikit-image==0.22.0
    
    # 5) InsightFace (sans build isolation pour éviter des surprises)
    RUN pip install cython scikit-build-core einops \
     && pip install --no-build-isolation insightface==0.7.0
    
    # Télécharger les modèles InsightFace depuis un repo HF connu
    RUN python - <<'PY'
import os
from huggingface_hub import snapshot_download
insightface_home = "/opt/insightface"
models_dir = f"{insightface_home}/models"
os.makedirs(models_dir, exist_ok=True)

# Télécharger antelopev2 depuis DIAMONIK7777/antelopev2
try:
    snapshot_download(
        repo_id="DIAMONIK7777/antelopev2",
        local_dir=f"{models_dir}/antelopev2",
        local_dir_use_symlinks=False
    )
    print(f"antelopev2 downloaded to {models_dir}/antelopev2")
    # Vérifier les fichiers
    import glob
    files = glob.glob(f"{models_dir}/antelopev2/**/*", recursive=True)
    print(f"Downloaded {len(files)} files")
    for f in files[:10]:
        print(f"  - {f}")
except Exception as e:
    print(f"antelopev2 download failed: {e}")
    import traceback
    traceback.print_exc()
PY
    
    # 6) IP-Adapter (FaceID Plus XL) — via ZIP (pas de git)
    RUN pip install --no-cache-dir ip-adapter
    
    # Sanity-check léger: vérifie juste que ip_adapter est installé
    RUN python -c "import ip_adapter; print('ip_adapter OK')"
    
    # Caches persistants
    RUN mkdir -p $HF_HOME $TRANSFORMERS_CACHE $INSIGHTFACE_HOME && chmod -R 777 $HF_HOME $INSIGHTFACE_HOME
    
    WORKDIR /app
    COPY src/ /app
    
    # Vérifier la syntaxe Python du handler
    RUN python -m py_compile /app/handler.py
    
    CMD ["python", "-u", "/app/handler.py"]