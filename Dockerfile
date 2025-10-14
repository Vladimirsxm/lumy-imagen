# --- bust cache si besoin ---
    ARG BUILD_NO=6

    FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
    
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
    
    # 3) Torch CUDA 12.1 + ONNXRuntime CPU
    RUN pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
     && pip install onnx==1.16.0 onnxruntime==1.18.0
    
    # 4) OpenCV + dépendances binaires
    RUN pip install --only-binary=:all: opencv-python-headless==4.9.0.80 scipy==1.11.4 scikit-image==0.22.0
    
    # 5) InsightFace (sans build isolation pour éviter des surprises)
    RUN pip install cython scikit-build-core einops \
     && pip install --no-build-isolation insightface==0.7.0
    
    # Télécharger les modèles InsightFace depuis HuggingFace au lieu de l'URL officielle
    RUN python - <<'PY'
import os
from huggingface_hub import hf_hub_download
insightface_home = os.environ.get("INSIGHTFACE_HOME", "/opt/insightface")
os.makedirs(insightface_home, exist_ok=True)
os.makedirs(f"{insightface_home}/models", exist_ok=True)

# Télécharger buffalo_l depuis HF
try:
    model_path = hf_hub_download(
        repo_id="public-data/insightface",
        filename="models/buffalo_l.zip",
        cache_dir=insightface_home
    )
    # Décompresser
    import zipfile
    with zipfile.ZipFile(model_path, 'r') as zip_ref:
        zip_ref.extractall(f"{insightface_home}/models/buffalo_l")
    print("buffalo_l downloaded from HuggingFace")
except Exception as e:
    print(f"buffalo_l download failed: {e}")
    # Fallback: essayer antelopev2
    try:
        model_path = hf_hub_download(
            repo_id="public-data/insightface",
            filename="models/antelopev2.zip",
            cache_dir=insightface_home
        )
        import zipfile
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall(f"{insightface_home}/models/antelopev2")
        print("antelopev2 downloaded from HuggingFace")
    except Exception as e2:
        print(f"antelopev2 download also failed: {e2}")
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