# --- bust cache si besoin ---
    ARG BUILD_NO=4

    FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
    
    ENV DEBIAN_FRONTEND=noninteractive \
        PYTHONUNBUFFERED=1 \
        PIP_NO_CACHE_DIR=1 \
        HF_HOME=/opt/hf_cache \
        TRANSFORMERS_CACHE=/opt/hf_cache/transformers \
        INSIGHTFACE_HOME=/opt/insightface \
        HF_HUB_ENABLE_HF_TRANSFER=1 \
        PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    
    # Outils + runtime OpenCV minimum
    RUN apt-get update && apt-get install -y \
        git git-lfs ffmpeg libgl1 libglib2.0-0 \
     && git lfs install \
     && rm -rf /var/lib/apt/lists/*
    
    # pip à jour
    RUN python -m pip install --upgrade pip setuptools wheel
    
    # 1) Bases
    RUN pip install numpy==1.26.4 "pillow>=10.0.0" requests boto3 runpod hf_transfer "huggingface_hub>=0.23"
    
    # 2) Diffusion stack
    RUN pip install diffusers==0.29.0 transformers==4.44.0 accelerate==0.33.0 safetensors==0.4.3
    
    # 3) ONNX + ONNXRuntime **CPU** (évite les conflits GPU pour insightface)
    RUN pip install onnx==1.16.0 onnxruntime==1.18.0
    
    # 4) OpenCV + DEPENDANCES BINAIRES AVANT insightface
    #    IMPORTANT: on force des wheels binaires pour éviter toute compile (SciPy/scikit-image).
    RUN pip install --only-binary=:all: opencv-python-headless==4.9.0.80 \
     && pip install --only-binary=:all: scipy==1.11.4 scikit-image==0.22.0
    
    # 5) InsightFace (utilise ORT CPU) — 0.7.3 fonctionne avec les pins ci-dessus
    RUN pip install insightface==0.7.3
    
    # Caches persistants
    RUN mkdir -p $HF_HOME $TRANSFORMERS_CACHE $INSIGHTFACE_HOME && chmod -R 777 $HF_HOME $INSIGHTFACE_HOME
    
    WORKDIR /app
    COPY src/ /app
    
    # (optionnel) sanity check rapide
    # RUN python -c "import insightface, onnxruntime, cv2; print('insightface', insightface.__version__, 'onnxruntime', onnxruntime.__version__, 'cv2', cv2.__version__)"
    
    CMD ["python", "-u", "/app/handler.py"]