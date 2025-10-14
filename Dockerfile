# --- bust cache si besoin ---
    ARG BUILD_NO=2

    FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
    
    ENV DEBIAN_FRONTEND=noninteractive \
        PYTHONUNBUFFERED=1 \
        PIP_NO_CACHE_DIR=1 \
        HF_HOME=/opt/hf_cache \
        TRANSFORMERS_CACHE=/opt/hf_cache/transformers \
        INSIGHTFACE_HOME=/opt/insightface \
        HF_HUB_ENABLE_HF_TRANSFER=1 \
        PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
    
    # Outils de base + libs runtime OpenCV
    RUN apt-get update && apt-get install -y git git-lfs ffmpeg libgl1 libglib2.0-0 && git lfs install && rm -rf /var/lib/apt/lists/*
    
    # pip à jour
    RUN python -m pip install --upgrade pip setuptools wheel
    
    # 1) Libs de base
    RUN pip install numpy==1.26.4 "pillow>=10.0.0" requests boto3 runpod hf_transfer "huggingface_hub>=0.23"
    
    # 2) Diffusers/Transformers/Accelerate/Safetensors
    RUN pip install diffusers==0.29.0 transformers==4.44.0 accelerate==0.33.0 safetensors==0.4.3
    
    # 3) ONNX + ONNX Runtime GPU (combo stable CUDA 12.1)
    RUN pip install onnx==1.16.0 onnxruntime-gpu==1.18.0
    
    # 4) OpenCV (headless) + InsightFace (wheels CUDA 12.1)
    RUN pip install opencv-python-headless==4.9.0.80 \
        && pip install --extra-index-url https://download.pytorch.org/whl/cu121 insightface==0.7.3
    
    # Caches persistants
    RUN mkdir -p $HF_HOME $TRANSFORMERS_CACHE $INSIGHTFACE_HOME && chmod -R 777 $HF_HOME $INSIGHTFACE_HOME
    
    # (Optionnel) pré-téléchargement supprimé pour éviter les erreurs heredoc
    
    WORKDIR /app
    
    # Copie uniquement le code handler (dans src/)
    COPY src/ /app
    
    # Si ton handler est à la racine du repo et s'appelle handler.py, rien à faire.
    # Si tu l'as dans un sous-dossier (ex: src/handler.py), assure-toi que COPY ci-dessus l'amène bien dans /app.
    
    CMD ["python", "-u", "/app/handler.py"]