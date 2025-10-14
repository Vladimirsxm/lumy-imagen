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
    RUN pip install cython scikit-build-core \
     && pip install --no-build-isolation insightface==0.7.0
    
    # 6) IP-Adapter (FaceID Plus XL) — via ZIP (pas de git)
    RUN pip install --no-cache-dir \
      "https://github.com/h94/IP-Adapter/archive/refs/heads/main.zip"
    
    # Sanity-check : échoue le build si la classe n’est pas importable
    RUN python - <<'PY'
    import sys
    try:
        import ip_adapter
        print("ip_adapter path:", ip_adapter.__file__)
        from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlusXL
        print("IPAdapterFaceIDPlusXL import OK")
    except Exception as e:
        print("IP-Adapter import FAILED:", repr(e))
        sys.exit(1)
    PY
    
    # Caches persistants
    RUN mkdir -p $HF_HOME $TRANSFORMERS_CACHE $INSIGHTFACE_HOME && chmod -R 777 $HF_HOME $INSIGHTFACE_HOME
    
    WORKDIR /app
    COPY src/ /app
    
    # Vérifier la syntaxe Python au build (utile)
    RUN python - <<'PY'
    import py_compile, traceback, sys
    try:
        py_compile.compile('/app/handler.py', doraise=True)
        print('py_compile: OK')
    except Exception:
        print('py_compile: FAILED')
        traceback.print_exc()
        sys.exit(1)
    PY
    
    CMD ["python", "-u", "/app/handler.py"]