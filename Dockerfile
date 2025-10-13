FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip
# Torch est déjà présent dans l'image de base. On ajoute seulement les libs nécessaires.
# Diffusers + Transformers + auxiliaires
RUN pip3 install diffusers==0.29.0 transformers==4.44.0 accelerate==0.33.0 safetensors==0.4.3 \
    pillow numpy boto3 requests runpod

# (Visage constant) FaceID avec InsightFace (GPU)
RUN pip3 install onnxruntime-gpu==1.18.1 insightface==0.7.3 opencv-python-headless==4.9.0.80
RUN pip3 install hf_transfer

# Optional: accélérations PyTorch pour attention
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app
COPY src/ /app
CMD ["python3", "/app/handler.py"]


