FROM nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip

# Torch CUDA 12.x + xformers
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install xformers==0.0.23.post1

# Diffusers + Transformers + auxiliaires
RUN pip3 install diffusers==0.29.0 transformers==4.44.0 accelerate==0.33.0 safetensors==0.4.3 \
    pillow numpy boto3 requests runpod

# (Visage constant) Insightface/InstantID – branché plus tard
RUN pip3 install insightface onnxruntime-gpu==1.18.0 controlnet-aux==0.0.8

WORKDIR /app
COPY src/ /app
CMD ["python3", "/app/handler.py"]


