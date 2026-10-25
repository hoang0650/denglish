FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# AIMX_GPU_DEPS=0 builds a small CPU-only image (puller / prefetch only).
ARG AIMX_GPU_DEPS=1
ENV PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/runpod-volume/huggingface \
    HF_XET_HIGH_PERFORMANCE=1 \
    AIMX_RUNTIME=serverless \
    AIMX_VOLUME=/runpod-volume
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.worker.txt requirements.gpu.txt requirements.tts.txt constraints-torch.txt ./
# Pin CUDA torch first; every later install runs with -c constraints-torch.txt.
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
 && pip install --no-cache-dir -c constraints-torch.txt -r requirements.worker.txt
RUN if [ "$AIMX_GPU_DEPS" = "1" ]; then \
      pip install --no-cache-dir -c constraints-torch.txt -r requirements.gpu.txt \
   && pip install --no-cache-dir -c constraints-torch.txt -r requirements.tts.txt \
   && pip install --no-cache-dir --no-deps chatterbox-tts==0.1.7 vieneu==3.8.3 neucodec==0.0.6 \
   && python -c "import torch; assert torch.__version__.startswith('2.8.0'), torch.__version__" ; \
    fi
COPY catalog.py db.py media.py handler.py pod_pull.py http_server.py prefetch_volume.py ./
COPY migrations ./migrations
COPY scripts ./scripts
RUN mkdir -p /runpod-volume/huggingface /runpod-volume/outputs
CMD ["python", "-u", "handler.py"]
