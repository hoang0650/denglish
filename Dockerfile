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
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 socat ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
# Tailscale (userspace mode, started by scripts/entrypoint.sh only when TS_AUTHKEY is set).
ARG TAILSCALE_VERSION=1.104.2
RUN curl -fsSL "https://pkgs.tailscale.com/stable/tailscale_${TAILSCALE_VERSION}_amd64.tgz" -o /tmp/ts.tgz \
 && tar -xzf /tmp/ts.tgz -C /tmp \
 && install -m 0755 "/tmp/tailscale_${TAILSCALE_VERSION}_amd64/tailscale" "/tmp/tailscale_${TAILSCALE_VERSION}_amd64/tailscaled" /usr/local/bin/ \
 && rm -rf /tmp/ts.tgz "/tmp/tailscale_${TAILSCALE_VERSION}_amd64" \
 && tailscale version
WORKDIR /app
COPY requirements.worker.txt requirements.gpu.txt requirements.tts.txt constraints-torch.txt requirements.lock.txt ./
# Pin CUDA torch first; every later install runs with -c requirements.lock.txt
# (full pinned set, generated with uv for Python 3.12 + torch 2.8.0/cu128).
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128
# Ubuntu ships cryptography and six as distro packages without a pip RECORD file,
# so pip cannot uninstall them; lay the pinned versions over them first.
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir --ignore-installed --no-deps -c requirements.lock.txt cryptography six \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
 && pip install --no-cache-dir -c requirements.lock.txt -r requirements.worker.txt
RUN if [ "$AIMX_GPU_DEPS" = "1" ]; then \
      pip install --no-cache-dir -c requirements.lock.txt -r requirements.gpu.txt \
   && pip install --no-cache-dir -c requirements.lock.txt -r requirements.tts.txt \
   && pip install --no-cache-dir --no-deps chatterbox-tts==0.1.7 vieneu==3.8.3 neucodec==0.0.6 \
   && python -c "import torch; assert torch.__version__.startswith('2.8.0'), torch.__version__" ; \
    fi
COPY catalog.py db.py media.py handler.py pod_pull.py http_server.py prefetch_volume.py ./
COPY migrations ./migrations
COPY scripts ./scripts
COPY artifacts.py ./
# The volume is for model weights/caches only; job files use /tmp/aimx-jobs.
RUN mkdir -p /runpod-volume/huggingface /tmp/aimx-jobs
# handler.py calls runpod.serverless.start({"handler": handler, ...}).
# The entrypoint only adds the optional Tailscale forwards in front of it.
CMD ["bash", "/app/scripts/entrypoint.sh"]
