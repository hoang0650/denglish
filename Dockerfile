FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
ENV PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    VIENEU_MODEL=pnnbao-ump/VieNeu-TTS-v3-Turbo \
    VIENEU_MODEL_DIR=/runpod-volume/vieneu-tts-v3-turbo \
    PHOWHISPER_MODEL=vinai/PhoWhisper-large \
    PHOWHISPER_MODEL_DIR=/runpod-volume/phowhisper-large \
    HF_HOME=/runpod-volume/huggingface
RUN apt-get update && apt-get install -y ffmpeg git tesseract-ocr libtesseract-dev tesseract-ocr-eng tesseract-ocr-deu && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# Debian may ship python3-cryptography without a pip RECORD. --ignore-installed
# still installs packages (incl. vieneu); it only skips the broken uninstall step.
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir --ignore-installed cryptography \
 && pip install --no-cache-dir --ignore-installed -r requirements.txt \
 && python -c "import vieneu; print('[build] vieneu', getattr(vieneu, '__version__', 'ok'))"
COPY handler.py tts_vieneu.py stt_phowhisper.py voice_assistant.py config.yaml .
RUN python -c "import tts_vieneu; assert len(tts_vieneu.PRESET_VOICES) == 20; print('[build] tts_vieneu', tts_vieneu.MODEL_ID)"
RUN python -c "import stt_phowhisper; assert stt_phowhisper.MODEL_ID == 'vinai/PhoWhisper-large'; print('[build] stt_phowhisper', stt_phowhisper.MODEL_ID)"
RUN mkdir -p \
      /runpod-volume/llama3-base \
      /runpod-volume/denglish-model \
      /runpod-volume/vieneu-tts-v3-turbo \
      /runpod-volume/phowhisper-large \
      /runpod-volume/huggingface
CMD ["python", "-u", "handler.py"]
