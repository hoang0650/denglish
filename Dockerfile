FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    tesseract-ocr-deu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# --- PHÉP THUẬT NẰM Ở ĐÂY: TẢI TRỰC TIẾP MODEL VÀO TRONG IMAGE ---
RUN pip install -U "huggingface_hub[cli]"
RUN huggingface-cli download unsloth/llama-3-8b-Instruct-bnb-4bit
RUN huggingface-cli download phgrouptechs/Denglish-8B-Instruct
# Tải luôn Whisper cho vào hộp
RUN python -c "import whisper; whisper.load_model('small', device='cpu')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]