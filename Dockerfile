# Nâng cấp lên PyTorch 2.8.0 và CUDA 12.8 để hỗ trợ trọn vẹn kiến trúc Blackwell (sm_120) của RTX 5090
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1

# Cập nhật OS và cài đặt các công cụ (FFmpeg cho STT, Tesseract cho OCR)
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

# Cài đặt thư viện Python
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]