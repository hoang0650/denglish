FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1

# Cài đặt các thư viện hệ thống cần thiết cho OCR và Audio
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    tesseract-ocr-deu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Copy file requirements trước để tận dụng Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# 2. [QUAN TRỌNG] Copy toàn bộ code vào thư mục /app
# Lệnh này sẽ đưa handler.py và các file liên quan vào container
COPY . /app

# Lệnh chạy chính thức
CMD ["python", "-u", "handler.py"]