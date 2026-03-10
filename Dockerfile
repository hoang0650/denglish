FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1

# Cài đặt công cụ hệ thống cho OCR và Audio
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    tesseract-ocr-deu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Cài đặt thư viện trước để tối ưu layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# 2. [QUAN TRỌNG] Copy toàn bộ code VÀ file test_input.json vào /app
# Đảm bảo file test_input.json nằm cùng thư mục với Dockerfile trên máy bạn
COPY . /app

# 3. Tạo sẵn thư mục mount để tránh lỗi logic
RUN mkdir -p /workspace/llama3-base /workspace/denglish-model

# Lệnh chạy chính thức sử dụng python3
CMD ["python3", "-u", "handler.py"]