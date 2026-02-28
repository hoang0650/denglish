FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y ffmpeg git tesseract-ocr libtesseract-dev tesseract-ocr-eng tesseract-ocr-deu && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r requirements.txt
COPY handler.py .
CMD ["python", "-u", "handler.py"]