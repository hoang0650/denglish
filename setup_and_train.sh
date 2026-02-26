#!/bin/bash

# Thoát ngay nếu có lỗi xảy ra
set -e

echo "========================================"
echo "AI English-German Tutor Training Script"
echo "========================================"

# 1. Cài đặt các thư viện cần thiết
echo "\n--- 1. Cài đặt các thư viện từ requirements.txt ---"
pip install -r requirements.txt

# 2. Kiểm tra và cấu hình biến môi trường
echo "\n--- 2. Kiểm tra và cấu hình biến môi trường ---"
if [ -z "$HF_TOKEN" ]; then
    echo "Cảnh báo: Biến môi trường HF_TOKEN chưa được đặt. Vui lòng đặt nó để upload model lên HuggingFace."
    echo "Ví dụ: export HF_TOKEN=\"hf_YOUR_HUGGINGFACE_TOKEN\""
fi

if [ -z "$WANDB_API_KEY" ]; then
    echo "Cảnh báo: Biến môi trường WANDB_API_KEY chưa được đặt. Vui lòng đặt nó để theo dõi huấn luyện với WandB."
    echo "Ví dụ: export WANDB_API_KEY=\"YOUR_WANDB_API_KEY\""
fi

# 3. Chuẩn bị dữ liệu
echo "\n--- 3. Chuẩn bị dữ liệu huấn luyện ---" 
python data_preparation.py

# 4. Huấn luyện mô hình
echo "\n--- 4. Bắt đầu huấn luyện mô hình ---"
python train.py

# 5. Upload mô hình lên HuggingFace (nếu HF_TOKEN được đặt)
echo "\n--- 5. Upload mô hình lên HuggingFace (nếu HF_TOKEN được đặt) ---"
if [ -n "$HF_TOKEN" ]; then
    python upload_hf.py
else
    echo "Bỏ qua bước upload lên HuggingFace vì HF_TOKEN chưa được đặt."
fi

echo "\n========================================"
echo "Quá trình huấn luyện và cấu hình hoàn tất!"
echo "========================================"