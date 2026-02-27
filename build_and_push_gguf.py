import os
import subprocess
from huggingface_hub import snapshot_download, HfApi

def run_command(command):
    """Hàm hỗ trợ chạy lệnh terminal từ Python"""
    print(f"\n[Denglish-AI] Đang chạy lệnh: {command}")
    subprocess.run(command, shell=True, check=True)

def main():
    # Cấu hình Repo và Tên file
    repo_id = "phgrouptechs/Denglish-8B-Instruct"
    gguf_f16_name = "Denglish-8B-Instruct-F16.gguf"
    gguf_q4_name = "Denglish-8B-Instruct-Q4_K_M.gguf"
    
    # 1. Tải bản gốc (Base weights) từ Hugging Face về máy (Cache)
    print("\n--- BƯỚC 1: Tải Model từ Hugging Face ---")
    model_path = snapshot_download(repo_id=repo_id)
    print(f"Model đã tải xong tại: {model_path}")
    
    # 2. Chuẩn bị công cụ llama.cpp
    print("\n--- BƯỚC 2: Cài đặt & Biên dịch llama.cpp ---")
    if not os.path.exists("llama.cpp"):
        run_command("git clone https://github.com/ggerganov/llama.cpp")
        # Biên dịch llama.cpp để lấy công cụ llama-quantize
        run_command("cd llama.cpp && pip install -r requirements.txt && make")
    else:
        print("Thư mục llama.cpp đã tồn tại, bỏ qua bước tải lại.")

    # 3. Chuyển đổi sang chuẩn GGUF (16-bit)
    print("\n--- BƯỚC 3: Chuyển đổi sang GGUF (F16) ---")
    if not os.path.exists(gguf_f16_name):
        run_command(f"python llama.cpp/convert_hf_to_gguf.py {model_path} --outfile {gguf_f16_name} --outtype f16")
    else:
        print(f"File {gguf_f16_name} đã tồn tại.")

    # 4. Ép xung (Quantize) xuống 4-bit (Siêu nhẹ, dành cho CPU)
    print("\n--- BƯỚC 4: Lượng tử hóa xuống Q4_K_M ---")
    if not os.path.exists(gguf_q4_name):
        run_command(f"./llama.cpp/llama-quantize {gguf_f16_name} {gguf_q4_name} q4_k_m")
    else:
        print(f"File {gguf_q4_name} đã tồn tại.")

    # 5. Upload file GGUF 4-bit lên lại Hugging Face
    print("\n--- BƯỚC 5: Đẩy file GGUF lên Hugging Face ---")
    api = HfApi()
    
    # Đẩy file 4-bit (File quan trọng nhất cho CPU)
    api.upload_file(
        path_or_fileobj=gguf_q4_name,
        path_in_repo=gguf_q4_name, # Tên hiển thị trên Hugging Face
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add Q4_K_M GGUF model for CPU inference"
    )
    
    print("\n🎉 HOÀN TẤT! Bạn có thể kiểm tra kho Hugging Face của mình.")

if __name__ == "__main__":
    main()