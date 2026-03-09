import os
import subprocess
from huggingface_hub import snapshot_download, HfApi

def run_command(command):
    print(f"\n[Denglish-AI] Đang chạy lệnh: {command}")
    subprocess.run(command, shell=True, check=True)

def main():
    repo_id = "phgrouptechs/Denglish-8B-Instruct"
    
    # ĐỔI ĐƯỜNG DẪN LƯU GGUF SANG Ổ /tmp ĐỂ KHÔNG BỊ FULL Ổ MẠNG
    gguf_f16_path = "/tmp/Denglish-8B-Instruct-F16.gguf"
    gguf_q4_path = "/tmp/Denglish-8B-Instruct-Q4_K_M.gguf"
    repo_filename = "Denglish-8B-Instruct-Q4_K_M.gguf" # Tên hiển thị trên web Hugging Face
    
    print("\n--- BƯỚC 1: Tải Model từ Hugging Face ---")
    model_path = snapshot_download(repo_id=repo_id)
    print(f"Model đã tải xong tại: {model_path}")
    
    print("\n--- BƯỚC 2: Cài đặt & Biên dịch llama.cpp bằng CMake ---")
    if not os.path.exists("llama.cpp"):
        run_command("git clone https://github.com/ggerganov/llama.cpp")
    
    # FIX LỖI: Cập nhật bắt buộc các thư viện xử lý token cho Llama 3
    print("\n[Denglish-AI] Cập nhật thư viện để fix lỗi TokenizersBackend và BpeVocab...")
    run_command("pip install --upgrade transformers tokenizers tiktoken sentencepiece")
    
    if not os.path.exists("llama.cpp/build"):
        run_command("cd llama.cpp && pip install -r requirements.txt && cmake -B build && cmake --build build --config Release")
    else:
        print("Đã biên dịch llama.cpp xong, bỏ qua bước build.")

    print("\n--- BƯỚC 3: Chuyển đổi sang GGUF (F16) ---")
    if not os.path.exists(gguf_f16_path):
        run_command(f"python llama.cpp/convert_hf_to_gguf.py {model_path} --outfile {gguf_f16_path} --outtype f16")
    else:
        print(f"File {gguf_f16_path} đã tồn tại.")

    print("\n--- BƯỚC 4: Lượng tử hóa xuống Q4_K_M ---")
    if not os.path.exists(gguf_q4_path):
        quantize_cmd = "./llama.cpp/build/bin/llama-quantize"
        if not os.path.exists(quantize_cmd):
            quantize_cmd = "./llama.cpp/build/llama-quantize"
            
        run_command(f"{quantize_cmd} {gguf_f16_path} {gguf_q4_path} q4_k_m")
    else:
        print(f"File {gguf_q4_path} đã tồn tại.")

    print("\n--- BƯỚC 5: Đẩy file GGUF lên Hugging Face ---")
    api = HfApi()
    
    api.upload_file(
        path_or_fileobj=gguf_q4_path,
        path_in_repo=repo_filename,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add Q4_K_M GGUF model for CPU inference"
    )
    
    print("\n🎉 HOÀN TẤT! Đã đẩy file 4-bit lên Hugging Face.")
    
    # Dọn dẹp file 16GB trung gian để tránh nặng máy
    if os.path.exists(gguf_f16_path):
        os.remove(gguf_f16_path)
        print("Đã dọn dẹp file tạm F16.")

if __name__ == "__main__":
    main()