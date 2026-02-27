import os
import subprocess
from huggingface_hub import snapshot_download, HfApi

def run_command(command):
    print(f"\n[Denglish-AI] Đang chạy lệnh: {command}")
    subprocess.run(command, shell=True, check=True)

def main():
    repo_id = "phgrouptechs/Denglish-8B-Instruct"
    gguf_f16_name = "Denglish-8B-Instruct-F16.gguf"
    gguf_q4_name = "Denglish-8B-Instruct-Q4_K_M.gguf"
    
    print("\n--- BƯỚC 1: Tải Model từ Hugging Face ---")
    model_path = snapshot_download(repo_id=repo_id)
    print(f"Model đã tải xong tại: {model_path}")
    
    print("\n--- BƯỚC 2: Cài đặt & Biên dịch llama.cpp bằng CMake ---")
    if not os.path.exists("llama.cpp"):
        run_command("git clone https://github.com/ggerganov/llama.cpp")
    
    # Sử dụng CMake thay cho make theo bản cập nhật mới nhất của llama.cpp
    if not os.path.exists("llama.cpp/build"):
        run_command("cd llama.cpp && pip install -r requirements.txt && cmake -B build && cmake --build build --config Release")
    else:
        print("Đã biên dịch llama.cpp xong, bỏ qua bước build.")

    print("\n--- BƯỚC 3: Chuyển đổi sang GGUF (F16) ---")
    if not os.path.exists(gguf_f16_name):
        run_command(f"python llama.cpp/convert_hf_to_gguf.py {model_path} --outfile {gguf_f16_name} --outtype f16")
    else:
        print(f"File {gguf_f16_name} đã tồn tại.")

    print("\n--- BƯỚC 4: Lượng tử hóa xuống Q4_K_M ---")
    if not os.path.exists(gguf_q4_name):
        # Đường dẫn file lượng tử hóa sau khi build bằng CMake thường nằm ở 1 trong 2 vị trí này
        quantize_cmd = "./llama.cpp/build/bin/llama-quantize"
        if not os.path.exists(quantize_cmd):
            quantize_cmd = "./llama.cpp/build/llama-quantize"
            
        run_command(f"{quantize_cmd} {gguf_f16_name} {gguf_q4_name} q4_k_m")
    else:
        print(f"File {gguf_q4_name} đã tồn tại.")

    print("\n--- BƯỚC 5: Đẩy file GGUF lên Hugging Face ---")
    api = HfApi()
    
    api.upload_file(
        path_or_fileobj=gguf_q4_name,
        path_in_repo=gguf_q4_name,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add Q4_K_M GGUF model for CPU inference (Built with CMake)"
    )
    
    print("\n🎉 HOÀN TẤT! Đã đẩy file 4-bit lên Hugging Face.")

if __name__ == "__main__":
    main()