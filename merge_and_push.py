import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Đọc đường dẫn LoRA từ file config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

outdir_path = config["model"]["output_dir"] 
hf_repo_id = "phgrouptechs/Denglish-8B-Instruct"

# ==========================================
# BƯỚC QUAN TRỌNG NHẤT: ĐIỀN BASE MODEL 16-BIT
# ==========================================
# Nếu lúc train bạn dùng mô hình 4-bit (vd: "unsloth/llama-3-8b-Instruct-bnb-4bit")
# Thì ở đây bạn BẮT BUỘC phải đổi tên thành bản gốc (chưa nén) của nó.
# Ví dụ: "meta-llama/Meta-Llama-3-8B-Instruct" hoặc "NousResearch/Meta-Llama-3-8B-Instruct"

base_model_16bit_id = "phgrouptechs/Denglish-8B-Instruct" 

print(f"1. Đang tải Bản gốc 16-bit: {base_model_16bit_id}...")
tokenizer = AutoTokenizer.from_pretrained(base_model_16bit_id)

# Tải bản 16-bit lên CPU để gộp (không bị lỗi 4-bit nữa)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_16bit_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu", 
)

print(f"2. Đang đọc LoRA từ {outdir_path} và Gộp (Merge)...")
# Đắp LoRA lên bản 16-bit
model = PeftModel.from_pretrained(base_model, outdir_path)
merged_model = model.merge_and_unload()

print(f"3. Đang đẩy mô hình HOÀN CHỈNH lên {hf_repo_id}...")
merged_model.push_to_hub(hf_repo_id)
tokenizer.push_to_hub(hf_repo_id)

print("--- Hoàn tất việc gộp và đẩy model! ---")