import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Đọc cấu hình từ file config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

base_model_id = config["model"]["id"]
outdir_path = config["model"]["output_dir"] # Thư mục chứa adapter sau khi train
hf_repo_id = "phgrouptechs/Denglish-8B-Instruct"

print("1. Đang tải Base Model và Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu", 
)

print(f"2. Đang đọc LoRA từ {outdir_path} và Gộp (Merge)...")
model = PeftModel.from_pretrained(base_model, outdir_path)
merged_model = model.merge_and_unload()

print(f"3. Đang đẩy mô hình HOÀN CHỈNH lên {hf_repo_id}...")
# Lệnh này sẽ tạo ra file config.json và các file weights lớn trên Hugging Face
merged_model.push_to_hub(hf_repo_id)
tokenizer.push_to_hub(hf_repo_id)

print("--- Hoàn tất việc gộp và đẩy model! ---")