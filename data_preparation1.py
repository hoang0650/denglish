import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset
import json
import os
import random

def standardize_format(instruction, input_text, output_text):
    """Hàm ép kiểu dữ liệu về chuẩn Alpaca để hợp nhất không bị lỗi Schema"""
    return {
        "instruction": str(instruction).strip() if instruction else "",
        "input": str(input_text).strip() if input_text else "",
        "output": str(output_text).strip() if output_text else ""
    }

def prepare_datasets():
    print("--- 🚀 LOADING AND STANDARDIZING MULTILINGUAL DATASETS ---")
    all_datasets = []

    # ==========================================
    # 0. CÁC DATASET CŨ CỦA BẠN (Alpaca, German, VN Mix)
    # ==========================================
    try:
        alpaca = load_dataset("yahma/alpaca-cleaned", split="train").select(range(10000))
        all_datasets.append(alpaca)
        print("✅ Loaded Alpaca Cleaned")
    except Exception as e: print(f"⚠️ Error Alpaca: {e}")

    try:
        german = load_dataset("philschmid/translated_tasks_de_google_52k", split="train").select(range(10000))
        all_datasets.append(german)
        print("✅ Loaded German Tasks")
    except Exception as e: print(f"⚠️ Error German: {e}")

    try:
        json_path = os.path.abspath("vn_mix_data.json")
        with open(json_path, "r", encoding="utf-8") as f:
            vn_mix_data = json.load(f)
        vn_ds = Dataset.from_list(vn_mix_data)
        all_datasets.append(vn_ds)
        print(f"Loaded Vietnamese Mix Data: {len(vn_ds)} samples")
    except FileNotFoundError:
        print("Warning: vn_mix_data.json not found. Skipping Vietnamese mix data.")
        vn_ds = Dataset.from_list([]) 
    except json.JSONDecodeError:
        print("Error: vn_mix_data.json is invalid JSON. Skipping.")
        vn_ds = Dataset.from_list([])

    # ==========================================
    # 1. VIETNAMESE INSTRUCTION (Trò chuyện tự nhiên)
    # ==========================================
    try:
        vi_sharegpt = load_dataset(
            "5CD-AI/Vietnamese-OpenGVLab-ShareGPT-4o-gg-translated", 
            "image_caption", 
            split="images"
        )
        
        vi_data = []
        
        # Lấy min để tránh lỗi out of range
        limit = min(59000, len(vi_sharegpt))
        
        for row in vi_sharegpt.select(range(limit)):
            # 2. Đổi tên cột cần lấy thành "conversations_vi"
            convs = row.get("conversations_vi", [])
            
            if len(convs) >= 2:
                vi_data.append(standardize_format(
                    instruction="Trả lời bằng tiếng Việt một cách tự nhiên và mượt mà.",
                    input_text=convs[0].get("value", ""),
                    output_text=convs[1].get("value", "")
                ))
                
        all_datasets.append(Dataset.from_list(vi_data))
        print(f"✅ Loaded Vietnamese Instruction: {len(vi_data)} samples")
    
    except Exception as e: print(f"⚠️ Error Vietnamese Instruction: {e}")

    # ==========================================
    # 2. CONVERSATION & DAILY DIALOG (Luyện nói)
    # ==========================================
    try:
        # Lấy bản tiếng Việt của Daily Dialog làm cốt lõi
        vi_dialog = load_dataset("vietgpt/daily_dialog_vi", split="train")
        dialog_data = []
        for row in vi_dialog.select(range(min(10000, len(vi_dialog)))):
            dialog_data.append(standardize_format(
                instruction="Tiếp tục cuộc hội thoại sau một cách tự nhiên:",
                input_text=row.get("dialog", [""])[0] if isinstance(row.get("dialog"), list) else "",
                output_text=row.get("dialog", ["", ""])[1] if isinstance(row.get("dialog"), list) and len(row.get("dialog")) > 1 else ""
            ))
        all_datasets.append(Dataset.from_list(dialog_data))
        print("✅ Loaded Daily Dialog (Vietnamese)")
    except Exception as e: print(f"⚠️ Error Dialog: {e}")

    # ==========================================
    # 3. GRAMMAR CORRECTION (Sửa lỗi ngữ pháp)
    # ==========================================
    try:
        # Sử dụng lang8 cleaned
        grammar_ds = load_dataset("rahuln2002/GED-lang8-cleaned", split="train")
        grammar_data = []
        for row in grammar_ds.select(range(min(200000, len(grammar_ds)))):
            grammar_data.append(standardize_format(
                instruction="Bạn là gia sư ngôn ngữ. Hãy sửa lỗi ngữ pháp cho câu sau và giữ nguyên ý nghĩa:",
                input_text=row.get("sentence", ""),
                output_text=row.get("corrected_sentence", "") or row.get("sentence", "")
            ))
        all_datasets.append(Dataset.from_list(grammar_data))
        print("✅ Loaded Grammar Correction")
    except Exception as e: print(f"⚠️ Error Grammar Correction: {e}")

    # ==========================================
    # 4. CEFR LEVEL (Phân loại trình độ A1-C2)
    # ==========================================
    try:
        cefr_ds = load_dataset("edesaras/CEFR-Sentence-Level-Annotations", split="train")
        cefr_data = []
        for row in cefr_ds.select(range(min(10000, len(cefr_ds)))):
            cefr_data.append(standardize_format(
                instruction=f"Hãy phân tích và viết một câu tiếng Anh ở trình độ {row.get('level', 'A2')}.",
                input_text="",
                output_text=row.get("sentence", "")
            ))
        all_datasets.append(Dataset.from_list(cefr_data))
        print("✅ Loaded CEFR Sentence Annotations")
    except Exception as e: print(f"⚠️ Error CEFR: {e}")

    # ==========================================
    # 5. GERMAN PARAPHRASE (deutsche-telekom/ger-backtrans-paraphrase)
    # Lấy từ cột 'de' và 'en_de' như trong ảnh Dataset Viewer
    # ==========================================
    try:
        para_ds = load_dataset("deutsche-telekom/ger-backtrans-paraphrase", split="train")
        para_data = []
        # Chạy mẫu 10,000 câu để tối ưu RAM
        for row in para_ds.select(range(min(21290000, len(para_ds)))):
            de_original = row.get("de", "")
            de_paraphrase = row.get("en_de", "") # Đây là bản back-translated paraphrase
            
            if de_original and de_paraphrase:
                para_data.append(standardize_format(
                    instruction="Hãy viết lại câu tiếng Đức sau đây theo một cách khác nhưng giữ nguyên ý nghĩa:",
                    input_text=de_original,
                    output_text=de_paraphrase
                ))
        all_datasets.append(Dataset.from_list(para_data))
        print(f"✅ Loaded German Paraphrase (Cột: de -> en_de): {len(para_data)} samples")
    except Exception as e: 
        print(f"⚠️ Error German Paraphrase: {e}")

    # ==========================================
    # 6. NLU EN-DE (deutsche-telekom/NLU-Evaluation-Data-en-de)
    # Lấy từ cột 'question' và 'answer_de' như trong ảnh Dataset Viewer
    # ==========================================
    try:
        nlu_ds = load_dataset("deutsche-telekom/NLU-Evaluation-Data-en-de", split="train")
        nlu_data = []
        for row in nlu_ds.select(range(min(25000, len(nlu_ds)))):
            en_question = row.get("question", "")
            de_answer = row.get("answer_de", "")
            intent = row.get("intent", "giao tiếp") # Lấy thêm intent để làm instruction phong phú
            
            if en_question and de_answer:
                nlu_data.append(standardize_format(
                    instruction=f"Dịch ý định '{intent}' sau đây từ tiếng Anh sang tiếng Đức chuẩn xác:",
                    input_text=en_question,
                    output_text=de_answer
                ))
        all_datasets.append(Dataset.from_list(nlu_data))
        print(f"✅ Loaded NLU Data (Cột: question -> answer_de): {len(nlu_data)} samples")
    except Exception as e: 
        print(f"⚠️ Error NLU Data: {e}")

    # ==========================================
    # MERGING & SAVING
    # ==========================================
    print("\n--- 🔄 Merging and Formatting ---")
    valid_datasets = [d for d in all_datasets if len(d) > 0]
    
    if not valid_datasets:
        print("❌ No datasets loaded successfully!")
        return

    # Gộp tất cả dataset (đã được chuẩn hóa schema) lại với nhau
    combined_dataset = concatenate_datasets(valid_datasets)
    
    # Xáo trộn dữ liệu (Shuffle) với seed cố định
    combined_dataset = combined_dataset.shuffle(seed=42)
    
    # Lưu xuống ổ đĩa
    output_path = "./processed_dataset"
    os.makedirs(output_path, exist_ok=True)
    combined_dataset.save_to_disk(output_path)
    print(f"🎉 Dataset prepared and saved to '{output_path}' with {len(combined_dataset)} total samples.")

if __name__ == "__main__":
    prepare_datasets()