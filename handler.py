import os
import runpod
import torch
import base64
import tempfile
import asyncio
import whisper
import edge_tts
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel # Bổ sung thư viện Peft để gộp LoRA

print("--- Đang khởi tạo Denglish AI Worker ---")

# 1. Tải Bộ Não (LLM) - Gộp Base Model và Denglish LoRA
BASE_MODEL_PATH = "/runpod-volume/llama3-base"
LORA_MODEL_PATH = "/runpod-volume/denglish-model"

print("Đang nạp Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

print("Đang nạp Base Model (Llama 3 4-bit)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

print("Đang gộp 'Não phụ' LoRA Denglish...")
model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)

# 2. Tải Đôi Tai (Whisper STT)
print("Đang nạp Whisper (small)...")
stt_model = whisper.load_model("small", device="cuda")

# Hàm tạo giọng nói (Edge TTS)
async def generate_speech(text, voice, output_path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

print("--- AI Worker Đã Sẵn Sàng ---")

def handler(job):
    job_input = job["input"]
    audio_base64 = job_input.get("audio_base64")
    target_lang = job_input.get("lang", "en") # "en" (Anh) hoặc "de" (Đức)
    
    if not audio_base64:
        return {"error": "Không tìm thấy dữ liệu âm thanh."}

    try:
        # BƯỚC 1: NGHE VÀ HIỂU (STT)
        audio_bytes = base64.b64decode(audio_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        # Whisper dịch audio thành text
        transcription = stt_model.transcribe(temp_audio_path)
        user_spoken_text = transcription["text"].strip()

        # BƯỚC 2: SỬA LỖI VÀ DẠY HỌC (LLM)
        lang_name = "English" if target_lang == "en" else "German"
        
        system_prompt = (
            f"You are a friendly and strict {lang_name} tutor for Vietnamese students. "
            f"The user tried to speak {lang_name}. Here is what they said: '{user_spoken_text}'. "
            f"Task: 1. Correct any grammatical or pronunciation mistakes. "
            f"2. Explain the correction clearly in Vietnamese. "
            f"3. Provide the perfectly corrected sentence in {lang_name} at the end."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Hãy sửa lỗi cho tôi."}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs, 
            max_new_tokens=400, 
            temperature=0.3, # Giữ nguyên mức 0.3 là cực kỳ hợp lý để AI sửa lỗi chuẩn xác
            pad_token_id=tokenizer.eos_token_id
        )
        
        ai_response = tokenizer.batch_decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)[0].strip()

        # BƯỚC 3: ĐỌC GIỌNG CHUẨN (TTS)
        voice_id = "en-US-EmmaNeural" if target_lang == "en" else "de-DE-KatjaNeural"
        tts_output_path = temp_audio_path.replace(".wav", "_ai.mp3")
        
        # Chạy Edge TTS
        asyncio.run(generate_speech(ai_response, voice_id, tts_output_path))
        
        # Chuyển audio AI đọc thành base64 để trả về
        with open(tts_output_path, "rb") as f:
            ai_audio_base64 = base64.b64encode(f.read()).decode('utf-8')

        # Dọn rác
        os.remove(temp_audio_path)
        os.remove(tts_output_path)

        return {
            "status": "success",
            "recognized_text": user_spoken_text,
            "ai_text": ai_response,
            "ai_audio_base64": ai_audio_base64
        }

    except Exception as e:
        # Dọn rác dự phòng nếu có lỗi xảy ra giữa chừng
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if 'tts_output_path' in locals() and os.path.exists(tts_output_path):
            os.remove(tts_output_path)
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})