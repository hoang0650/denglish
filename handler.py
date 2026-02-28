import runpod
import torch
import base64
import tempfile
import os
import asyncio
import whisper
import edge_tts
from transformers import AutoModelForCausalLM, AutoTokenizer

print("--- Đang khởi tạo Denglish AI Worker ---")

# 1. Tải Bộ Não (LLM) - Lấy model đã gộp của bạn
MODEL_ID = "phgrouptechs/Denglish-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

# 2. Tải Đôi Tai (Whisper STT) - Theo cấu hình trong config của bạn là 'small'
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
            temperature=0.3, # Để nhiệt độ thấp cho việc sửa lỗi chính xác
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
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})