import os
import runpod
import torch
import base64
import tempfile
import asyncio
import io
import whisper
import edge_tts
import pytesseract
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("--- Đang khởi tạo Denglish AI Worker (Đa phương tiện) ---")

# 1. NẠP LLM (BỘ NÃO)
BASE_MODEL_PATH = "/runpod-volume/llama3-base"
LORA_MODEL_PATH = "/runpod-volume/denglish-model"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)

# 2. NẠP WHISPER (ĐÔI TAI)
stt_model = whisper.load_model("small", device="cuda")

# HÀM TTS (GIỌNG ĐỌC)
async def generate_speech(text, voice, output_path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

print("--- AI Worker Đã Sẵn Sàng ---")

def handler(job):
    job_input = job.get("input", {})
    
    # Lấy các trường dữ liệu từ Payload của Railway gửi sang
    text_input = job_input.get("text")
    image_base64 = job_input.get("image_base64")
    audio_base64 = job_input.get("audio_base64")  # Dùng chung cho Voice/Audio
    target_lang = job_input.get("lang", "en")     # "en" hoặc "de"
    
    user_extracted_text = ""
    input_type = "unknown"
    temp_files = [] # Danh sách lưu file tạm để dọn rác

    try:
        # ==========================================
        # BƯỚC 1: NHẬN DIỆN VÀ TRÍCH XUẤT VĂN BẢN
        # ==========================================
        if audio_base64:
            input_type = "audio"
            audio_bytes = base64.b64decode(audio_base64)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name
                temp_files.append(temp_audio_path)
            
            # STT
            transcription = stt_model.transcribe(temp_audio_path)
            user_extracted_text = transcription["text"].strip()
            
        elif image_base64:
            input_type = "image"
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes))
            
            # OCR (eng cho tiếng Anh, deu cho tiếng Đức)
            ocr_lang = "eng" if target_lang == "en" else "deu"
            user_extracted_text = pytesseract.image_to_string(image, lang=ocr_lang).strip()
            
        elif text_input:
            input_type = "text"
            user_extracted_text = text_input.strip()
            
        else:
            return {"error": "Vui lòng cung cấp ít nhất một trường: text, image_base64, hoặc audio_base64."}

        # Nếu trích xuất thất bại (file rỗng, ảnh mờ không thấy chữ...)
        if not user_extracted_text:
            return {"error": f"Không thể trích xuất được văn bản từ dữ liệu đầu vào ({input_type})."}

        # ==========================================
        # BƯỚC 2: GIA SƯ AI SỬA LỖI (LLM)
        # ==========================================
        lang_name = "English" if target_lang == "en" else "German"
        
        # Nhắc nhở AI nếu là ảnh thì có thể có lỗi do OCR đọc nhầm
        if input_type == "image":
            context_msg = f"The user uploaded an image of their {lang_name} exercise. Here is the extracted text from OCR: '{user_extracted_text}'. Note that OCR might have some typos."
        else:
            context_msg = f"The user provided a {lang_name} input: '{user_extracted_text}'."

        system_prompt = (
            f"You are a friendly and strict {lang_name} tutor for Vietnamese students. "
            f"{context_msg} "
            f"Task: 1. Correct any grammatical, spelling, or pronunciation mistakes. "
            f"2. Explain the corrections clearly in Vietnamese. "
            f"3. Provide the perfectly corrected sentence in {lang_name} at the very end."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Hãy chấm bài và sửa lỗi cho tôi."}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs, 
            max_new_tokens=400, 
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id
        )
        
        ai_response = tokenizer.batch_decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)[0].strip()

        # ==========================================
        # BƯỚC 3: AI ĐỌC KẾT QUẢ TRẢ VỀ (TTS)
        # ==========================================
        voice_id = "en-US-EmmaNeural" if target_lang == "en" else "de-DE-KatjaNeural"
        
        # Tạo file tạm cho audio xuất ra
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_tts:
            tts_output_path = temp_tts.name
            temp_files.append(tts_output_path)
        
        asyncio.run(generate_speech(ai_response, voice_id, tts_output_path))
        
        with open(tts_output_path, "rb") as f:
            ai_audio_base64 = base64.b64encode(f.read()).decode('utf-8')

        # ==========================================
        # TRẢ KẾT QUẢ VỀ CHO RAILWAY
        # ==========================================
        return {
            "status": "success",
            "input_type": input_type,
            "recognized_text": user_extracted_text,
            "ai_text": ai_response,
            "ai_audio_base64": ai_audio_base64
        }

    except Exception as e:
        return {"error": f"Lỗi trong quá trình xử lý: {str(e)}"}

    finally:
        # DỌN RÁC: Xóa sạch file tạm (ảnh, âm thanh) để không bị đầy ổ cứng 50GB
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})