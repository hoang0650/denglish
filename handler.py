import os
import runpod
import torch
import base64
import tempfile
import asyncio
import io
import re
import whisper
import edge_tts
import pytesseract
import threading
from PIL import Image
from pydub import AudioSegment
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("--- Đang khởi tạo Siêu Gia Sư Denglish AGI (Vi-En-De) ---")

# ==========================================
# 1. NẠP MÔ HÌNH (Sử dụng dung lượng Network Volume)
# ==========================================
BASE_MODEL_PATH = "/runpod-volume/llama3-base" 
LORA_MODEL_PATH = "/runpod-volume/denglish-model"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
stt_model = whisper.load_model("small", device="cuda")

# ==========================================
# 2. MODULE XỬ LÝ ÂM THANH TAM NGỮ
# ==========================================
async def generate_trilingual_audio(full_text, output_path):
    """Tách văn bản và ghép giọng chuẩn theo từng ngôn ngữ"""
    # Tách các dòng văn bản để nhận diện ngôn ngữ
    lines = [line.strip() for line in full_text.split('\n') if line.strip()]
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=400)

    for line in lines:
        # Chọn giọng đọc dựa trên từ khóa trong dòng
        if any(k in line.upper() for k in ["ENGLISH:", "TIẾNG ANH:", "ANH NGỮ:"]):
            voice = "en-US-EmmaNeural"
        elif any(k in line.upper() for k in ["GERMAN:", "TIẾNG ĐỨC:", "DEUTSCH:"]):
            voice = "de-DE-KatjaNeural"
        else:
            voice = "vi-VN-HoaiMyNeural" # Mặc định là tiếng Việt giải thích

        temp_p = tempfile.mktemp(suffix=".mp3")
        communicate = edge_tts.Communicate(line, voice)
        await communicate.save(temp_p)
        
        segment = AudioSegment.from_mp3(temp_p)
        combined += segment + silence
        if os.path.exists(temp_p): os.remove(temp_p)

    combined.export(output_path, format="mp3")

# ==========================================
# 3. WORKER HANDLER CHÍNH
# ==========================================
def handler(job):
    job_input = job.get("input", {})
    text_input = job_input.get("text")
    image_base64 = job_input.get("image_base64")
    audio_base64 = job_input.get("audio_base64")
    
    # Lấy lịch sử hội thoại hoặc profile người dùng từ Client gửi lên (nếu có)
    user_profile = job_input.get("user_profile")
    topic = job_input.get("topic")
    
    user_text = ""
    input_source = ""
    temp_files = []

    try:
        # BƯỚC 1: NHẬN DIỆN ĐẦU VÀO (Giữ nguyên logic STT/OCR của bạn)
        if audio_base64:
            input_source = "audio"
            audio_bytes = base64.b64decode(audio_base64)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                temp_audio_path = tmp.name
                temp_files.append(temp_audio_path)
            
            # Whisper trích xuất văn bản
            result = stt_model.transcribe(temp_audio_path)
            user_text = result["text"].strip()
            input_context = f"Người dùng vừa NÓI trực tiếp: '{user_text}'."
            
        elif image_base64:
            input_source = "image"
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes))
            # Quét song ngữ Anh-Đức trên ảnh
            user_text = pytesseract.image_to_string(image, lang="eng+deu").strip()
            input_context = f"Bạn vừa XEM một hình ảnh bài tập. Văn bản trích xuất (OCR) phân tích và giải bài tập theo yêu cầu: '{user_text}'."
            
        elif text_input:
            input_source = "text"
            user_text = text_input.strip()
            input_context = f"Người dùng gửi TIN NHẮN văn bản: '{user_text}'."
        else:
            return {"error": "Thiếu dữ liệu đầu vào (text/image/audio)."}

        if not user_text:
            return {"error": "Không thể nhận diện nội dung đầu vào."}

        # BƯỚC 2: GIA SƯ AI TAM NGỮ (LLM)
        system_prompt = (
            "Bạn là Denglish AGI - Một gia sư luyện nói Face-to-Face siêu thông minh Tiếng Anh/Đức cho người Việt. "
            f"Chủ đề hiện tại: {topic}. Trình độ người dùng: {user_profile}.\n\n"
            "NHIỆM VỤ CỦA BẠN:\n"
            "1. PHẢN HỒI REAL-TIME: Trả lời ngắn gọn, tự nhiên như đang nói chuyện trực tiếp.\n"
            "2. ĐÁNH GIÁ (Critique): Nếu người dùng nói sai, hãy chỉ ra điểm yếu (phát âm, dùng từ) một cách khéo léo bằng tiếng Việt.\n"
            "3. PHÁT HUY ĐIỂM MẠNH: Khen ngợi nếu họ dùng cấu trúc hay.\n"
            "4. DẪN DẮT: Luôn kết thúc bằng một câu hỏi gợi mở để người dùng tiếp tục nói theo chủ đề.\n\n"
            "CẤU TRÚC PHẢN HỒI (BẮT BUỘC):\n"
            "Tiếng Việt: [Nhận xét điểm mạnh/yếu + Giải thích nhanh]\n"
            "Tiếng Anh: [Câu phản hồi chuẩn + Câu hỏi gợi mở]\n"
            "Tiếng Đức: [Câu phản hồi chuẩn + Câu hỏi gợi mở]\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text if user_text else "Bắt đầu buổi luyện nói."}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.3, pad_token_id=tokenizer.eos_token_id)
        ai_response = tokenizer.batch_decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)[0].strip()

        # BƯỚC 3: TẠO ÂM THANH TAM NGỮ (Threading + pydub)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
            final_audio_path = tmp_mp3.name
            temp_files.append(final_audio_path)

        def run_tts():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(generate_trilingual_audio(ai_response, final_audio_path))
            loop.close()

        tts_thread = threading.Thread(target=run_tts)
        tts_thread.start()
        tts_thread.join()

        with open(final_audio_path, "rb") as f:
            audio_base64_out = base64.b64encode(f.read()).decode('utf-8')

        return {
            "status": "success",
            "input_detected": input_source,
            "original_text": user_text,
            "ai_response_text": ai_response,
            "ai_response_audio": audio_base64_out
        }

    except Exception as e:
        return {"error": f"Lỗi hệ thống: {str(e)}"}
    finally:
        for f in temp_files:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})