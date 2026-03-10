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


try:
    BASE_MODEL_PATH = "/workspace/llama3-base"
    LORA_MODEL_PATH = "/workspace/denglish-model"

    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(f"Không thấy Base Model tại {BASE_MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
    
    # RTX 5090 cần cấu hình này để tối ưu
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto", # Hoặc "cuda"
        local_files_only=True
    )
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH, local_files_only=True)
    stt_model = whisper.load_model("small", device="cuda")
    
    print("--- [Denglish-AI] Model loaded successfully! ---")

except Exception as e:
    print(f"❌ LỖI KHỞI TẠO NGHIÊM TRỌNG: {str(e)}")
    # Giữ worker sống để bạn kịp đọc Log thay vì báo Unhealthy rồi tắt
    import time
    time.sleep(600)

# ==========================================
# 2. MODULE XỬ LÝ ÂM THANH TAM NGỮ
# ==========================================
async def generate_trilingual_audio(full_text, output_path):
    """Tách văn bản và ghép giọng chuẩn: Anh, Đức, Việt"""
    lines = [line.strip() for line in full_text.split('\n') if line.strip()]
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=500)

    for line in lines:
        upper_line = line.upper()
        # Xác định giọng đọc và làm sạch văn bản
        if any(k in upper_line for k in ["ENGLISH:", "TIẾNG ANH:", "ANH NGỮ:"]):
            voice = "en-US-EmmaNeural"
            clean_text = re.sub(r'^.*?:', '', line).strip()
        elif any(k in upper_line for k in ["GERMAN:", "TIẾNG ĐỨC:", "DEUTSCH:"]):
            voice = "de-DE-KatjaNeural"
            clean_text = re.sub(r'^.*?:', '', line).strip()
        else:
            voice = "vi-VN-HoaiMyNeural"
            # Loại bỏ prefix Tiếng Việt nếu có, nếu không giữ nguyên
            clean_text = re.sub(r'^.*?:', '', line).strip() if ":" in line[:15] else line

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_part:
            temp_p = tmp_part.name
        
        try:
            communicate = edge_tts.Communicate(clean_text, voice)
            await communicate.save(temp_p)
            segment = AudioSegment.from_mp3(temp_p)
            combined += segment + silence
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
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
    
    lang = job_input.get("lang", "en")  # en, de
    action = job_input.get("action", "chat") 
    target_level = job_input.get("level", "A1") 
    test_count = job_input.get("test_count", 5)
    test_context = job_input.get("test_context", "")
    username = job_input.get("username", "Học viên")
    topic = job_input.get("topic", "General Conversation")
    
    user_text = ""
    input_source = ""
    temp_files = []

    try:
        # --- BƯỚC 1: XỬ LÝ ĐẦU VÀO ---
        if audio_base64:
            input_source = "audio"
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(base64.b64decode(audio_base64))
                temp_audio_path = tmp.name
                temp_files.append(temp_audio_path)
            result = stt_model.transcribe(temp_audio_path)
            user_text = result["text"].strip()
        
        elif image_base64:
            input_source = "image"
            image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
            user_text = pytesseract.image_to_string(image, lang="eng+deu").strip()
        
        elif text_input:
            input_source = "text"
            user_text = text_input.strip()

        else:
            return {"error": "Thiếu dữ liệu đầu vào (text/image/audio)."}

        if not user_text and "generate_test" not in action:
            return {"error": "Không thể trích xuất nội dung từ đầu vào."}

        # --- BƯỚC 2: XÂY DỰNG PROMPT THEO LOGIC CỦA BẠN ---
        lang_name = "Tiếng Anh" if lang == "en" else "Tiếng Đức"
        target_lang_key = "English" if lang == "en" else "German"

        if action == "generate_test_en" or (action == "generate_test" and lang == "en"):
            system_prompt = (
                f"Bạn là Giám khảo khảo thí ngôn ngữ. Hãy tạo một bài kiểm tra ngắn gồm {test_count} câu hỏi "
                f"để đánh giá trình độ {lang_name} ở cấp độ {target_level}.\n"
                "Yêu cầu xuất ra:\n"
                "Tiếng Việt: [Lời chào và hướng dẫn làm bài]\n"
                f"{lang_name}: [{test_count} câu hỏi {target_lang_key}]"
            )
            user_msg = "Hãy ra đề kiểm tra cho tôi."
            
        elif action == "generate_test_de" or (action == "generate_test" and lang == "de"):
            system_prompt = (
                f"Bạn là Giám khảo khảo thí ngôn ngữ. Hãy tạo một bài kiểm tra ngắn gồm {test_count} câu hỏi "
                f"để đánh giá trình độ {lang_name} ở cấp độ {target_level}.\n"
                "Yêu cầu xuất ra:\n"
                "Tiếng Việt: [Lời chào và hướng dẫn làm bài]\n"
                f"{lang_name}: [{test_count} câu hỏi {target_lang_key}]"
            )
            user_msg = "Hãy ra đề kiểm tra cho tôi."

        elif "grade_test" in action:
            system_prompt = (
                f"Bạn là Giám khảo khảo thí vô cùng nghiêm khắc. ({username}) vừa nộp bài làm.\n"
                f"Đề bài gốc: '{test_context}'.\n"
                f"Bài làm của học viên: '{user_text}'.\n\n"
                "NHIỆM VỤ ĐÁNH GIÁ:\n"
                "1. Chấm điểm tổng quát (ví dụ: 7.5/10).\n"
                f"2. Xác định trình độ thực tế hiện tại của {username} (A1-C2).\n"
                "3. Trình bày theo cấu trúc BẮT BUỘC sau:\n"
                "Tiếng Việt: [Điểm số] - [Trình độ đánh giá] - [Nhận xét chi tiết lỗi sai và điểm mạnh]\n"
                f"Tiếng {lang_name}: [Đáp án/Câu sửa chuẩn xác hoàn toàn]\n"
            )
            user_msg = "Chấm điểm bài làm cho tôi."

        else: # Mặc định là Chat/Luyện nói
            system_prompt = (
                f"Bạn là Denglish AI - AI chuyên luyện nói Face-to-Face {lang_target} cho học viên người Việt Nam với chủ đề {topic} theo cấp độ {target_level}.\n"
                f"Người dùng vừa NÓI: '{user_text}'.\n"
                "NHIỆM VỤ CỦA BẠN:\n"
                "1. PHẢN HỒI REAL-TIME: Trả lời ngắn gọn, tự nhiên như đang nói chuyện trực tiếp.\n"
                "2. ĐÁNH GIÁ (Critique): Nếu người dùng nói sai, hãy chỉ ra điểm yếu (phát âm, dùng từ) một cách khéo léo bằng tiếng Việt.\n"
                "3. PHÁT HUY ĐIỂM MẠNH: Khen ngợi nếu họ dùng cấu trúc hay.\n"
                "4. DẪN DẮT: Luôn kết thúc bằng một câu hỏi gợi mở để người dùng tiếp tục nói theo chủ đề.\n\n"
                "PHẢN HỒI REAL-TIME BẮT BUỘC:\n"
                "Tiếng Việt: [Nhận xét nhanh điểm mạnh/yếu + Giải thích]\n"
                f"{lang_name}: [Câu phản hồi chuẩn + Câu hỏi gợi mở]\n"
            )
            user_msg = user_text if user_text else "Bắt đầu hội thoại."

        # Áp dụng template Llama 3
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        # Sinh văn bản (Sử dụng RTX 5090 cực nhanh)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=1024, 
                temperature=0.4, 
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        ai_response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

        # --- BƯỚC 3: TẠO ÂM THANH ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_out:
            final_audio_path = tmp_out.name
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
            "input_detected": user_text,
            "ai_response_text": ai_response,
            "ai_response_audio": audio_base64_out
        }

    except Exception as e:
        return {"error": f"Lỗi: {str(e)}"}
    finally:
        # Giải phóng VRAM RTX 5090 và xóa file tạm
        torch.cuda.empty_cache()
        for f in temp_files:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})