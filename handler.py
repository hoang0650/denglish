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

print("--- Đang khởi tạo Siêu Gia Sư Denglish (Vi-En-De) ---")

# Cấu hình đường dẫn
BASE_MODEL_PATH = "/runpod-volume/llama3-base" 
LORA_MODEL_PATH = "/runpod-volume/denglish-model"

# Nạp Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
# Thêm pad_token nếu chưa có để tránh lỗi batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto" # Thay cuda bằng auto để tối ưu phân bổ
)
model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
stt_model = whisper.load_model("small", device="cuda")

async def generate_trilingual_audio(full_text, output_path):
    """Tách văn bản và ghép giọng chuẩn theo từng ngôn ngữ"""
    lines = [line.strip() for line in full_text.split('\n') if line.strip()]
    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=400)

    for line in lines:
        # Nhận diện giọng đọc thông minh hơn qua prefix
        upper_line = line.upper()
        if any(k in upper_line for k in ["ENGLISH:", "TIẾNG ANH:", "ANH NGỮ:"]):
            voice = "en-US-EmmaNeural"
            clean_text = re.sub(r'^.*?:', '', line).strip()
        elif any(k in upper_line for k in ["GERMAN:", "TIẾNG ĐỨC:", "DEUTSCH:"]):
            voice = "de-DE-KatjaNeural"
            clean_text = re.sub(r'^.*?:', '', line).strip()
        else:
            voice = "vi-VN-HoaiMyNeural"
            clean_text = line

        # Tạo file tạm an toàn
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            temp_p = tmp.name
        
        try:
            communicate = edge_tts.Communicate(clean_text, voice)
            await communicate.save(temp_p)
            segment = AudioSegment.from_mp3(temp_p)
            combined += segment + silence
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
    lang = job_input.get("lang") # Ngôn ngữ muốn test (en, de, vi)
    action = job_input.get("action") # "chat", "generate_test", "grade_test"
    target_level = job_input.get("level") # Mức độ CEFR muốn test (A1, A2, B1, B2, C1, C2)
    test_count = job_input.get("test_count") # Số lượng câu hỏi trong đề (dùng khi để đặt số lượng câu)
    test_context = job_input.get("test_context") # Nội dung đề bài (dùng khi chấm điểm)
    username = job_input.get("username")
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
            
            result = stt_model.transcribe(temp_audio_path)
            user_text = result["text"].strip()
            input_context = f"Người dùng vừa NÓI trực tiếp: '{user_text}'."
        elif image_base64:
            input_source = "image"
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes))
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
        if action == "generate_test_en":
            system_prompt = (
                f"Bạn là Giám khảo khảo thí ngôn ngữ. Hãy tạo một bài kiểm tra ngắn gồm {test_count} câu hỏi "
                f"để đánh giá trình độ tiếng Anh ở cấp độ {target_level}.\n"
                "Yêu cầu xuất ra:\n"
                "Tiếng Việt: [Lời chào và hướng dẫn làm bài]\n"
                f"Tiếng Anh: [{test_count} câu hỏi tiếng Anh]"
            )
            user_msg = "Hãy ra đề kiểm tra cho tôi."

        elif action == "generate_test_de":
            system_prompt = (
                f"Bạn là Giám khảo khảo thí ngôn ngữ. Hãy tạo một bài kiểm tra ngắn gồm {test_count} câu hỏi "
                f"để đánh giá trình độ tiếng Đức ở cấp độ {target_level}.\n"
                "Yêu cầu xuất ra:\n"
                "Tiếng Việt: [Lời chào và hướng dẫn làm bài]\n"
                f"Tiếng Đức: [{test_count} câu hỏi tiếng Đức]"
            )
            user_msg = "Hãy ra đề kiểm tra cho tôi."

        elif action == "grade_test_en":
            system_prompt = (
                f"Bạn là Giám khảo khảo thí vô cùng nghiêm khắc. ({username}) vừa nộp bài làm.\n"
                f"Đề bài gốc: '{test_context}'.\n"
                f"Bài làm của học viên: '{user_text}'.\n\n"
                "NHIỆM VỤ ĐÁNH GIÁ:\n"
                "1. Chấm điểm tổng quát (ví dụ: 7.5/10).\n"
                "2. Xác định trình độ thực tế hiện tại của học viên (A1 - C2).\n"
                "3. Trình bày theo cấu trúc BẮT BUỘC sau:\n"
                "   Tiếng Việt: [Điểm số] - [Trình độ đánh giá] - [Nhận xét chi tiết lỗi sai và điểm mạnh]\n"
                "   Tiếng Anh: [Đáp án/Câu sửa chuẩn xác hoàn toàn]\n"
            )
            user_msg = "Đây là bài làm của tôi, hãy chấm điểm và đánh giá năng lực."

        elif action == "grade_test_de":
            system_prompt = (
                f"Bạn là Giám khảo khảo thí vô cùng nghiêm khắc. ({username}) vừa nộp bài làm.\n"
                f"Đề bài gốc: '{test_context}'.\n"
                f"Bài làm của học viên: '{user_text}'.\n\n"
                "NHIỆM VỤ ĐÁNH GIÁ:\n"
                "1. Chấm điểm tổng quát (ví dụ: 7.5/10).\n"
                "2. Xác định trình độ thực tế hiện tại của học viên (A1 - C2).\n"
                "3. Trình bày theo cấu trúc BẮT BUỘC sau:\n"
                "   Tiếng Việt: [Điểm số] - [Trình độ đánh giá] - [Nhận xét chi tiết lỗi sai và điểm mạnh]\n"
                "   Tiếng Đức: [Đáp án/Câu sửa chuẩn xác hoàn toàn]\n"
            )
            user_msg = "Đây là bài làm của tôi, hãy chấm điểm và đánh giá năng lực."

        elif action == "speak_en":
            system_prompt = (
                f"Bạn là Denglish AI - AI chuyên luyện nói Face-to-Face tiếng Anh cho học viên người Việt Nam với chủ đề {topic} theo cấp độ {target_level}.\n"
                f"Người dùng vừa NÓI: '{user_text}'.\n"
                "NHIỆM VỤ CỦA BẠN:\n"
                "1. PHẢN HỒI REAL-TIME: Trả lời ngắn gọn, tự nhiên như đang nói chuyện trực tiếp.\n"
                "2. ĐÁNH GIÁ (Critique): Nếu người dùng nói sai, hãy chỉ ra điểm yếu (phát âm, dùng từ) một cách khéo léo bằng tiếng Việt.\n"
                "3. PHÁT HUY ĐIỂM MẠNH: Khen ngợi nếu họ dùng cấu trúc hay.\n"
                "4. DẪN DẮT: Luôn kết thúc bằng một câu hỏi gợi mở để người dùng tiếp tục nói theo chủ đề.\n\n"
                "PHẢN HỒI REAL-TIME BẮT BUỘC:\n"
                "Tiếng Việt: [Nhận xét nhanh điểm mạnh/yếu + Giải thích]\n"
                "Tiếng Anh: [Câu phản hồi chuẩn + Câu hỏi gợi mở]\n"
            )
            user_msg = "Bắt đầu hội thoại."

        elif action == "speak_de":
            system_prompt = (
                f"Bạn là Denglish AI - AI chuyên luyện nói Face-to-Face tiếng Đức cho học viên người Việt Nam với chủ đề {topic} theo cấp độ {target_level}.\n"
                f"Người dùng vừa NÓI: '{user_text}'.\n"
                "NHIỆM VỤ CỦA BẠN:\n"
                "1. PHẢN HỒI REAL-TIME: Trả lời ngắn gọn, tự nhiên như đang nói chuyện trực tiếp.\n"
                "2. ĐÁNH GIÁ (Critique): Nếu người dùng nói sai, hãy chỉ ra điểm yếu (phát âm, dùng từ) một cách khéo léo bằng tiếng Việt.\n"
                "3. PHÁT HUY ĐIỂM MẠNH: Khen ngợi nếu họ dùng cấu trúc hay.\n"
                "4. DẪN DẮT: Luôn kết thúc bằng một câu hỏi gợi mở để người dùng tiếp tục nói theo chủ đề.\n\n"
                "PHẢN HỒI REAL-TIME BẮT BUỘC:\n"
                "Tiếng Việt: [Nhận xét nhanh điểm mạnh/yếu + Giải thích]\n"
                "Tiếng Đức: [Câu phản hồi chuẩn + Câu hỏi gợi mở]\n"
            )
            user_msg = "Bắt đầu hội thoại."

        else: # Mặc định là "chat" (Luyện nói Face-to-Face như trước)
            system_prompt = (
                f"Bạn là Denglish AI - AI chuyên luyện nói Face-to-Face Tiếng {lang} cho học viên người Việt Nam với chủ đề {topic} theo cấp độ {target_level}.\n"
                f"Người dùng vừa NÓI: '{user_text}'.\n"
                "NHIỆM VỤ CỦA BẠN:\n"
                "1. PHẢN HỒI REAL-TIME: Trả lời ngắn gọn, tự nhiên như đang nói chuyện trực tiếp.\n"
                "2. ĐÁNH GIÁ (Critique): Nếu người dùng nói sai, hãy chỉ ra điểm yếu (phát âm, dùng từ) một cách khéo léo bằng tiếng Việt.\n"
                "3. PHÁT HUY ĐIỂM MẠNH: Khen ngợi nếu họ dùng cấu trúc hay.\n"
                "4. DẪN DẮT: Luôn kết thúc bằng một câu hỏi gợi mở để người dùng tiếp tục nói theo chủ đề.\n\n"
                "PHẢN HỒI REAL-TIME BẮT BUỘC:\n"
                "Tiếng Việt: [Nhận xét nhanh điểm mạnh/yếu + Giải thích]\n"
                f"Tiếng {lang}: [Câu phản hồi chuẩn + Câu hỏi gợi mở]\n"
            )
            user_msg = user_text if user_text else "Bắt đầu hội thoại."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.4, pad_token_id=tokenizer.eos_token_id)
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
            "action_executed": action,
            "input_detected": input_source,
            "detected_input": user_text,
            "ai_response_text": ai_response,
            "ai_response_audio": audio_base64_out
        }

    except Exception as e:
        return {"error": f"Lỗi hệ thống: {str(e)}"}
    finally:
        torch.cuda.empty_cache()
        for f in temp_files:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})