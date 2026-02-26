# Kiến trúc Hệ thống Gia sư Tiếng Anh - Đức (Hỗ trợ Tiếng Việt)

## 1. Thành phần Mô hình (LLM)
- **Base Model**: Llama-3-8B hoặc Mistral-7B (Hỗ trợ đa ngôn ngữ tốt).
- **Fine-tuning**: Sử dụng kỹ thuật **QLoRA** để tiết kiệm tài nguyên.
- **Datasets**:
    - `yahma/alpaca-cleaned`: Dữ liệu chỉ dẫn tiếng Anh tổng quát.
    - `philschmid/translated_tasks_de_google_52k`: Dữ liệu chỉ dẫn tiếng Đức.
    - `OpenAssistant/oasst2`: Dữ liệu hội thoại đa ngôn ngữ chất lượng cao.
    - **Custom Vietnamese Mix**: Thêm các cặp câu (Sai ngữ pháp/phát âm -> Giải thích bằng tiếng Việt) để AI biết cách sửa lỗi.

## 2. Thành phần Voice (STT & TTS)
- **STT (Speech-to-Text)**: `faster-whisper` (Model: `small` hoặc `medium`). Hỗ trợ cực tốt Anh, Đức, Việt.
- **TTS (Text-to-Speech)**: 
    - `edge-tts`: Miễn phí, chất lượng cao, hỗ trợ nhiều giọng đọc (Anh, Đức, Việt) với độ trễ thấp.
    - Dự phòng: `gTTS` hoặc `Coqui TTS`.

## 3. Pipeline Hoạt động
1. **Input**: Người dùng nói (Audio).
2. **STT**: Chuyển Audio -> Text (Nhận diện ngôn ngữ tự động).
3. **LLM**: 
    - Nhận Text + System Prompt (Gia sư).
    - Phân tích lỗi (nếu có).
    - Trả lời bằng tiếng Anh/Đức và giải thích bằng tiếng Việt nếu cần.
4. **TTS**: Chuyển Text trả lời -> Audio.
5. **Output**: Phát âm thanh cho người dùng.

## 4. Kế hoạch Code
- `data_preparation.py`: Tải và trộn 3 dataset + mix tiếng Việt.
- `train.py`: Script fine-tuning sử dụng thư viện `peft` và `trl`.
- `voice_assistant.py`: Pipeline tích hợp STT, LLM, TTS.
- `upload_hf.py`: Script upload model và adapter lên HuggingFace.
