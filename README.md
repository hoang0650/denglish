# AI English-German Tutor with Voice & Vietnamese Support

Dự án này cung cấp bộ code Python để xây dựng một gia sư AI đa năng, có khả năng nghe, hiểu và trả lời bằng giọng nói tiếng Anh và tiếng Đức, đồng thời sử dụng tiếng Việt để giải thích lỗi sai.

## 1. Cấu trúc dự án
```
./
├── data_preparation.py
├── train.py
├── voice_assistant.py
├── upload_hf.py
├── requirements.txt
├── README.md
├── config.yaml
├── vn_mix_data.json
└── system_architecture.md
```

-   `data_preparation.py`: Script tải, xử lý và kết hợp các bộ dữ liệu huấn luyện, bao gồm `yahma/alpaca-cleaned` (tiếng Anh), `philschmid/translated_tasks_de_google_52k` (tiếng Đức), `OpenAssistant/oasst2` (đa ngôn ngữ). Dữ liệu sửa lỗi tiếng Việt được đọc từ `vn_mix_data.json`.
-   `train.py`: Script thực hiện fine-tuning mô hình ngôn ngữ lớn (LLM) sử dụng kỹ thuật QLoRA, tích hợp theo dõi với Weights & Biases (WandB).
-   `voice_assistant.py`: Script triển khai pipeline giao tiếp bằng giọng nói, tích hợp Speech-to-Text (STT) với `faster-whisper`, LLM đã fine-tuned, và Text-to-Speech (TTS) với `edge-tts`.
-   `upload_hf.py`: Script hỗ trợ tải mô hình đã huấn luyện lên HuggingFace Hub.
-   `requirements.txt`: Liệt kê tất cả các thư viện Python cần thiết để chạy dự án.
-   `README.md`: File hướng dẫn này.
-   `system_architecture.md`: Mô tả chi tiết kiến trúc hệ thống và các thành phần được sử dụng.
-   `config.yaml`: File cấu hình chứa các tham số cho mô hình, huấn luyện, WandB, HuggingFace, voice và vision.

## 2. Hướng dẫn cài đặt

Để cài đặt các thư viện cần thiết, bạn có thể sử dụng `pip` với file `requirements.txt`:

```bash
nvidia-smi
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

**Lưu ý**: Một số thư viện như `torch` có thể yêu cầu cài đặt cụ thể tùy thuộc vào cấu hình GPU của bạn. Vui lòng tham khảo tài liệu chính thức của PyTorch để biết thêm chi tiết.

## 3. Cấu hình dự án

### 3.1. File `config.yaml`

Chỉnh sửa file `config.yaml` để cấu hình các tham số của dự án. Đặc biệt chú ý đến các phần sau:

-   `wandb.entity`: Thay thế `"your_wandb_entity"` bằng entity của bạn trên Weights & Biases.
-   `huggingface.repo_name`: Thay thế `"your-username/english-german-tutor-lora"` bằng username và tên repository mong muốn của bạn trên HuggingFace.

### 3.2. Biến môi trường (Environment Variables)

Để bảo mật các thông tin nhạy cảm, dự án sử dụng biến môi trường cho HuggingFace Token và WandB API Key. Bạn cần thiết lập các biến này trước khi chạy các script liên quan:

-   **`HF_TOKEN`**: HuggingFace API Token của bạn (có quyền ghi nếu bạn muốn upload model).
-   **`WANDB_API_KEY`**: API Key của bạn trên Weights & Biases.

Bạn có thể thiết lập chúng trong terminal như sau:

```bash
export HF_TOKEN="hf_YOUR_HUGGINGFACE_TOKEN"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
```

Hoặc thêm vào file `.bashrc`, `.zshrc` hoặc `.profile` để chúng được tải tự động mỗi khi bạn mở terminal mới.

## 4. Quy trình thực hiện thủ công

Thực hiện các bước sau theo thứ tự để xây dựng và chạy mô hình gia sư AI:

### Bước 1: Chuẩn bị dữ liệu

Chạy script `data_preparation.py` để tải về, xử lý và kết hợp các bộ dữ liệu. Script này sẽ đọc dữ liệu sửa lỗi tiếng Việt từ `vn_mix_data.json` (nếu tồn tại) và tạo ra một thư mục `processed_dataset` chứa dữ liệu đã chuẩn bị.

```bash
python data_preparation.py
```

### Bước 2: Huấn luyện mô hình (Fine-tuning)

Chạy script `train.py` để fine-tune mô hình LLM. Quá trình này yêu cầu GPU với VRAM đủ lớn (khuyến nghị > 12GB cho Llama-3-8B QLoRA). Mô hình đã huấn luyện sẽ được lưu vào thư mục `./tutor_model_output`. Quá trình huấn luyện sẽ được theo dõi trên WandB.

```bash
python train.py
```

### Bước 3: Chạy trợ lý giọng nói (Voice & Vision)

Script `voice_assistant.py` sẽ triển khai pipeline giao tiếp bằng giọng nói và xử lý hình ảnh. Bạn có thể cung cấp một file audio đầu vào (ví dụ: `user_speech.wav`) và/hoặc một file hình ảnh (ví dụ: `grammar_exercise.png`) để kiểm tra chức năng này. Mô hình sẽ chuyển đổi giọng nói thành văn bản, trích xuất văn bản từ hình ảnh (nếu có), xử lý bằng LLM và phản hồi bằng giọng nói.

**Để sử dụng tính năng Vision (OCR):**

1.  **Cài đặt Tesseract OCR**: Bạn cần cài đặt Tesseract OCR trên hệ thống của mình. Ví dụ trên Ubuntu:
    ```bash
    sudo apt update
    sudo apt install tesseract-ocr
    sudo apt install tesseract-ocr-eng tesseract-ocr-deu tesseract-ocr-vie
    ```
    Đối với các hệ điều hành khác, vui lòng tham khảo hướng dẫn cài đặt Tesseract OCR chính thức.
2.  **Cấu hình `config.yaml`**: Đảm bảo `vision.ocr_enabled` được đặt là `True` và `vision.ocr_lang` chứa các ngôn ngữ bạn muốn nhận diện (ví dụ: `eng+deu+vie`).
3.  **Đặt file hình ảnh**: Đặt file hình ảnh bạn muốn AI phân tích (ví dụ: `grammar_exercise.png`) vào cùng thư mục với script `voice_assistant.py` hoặc chỉnh sửa đường dẫn trong script.

``python voice_assistant.py
```

**Lưu ý**: Hiện tại, script này chỉ là một ví dụ đơn giản. Để có một ứng dụng tương tác thực tế, bạn sẽ cần tích hợp nó với một giao diện người dùng (UI) hoặc một hệ thống xử lý audio/hình ảnh thời gian thực.## Bước 4: Tải mô hình lên HuggingFace Hub

Sau khi huấn luyện xong, bạn có thể tải mô hình lên HuggingFace Hub bằng cách chạy script `upload_hf.py`. Đảm bảo biến môi trường `HF_TOKEN` đã được thiết lập.

```bash
python upload_hf.py
```

## 5. Quy trình tự động hóa với `setup_and_train.sh`

Để đơn giản hóa quá trình cài đặt và huấn luyện, bạn có thể sử dụng script `setup_and_train.sh`. Script này sẽ tự động thực hiện các bước sau:

1.  Cài đặt tất cả các thư viện từ `requirements.txt`.
2.  Kiểm tra các biến môi trường `HF_TOKEN` và `WANDB_API_KEY`.
3.  Chuẩn bị dữ liệu huấn luyện bằng cách chạy `data_preparation.py`.
4.  Bắt đầu quá trình huấn luyện mô hình bằng cách chạy `train.py`.
5.  Nếu `HF_TOKEN` được thiết lập, script sẽ tự động upload mô hình đã huấn luyện lên HuggingFace Hub bằng cách chạy `upload_hf.py`.

**Cách sử dụng:**

Trước khi chạy, đảm bảo bạn đã cấp quyền thực thi cho script:

```bash
chmod +x setup_and_train.sh
```

Sau đó, bạn có thể chạy script:

```bash
./setup_and_train.sh
```

**Lưu ý quan trọng**: Đảm bảo rằng các biến môi trường `HF_TOKEN` và `WANDB_API_KEY` đã được thiết lập trong môi trường shell của bạn trước khi chạy script này để quá trình upload và theo dõi WandB diễn ra thành công.

## 6. Điểm nổi bật của dự án

-   **Học tập đa ngôn ngữ**: Mô hình được thiết kế để hỗ trợ người dùng học cả tiếng Anh và tiếng Đức một cách hiệu quả.
-   **Phản hồi thông minh**: Sử dụng LLM để cung cấp các phản hồi ngữ cảnh, giúp người học hiểu sâu hơn về ngôn ngữ.
-   **Hỗ trợ tiếng Việt**: Một tính năng độc đáo là khả năng tự động phát hiện lỗi ngữ pháp hoặc phát âm trong tiếng Anh/Đức và giải thích, sửa lỗi bằng tiếng Việt, giúp người học Việt Nam dễ dàng tiếp thu.
-   **Giao tiếp tự nhiên**: Tích hợp công nghệ Speech-to-Text và Text-to-Speech hiện đại để tạo ra trải nghiệm giao tiếp mượt mà và tự nhiên.
-   **Theo dõi huấn luyện với WandB**: Giúp bạn dễ dàng theo dõi, phân tích và tối ưu hóa quá trình huấn luyện mô hình.

## 7. Kiến trúc hệ thống

Để hiểu rõ hơn về cách các thành phần hoạt động cùng nhau, vui lòng tham khảo file `system_architecture.md`.
