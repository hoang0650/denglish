# AI Markets RunPod worker (`aimx-runpod-worker`)

GPU worker for **ai.aimarkets.vn**. One image, three runtimes:

| `AIMX_RUNTIME` | What it does |
|---|---|
| `serverless` (default) | RunPod Serverless handler (`handler.py`) |
| `pod` | Pulls jobs from the marketplace Redis queue (`pod_pull.py`) |
| `http` | Small HTTP API on `PORT` (`http_server.py`) |

The old Denglish tutor (Llama-3-70B LoRA, PhoWhisper, `voice_assistant.py`, `train.py`) is **not** in the image.

---

## Tiếng Việt

### Mô hình (chỉ giấy phép dùng thương mại)

| Key | Hugging Face | Giấy phép | Việc | Mặc định | ~GB |
|---|---|---|---|---|---|
| `vieneu-tts` | `pnnbao-ump/VieNeu-TTS` (0.5B) | Apache-2.0 | TTS tiếng Việt + clone giọng | ✅ | 2 |
| `vieneu-tts-0.3b` | `pnnbao-ump/VieNeu-TTS-0.3B` | Apache-2.0 | TTS tiếng Việt nhanh (~2x) | ✅ | 1 |
| `neucodec` | `neuphonic/neucodec` | Apache-2.0 | codec của VieNeu (cần `HF_TOKEN`) | ✅ | 2 |
| `w2v-bert-2.0` | `facebook/w2v-bert-2.0` | MIT | encoder cho neucodec | ✅ | 3 |
| `chatterbox` | `ResembleAI/chatterbox` | MIT | TTS 23 ngôn ngữ khác (không có tiếng Việt) | ✅ | 14 |
| `z-image-turbo` | `Tongyi-MAI/Z-Image-Turbo` | Apache-2.0 | ảnh | ✅ | 33 |
| `flux-schnell` | `black-forest-labs/FLUX.1-schnell` | Apache-2.0 | ảnh (gated, cần `HF_TOKEN`) | tuỳ chọn | 34 |
| `wan2.2-ti2v-5b` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | Apache-2.0 | video (fast/talking/film) | ✅ | 34 |
| `cogvideox-2b` | `THUDM/CogVideoX-2b` | Apache-2.0 | video | tuỳ chọn | 14 |
| `birefnet` | `ZhengPeng7/BiRefNet` | MIT | tách nền | ✅ | 1 |
| `whisper-large-v3-turbo` | `openai/whisper-large-v3-turbo` | MIT | nhận dạng giọng nói | ✅ | 2 |
| `paddleocr-vl-1.5` | `PaddlePaddle/PaddleOCR-VL-1.5` | Apache-2.0 | OCR | ✅ | 2 |
| `vn-embedding` | `AITeamVN/Vietnamese_Embedding` | Apache-2.0 | embedding | ✅ | 3 |
| `vn-reranker` | `AITeamVN/Vietnamese_Reranker` | Apache-2.0 | rerank | ✅ | 3 |
| `ace-step-1.5` | `ACE-Step/Ace-Step1.5` | MIT | nhạc (sidecar, **chưa nối xong**) | ✅ | 11 |
| `qwen3.6-35b-a3b-awq` | `QuantTrio/Qwen3.6-35B-A3B-AWQ` | Apache-2.0 | LLM qua vLLM | ✅ | 26 |

Mặc định ≈137 GB, tất cả ≈185 GB → network volume **200 GB**. GPU mục tiêu: 1× 48 GB (A40 / RTX A6000).

**Định tuyến TTS:** `language: "vi"` (hoặc văn bản có dấu tiếng Việt) → VieNeu-TTS; ngôn ngữ khác → Chatterbox.
Chọn bản nhanh: `"model": "vieneu-0.3b"`. Giọng có sẵn: `Vinh, Binh, Tuyen, Doan, Ly, Ngoc`
(tên cũ như `ngoc-huyen`, `pham-tuyen` được map tự động). Clone giọng: gửi `ref_audio_url` (chỉ https công khai),
`ref_audio_base64` hoặc `ref_audio_path` + `ref_text` (nếu thiếu `ref_text` thì dùng Whisper để chép lại).

> **Lưu ý giấy phép VieNeu-TTS:** model VieNeu-TTS và VieNeu-TTS-0.3B phát hành theo **Apache-2.0**. Bộ dữ liệu
> huấn luyện `pnnbao-ump/VieNeu-TTS-1000h` là **CC-BY-NC-4.0**. Chủ dự án đã xem xét và chấp nhận dùng VieNeu
> dựa trên giấy phép Apache-2.0 của model (quyết định ngày 25/10/2026).

Bị chặn: `meta-llama/Meta-Llama-3-70B*`, `black-forest-labs/FLUX.1-dev` (phi thương mại), các tên `llama`, `denglish`, `lora`, `phowhisper`, `flux-dev`.

### Tải model lên volume

```bash
python prefetch_volume.py --list                 # bố cục, dung lượng, đã có / còn thiếu
python prefetch_volume.py                        # tải tất cả model mặc định (tiếp tục được nếu bị ngắt)
python prefetch_volume.py --models vieneu,vieneu-0.3b,neucodec,w2v-bert-2.0
python prefetch_volume.py --models all --json
```

`HF_TOKEN` chỉ cần cho repo gated (`flux-schnell`, `neucodec`; vào trang model trên huggingface.co bấm chấp nhận điều khoản trước). Token không bao giờ được in ra. Hoặc gửi job `{"action": "prefetch", "models": [...]}`.

### Redis (hàng đợi) và Postgres (lịch sử job)

* `REDIS_URL` (`redis://` hoặc `rediss://` cho TLS), `REDIS_KEY_PREFIX=aimarkets` – giống hệt `ai-marketplace-ai`
  (`aimarkets:aimx:q:paid|free`, `aimarkets:aimx:job:<id>`, `aimarkets:aimx:running`).
* `AIMX_DATABASE_URL` (hoặc `DATABASE_URL`), ví dụ `postgresql://aimx_worker:<mật khẩu>@<host>:5432/aimx_worker?sslmode=require`.
  Worker tự tạo bảng `aimx_jobs`, `aimx_volume_status` (`CREATE TABLE IF NOT EXISTS`). Postgres lỗi thì job vẫn chạy.
* Redis/Postgres trên VPS hiện **không** mở ra internet. Xem phần "Connecting from RunPod" bên dưới.

### Chạy test

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt   # Windows: .venv\Scripts\pip
python -m pytest -q
# tuỳ chọn: AIMX_TEST_REDIS_URL=redis://127.0.0.1:6379/15 AIMX_TEST_DATABASE_URL=postgresql://... python -m pytest -q
```

---

## English

### Jobs

`action` (serverless `input.action`, HTTP path, or queue `kind`):

| action | HTTP | model |
|---|---|---|
| `tts`, `clone` | `/v1/tts` | VieNeu-TTS for Vietnamese, Chatterbox otherwise |
| `image` | `/v1/image` | Z-Image-Turbo (`"model": "flux"` for FLUX.1-schnell) |
| `fast`, `talking`, `film` | `/v1/video/fast`, `/v1/video/talking`, `/v1/film` | Wan2.2-TI2V-5B (`"model": "cogvideox"`) |
| `bg_remove` | `/v1/bg_remove`, `/v1/image/remove-bg` | BiRefNet |
| `stt` | `/v1/stt` | Whisper large-v3-turbo |
| `ocr` | `/v1/ocr` | PaddleOCR-VL-1.5 |
| `embed` / `rerank` | `/v1/embed`, `/v1/rerank` | AITeamVN embedding / reranker |
| `llm` | `/v1/llm` | Qwen3.6-35B-A3B-AWQ via the vLLM sidecar (`scripts/start_vllm.sh`) |
| `music` | `/v1/music` | ACE-Step 1.5 — **stub**: forwards to `AIMX_ACESTEP_URL` only |
| `train` | `/v1/train` | hand-off (ACT / Diffusion policy) |
| `prefetch`, `volume_status` | `/v1/prefetch` | volume management |

Tutor actions (`chat`, `agent_turn`, `generate_test`, `grade_test`) are rejected.
Media inputs accept `<name>_base64`, `<name>_url` (public https only; private, loopback, metadata hosts,
credentials in URLs and redirects to internal hosts are refused) or `<name>_path` (must be on the volume).

**VieNeu-TTS license note:** the VieNeu-TTS and VieNeu-TTS-0.3B model weights are **Apache-2.0**. The training
dataset `pnnbao-ump/VieNeu-TTS-1000h` is **CC-BY-NC-4.0**; the owner reviewed this and accepted shipping VieNeu
on the basis of the model's Apache-2.0 license (decision of 2026-10-25).

### Status

Implemented, unit-tested without a GPU (fake engines), **not yet run on a GPU**: tts (VieNeu + Chatterbox),
image, video, bg_remove, stt, ocr, embed, rerank, llm proxy. `music` is a stub until the ACE-Step API sidecar
is deployed. Prefetch was exercised for real against Hugging Face (VieNeu-TTS-0.3B, BiRefNet).

### Image

`Dockerfile` keeps `torch==2.8.0` / `torchaudio==2.8.0` / `torchvision==0.23.0` from the cu128 index and
installs everything else with `-c constraints-torch.txt`. `chatterbox-tts`, `vieneu` and `neucodec` are
installed with `--no-deps` because their own pins would replace torch; their runtime deps are listed in
`requirements.tts.txt`. Build a CPU-only puller image with `--build-arg AIMX_GPU_DEPS=0`.
vLLM pins a newer torch, so it runs from its own venv on the volume (see `scripts/start_vllm.sh`).
CI tags `<DOCKER_USERNAME>/aimx-runpod-worker:sha-<commit>` on pushes to master/main after the tests pass.

### Environment

See `.env.example` (names only). The important ones:

* `AIMX_RUNTIME`, `AIMX_VOLUME`, `HF_HOME`, `HF_TOKEN` (gated repos only)
* `REDIS_URL`, `REDIS_KEY_PREFIX=aimarkets`, `REDIS_TLS_CA_FILE`
* `AIMX_DATABASE_URL` / `DATABASE_URL`, `AIMX_DB_SSLMODE`
* `AIMX_HTTP_TOKEN` (HTTP mode bearer token), `AIMX_VLLM_*`, `AIMX_ACESTEP_URL`
* `AIMX_VIENEU_DEFAULT` (`vieneu-tts` or `vieneu-tts-0.3b`), `AIMX_VIENEU_VOICE`, `AIMX_TTS_DEFAULT_LANG`

### Connecting from RunPod to Redis / Postgres on the VPS

RunPod workers have changing IPs, so allow-listing IPs does not work. Two options:

1. **Preferred – private tunnel (Tailscale or WireGuard).** Put the VPS and the pod on one tailnet and use
   the tailnet address in `REDIS_URL` / `AIMX_DATABASE_URL`. Nothing is published to the internet.
2. **TLS endpoints with strong, least-privilege credentials.** Publish Redis and Postgres through Traefik TCP
   routers with TLS (`HostSNI`), then:
   * Postgres: role `aimx_worker` (LOGIN, no superuser) owning database `aimx_worker`, `ssl=on`, `hostssl`
     rules only; worker uses `sslmode=require` (or `verify-full`).
   * Redis: ACL user restricted to `~aimarkets:aimx:*` and the commands the puller needs
     (`get set zrevrange zrem sadd srem smembers ping`), `rediss://` from the worker.

The marketplace must add the new kinds (`bg_remove`, `stt`, `ocr`, `embed`, `rerank`, `music`, `llm`) to
`providers/aimx_models.py` before enqueuing them for pod mode; serverless/HTTP accept them now.
Artifacts are written to `/runpod-volume/outputs` (no object-storage upload yet).
