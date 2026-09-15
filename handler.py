"""Denglish RunPod Serverless / Public Endpoint worker.

Host this image as a RunPod Serverless endpoint (GPU + network volume).
AI Markets AI (`denglish-api` / ai.aimarkets.vn) calls it with:

    POST https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync
    Authorization: Bearer {RUNPOD_API_KEY}
    {"input": {"text"|"prompt"|"messages": ..., "action": "chat"|"agent_turn"|"tts"|"stt"|"list_voices"|...}}

RunPod wraps this handler's return value as `output` on the job.
"""

from __future__ import annotations

import base64
import io
import os
import re
import tempfile

import runpod
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import tts_vieneu as vieneu_tts
import stt_phowhisper as phowhisper

try:
    import pytesseract
except ImportError:
    pytesseract = None


TTS_ACTIONS = {"tts", "voice_clone", "clone", "list_voices", "voices"}
STT_ACTIONS = {"stt", "transcribe", "asr"}
MODEL = os.environ.get("DENGLISH_MODEL_ID", "denglish-lora")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_CANDIDATES = [
    os.environ.get("DENGLISH_BASE_MODEL", "").strip(),
    "/runpod-volume/llama3-base",
    "/workspace/llama3-base",
]
LORA_CANDIDATES = [
    os.environ.get("DENGLISH_LORA_MODEL", "").strip(),
    "/runpod-volume/denglish-model",
    "/workspace/denglish-model",
]
# VieNeu-TTS-v3-Turbo — same network volume layout as llama / LoRA.
TTS_CANDIDATES = [
    os.environ.get("VIENEU_MODEL_DIR", "").strip(),
    "/runpod-volume/vieneu-tts-v3-turbo",
    "/workspace/vieneu-tts-v3-turbo",
]
STT_CANDIDATES = [
    os.environ.get("PHOWHISPER_MODEL_DIR", "").strip(),
    "/runpod-volume/phowhisper-large",
    "/workspace/phowhisper-large",
]

tokenizer = None
model = None
LOAD_ERROR = None
TTS_CACHE_DIR = ""
STT_CACHE_DIR = ""


def _first_dir(candidates):
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return ""


def _load_models():
    global tokenizer, model, LOAD_ERROR
    base = _first_dir(BASE_CANDIDATES)
    lora = _first_dir(LORA_CANDIDATES)
    if not base:
        raise FileNotFoundError(
            "Base model not found. Mount network volume at /runpod-volume/llama3-base "
            "or set DENGLISH_BASE_MODEL."
        )
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=dtype,
        device_map="auto" if DEVICE == "cuda" else None,
        local_files_only=True,
    )
    if lora:
        model_local = PeftModel.from_pretrained(base_model, lora, local_files_only=True)
    else:
        model_local = base_model
    if DEVICE == "cpu":
        model_local = model_local.to("cpu")
    model = model_local
    print(f"[denglish-worker] loaded base={base} lora={lora or 'none'} device={DEVICE}")


try:
    _load_models()
except Exception as exc:  # noqa: BLE001
    LOAD_ERROR = str(exc)
    print(f"[denglish-worker] load failed: {LOAD_ERROR}")

# Prefetch / reuse VieNeu on the network volume so cold starts skip Hub re-download.
try:
    vieneu_tts.configure_hf_cache()
    TTS_CACHE_DIR = vieneu_tts.ensure_model(
        repo_id=vieneu_tts.MODEL_ID,
        download=os.environ.get("VIENEU_DOWNLOAD", "1") != "0",
        extra_candidates=TTS_CANDIDATES,
    )
    print(
        f"[denglish-worker] tts_vieneu ready model={vieneu_tts.MODEL_ID} "
        f"cache={TTS_CACHE_DIR} voices={len(vieneu_tts.PRESET_VOICES)}"
    )
except Exception as exc:  # noqa: BLE001
    print(f"[denglish-worker] tts ensure deferred: {exc}")
    print(
        f"[denglish-worker] tts_vieneu lazy model={vieneu_tts.MODEL_ID} "
        f"voices={len(vieneu_tts.PRESET_VOICES)}"
    )

# Prefetch / reuse PhoWhisper-large on the same network volume.
try:
    phowhisper.configure_hf_cache()
    STT_CACHE_DIR = phowhisper.ensure_model(
        repo_id=phowhisper.MODEL_ID,
        download=os.environ.get("PHOWHISPER_DOWNLOAD", "1") != "0",
        extra_candidates=STT_CANDIDATES,
    )
    print(
        f"[denglish-worker] stt_phowhisper ready model={phowhisper.MODEL_ID} "
        f"cache={STT_CACHE_DIR}"
    )
except Exception as exc:  # noqa: BLE001
    print(f"[denglish-worker] stt ensure deferred: {exc}")
    print(f"[denglish-worker] stt_phowhisper lazy model={phowhisper.MODEL_ID}")


def _text_from_messages(messages) -> str:
    if not isinstance(messages, list):
        return ""
    parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text") or "").strip())
    return "\n".join(p for p in parts if p)


def _normalize_input(job: dict) -> dict:
    raw = job.get("input") if isinstance(job, dict) else {}
    if not isinstance(raw, dict):
        raw = {"text": str(raw)}
    inp = dict(raw)
    text = str(inp.get("text") or inp.get("prompt") or inp.get("message") or "").strip()
    if not text:
        text = _text_from_messages(inp.get("messages")).strip()
    inp["text"] = text
    if inp.get("image") and not inp.get("image_base64"):
        img = str(inp["image"])
        if img.startswith("data:") and "," in img:
            inp["image_base64"] = img.split(",", 1)[1]
        elif len(img) > 256 and "://" not in img[:16]:
            inp["image_base64"] = img
    if inp.get("audio") and not inp.get("audio_base64"):
        inp["audio_base64"] = inp["audio"]
    return inp


def _wants_tts(inp: dict, action: str) -> bool:
    if action == "agent_turn" or action in STT_ACTIONS:
        return False
    if action in TTS_ACTIONS:
        return action not in {"list_voices", "voices"}
    if "tts" in inp:
        return bool(inp.get("tts"))
    if "want_audio" in inp:
        return bool(inp.get("want_audio"))
    return False


def _materialize_ref(inp: dict, action: str, temp_files: list) -> str:
    url = str(inp.get("ref_audio") or inp.get("ref_audio_url") or inp.get("audio_url") or "").strip()
    b64 = inp.get("ref_audio_base64")
    if action in {"tts", "voice_clone", "clone"}:
        b64 = b64 or inp.get("audio_base64")
    path = ""
    if url.startswith("http://") or url.startswith("https://"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            path = tmp.name
        vieneu_tts.download_ref(url, path)
        temp_files.append(path)
        return path
    if b64:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            path = tmp.name
        vieneu_tts.write_ref_bytes(vieneu_tts.decode_audio_b64(str(b64)), path)
        temp_files.append(path)
        return path
    return path


def _materialize_stt_audio(inp: dict, temp_files: list) -> str:
    """Buyer speech for PhoWhisper — never the seller clone ref_audio."""
    url = str(
        inp.get("stt_audio_url")
        or inp.get("speech_url")
        or inp.get("audio_url")
        or ""
    ).strip()
    b64 = inp.get("audio_base64") or inp.get("audio") or inp.get("speech_base64")
    path = ""
    if url.startswith("http://") or url.startswith("https://"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            path = tmp.name
        vieneu_tts.download_ref(url, path)
        temp_files.append(path)
        return path
    if b64:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            path = tmp.name
        vieneu_tts.write_ref_bytes(vieneu_tts.decode_audio_b64(str(b64)), path)
        temp_files.append(path)
        return path
    return path


def _is_health(inp: dict) -> bool:
    if inp.get("health") or inp.get("health_check") or inp.get("ping"):
        return True
    keys = {k for k, v in inp.items() if v not in (None, "", [], {})}
    return not keys


def _strip_memory_delta(text: str) -> str:
    return re.sub(
        r"MEMORY_DELTA\s*[:=]?\s*\{.*\}\s*$",
        "",
        text or "",
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()


def _extract_memory_delta(ai_response: str, user_text: str) -> dict:
    import json

    m = re.search(
        r"MEMORY_DELTA\s*[:=]?\s*(\{.*\})\s*$",
        ai_response or "",
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict):
                return {
                    "facts": parsed.get("facts") or [],
                    "preferences": parsed.get("preferences") or [],
                    "habits": parsed.get("habits") or [],
                    "entities": parsed.get("entities") or [],
                }
        except Exception:  # noqa: BLE001
            pass
    lower = (user_text or "").lower()
    delta = {"facts": [], "preferences": [], "habits": [], "entities": []}
    if user_text:
        delta["facts"].append({"text": f"User said: {user_text[:220]}", "importance": 0.45})
    if "prefer" in lower or "thích" in lower:
        delta["preferences"].append({"key": "stated_preference", "value": user_text[:160]})
    if any(k in lower for k in ("code", "python", "nestjs", "angular", "typescript")):
        delta["habits"].append({"action": "asks_for_code", "context": "engineering"})
        delta["entities"].append({"name": "Engineering", "type": "Topic", "rel": "OFTEN_ASKS"})
    return delta


def _health_payload():
    return {
        "ok": True,
        "status": "success",
        "text": "ok",
        "model": MODEL,
        "device": DEVICE,
        "loaded": LOAD_ERROR is None,
        "error": LOAD_ERROR,
        "tts_model": vieneu_tts.MODEL_ID,
        "tts_cache": TTS_CACHE_DIR or vieneu_tts.resolve_local_model_dir(TTS_CANDIDATES) or "",
        "stt_model": phowhisper.MODEL_ID,
        "stt_cache": STT_CACHE_DIR or phowhisper.resolve_local_model_dir(STT_CANDIDATES) or "",
        "voices": len(vieneu_tts.PRESET_VOICES),
        "meta": {"provider": "runpod_serverless", "action": "health"},
    }


def handler(job):
    inp = _normalize_input(job if isinstance(job, dict) else {})
    if _is_health(inp):
        return _health_payload()

    action = str(inp.get("action") or "chat")
    lang = inp.get("lang", "en")
    temp_files = []

    if action in {"list_voices", "voices"}:
        voices = vieneu_tts.list_voices()
        return {
            "status": "success",
            "text": f"{len(voices)} preset voices",
            "voices": voices,
            "output": {"kind": "voices", "voices": voices, "model": vieneu_tts.MODEL_ID},
            "meta": {"model": vieneu_tts.MODEL_ID, "provider": "runpod_serverless", "action": action},
        }

    if action in {"tts", "voice_clone", "clone"}:
        try:
            ref = _materialize_ref(inp, action, temp_files)
            result = vieneu_tts.synthesize(
                inp.get("text") or "",
                voice=str(inp.get("voice") or inp.get("speaker") or ""),
                lang=str(lang),
                ref_audio=ref,
                denoise=inp.get("denoise", True),
            )
            if result.get("error"):
                return {"error": result["error"]}
            chars = max(1, len(str(inp.get("text") or "")))
            return {
                "status": "success",
                "text": result.get("text"),
                "ai_response_text": result.get("text"),
                "ai_response_audio": result.get("audio_base64"),
                "output": result,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "unit": "characters",
                    "quantity": chars,
                },
                "meta": {
                    "model": vieneu_tts.MODEL_ID,
                    "provider": "runpod_serverless",
                    "action": action,
                    "cloned": result.get("cloned"),
                    "voice": result.get("voice"),
                },
            }
        except Exception as e:  # noqa: BLE001
            return {"error": f"VieNeu TTS: {e}"}
        finally:
            for f in temp_files:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

    if action in STT_ACTIONS:
        try:
            path = _materialize_stt_audio(inp, temp_files)
            if not path:
                return {"error": "Missing audio (audio_base64 or audio_url) for STT."}
            lang_stt = str(inp.get("language") or inp.get("lang") or "vi")
            result = phowhisper.transcribe(path, language=lang_stt)
            text = str(result.get("text") or "").strip()
            chars = max(1, len(text))
            return {
                "status": "success",
                "text": text,
                "ai_response_text": text,
                "input_detected": text,
                "output": {
                    "kind": "text",
                    "text": text,
                    "language": result.get("language") or lang_stt,
                    "model": result.get("model") or phowhisper.MODEL_ID,
                },
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "unit": "characters",
                    "quantity": chars,
                },
                "meta": {
                    "model": phowhisper.MODEL_ID,
                    "provider": "runpod_serverless",
                    "action": action,
                    "language": result.get("language") or lang_stt,
                },
            }
        except Exception as e:  # noqa: BLE001
            return {"error": f"PhoWhisper STT: {e}"}
        finally:
            for f in temp_files:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

    if LOAD_ERROR or model is None or tokenizer is None:
        return {"error": f"Worker model not loaded: {LOAD_ERROR or 'unknown'}"}

    text_input = inp.get("text")
    image_base64 = inp.get("image_base64")
    audio_base64 = inp.get("audio_base64") if action not in TTS_ACTIONS else None

    target_level = inp.get("level", "A1")
    test_count = inp.get("test_count", 5)
    test_context = inp.get("test_context", "")
    username = inp.get("username", "Học viên")
    topic = inp.get("topic", "General Conversation")
    memory_context = inp.get("memory_context") or ""

    user_text = ""
    input_source = ""
    temp_files = []

    try:
        if audio_base64:
            input_source = "audio"
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(base64.b64decode(audio_base64))
                temp_audio_path = tmp.name
                temp_files.append(temp_audio_path)
            result = phowhisper.transcribe(temp_audio_path, language=str(lang or "vi"))
            user_text = str(result.get("text") or "").strip()

        elif image_base64:
            input_source = "image"
            if pytesseract is None:
                return {"error": "OCR (pytesseract) is not available on this worker."}
            image = Image.open(io.BytesIO(base64.b64decode(image_base64)))
            user_text = pytesseract.image_to_string(image, lang="eng+deu").strip()

        elif text_input:
            input_source = "text"
            user_text = text_input.strip()

        else:
            return {"error": "Missing input (text/prompt/messages, image_base64, or audio_base64)."}

        if not user_text and "generate_test" not in action:
            return {"error": "Could not extract text from the input."}

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

        elif action == "agent_turn":
            system_prompt = (
                "You are a marketplace hire-agent running on RunPod. "
                "Use the persistent memory pack to personalize replies and infer user habits.\n"
                "Do not invent memories that are not supported.\n\n"
                f"{memory_context}\n\n"
                "After your normal reply, append ONE line exactly in this form:\n"
                'MEMORY_DELTA: {"facts":[{"text":"...","importance":0.7}],'
                '"preferences":[{"key":"tone","value":"concise"}],'
                '"habits":[{"action":"asks_for_code","context":"python"}],'
                '"entities":[{"name":"NestJS","type":"Topic","rel":"OFTEN_ASKS"}]}\n'
                "Use empty arrays when nothing new was learned."
            )
            user_msg = user_text if user_text else "Hello"

        else:
            system_prompt = (
                f"Bạn là Denglish AI - AI chuyên luyện nói Face-to-Face {target_lang_key} cho học viên người Việt Nam với chủ đề {topic} theo cấp độ {target_level}.\n"
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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt")
        if DEVICE == "cuda":
            inputs = inputs.to("cuda")

        input_tokens = int(inputs.input_ids.shape[-1])
        max_new = int(inp.get("max_new_tokens") or inp.get("max_tokens") or 1024)
        temperature = float(inp.get("temperature") or 0.4)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][len(inputs.input_ids[0]) :]
        output_tokens = int(generated_ids.shape[-1])
        ai_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        memory_delta = None
        visible_text = ai_response
        if action == "agent_turn":
            memory_delta = _extract_memory_delta(ai_response, user_text)
            visible_text = _strip_memory_delta(ai_response)

        audio_base64_out = None
        tts_meta = None
        if _wants_tts(inp, action):
            try:
                ref = _materialize_ref(inp, action, temp_files)
                spoken = vieneu_tts.synthesize(
                    visible_text,
                    voice=str(inp.get("voice") or inp.get("speaker") or ""),
                    lang=str(lang),
                    ref_audio=ref,
                    denoise=inp.get("denoise", True),
                )
                if spoken.get("error"):
                    raise RuntimeError(spoken["error"])
                audio_base64_out = spoken.get("audio_base64")
                tts_meta = {"voice": spoken.get("voice"), "cloned": spoken.get("cloned"), "tts_model": vieneu_tts.MODEL_ID}
            except Exception as e:  # noqa: BLE001
                return {"error": f"VieNeu TTS: {e}"}

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "unit": "tokens",
            "quantity": input_tokens + output_tokens,
        }
        output = {
            "kind": "text",
            "input_detected": user_text,
            "ai_response_text": visible_text,
            "ai_response_audio": audio_base64_out,
            "text": visible_text,
        }
        if audio_base64_out:
            output["audio_base64"] = audio_base64_out
            output["kind"] = "audio"
            output["format"] = "wav"
            if tts_meta:
                output.update(tts_meta)
        if memory_delta is not None:
            output["memory_delta"] = memory_delta
        return {
            "status": "success",
            "input_detected": user_text,
            "ai_response_text": visible_text,
            "ai_response_audio": audio_base64_out,
            "text": visible_text,
            "output": output,
            "memory_delta": memory_delta,
            "usage": usage,
            "meta": {
                "model": MODEL,
                "provider": "runpod_serverless",
                "input_source": input_source,
                "action": action,
                "tts_model": vieneu_tts.MODEL_ID if audio_base64_out else None,
            },
        }

    except Exception as e:  # noqa: BLE001
        return {"error": f"Lỗi: {str(e)}"}
    finally:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
