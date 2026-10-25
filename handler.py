"""AI Markets RunPod worker.

Serves commercially licensed checkpoints from /runpod-volume (see catalog.py):
TTS/voice clone, image, video, background removal, speech-to-text, OCR,
Vietnamese embeddings and reranking, music (ACE-Step sidecar), and an LLM
(vLLM sidecar). Training is a hand-off. Denglish tutor actions (chat,
agent_turn, generate_test, grade_test) and non-commercial weights are rejected.

AIMX_RUNTIME=serverless (default) starts runpod.serverless.
AIMX_RUNTIME=pod pulls the Redis queue shared with ai-marketplace-ai.
AIMX_RUNTIME=http serves the same jobs over HTTP for a pod puller.

Every runtime records jobs in Postgres (db.py) when DATABASE_URL is set; a DB
outage never fails a job.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from collections import OrderedDict
from typing import Any

import catalog
import db
import media

VOLUME = catalog.VOLUME
# Hub-cache models (neucodec, w2v-bert) live under the volume too.
os.environ.setdefault("HF_HOME", f"{VOLUME}/huggingface")

COMMERCIAL_MODELS: dict[str, dict] = {}
for _key, _row in catalog.MODELS.items():
    _tasks = list(_row["tasks"])
    if "tts" in _tasks:
        _tasks.append("clone")
    COMMERCIAL_MODELS[_key] = {
        "key": _key,
        "id": _row["id"],
        "license": _row["license"],
        "tasks": _tasks,
        "volume": f"{VOLUME}/{_row['dir']}",
        "default": bool(_row.get("default")),
        "languages": list(_row.get("languages") or []),
        "notes": _row.get("notes", ""),
    }

TRAIN_POLICIES = {
    "act": {"license": "Apache-2.0", "upstream_license": "MIT", "implementation": "huggingface/lerobot"},
    "diffusion": {"license": "Apache-2.0", "upstream_license": "MIT", "implementation": "huggingface/lerobot"},
}

BLOCKED_MODELS = dict(catalog.BLOCKED_MODELS)

# Denglish tutor actions. Speech-to-text is served again (Whisper, MIT).
BLOCKED_ACTIONS = frozenset({"chat", "agent_turn", "generate_test", "grade_test"})

_TASK = {
    "tts": "tts",
    "clone": "tts",
    "voice_clone": "tts",
    "image": "image",
    "fast": "fast",
    "talking": "talking",
    "film": "film",
    "bg_remove": "bg_remove",
    "remove_bg": "bg_remove",
    "background_removal": "bg_remove",
    "stt": "stt",
    "transcribe": "stt",
    "asr": "stt",
    "ocr": "ocr",
    "embed": "embed",
    "embedding": "embed",
    "embeddings": "embed",
    "rerank": "rerank",
    "music": "music",
    "llm": "llm",
    "generate": "llm",
    "completion": "llm",
}

VIDEO_KINDS = catalog.VIDEO_KINDS
LIGHT_KINDS = catalog.LIGHT_KINDS
TRAIN_KINDS = catalog.TRAIN_KINDS

_KIND_FROM_PATH = {
    "/v1/tts": "tts",
    "/v1/image": "image",
    "/v1/video/fast": "fast",
    "/v1/video/talking": "talking",
    "/v1/film": "film",
    "/v1/train": "train",
    "/v1/bg_remove": "bg_remove",
    "/v1/image/remove-bg": "bg_remove",
    "/v1/stt": "stt",
    "/v1/ocr": "ocr",
    "/v1/embed": "embed",
    "/v1/rerank": "rerank",
    "/v1/music": "music",
    "/v1/llm": "llm",
    "/v1/prefetch": "prefetch",
}

# Chatterbox Multilingual language ids (model card). Vietnamese is not one of them.
CHATTERBOX_LANGS = frozenset({
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms",
    "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
})

# Tasks whose weights are not loaded in this process.
_SIDECAR_TASKS = {"llm", "music"}


def _sandbox() -> bool:
    return os.environ.get("AIMX_WORKER_SANDBOX", "").strip().lower() in {"1", "true", "yes"}


def _ready(path: str) -> bool:
    try:
        if not os.path.isdir(path):
            return False
        return any(True for entry in os.scandir(path) if not entry.name.startswith("."))
    except OSError:
        return False


_VI_CHARS = frozenset("ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")


def tts_language(payload: dict | None) -> str:
    """Explicit language/language_id wins; otherwise Vietnamese letters in the text mean 'vi'."""
    body = payload or {}
    raw = str(body.get("language") or body.get("language_id") or body.get("lang") or "").strip().lower()
    if raw:
        return "vi" if raw in {"vi", "vie", "vi-vn", "vi_vn", "vietnamese", "tiếng việt"} else raw.split("-")[0]
    text = str(body.get("text") or body.get("prompt") or "").lower()
    if any(ch in _VI_CHARS for ch in text):
        return "vi"
    return os.environ.get("AIMX_TTS_DEFAULT_LANG", "en").strip().lower() or "en"


def _tts_route(payload: dict | None) -> dict:
    """Vietnamese -> VieNeu-TTS (0.5B), every other language -> Chatterbox."""
    if tts_language(payload) == "vi":
        key = os.environ.get("AIMX_VIENEU_DEFAULT", "vieneu-tts").strip() or "vieneu-tts"
        return COMMERCIAL_MODELS.get(key) or COMMERCIAL_MODELS["vieneu-tts"]
    return COMMERCIAL_MODELS["chatterbox"]


def model_for(task: str, override: str = "", payload: dict | None = None) -> dict:
    wanted = (override or "").strip()
    if not wanted and task in {"tts", "clone"}:
        return _tts_route(payload)
    why = catalog.blocked_reason(wanted) if wanted else ""
    if why:
        raise ValueError(why)
    if wanted:
        for key, row in COMMERCIAL_MODELS.items():
            if wanted not in {key, row["id"], row["volume"]} and catalog.ALIASES.get(wanted.lower()) != key:
                continue
            if task not in row["tasks"]:
                raise ValueError(f"model '{wanted}' does not serve {task}")
            return row
        raise ValueError(f"model '{wanted}' is not on the commercial allowlist for {task}")
    rows = [row for row in COMMERCIAL_MODELS.values() if task in row["tasks"]]
    rows.sort(key=lambda row: 0 if row["default"] else 1)
    if rows:
        return rows[0]
    raise ValueError(f"no commercial model for {task}")


def health_body() -> dict:
    models = []
    for row in COMMERCIAL_MODELS.values():
        models.append({
            "key": row["key"],
            "id": row["id"],
            "license": row["license"],
            "tasks": list(row["tasks"]),
            "default": row["default"],
            "volume": row["volume"],
            "ready": _ready(row["volume"]),
        })
    return {
        "ok": True,
        "service": "aimx-runpod-worker",
        "runtime": _runtime(),
        "volume": VOLUME,
        "models": models,
        "blocked": sorted(BLOCKED_MODELS),
        "db": db.enabled(),
    }


def _runtime() -> str:
    return (os.environ.get("AIMX_RUNTIME") or "serverless").strip().lower()


def _hand_off(payload: dict) -> dict:
    policy = str(payload.get("policy") or payload.get("train_policy") or "").strip().lower()
    row = TRAIN_POLICIES.get(policy)
    if not row:
        return {"status": "failed", "error": f"policy '{policy}' is not allowed"}
    return {
        "status": "handed_off",
        "backend": "pod",
        "license": row["license"],
        "implementation": row["implementation"],
        "policy": policy,
        "gpu_seconds": 0,
        "stage": "handed_off",
    }


def _sandbox_ok(task: str, row: dict) -> dict:
    out: dict[str, Any] = {
        "status": "succeeded",
        "model": row["id"],
        "license": row["license"],
        "gpu_seconds": 0.01,
        "artifact_url": "sandbox://aimx",
    }
    if task in {"stt", "ocr", "llm"}:
        out["text"] = ""
    if task == "tts":
        out["language"] = "vi" if row["key"].startswith("vieneu") else "auto"
    return out


# ---------------------------------------------------------------- model cache

_cache: "OrderedDict[str, Any]" = OrderedDict()


def _heavy_limit() -> int:
    try:
        return max(1, int(os.environ.get("AIMX_HEAVY_CACHE", "1") or "1"))
    except ValueError:
        return 1


def _free_gpu() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _get_cached(name: str, heavy: bool, loader):
    """Keep light models resident; keep at most AIMX_HEAVY_CACHE heavy pipelines on the GPU."""
    key = ("heavy:" if heavy else "light:") + name
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    if heavy:
        heavy_keys = [k for k in _cache if k.startswith("heavy:")]
        while heavy_keys and len(heavy_keys) >= _heavy_limit():
            _cache.pop(heavy_keys.pop(0), None)
            _free_gpu()
    obj = loader()
    _cache[key] = obj
    return obj


def _device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype():
    import torch

    return torch.bfloat16 if _device() == "cuda" else torch.float32


# ---------------------------------------------------------------- execution

def _run(task: str, row: dict, payload: dict) -> dict:
    started = time.perf_counter()
    try:
        out = _infer(task, row, payload)
    except ImportError as exc:
        return {
            "status": "failed",
            "model": row["id"],
            "error": f"install requirements.gpu.txt on the volume (torch stays 2.8.0+cu128): {exc}",
        }
    except media.MediaError as exc:
        return {"status": "failed", "model": row["id"], "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "model": row["id"], "error": str(exc)[:500]}
    body = {
        "status": "succeeded",
        "model": row["id"],
        "license": row["license"],
        "gpu_seconds": round(time.perf_counter() - started, 3),
    }
    body.update(out)
    return body


def _out(name: str, ext: str) -> str:
    os.makedirs(f"{VOLUME}/outputs", exist_ok=True)
    return f"{VOLUME}/outputs/{name}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}.{ext}"


def _infer(task: str, row: dict, payload: dict) -> dict:
    if task == "tts":
        if row["key"].startswith("vieneu"):
            return _tts_vieneu(row, payload, _out("tts", "wav"))
        dest = _tts(row["volume"], payload, _out("tts", "wav"))
        return {"artifact_url": dest, "audio_url": dest, "language": tts_language(payload)}
    if task == "image":
        dest = _image(row, payload, _out("image", "png"))
        return {"artifact_url": dest, "image_url": dest}
    if task in VIDEO_KINDS:
        dest = _video(row, payload, _out("video", "mp4"))
        return {"artifact_url": dest, "video_url": dest}
    if task == "bg_remove":
        dest = _bg_remove(row["volume"], payload, _out("cutout", "png"))
        return {"artifact_url": dest, "image_url": dest}
    if task == "stt":
        return _stt(row["volume"], payload)
    if task == "ocr":
        return _ocr(row["volume"], payload)
    if task == "embed":
        return _embed(row["volume"], payload)
    if task == "rerank":
        return _rerank(row["volume"], payload)
    if task == "music":
        return _music(row, payload)
    if task == "llm":
        return _llm(row, payload)
    raise ValueError(f"unknown task {task}")


def _text_of(payload: dict) -> str:
    return str(payload.get("text") or payload.get("prompt") or "")


def _tts(path: str, payload: dict, dest: str) -> str:
    language = tts_language(payload)
    if language not in CHATTERBOX_LANGS:
        hint = " Use VieNeu-TTS (model 'vieneu') for Vietnamese." if language == "vi" else ""
        raise ValueError(f"Chatterbox does not support language '{language}' (no Vietnamese).{hint} Supported: "
                         + ",".join(sorted(CHATTERBOX_LANGS)))
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = _device()

    def load():
        if hasattr(ChatterboxMultilingualTTS, "from_local"):
            return ChatterboxMultilingualTTS.from_local(path, device)
        return ChatterboxMultilingualTTS.from_pretrained(device=device)

    model = _get_cached("chatterbox", False, load)
    ref = media.resolve(payload, "audio", VOLUME) or media.resolve(payload, "ref_audio", VOLUME)
    kwargs: dict[str, Any] = {}
    if ref:
        kwargs["audio_prompt_path"] = ref
    wav = model.generate(_text_of(payload), language_id=language, **kwargs)
    import torchaudio

    tensor = wav.cpu() if hasattr(wav, "cpu") else wav
    if getattr(tensor, "dim", lambda: 1)() == 1:
        tensor = tensor.unsqueeze(0)
    torchaudio.save(dest, tensor, int(getattr(model, "sr", 24000)))
    return dest


# ---------------------------------------------------------------- VieNeu-TTS
# Preset voices shipped in voices.json of pnnbao-ump/VieNeu-TTS and -0.3B.
# Names from the old tts_vieneu.py roster (v3 Turbo) map onto the closest preset.
VIENEU_PRESETS = {
    "vinh": "Vinh", "vĩnh": "Vinh", "xuan-vinh": "Vinh", "xuân vĩnh": "Vinh",
    "binh": "Binh", "bình": "Binh", "thanh-binh": "Binh", "thanh bình": "Binh",
    "tuyen": "Tuyen", "tuyên": "Tuyen", "pham-tuyen": "Tuyen", "phạm tuyên": "Tuyen",
    "doan": "Doan", "đoan": "Doan", "doan-trang": "Doan", "đoan trang": "Doan", "thuc-doan": "Doan",
    "thục đoan": "Doan",
    "ly": "Ly", "truc-ly": "Ly", "trúc ly": "Ly",
    "ngoc": "Ngoc", "ngọc": "Ngoc", "ngoc-huyen": "Ngoc", "ngọc huyền": "Ngoc",
}
VIENEU_SAMPLE_RATE = 24000


def vieneu_voice(name: Any) -> str:
    raw = str(name or "").strip()
    if not raw:
        return os.environ.get("AIMX_VIENEU_VOICE", "").strip()
    return VIENEU_PRESETS.get(raw.lower(), raw)


def _vieneu_codec() -> str:
    return os.environ.get("AIMX_VIENEU_CODEC", "neuphonic/neucodec").strip() or "neuphonic/neucodec"


def _vieneu_engine(row: dict):
    from vieneu import Vieneu

    device = _device()
    codec = _vieneu_codec()
    codec_device = "cpu" if "onnx" in codec.lower() else device

    def load():
        engine = Vieneu(
            mode="standard",
            backbone_repo=row["volume"],
            backbone_device=device,
            codec_repo=codec,
            codec_device=codec_device,
            gguf_filename=None,  # the SDK default points at a GGUF file; use the safetensors weights
        )
        # The 0.5B model uses the chat prompt; the SDK keys that off the path suffix.
        engine.use_chat_format = row["id"] == "pnnbao-ump/VieNeu-TTS"
        return engine

    return _get_cached(row["key"], False, load)


def _ref_text_for(payload: dict, ref: str) -> str:
    text = str(payload.get("ref_text") or payload.get("reference_text") or "").strip()
    if text:
        return text
    whisper = COMMERCIAL_MODELS.get("whisper-large-v3-turbo")
    if not whisper or not _ready(whisper["volume"]):
        raise ValueError("voice clone needs ref_text (what the reference audio says), or whisper on the volume")
    return str(_stt(whisper["volume"], {"language": "vi"}, src=ref).get("text") or "").strip()


def _tts_vieneu(row: dict, payload: dict, dest: str) -> dict:
    text = _text_of(payload).strip()
    if not text:
        raise ValueError("Missing text for TTS")
    codec_row = COMMERCIAL_MODELS.get("neucodec")
    if _vieneu_codec() == "neuphonic/neucodec" and codec_row and not _ready(codec_row["volume"]):
        raise ValueError("VieNeu codec not on volume: run prefetch with neucodec and w2v-bert-2.0 (HF_TOKEN needed)")
    engine = _vieneu_engine(row)
    ref = None
    for prefix in ("ref_audio", "audio", "speaker_audio"):
        ref = media.resolve(payload, prefix, VOLUME, ".wav")
        if ref:
            break
    kwargs: dict[str, Any] = {
        "temperature": float(payload.get("temperature") or 1.0),
        "top_k": int(payload.get("top_k") or 50),
    }
    voice = ""
    if ref:
        kwargs["ref_audio"] = ref
        kwargs["ref_text"] = _ref_text_for(payload, ref)
    else:
        voice = vieneu_voice(payload.get("voice") or payload.get("speaker"))
        try:
            kwargs["voice"] = engine.get_preset_voice(voice or None)
        except ValueError as exc:
            presets = [name for _desc, name in engine.list_preset_voices()]
            raise ValueError(f"unknown VieNeu voice '{voice}'; presets: {', '.join(presets)}") from exc
    audio = engine.infer(text, **kwargs)
    engine.save(audio, dest)
    return {
        "artifact_url": dest,
        "audio_url": dest,
        "language": "vi",
        "voice": "cloned" if ref else (voice or "default"),
        "cloned": bool(ref),
        "sample_rate": int(getattr(engine, "sample_rate", VIENEU_SAMPLE_RATE)),
    }


_IMAGE_DEFAULTS = {
    "z-image-turbo": {"steps": 9, "guidance_scale": 0.0},
    "flux-schnell": {"steps": 4, "guidance_scale": 0.0},
}


def _image(row: dict, payload: dict, dest: str) -> str:
    import torch
    from diffusers import DiffusionPipeline

    def load():
        pipe = DiffusionPipeline.from_pretrained(row["volume"], torch_dtype=_dtype(), local_files_only=True)
        return pipe.to(_device())

    pipe = _get_cached(row["key"], True, load)
    defaults = _IMAGE_DEFAULTS.get(row["key"], {"steps": 20, "guidance_scale": 3.5})
    kwargs: dict[str, Any] = {
        "prompt": _text_of(payload),
        "num_inference_steps": int(payload.get("steps") or defaults["steps"]),
        "guidance_scale": float(payload.get("guidance_scale", defaults["guidance_scale"]) or 0.0),
        "height": int(payload.get("height") or 1024),
        "width": int(payload.get("width") or 1024),
    }
    if payload.get("seed") is not None:
        kwargs["generator"] = torch.Generator(_device()).manual_seed(int(payload["seed"]))
    image = pipe(**kwargs).images[0]
    image.save(dest)
    return dest


def _video(row: dict, payload: dict, dest: str) -> str:
    import torch
    from diffusers.utils import export_to_video, load_image

    image_path = media.resolve(payload, "image", VOLUME)
    if row["key"] == "wan2.2-ti2v-5b":
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline

        cls = WanImageToVideoPipeline if image_path else WanPipeline

        def load():
            vae = AutoencoderKLWan.from_pretrained(row["volume"], subfolder="vae", torch_dtype=torch.float32,
                                                   local_files_only=True)
            pipe = cls.from_pretrained(row["volume"], vae=vae, torch_dtype=_dtype(), local_files_only=True)
            return pipe.to(_device())

        pipe = _get_cached(f"{row['key']}:{cls.__name__}", True, load)
        kwargs: dict[str, Any] = {
            "prompt": _text_of(payload),
            "negative_prompt": str(payload.get("negative_prompt") or ""),
            "height": int(payload.get("height") or 704),
            "width": int(payload.get("width") or 1280),
            "num_frames": int(payload.get("num_frames") or 121),
            "num_inference_steps": int(payload.get("steps") or 50),
            "guidance_scale": float(payload.get("guidance_scale") or 5.0),
        }
        if image_path:
            kwargs["image"] = load_image(image_path)
        fps = int(payload.get("fps") or 24)
    else:
        from diffusers import DiffusionPipeline

        def load():
            pipe = DiffusionPipeline.from_pretrained(row["volume"], torch_dtype=_dtype(), local_files_only=True)
            return pipe.to(_device())

        pipe = _get_cached(row["key"], True, load)
        kwargs = {
            "prompt": _text_of(payload),
            "num_inference_steps": int(payload.get("steps") or 50),
            "guidance_scale": float(payload.get("guidance_scale") or 6.0),
        }
        fps = int(payload.get("fps") or 8)
    if payload.get("seed") is not None:
        kwargs["generator"] = torch.Generator(_device()).manual_seed(int(payload["seed"]))
    frames = pipe(**kwargs).frames[0]
    export_to_video(frames, dest, fps=fps)
    return dest


def _bg_remove(path: str, payload: dict, dest: str) -> str:
    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    src = media.resolve(payload, "image", VOLUME)
    if not src:
        raise ValueError("bg_remove needs image_url, image_base64, or image_path")

    def load():
        model = AutoModelForImageSegmentation.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        model.to(_device()).eval()
        if _device() == "cuda":
            model.half()
        return model

    model = _get_cached("birefnet", False, load)
    image = Image.open(src).convert("RGB")
    size = int(payload.get("size") or 1024)
    prep = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    batch = prep(image).unsqueeze(0).to(_device())
    if _device() == "cuda":
        batch = batch.half()
    with torch.no_grad():
        preds = model(batch)[-1].sigmoid().cpu()
    mask = transforms.ToPILImage()(preds[0].squeeze().float()).resize(image.size)
    image.putalpha(mask)
    image.save(dest)
    return dest


def _stt(path: str, payload: dict, src: str | None = None) -> dict:
    import torch
    from transformers import pipeline

    src = src or media.resolve(payload, "audio", VOLUME)
    if not src:
        raise ValueError("stt needs audio_url, audio_base64, or audio_path")

    def load():
        return pipeline(
            "automatic-speech-recognition",
            model=path,
            torch_dtype=torch.float16 if _device() == "cuda" else torch.float32,
            device=0 if _device() == "cuda" else -1,
        )

    asr = _get_cached("whisper", False, load)
    import librosa

    audio, sr = librosa.load(src, sr=16000, mono=True)
    generate_kwargs: dict[str, Any] = {"task": "transcribe"}
    if payload.get("language"):
        generate_kwargs["language"] = str(payload["language"])
    result = asr({"raw": audio, "sampling_rate": sr}, return_timestamps=bool(payload.get("timestamps", True)),
                 generate_kwargs=generate_kwargs, chunk_length_s=30, batch_size=8)
    out: dict[str, Any] = {"text": str(result.get("text") or "").strip(), "duration_sec": round(len(audio) / sr, 2)}
    if payload.get("language"):
        out["language"] = str(payload["language"])
    if result.get("chunks"):
        out["chunks"] = [{"text": c.get("text"), "timestamp": list(c.get("timestamp") or [])} for c in result["chunks"]]
    return out


_OCR_PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}


def _ocr(path: str, payload: dict) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    src = media.resolve(payload, "image", VOLUME)
    if not src:
        raise ValueError("ocr needs image_url, image_base64, or image_path")
    task = str(payload.get("task") or "ocr").lower()
    if task not in _OCR_PROMPTS:
        raise ValueError(f"ocr task must be one of {', '.join(_OCR_PROMPTS)}")

    def load():
        model = AutoModelForImageTextToText.from_pretrained(path, torch_dtype=torch.bfloat16, local_files_only=True)
        return model.to(_device()).eval(), AutoProcessor.from_pretrained(path, local_files_only=True)

    model, processor = _get_cached("paddleocr-vl", True, load)
    image = Image.open(src).convert("RGB")
    max_pixels = 2048 * 28 * 28 if task == "spotting" else 1280 * 28 * 28
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": _OCR_PROMPTS[task]}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        images_kwargs={"size": {"shortest_edge": processor.image_processor.min_pixels, "longest_edge": max_pixels}},
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=int(payload.get("max_new_tokens") or 1024))
    text = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])
    return {"text": text, "stage": task}


def _texts(payload: dict, key: str) -> list[str]:
    value = payload.get(key)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


def _embed(path: str, payload: dict) -> dict:
    from sentence_transformers import SentenceTransformer

    texts = _texts(payload, "texts") or _texts(payload, "input") or _texts(payload, "text")
    if not texts:
        raise ValueError("embed needs texts")
    if len(texts) > int(os.environ.get("AIMX_EMBED_MAX_TEXTS", "256") or "256"):
        raise ValueError("too many texts in one embed job")
    model = _get_cached("vn-embedding", False, lambda: SentenceTransformer(path, device=_device(), local_files_only=True))
    model.max_seq_length = int(payload.get("max_length") or 2048)
    vectors = model.encode(texts, normalize_embeddings=bool(payload.get("normalize", True)), convert_to_numpy=True)
    return {"embeddings": [[round(float(x), 6) for x in v] for v in vectors], "dim": int(vectors.shape[1])}


def _rerank(path: str, payload: dict) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    query = str(payload.get("query") or "")
    docs = _texts(payload, "documents") or _texts(payload, "docs")
    if not query or not docs:
        raise ValueError("rerank needs query and documents")

    def load():
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        if _device() == "cuda":
            model = model.half()
        return tok, model.to(_device()).eval()

    tok, model = _get_cached("vn-reranker", False, load)
    pairs = [[query, d] for d in docs]
    with torch.no_grad():
        inputs = tok(pairs, padding=True, truncation=True, return_tensors="pt",
                     max_length=int(payload.get("max_length") or 2304)).to(_device())
        scores = model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()
    ranking = sorted(range(len(docs)), key=lambda i: -scores[i])
    return {"scores": [round(s, 5) for s in scores], "ranking": ranking}


def _post_json(url: str, body: dict, timeout: float, token: str = "") -> dict:
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"sidecar HTTP {exc.code}: {exc.read()[:300].decode('utf-8', 'replace')}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"sidecar not reachable at {url.split('?')[0]}: {type(exc).__name__}") from exc


def _llm(row: dict, payload: dict) -> dict:
    """Proxy to the vLLM OpenAI server started by scripts/start_vllm.sh."""
    base = (os.environ.get("AIMX_VLLM_URL") or "http://127.0.0.1:8001/v1").rstrip("/")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        prompt = _text_of(payload)
        if not prompt:
            raise ValueError("llm needs messages or prompt")
        messages = [{"role": "user", "content": prompt}]
        if payload.get("system"):
            messages.insert(0, {"role": "system", "content": str(payload["system"])})
    body: dict[str, Any] = {
        "model": os.environ.get("AIMX_VLLM_MODEL") or "qwen3.6-35b-a3b",
        "messages": messages,
        "max_tokens": min(int(payload.get("max_tokens") or 1024), int(os.environ.get("AIMX_LLM_MAX_TOKENS", "8192") or "8192")),
        "temperature": float(payload.get("temperature", 0.7)),
    }
    for key in ("top_p", "stop", "response_format", "tools", "tool_choice", "chat_template_kwargs"):
        if key in payload:
            body[key] = payload[key]
    timeout = float(os.environ.get("AIMX_TIMEOUT_LLM", "300") or "300")
    resp = _post_json(f"{base}/chat/completions", body, timeout, os.environ.get("AIMX_VLLM_API_KEY", ""))
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    out: dict[str, Any] = {"text": msg.get("content") or "", "usage": resp.get("usage") or {}}
    if msg.get("reasoning_content"):
        out["reasoning"] = msg["reasoning_content"]
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    return out


def _music(row: dict, payload: dict) -> dict:
    """STUB: ACE-Step 1.5 runs as its own API sidecar (`acestep-api`, separate venv).

    Set AIMX_ACESTEP_URL to forward the payload to `{AIMX_ACESTEP_URL}{AIMX_ACESTEP_PATH}`
    unchanged; the response must carry an audio path or URL. The exact request
    shape follows the ACE-Step API version you deploy, so it is not mapped here.
    """
    base = (os.environ.get("AIMX_ACESTEP_URL") or "").rstrip("/")
    if not base:
        raise RuntimeError("music is not wired yet: start the ACE-Step 1.5 API sidecar and set AIMX_ACESTEP_URL")
    path = os.environ.get("AIMX_ACESTEP_PATH") or "/release_task"
    resp = _post_json(base + path, payload, float(os.environ.get("AIMX_TIMEOUT_MUSIC", "600") or "600"))
    url = resp.get("audio_url") or resp.get("audio_path") or resp.get("url") or ""
    out: dict[str, Any] = {"stage": "sidecar"}
    if url:
        out["audio_url"] = url
        out["artifact_url"] = url
    else:
        out["result"] = resp
    return out


def _prefetch(payload: dict) -> dict:
    import prefetch_volume

    if payload.get("list") or payload.get("status_only"):
        return {"status": "succeeded", **prefetch_volume.status(payload.get("root") or None)}
    result = prefetch_volume.prefetch(payload.get("models"), root=payload.get("root") or None)
    result["status"] = "succeeded" if result.get("ok") else "failed"
    if not result.get("ok"):
        result["error"] = result.get("text")
    return result


def execute(kind: str, payload: dict | None) -> dict:
    body = payload if isinstance(payload, dict) else {}
    action = str(kind or body.get("action") or body.get("kind") or "").strip().lower()
    if action in BLOCKED_ACTIONS:
        return {"status": "failed", "error": f"action '{action}' is not an AI Markets workload"}
    if action == "train" or body.get("hand_off") is True or str(body.get("hand_off")).lower() == "true":
        return _hand_off(body)
    if action == "health" or action == "":
        return health_body()
    if action in {"prefetch", "volume_status"}:
        if action == "volume_status":
            body = {**body, "list": True}
        return _prefetch(body)
    task = _TASK.get(action)
    if not task:
        return {"status": "failed", "error": f"unknown action '{action}'"}
    try:
        row = model_for(task, str(body.get("model") or ""), body)
    except ValueError as exc:
        return {"status": "failed", "error": str(exc)}
    if _sandbox():
        return _sandbox_ok(task, row)
    if task not in _SIDECAR_TASKS and not _ready(row["volume"]):
        return {"status": "failed", "model": row["id"], "error": f"model not on volume: {row['volume']}"}
    return _run(task, row, body)


def run_job(kind: str, payload: dict | None, meta: dict | None = None) -> dict:
    """execute() plus a Postgres record. Used by serverless and HTTP runtimes."""
    meta = dict(meta or {})
    body = payload if isinstance(payload, dict) else {}
    job = {
        "id": str(meta.get("id") or body.get("job_id") or uuid.uuid4().hex),
        "kind": str(kind or body.get("action") or ""),
        "tier": meta.get("tier") or body.get("tier"),
        "priority": meta.get("priority") or body.get("priority"),
        "backend": meta.get("backend") or _runtime(),
        "createdAt": meta.get("createdAt") or time.time(),
        "startedAt": time.time(),
        "status": "running",
    }
    recording = job["kind"] not in {"health", ""} and db.enabled()
    if recording:
        db.job_started(job, runtime=_runtime())
    try:
        result = execute(kind, body)
    except Exception as exc:  # noqa: BLE001
        result = {"status": "failed", "error": str(exc)[:500]}
    if recording:
        status = str(result.get("status") or "succeeded")
        if status == "handed_off":
            status = "succeeded"
        done = {**job, "status": status, "finishedAt": time.time(),
                "gpu_seconds": result.get("gpu_seconds"), "error": result.get("error")}
        db.job_finished(done, runtime=_runtime(), output=result)
    return result


def handler(event: dict | None) -> dict:
    event = event or {}
    incoming = event.get("input") if isinstance(event.get("input"), dict) else event
    if not isinstance(incoming, dict):
        incoming = {}
    action = str(incoming.get("action") or incoming.get("kind") or "health")
    if action == "health":
        return health_body()
    return run_job(action, incoming, {"id": event.get("id") or incoming.get("job_id"), "backend": "serverless"})


def concurrency_modifier(_current: int = 1) -> int:
    return 1


if __name__ == "__main__":
    runtime = _runtime()
    if runtime == "pod":
        import pod_pull

        pod_pull.main()
    elif runtime == "http":
        import http_server

        http_server.main()
    else:
        import runpod

        runpod.serverless.start({"handler": handler, "concurrency_modifier": concurrency_modifier})
