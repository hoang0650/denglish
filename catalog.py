"""Commercial model allowlist for the AI Markets RunPod worker.

One table drives handler.py (what a job may load), prefetch_volume.py (what is
downloaded onto the network volume) and the /health report. Every entry was
checked on Hugging Face on 2026-10-25: the repo exists and the license tag is
the one listed here. Target GPU: one 48 GB A40 / RTX A6000.

`default: True` models are downloaded by a plain `prefetch_volume.py` run.
Optional models (FLUX.1-schnell, CogVideoX-2b) are selectable per job with
"model": "<key or repo id>" once they are on the volume.

VieNeu-TTS (Vietnamese TTS + voice clone): the model repos are Apache-2.0. Its
training dataset pnnbao-ump/VieNeu-TTS-1000h is CC-BY-NC-4.0; the owner reviewed
that and chose to ship VieNeu on the basis of the model's Apache-2.0 license
(decision of 2026-10-25).
"""

from __future__ import annotations

import os
from typing import Any

VOLUME = os.environ.get("AIMX_VOLUME", "/runpod-volume")

# Kinds the marketplace queue knows today (providers/aimx_models.py in
# ai-marketplace-ai). Admission control in pod mode uses exactly these sets.
VIDEO_KINDS = frozenset({"fast", "talking", "film", "i2v"})
LIGHT_KINDS = frozenset({"tts", "image", "image_edit"})
TRAIN_KINDS = frozenset({"train"})
QUEUE_KINDS = VIDEO_KINDS | LIGHT_KINDS | TRAIN_KINDS

# Extra kinds this worker serves. The marketplace must add them to
# aimx_models.ALL_KINDS before it enqueues them; serverless and HTTP accept
# them now.
EXTRA_KINDS = frozenset({"bg_remove", "stt", "ocr", "embed", "rerank", "music", "llm"})

_DIFFUSERS_IGNORE = ["*.md", "assets/*", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.mp4", "*.webp"]

MODELS: dict[str, dict[str, Any]] = {
    "chatterbox": {
        "id": "ResembleAI/chatterbox",
        "license": "MIT",
        "tasks": ["tts"],
        "dir": "chatterbox",
        "default": True,
        "approx_gb": 14,
        "allow": ["*.safetensors", "*.pt", "*.json", "*.txt", "*.yaml", "*.model"],
        "need": [".safetensors"],
        "notes": "Non-Vietnamese TTS and zero-shot voice clone. Chatterbox Multilingual has 23 languages and NO Vietnamese.",
    },
    "vieneu-tts": {
        "id": "pnnbao-ump/VieNeu-TTS",
        "license": "Apache-2.0",
        "base": "neuphonic/neutts-air (Apache-2.0)",
        "dataset_license": "pnnbao-ump/VieNeu-TTS-1000h: CC-BY-NC-4.0 (owner accepted, 2026-10-25)",
        "tasks": ["tts"],
        "languages": ["vi"],
        # The vieneu SDK switches on the chat prompt only when the path ends in
        # "pnnbao-ump/VieNeu-TTS", so keep the repo id as the folder name.
        "dir": "pnnbao-ump/VieNeu-TTS",
        "default": True,
        "approx_gb": 2,
        "ignore": ["*.md", "*.gguf"],
        "need": ["model.safetensors", "voices.json"],
        "notes": "Default Vietnamese TTS (0.5B) + voice clone. vieneu SDK mode 'standard' with the NeuCodec codec.",
    },
    "vieneu-tts-0.3b": {
        "id": "pnnbao-ump/VieNeu-TTS-0.3B",
        "license": "Apache-2.0",
        "dataset_license": "pnnbao-ump/VieNeu-TTS-1000h: CC-BY-NC-4.0 (owner accepted, 2026-10-25)",
        "tasks": ["tts"],
        "languages": ["vi"],
        "dir": "pnnbao-ump/VieNeu-TTS-0.3B",
        "default": True,
        "approx_gb": 1,
        "ignore": ["*.md", "*.gguf"],
        "need": ["model.safetensors", "voices.json"],
        "notes": "Faster Vietnamese TTS (0.3B, ~2x faster). Pick with \"model\": \"vieneu-0.3b\".",
    },
    "neucodec": {
        "id": "neuphonic/neucodec",
        "license": "Apache-2.0",
        "tasks": ["codec"],
        # Loaded by repo id through the HF cache (neucodec checks the exact id).
        "hub": True,
        "dir": "huggingface/hub/models--neuphonic--neucodec",
        "default": True,
        "approx_gb": 2,
        "allow": ["pytorch_model.bin", "meta.yaml", "config.json"],
        "need": ["pytorch_model.bin", "meta.yaml"],
        "gated": True,
        "notes": "VieNeu audio codec (encode for clone, decode). Auto-approved gate: accept on HF and set HF_TOKEN.",
    },
    "w2v-bert-2.0": {
        "id": "facebook/w2v-bert-2.0",
        "license": "MIT",
        "tasks": ["codec"],
        "hub": True,
        "dir": "huggingface/hub/models--facebook--w2v-bert-2.0",
        "default": True,
        "approx_gb": 3,
        "allow": ["config.json", "model.safetensors", "preprocessor_config.json"],
        "need": ["model.safetensors", "preprocessor_config.json"],
        "notes": "Semantic encoder NeuCodec loads by repo id.",
    },
    "z-image-turbo": {
        "id": "Tongyi-MAI/Z-Image-Turbo",
        "license": "Apache-2.0",
        "tasks": ["image"],
        "dir": "z-image-turbo",
        "default": True,
        "approx_gb": 33,
        "ignore": _DIFFUSERS_IGNORE,
        "need": ["model_index.json"],
        "notes": "Default text-to-image. diffusers ZImagePipeline, 9 steps, guidance 0.",
    },
    "flux-schnell": {
        "id": "black-forest-labs/FLUX.1-schnell",
        "license": "Apache-2.0",
        "tasks": ["image"],
        "dir": "flux-schnell",
        "default": False,
        "approx_gb": 34,
        # diffusers layout only; skips the 24 GB single-file checkpoint.
        "ignore": _DIFFUSERS_IGNORE + ["flux1-schnell.safetensors", "ae.safetensors"],
        "need": ["model_index.json"],
        "gated": True,
        "notes": "Optional image model. Gated on HF: accept the terms and set HF_TOKEN to download.",
    },
    "wan2.2-ti2v-5b": {
        "id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "license": "Apache-2.0",
        "tasks": ["fast", "talking", "film"],
        "dir": "wan2.2-ti2v-5b",
        "default": True,
        "approx_gb": 34,
        "ignore": _DIFFUSERS_IGNORE,
        "need": ["model_index.json"],
        "notes": "Default text/image-to-video, 720p 24 fps.",
    },
    "wan2.2-i2v-a14b": {
        "id": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "revision": "596658fd9ca6b7b71d5057529bbf319ecbc61d74",
        "license": "Apache-2.0",
        "tasks": ["i2v"],
        "dir": "wan2.2-i2v-a14b",
        "default": False,
        "manual": True,
        "approx_gb": 72,
        "ignore": _DIFFUSERS_IGNORE,
        "need": ["model_index.json", "AIMX_SOURCE.json"],
        "lora": {"id": "lightx2v/Wan2.2-Lightning", "revision": "18bccf8884ec0a078eed79785eb4ef13ea16ce1e",
                 "license": "Apache-2.0", "dir": "lightning",
                 "files": ["high_noise_model.safetensors", "low_noise_model.safetensors"]},
        "notes": "AI Film image-to-video, 720p 16 fps. Two 14B experts stored in bf16 (converted from the fp32 "
                 "release) + 4-step Lightning LoRA (Apache-2.0). Placed by scripts/fetch_film_models.py, "
                 "not by prefetch (the fp32 release would not fit).",
    },
    "qwen-image-edit-2511": {
        "id": "Qwen/Qwen-Image-Edit-2511",
        "revision": "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9",
        "license": "Apache-2.0",
        "tasks": ["image_edit"],
        "dir": "qwen-image-edit-2511",
        "default": False,
        "manual": True,
        "approx_gb": 59,
        "ignore": _DIFFUSERS_IGNORE,
        "need": ["model_index.json", "AIMX_SOURCE.json"],
        "lora": {"id": "lightx2v/Qwen-Image-Edit-2511-Lightning", "revision": "107d238d68b8387e32843d0b4a2e24ceb0d3ad86",
                 "license": "Apache-2.0", "dir": "lightning",
                 "files": ["Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors"]},
        "notes": "AI Film keyframes / image edit, 1-3 reference images, 8-step Lightning LoRA (Apache-2.0).",
    },
    "cogvideox-2b": {
        "id": "THUDM/CogVideoX-2b",
        "license": "Apache-2.0",
        "tasks": ["fast", "talking", "film"],
        "dir": "cogvideox-2b",
        "default": False,
        "approx_gb": 14,
        "ignore": _DIFFUSERS_IGNORE,
        "need": ["model_index.json"],
        "notes": "Optional video model. HF redirects THUDM/ to zai-org/CogVideoX-2b.",
    },
    "birefnet": {
        "id": "ZhengPeng7/BiRefNet",
        "license": "MIT",
        "tasks": ["bg_remove"],
        "dir": "birefnet",
        "default": True,
        "approx_gb": 1,
        "allow": ["*.py", "*.json", "*.safetensors", "*.txt"],
        "need": [".safetensors"],
        "notes": "Background removal (trust_remote_code from the volume copy).",
    },
    "whisper-large-v3-turbo": {
        "id": "openai/whisper-large-v3-turbo",
        "license": "MIT",
        "tasks": ["stt"],
        "dir": "whisper-large-v3-turbo",
        "default": True,
        "approx_gb": 2,
        "allow": ["*.json", "*.safetensors", "*.txt"],
        "need": [".safetensors"],
        "notes": "Speech-to-text, Vietnamese included.",
    },
    "paddleocr-vl-1.5": {
        "id": "PaddlePaddle/PaddleOCR-VL-1.5",
        "license": "Apache-2.0",
        "tasks": ["ocr"],
        "dir": "paddleocr-vl-1.5",
        "default": True,
        "approx_gb": 2,
        "ignore": _DIFFUSERS_IGNORE,
        "need": [".safetensors"],
        "notes": "Document OCR through transformers>=5 (AutoModelForImageTextToText).",
    },
    "vn-embedding": {
        "id": "AITeamVN/Vietnamese_Embedding",
        "license": "Apache-2.0",
        "tasks": ["embed"],
        "dir": "vn-embedding",
        "default": True,
        "approx_gb": 3,
        "ignore": ["onnx/*", "*.md"],
        "need": [".safetensors"],
        "notes": "Vietnamese sentence embeddings (sentence-transformers).",
    },
    "vn-reranker": {
        "id": "AITeamVN/Vietnamese_Reranker",
        "license": "Apache-2.0",
        "tasks": ["rerank"],
        "dir": "vn-reranker",
        "default": True,
        "approx_gb": 3,
        "ignore": ["onnx/*", "*.md"],
        "need": [".safetensors"],
        "notes": "Vietnamese cross-encoder reranker.",
    },
    "ace-step-1.5": {
        "id": "ACE-Step/Ace-Step1.5",
        "license": "MIT",
        "tasks": ["music"],
        "dir": "ace-step-1.5",
        "default": True,
        "approx_gb": 11,
        "ignore": ["*.md", "assets/*", "*.png", "*.jpg"],
        "need": [".safetensors"],
        "notes": "Text-to-music. Served by the ACE-Step 1.5 API sidecar (AIMX_ACESTEP_URL).",
    },
    "qwen3.6-35b-a3b-awq": {
        "id": "QuantTrio/Qwen3.6-35B-A3B-AWQ",
        "license": "Apache-2.0",
        "base": "Qwen/Qwen3.6-35B-A3B (Apache-2.0)",
        "tasks": ["llm"],
        "dir": "qwen3.6-35b-a3b-awq",
        "default": True,
        "approx_gb": 26,
        "ignore": ["*.md"],
        "need": [".safetensors", "config.json"],
        "notes": "LLM served by vLLM on the pod (scripts/start_vllm.sh), OpenAI API on AIMX_VLLM_URL.",
    },
}

ALIASES: dict[str, str] = {
    "tts": "chatterbox", "voice": "chatterbox", "clone": "chatterbox",
    "vieneu": "vieneu-tts", "vieneu-0.5b": "vieneu-tts", "vieneu-tts-0.5b": "vieneu-tts", "vi-tts": "vieneu-tts",
    "vieneu-0.3b": "vieneu-tts-0.3b", "vieneu-fast": "vieneu-tts-0.3b", "vieneu-tts-0.3b": "vieneu-tts-0.3b",
    "codec": "neucodec",
    "zimage": "z-image-turbo", "z-image": "z-image-turbo", "image": "z-image-turbo",
    "flux": "flux-schnell", "flux1": "flux-schnell", "flux.1": "flux-schnell", "flux1-schnell": "flux-schnell",
    "schnell": "flux-schnell",
    "wan": "wan2.2-ti2v-5b", "wan2.2": "wan2.2-ti2v-5b", "video": "wan2.2-ti2v-5b",
    "wan-i2v": "wan2.2-i2v-a14b", "wan2.2-i2v": "wan2.2-i2v-a14b", "wan-2-2-i2v-720": "wan2.2-i2v-a14b",
    "i2v": "wan2.2-i2v-a14b",
    "qwen-image-edit": "qwen-image-edit-2511", "image-edit": "qwen-image-edit-2511", "image_edit": "qwen-image-edit-2511",
    "cogvideox": "cogvideox-2b", "cogvideo": "cogvideox-2b", "zai-org/cogvideox-2b": "cogvideox-2b",
    "bg": "birefnet", "bg_remove": "birefnet", "rembg": "birefnet",
    "whisper": "whisper-large-v3-turbo", "stt": "whisper-large-v3-turbo",
    "ocr": "paddleocr-vl-1.5", "paddleocr": "paddleocr-vl-1.5", "paddleocr-vl": "paddleocr-vl-1.5",
    "embedding": "vn-embedding", "embed": "vn-embedding",
    "reranker": "vn-reranker", "rerank": "vn-reranker",
    "music": "ace-step-1.5", "ace-step": "ace-step-1.5", "acestep": "ace-step-1.5",
    "llm": "qwen3.6-35b-a3b-awq", "qwen": "qwen3.6-35b-a3b-awq", "qwen3.6": "qwen3.6-35b-a3b-awq",
    "qwen/qwen3.6-35b-a3b": "qwen3.6-35b-a3b-awq",
}
for _key, _row in MODELS.items():
    ALIASES.setdefault(_key, _key)
    ALIASES.setdefault(_row["id"].lower(), _key)

# Exact repo ids and short names that must never load.
BLOCKED_MODELS: dict[str, str] = {
    "meta-llama/Meta-Llama-3-70B": "70B tutor weights are not an AI Markets workload.",
    "meta-llama/Meta-Llama-3-70B-Instruct": "70B tutor weights are not an AI Markets workload.",
    "black-forest-labs/FLUX.1-dev": "FLUX.1-dev is non-commercial.",
}
BLOCKED_NAMES: dict[str, str] = {
    "llama": "Llama / Denglish weights are not part of AI Markets.",
    "llama3": "Llama / Denglish weights are not part of AI Markets.",
    "denglish": "Llama / Denglish weights are not part of AI Markets.",
    "denglish-lora": "Llama / Denglish weights are not part of AI Markets.",
    "lora": "Llama / Denglish weights are not part of AI Markets.",
    "phowhisper": "PhoWhisper is not on the allowlist; use whisper-large-v3-turbo.",
    "flux-dev": "FLUX.1-dev is non-commercial.",
    "comfyui": "The worker uses diffusers directly; ComfyUI is not needed.",
}

DEFAULT_MODELS = tuple(k for k, row in MODELS.items() if row.get("default"))


def volume_path(key: str, root: str | None = None) -> str:
    return os.path.normpath(os.path.join(root or VOLUME, MODELS[key]["dir"]))


def blocked_reason(name: str) -> str:
    raw = str(name or "").strip()
    if raw in BLOCKED_MODELS:
        return BLOCKED_MODELS[raw]
    low = raw.lower()
    for repo, why in BLOCKED_MODELS.items():
        if repo.lower() == low:
            return why
    return BLOCKED_NAMES.get(low, "")


def resolve(name: str) -> str:
    """Return the catalog key for a key, alias, repo id, or volume path. Raises ValueError."""
    raw = str(name or "").strip()
    why = blocked_reason(raw)
    if why:
        raise ValueError(why)
    low = raw.lower()
    if low in ALIASES:
        return ALIASES[low]
    for key in MODELS:
        if raw in {volume_path(key), volume_path(key).rstrip("/")}:
            return key
    raise ValueError(f"model '{raw}' is not on the commercial allowlist")


def defaults_for(task: str) -> list[str]:
    """Catalog keys that serve a task, default model first."""
    keys = [k for k, row in MODELS.items() if task in row["tasks"]]
    return sorted(keys, key=lambda k: 0 if MODELS[k].get("default") else 1)


def total_gb(keys: Any = None) -> int:
    chosen = list(MODELS) if keys is None else list(keys)
    return int(sum(MODELS[k]["approx_gb"] for k in chosen))
