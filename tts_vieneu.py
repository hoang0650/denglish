"""VieNeu-TTS v3 Turbo — 20 preset voices + instant clone.

Model: pnnbao-ump/VieNeu-TTS-v3-Turbo (vieneu SDK ≥ 3.3.0)
Preset roster is from the model card (SDK v3.3.0). Call by `voice` name; clone with `ref_audio`.
"""

from __future__ import annotations

import base64
import os
import socket
import tempfile
from ipaddress import ip_address
from typing import Any, Optional
from urllib.error import URLError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

MODEL_ID = os.environ.get("VIENEU_MODEL", "pnnbao-ump/VieNeu-TTS-v3-Turbo")
DEFAULT_VOICE = os.environ.get("VIENEU_DEFAULT_VOICE", "Adam")
SAMPLE_RATE = 48000
READY_MARKER = ".vieneu_ready"

# Same pattern as handler BASE_CANDIDATES / LORA_CANDIDATES — network volume first.
TTS_CANDIDATES = [
    os.environ.get("VIENEU_MODEL_DIR", "").strip(),
    "/runpod-volume/vieneu-tts-v3-turbo",
    "/workspace/vieneu-tts-v3-turbo",
    os.path.join(os.getcwd(), "models", "vieneu-tts-v3-turbo"),
]


def configure_hf_cache() -> str:
    """Point Hugging Face caches at the RunPod network volume when present."""
    for root in ("/runpod-volume", "/workspace"):
        if os.path.isdir(root):
            hf = os.path.join(root, "huggingface")
            os.makedirs(hf, exist_ok=True)
            os.environ.setdefault("HF_HOME", hf)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf, "hub"))
            os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf, "transformers"))
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            return hf
    return ""


def tts_model_candidates(extra: Optional[list[str]] = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for path in list(extra or []) + TTS_CANDIDATES:
        p = str(path or "").strip()
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def preferred_tts_dir(extra: Optional[list[str]] = None) -> str:
    for path in tts_model_candidates(extra):
        parent = os.path.dirname(path) or "."
        if os.path.isdir(path):
            return path
        if os.path.isdir(parent) and os.access(parent, os.W_OK):
            return path
    return "/runpod-volume/vieneu-tts-v3-turbo"


def _looks_like_tts_model(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, READY_MARKER)):
        return True
    names = set(os.listdir(path))
    markers = (
        "config.json",
        "model.onnx",
        "model.onnx.data",
        "tokenizer.json",
        "preprocessor_config.json",
    )
    if any(m in names for m in markers):
        return True
    return any(n.endswith((".safetensors", ".onnx", ".bin", ".pt")) for n in names)


def resolve_local_model_dir(extra: Optional[list[str]] = None) -> str:
    for path in tts_model_candidates(extra):
        if _looks_like_tts_model(path):
            return path
    return ""


def ensure_model(
    *,
    repo_id: str = "",
    download: bool = True,
    extra_candidates: Optional[list[str]] = None,
) -> str:
    """Return local TTS dir; download from Hugging Face onto the volume if missing."""
    configure_hf_cache()
    repo = str(repo_id or MODEL_ID).strip() or MODEL_ID
    existing = resolve_local_model_dir(extra_candidates)
    if existing:
        print(f"[vieneu] using cached model dir={existing}")
        return existing
    if not download:
        raise FileNotFoundError(
            "VieNeu model not found on volume. Mount /runpod-volume/vieneu-tts-v3-turbo "
            "or set VIENEU_MODEL_DIR / run voice_assistant --ensure-tts."
        )
    target = preferred_tts_dir(extra_candidates)
    os.makedirs(target, exist_ok=True)
    print(f"[vieneu] downloading {repo} → {target}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to cache VieNeu on the volume") from exc
    snapshot_download(
        repo_id=repo,
        local_dir=target,
        resume_download=True,
    )
    marker = os.path.join(target, READY_MARKER)
    with open(marker, "w", encoding="utf-8") as f:
        f.write(f"{repo}\n")
    print(f"[vieneu] cached model ready dir={target}")
    return target


_engine = None
_engine_error: Optional[str] = None
_engine_dir: str = ""

PRESET_VOICES: list[dict[str, str]] = [
    {"id": "adam", "name": "Adam", "region": "Nam", "character": "Natural", "gender": "male"},
    {"id": "pham-tuyen", "name": "Phạm Tuyên", "region": "Bắc", "character": "Natural", "gender": "male"},
    {"id": "minh-duc", "name": "Minh Đức", "region": "Bắc", "character": "News", "gender": "male"},
    {"id": "thanh-binh", "name": "Thanh Bình", "region": "Bắc", "character": "Storytelling", "gender": "male"},
    {"id": "ngoc-huyen", "name": "Ngọc Huyền", "region": "Bắc", "character": "Natural", "gender": "female"},
    {"id": "truc-ly", "name": "Trúc Ly", "region": "Bắc", "character": "Natural", "gender": "female"},
    {"id": "doan-trang", "name": "Đoan Trang", "region": "Bắc", "character": "Natural", "gender": "female"},
    {"id": "ngoc-linh", "name": "Ngọc Linh", "region": "Bắc", "character": "Storytelling", "gender": "female"},
    {"id": "mai-anh", "name": "Mai Anh", "region": "Bắc", "character": "News", "gender": "female"},
    {"id": "quynh-anh", "name": "Quỳnh Anh", "region": "Bắc", "character": "Audiobook", "gender": "female"},
    {"id": "quang-son", "name": "Quang Sơn", "region": "Trung", "character": "Natural", "gender": "male"},
    {"id": "ngoc-tran", "name": "Ngọc Trân", "region": "Trung", "character": "Natural", "gender": "female"},
    {"id": "xuan-vinh", "name": "Xuân Vĩnh", "region": "Nam", "character": "Natural", "gender": "male"},
    {"id": "thai-son", "name": "Thái Sơn", "region": "Nam", "character": "Storytelling", "gender": "male"},
    {"id": "minh-triet", "name": "Minh Triết", "region": "Nam", "character": "News", "gender": "male"},
    {"id": "duc-tri", "name": "Đức Trí", "region": "Nam", "character": "Audiobook", "gender": "male"},
    {"id": "thuc-doan", "name": "Thục Đoan", "region": "Nam", "character": "Storytelling", "gender": "female"},
    {"id": "thuy-dung", "name": "Thùy Dung", "region": "Nam", "character": "News", "gender": "female"},
    {"id": "my-duyen", "name": "Mỹ Duyên", "region": "Nam", "character": "Audiobook", "gender": "female"},
    {"id": "kim-thanh", "name": "Kim Thanh", "region": "Nam", "character": "Audiobook", "gender": "female"},
]

_BY_KEY: dict[str, dict[str, str]] = {}
for _v in PRESET_VOICES:
    _BY_KEY[_v["id"]] = _v
    _BY_KEY[_v["name"].lower()] = _v
    _BY_KEY[_v["name"].casefold()] = _v

_engine = None
_engine_error: Optional[str] = None


def list_voices() -> list[dict[str, str]]:
    return [dict(v) for v in PRESET_VOICES]


def resolve_voice(name: Optional[str], lang: str = "") -> str:
    raw = str(name or "").strip()
    if not raw:
        if str(lang).lower().startswith("vi"):
            return "Ngọc Huyền"
        return DEFAULT_VOICE
    hit = _BY_KEY.get(raw.lower()) or _BY_KEY.get(raw.casefold())
    if hit:
        return hit["name"]
    return raw


def _build_engine(model_ref: str):
    from vieneu import Vieneu

    try:
        return Vieneu(backbone_repo=model_ref)
    except TypeError:
        try:
            return Vieneu(mode="v3turbo", backbone_repo=model_ref)
        except TypeError:
            try:
                return Vieneu(mode="v3turbo")
            except TypeError:
                return Vieneu()


def _get_engine():
    global _engine, _engine_error, _engine_dir
    if _engine is not None:
        return _engine
    if _engine_error:
        raise RuntimeError(_engine_error)
    try:
        from vieneu import Vieneu  # noqa: F401
    except ImportError as exc:
        _engine_error = "vieneu is not installed. pip install 'vieneu>=3.3.0'"
        raise RuntimeError(_engine_error) from exc
    try:
        configure_hf_cache()
        local = ""
        try:
            local = ensure_model(repo_id=MODEL_ID, download=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[vieneu] volume ensure failed ({exc}); falling back to hub id={MODEL_ID}")
            local = resolve_local_model_dir()
        model_ref = local or MODEL_ID
        _engine = _build_engine(model_ref)
        _engine_dir = local or MODEL_ID
        print(f"[vieneu] ready model={_engine_dir} default_voice={DEFAULT_VOICE}")
        return _engine
    except Exception as exc:  # noqa: BLE001
        _engine_error = str(exc)
        raise RuntimeError(_engine_error) from exc


_BLOCKED_HOSTS = {
    "localhost",
    "metadata",
    "metadata.google.internal",
    "metadata.internal",
}
_MAX_REF_BYTES = 12 * 1024 * 1024


def _is_blocked_ip(raw: str) -> bool:
    try:
        ip = ip_address(raw.split("%")[0])
    except ValueError:
        return False
    return bool(
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def assert_safe_http_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http(s) URLs are allowed")
    if parsed.username or parsed.password:
        raise ValueError("URLs with credentials are not allowed")
    host = (parsed.hostname or "").strip(".").lower()
    if not host or host in _BLOCKED_HOSTS or host.endswith(".localhost") or host.endswith(".internal") or host.endswith(".local"):
        raise ValueError("URL host is not allowed")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("URL host could not be resolved") from exc
    for info in infos:
        addr = info[4][0]
        if addr.startswith("::ffff:"):
            addr = addr[7:]
        if _is_blocked_ip(addr):
            raise ValueError("URL resolves to a private or metadata address")
    return parsed.geturl()


class _SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        assert_safe_http_url(urljoin(req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_ref(url: str, dest: str) -> str:
    safe = assert_safe_http_url(url)
    req = Request(safe, headers={"User-Agent": "AI-Markets-VieNeu/1.0"})
    opener = build_opener(_SafeRedirect)
    try:
        with opener.open(req, timeout=20) as res, open(dest, "wb") as out:
            chunk = res.read(_MAX_REF_BYTES + 1)
            if len(chunk) > _MAX_REF_BYTES:
                raise ValueError("Reference audio is too large")
            out.write(chunk)
    except URLError as exc:
        raise ValueError(f"Could not download reference audio: {exc}") from exc
    return dest


def write_ref_bytes(data: bytes, dest: str) -> str:
    with open(dest, "wb") as out:
        out.write(data)
    return dest


def decode_audio_b64(raw: str) -> bytes:
    s = str(raw or "").strip()
    if "," in s and s.lower().startswith("data:"):
        s = s.split(",", 1)[1]
    return base64.b64decode(s)


def _save_wav(engine, audio: Any, path: str) -> str:
    if hasattr(engine, "save"):
        engine.save(audio, path)
        return path
    import numpy as np

    wav = audio[0] if isinstance(audio, (tuple, list)) else audio
    arr = np.asarray(wav)
    if arr.ndim > 1:
        arr = arr.squeeze()
    try:
        import soundfile as sf

        sf.write(path, arr, SAMPLE_RATE)
    except Exception:
        import wave

        pcm = (arr.clip(-1, 1) * 32767).astype("int16")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())
    return path


def synthesize(
    text: str,
    *,
    voice: str = "",
    lang: str = "",
    ref_audio: str = "",
    denoise: bool = True,
) -> dict[str, Any]:
    """Return wav bytes + metadata. `ref_audio` is a local file path for cloning."""
    text = str(text or "").strip()
    if not text:
        return {"error": "Missing text/prompt for TTS."}

    engine = _get_engine()
    voice_name = resolve_voice(voice, lang)
    kwargs: dict[str, Any] = {}
    cloned = bool(ref_audio)
    if cloned:
        kwargs["ref_audio"] = ref_audio
        kwargs["denoise"] = bool(denoise)
    else:
        kwargs["voice"] = voice_name

    audio = engine.infer(text, **kwargs)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = tmp.name
    try:
        _save_wav(engine, audio, wav_path)
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    b64 = base64.b64encode(wav_bytes).decode("ascii")
    return {
        "kind": "audio",
        "text": text,
        "audio_base64": b64,
        "ai_response_audio": b64,
        "audio_url": f"data:audio/wav;base64,{b64}",
        "format": "wav",
        "sample_rate": SAMPLE_RATE,
        "voice": voice_name if not cloned else "cloned",
        "cloned": cloned,
        "model": _engine_dir or MODEL_ID,
        "status": "success",
    }
