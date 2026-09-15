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

# Authoritative 20 presets (SDK v3.3.0). `id` is a URL-safe alias; `name` is what Vieneu.infer(voice=) expects.
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


def _get_engine():
    global _engine, _engine_error
    if _engine is not None:
        return _engine
    if _engine_error:
        raise RuntimeError(_engine_error)
    try:
        from vieneu import Vieneu
    except ImportError as exc:
        _engine_error = "vieneu is not installed. pip install 'vieneu>=3.3.0'"
        raise RuntimeError(_engine_error) from exc
    try:
        try:
            _engine = Vieneu(backbone_repo=MODEL_ID)
        except TypeError:
            try:
                _engine = Vieneu(mode="v3turbo")
            except TypeError:
                _engine = Vieneu()
        print(f"[vieneu] ready model={MODEL_ID} default_voice={DEFAULT_VOICE}")
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
        "model": MODEL_ID,
        "status": "success",
    }
