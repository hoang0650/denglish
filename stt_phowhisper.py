"""vinai/PhoWhisper-large — Vietnamese ASR for marketplace STT.

Cache on the RunPod network volume (same layout as VieNeu / Llama):

    /runpod-volume/phowhisper-large
    /runpod-volume/huggingface
"""

from __future__ import annotations

import os
from typing import Any, Optional

MODEL_ID = os.environ.get("PHOWHISPER_MODEL", "vinai/PhoWhisper-large")
READY_MARKER = ".phowhisper_ready"

STT_CANDIDATES = [
    os.environ.get("PHOWHISPER_MODEL_DIR", "").strip(),
    "/runpod-volume/phowhisper-large",
    "/workspace/phowhisper-large",
    os.path.join(os.getcwd(), "models", "phowhisper-large"),
]

_processor = None
_model = None
_model_error: Optional[str] = None
_model_dir = ""


def configure_hf_cache() -> str:
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


def _looks_like_model(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, READY_MARKER)):
        return True
    names = set(os.listdir(path))
    return "config.json" in names and any(
        n.endswith((".safetensors", ".bin", ".pt")) or n.startswith("model") for n in names
    )


def resolve_local_model_dir(extra: Optional[list[str]] = None) -> str:
    for path in list(extra or []) + STT_CANDIDATES:
        p = str(path or "").strip()
        if _looks_like_model(p):
            return p
    return ""


def preferred_stt_dir(extra: Optional[list[str]] = None) -> str:
    for path in list(extra or []) + STT_CANDIDATES:
        p = str(path or "").strip()
        if not p:
            continue
        parent = os.path.dirname(p) or "."
        if os.path.isdir(p) or (os.path.isdir(parent) and os.access(parent, os.W_OK)):
            return p
    return "/runpod-volume/phowhisper-large"


def ensure_model(
    *,
    repo_id: str = "",
    download: bool = True,
    extra_candidates: Optional[list[str]] = None,
) -> str:
    configure_hf_cache()
    repo = str(repo_id or MODEL_ID).strip() or MODEL_ID
    existing = resolve_local_model_dir(extra_candidates)
    if existing:
        print(f"[phowhisper] using cached model dir={existing}")
        return existing
    if not download:
        raise FileNotFoundError(
            "PhoWhisper not found on volume. Mount /runpod-volume/phowhisper-large "
            "or set PHOWHISPER_MODEL_DIR."
        )
    target = preferred_stt_dir(extra_candidates)
    os.makedirs(target, exist_ok=True)
    print(f"[phowhisper] downloading {repo} → {target}")
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo, local_dir=target, resume_download=True)
    with open(os.path.join(target, READY_MARKER), "w", encoding="utf-8") as f:
        f.write(f"{repo}\n")
    print(f"[phowhisper] cached model ready dir={target}")
    return target


def _load_engine(local_dir: str):
    global _processor, _model, _model_dir
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    src = local_dir if local_dir and os.path.isdir(local_dir) else MODEL_ID
    _processor = WhisperProcessor.from_pretrained(src)
    _model = WhisperForConditionalGeneration.from_pretrained(
        src,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    _model.to(device)
    _model.eval()
    _model_dir = src
    print(f"[phowhisper] ready model={src} device={device}")
    return _processor, _model


def get_engine():
    global _processor, _model, _model_error
    if _processor is not None and _model is not None:
        return _processor, _model
    if _model_error:
        raise RuntimeError(_model_error)
    try:
        local = ""
        try:
            local = ensure_model(download=os.environ.get("PHOWHISPER_DOWNLOAD", "1") != "0")
        except Exception as exc:  # noqa: BLE001
            print(f"[phowhisper] volume ensure failed ({exc}); falling back to hub id={MODEL_ID}")
            local = resolve_local_model_dir()
        return _load_engine(local)
    except Exception as exc:  # noqa: BLE001
        _model_error = str(exc)
        raise RuntimeError(_model_error) from exc


def _load_mono_16k(audio_path: str):
    """Load wav/mp3/m4a as float32 mono 16 kHz (PhoWhisper ASR)."""
    import numpy as np
    import torch

    wav = None
    sr = 0
    try:
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        wav = waveform.squeeze(0).float().numpy()
        sr = int(sr)
    except Exception:
        wav = None
    if wav is None:
        try:
            import soundfile as sf

            raw, sr = sf.read(audio_path, always_2d=False)
            if getattr(raw, "ndim", 1) > 1:
                raw = np.mean(raw, axis=1)
            wav = np.asarray(raw, dtype=np.float32)
            sr = int(sr)
        except Exception:
            wav = None
    if wav is None:
        import subprocess
        import tempfile

        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", tmp],
                check=True,
                capture_output=True,
            )
            import soundfile as sf

            wav, sr = sf.read(tmp, always_2d=False)
            wav = np.asarray(wav, dtype=np.float32)
            sr = 16000
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    if wav is None:
        raise RuntimeError(f"Could not decode audio: {audio_path}")
    wav = np.asarray(wav, dtype=np.float32)
    if sr != 16000:
        import torchaudio.functional as F

        tensor = torch.from_numpy(wav).float().unsqueeze(0)
        wav = F.resample(tensor, sr, 16000).squeeze(0).numpy()
        sr = 16000
    return wav, sr


def transcribe(audio_path: str, language: str = "vi") -> dict[str, Any]:
    """Return {text, language, model} from a local wav/mp3 path."""
    import torch

    processor, model = get_engine()
    wav, sr = _load_mono_16k(audio_path)
    device = next(model.parameters()).device
    inputs = processor(wav, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device=device, dtype=model.dtype)
    lang = (language or "vi").split("-")[0].lower()
    forced = None
    try:
        forced = processor.get_decoder_prompt_ids(language=lang, task="transcribe")
    except Exception:
        forced = None
    with torch.no_grad():
        predicted = model.generate(
            input_features,
            forced_decoder_ids=forced,
            max_new_tokens=256,
        )
    text = processor.batch_decode(predicted, skip_special_tokens=True)[0].strip()
    return {"text": text, "language": lang, "model": _model_dir or MODEL_ID}
