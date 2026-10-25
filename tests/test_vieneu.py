"""VieNeu-TTS routing and inference glue, with a fake vieneu engine."""

from __future__ import annotations

import os

import pytest

import handler


class FakeEngine:
    sample_rate = 24000

    def __init__(self):
        self.calls = []

    def list_preset_voices(self):
        return [(d, n) for n, d in (("Vinh", "Vĩnh"), ("Binh", "Bình"), ("Tuyen", "Tuyên"), ("Doan", "Đoan"),
                                    ("Ly", "Ly"), ("Ngoc", "Ngọc"))]

    def get_preset_voice(self, name=None):
        names = [n for _d, n in self.list_preset_voices()]
        if name is not None and name not in names:
            raise ValueError("nope")
        return {"codes": [1, 2], "text": f"ref {name or 'Vinh'}"}

    def infer(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return [0.0] * 10

    def save(self, audio, path):
        with open(path, "wb") as fh:
            fh.write(b"RIFF")


@pytest.fixture()
def live(monkeypatch, tmp_path):
    monkeypatch.setenv("AIMX_WORKER_SANDBOX", "0")
    monkeypatch.setattr(handler, "VOLUME", str(tmp_path))
    for key in ("vieneu-tts", "vieneu-tts-0.3b", "neucodec"):
        vol = tmp_path / key
        vol.mkdir()
        (vol / "model.safetensors").write_text("x")
        monkeypatch.setitem(handler.COMMERCIAL_MODELS[key], "volume", str(vol))
    engine = FakeEngine()
    used = []
    monkeypatch.setattr(handler, "_vieneu_engine", lambda row: used.append(row["key"]) or engine)
    return engine, used


def test_language_routing():
    assert handler.tts_language({"text": "Xin chào các bạn"}) == "vi"
    assert handler.tts_language({"text": "hello", "language": "vi-VN"}) == "vi"
    assert handler.tts_language({"text": "hello"}) == "en"
    assert handler.tts_language({"text": "bonjour", "language": "fr"}) == "fr"
    assert handler.model_for("tts", "", {"text": "Xin chào"})["id"] == "pnnbao-ump/VieNeu-TTS"
    assert handler.model_for("tts", "", {"text": "hi", "language": "vi"})["id"] == "pnnbao-ump/VieNeu-TTS"
    assert handler.model_for("tts", "", {"text": "hi", "language": "de"})["id"] == "ResembleAI/chatterbox"
    assert handler.model_for("tts", "vieneu-0.3b", {"text": "Xin chào"})["id"] == "pnnbao-ump/VieNeu-TTS-0.3B"
    assert handler.model_for("tts", "pnnbao-ump/VieNeu-TTS-0.3B")["key"] == "vieneu-tts-0.3b"
    assert handler.model_for("clone", "", {"text": "Xin chào"})["id"] == "pnnbao-ump/VieNeu-TTS"


def test_default_vi_model_env(monkeypatch):
    monkeypatch.setenv("AIMX_VIENEU_DEFAULT", "vieneu-tts-0.3b")
    assert handler.model_for("tts", "", {"text": "Xin chào"})["id"] == "pnnbao-ump/VieNeu-TTS-0.3B"


def test_preset_names_map():
    assert handler.vieneu_voice("Ngọc Huyền") == "Ngoc"
    assert handler.vieneu_voice("pham-tuyen") == "Tuyen"
    assert handler.vieneu_voice("Binh") == "Binh"


def test_vieneu_preset_synthesis(live):
    engine, used = live
    out = handler.execute("tts", {"text": "Xin chào", "voice": "ngoc-huyen"})
    assert out["status"] == "succeeded", out
    assert out["model"] == "pnnbao-ump/VieNeu-TTS" and out["license"] == "Apache-2.0"
    assert out["voice"] == "Ngoc" and out["cloned"] is False and out["sample_rate"] == 24000
    assert os.path.isfile(out["artifact_url"])
    assert engine.calls[0][1]["voice"]["text"] == "ref Ngoc"
    assert used == ["vieneu-tts"]


def test_vieneu_fast_option_and_unknown_voice(live):
    engine, used = live
    out = handler.execute("tts", {"text": "Xin chào", "model": "vieneu-fast"})
    assert out["status"] == "succeeded" and out["model"] == "pnnbao-ump/VieNeu-TTS-0.3B"
    bad = handler.execute("tts", {"text": "Xin chào", "voice": "Nobody"})
    assert bad["status"] == "failed" and "Ngoc" in bad["error"]


def test_vieneu_clone_with_ref_text(live):
    engine, _ = live
    out = handler.execute("tts", {"text": "Xin chào", "ref_audio_base64": "UklGRg==", "ref_text": "mẫu giọng"})
    assert out["status"] == "succeeded" and out["cloned"] is True and out["voice"] == "cloned"
    kwargs = engine.calls[-1][1]
    assert kwargs["ref_text"] == "mẫu giọng" and os.path.isfile(kwargs["ref_audio"]) and "voice" not in kwargs


def test_vieneu_clone_blocks_internal_ref_urls(live):
    out = handler.execute("tts", {"text": "Xin chào", "ref_audio_url": "https://169.254.169.254/latest/x.wav",
                                  "ref_text": "a"})
    assert out["status"] == "failed" and "not public" in out["error"]


def test_vieneu_clone_without_ref_text_or_whisper(live, monkeypatch):
    monkeypatch.setitem(handler.COMMERCIAL_MODELS["whisper-large-v3-turbo"], "volume", "/nonexistent")
    out = handler.execute("tts", {"text": "Xin chào", "ref_audio_base64": "UklGRg=="})
    assert out["status"] == "failed" and "ref_text" in out["error"]


def test_vieneu_needs_codec(live, monkeypatch):
    monkeypatch.setitem(handler.COMMERCIAL_MODELS["neucodec"], "volume", "/nonexistent")
    out = handler.execute("tts", {"text": "Xin chào"})
    assert out["status"] == "failed" and "neucodec" in out["error"]


def test_english_still_goes_to_chatterbox(live):
    out = handler.execute("tts", {"text": "hello there", "language": "en"})
    # chatterbox volume is not set up in this fixture
    assert out["model"] == "ResembleAI/chatterbox"
