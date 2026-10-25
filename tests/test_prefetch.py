from __future__ import annotations

import json
import os

import pytest

import catalog
import db
import handler
import prefetch_volume as pv
from fakes import FakeConn


def test_defaults_and_all():
    assert pv.select_models(None) == list(catalog.DEFAULT_MODELS)
    assert "flux-schnell" not in pv.select_models(None)
    assert "cogvideox-2b" not in pv.select_models(None)
    assert set(pv.select_models("all")) == set(catalog.MODELS)


def test_aliases_and_blocks():
    assert pv.select_models("flux,wan,whisper,llm") == ["flux-schnell", "wan2.2-ti2v-5b", "whisper-large-v3-turbo",
                                                        "qwen3.6-35b-a3b-awq"]
    assert pv.select_models(["Tongyi-MAI/Z-Image-Turbo"]) == ["z-image-turbo"]
    assert pv.select_models("vieneu,vieneu-0.3b") == ["vieneu-tts", "vieneu-tts-0.3b"]
    assert pv.select_models("pnnbao-ump/VieNeu-TTS") == ["vieneu-tts"]
    for bad in ("flux-dev", "black-forest-labs/FLUX.1-dev", "llama", "phowhisper", "nope"):
        with pytest.raises(ValueError):
            pv.select_models(bad)


def test_every_entry_has_layout_and_license():
    for key, row in catalog.MODELS.items():
        assert row["license"] in {"MIT", "Apache-2.0"}
        assert row["dir"] and row["approx_gb"] > 0 and row["need"]
    assert catalog.MODELS["flux-schnell"]["default"] is False
    assert catalog.MODELS["chatterbox"]["default"] is True
    for key in ("vieneu-tts", "vieneu-tts-0.3b", "neucodec", "w2v-bert-2.0"):
        assert catalog.MODELS[key]["default"] is True
    assert "CC-BY-NC-4.0" in catalog.MODELS["vieneu-tts"]["dataset_license"]
    assert catalog.volume_path("vieneu-tts").endswith(os.path.join("pnnbao-ump", "VieNeu-TTS"))


def test_hub_models_fill_the_hf_cache(tmp_path, monkeypatch):
    seen = {}

    def fake_snapshot(**kwargs):
        seen.update(kwargs)
        snap = tmp_path / "huggingface" / "hub" / "models--neuphonic--neucodec" / "snapshots" / "abc"
        snap.mkdir(parents=True)
        (snap / "pytorch_model.bin").write_text("x")
        (snap / "meta.yaml").write_text("x")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    out = pv.prefetch("neucodec", root=str(tmp_path))
    assert out["models"]["neucodec"]["status"] == "downloaded"
    assert seen["cache_dir"] == str(tmp_path / "huggingface" / "hub") and "local_dir" not in seen
    assert seen["allow_patterns"] == ["pytorch_model.bin", "meta.yaml", "config.json"]


def test_status_layout(tmp_path):
    body = pv.status(str(tmp_path))
    assert body["volume"] == str(tmp_path)
    assert body["models"]["birefnet"]["path"] == os.path.join(str(tmp_path), "birefnet")
    assert body["models"]["birefnet"]["ready"] is False
    assert body["default_gb"] < body["all_gb"]


def _fake_download(row, dest):
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "model_index.json"), "w") as fh:
        fh.write("{}")
    with open(os.path.join(dest, "config.json"), "w") as fh:
        fh.write("{}")
    with open(os.path.join(dest, "w.safetensors"), "w") as fh:
        fh.write("x")


def test_prefetch_downloads_then_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "_download", _fake_download)
    first = pv.prefetch(["birefnet", "z-image-turbo"], root=str(tmp_path))
    assert first["ok"] and first["models"]["birefnet"]["status"] == "downloaded"
    assert json.load(open(tmp_path / "birefnet" / pv.STAMP))["license"] == "MIT"
    again = pv.prefetch("birefnet", root=str(tmp_path))
    assert again["models"]["birefnet"]["status"] == "cached"


def test_incomplete_files_are_resumed(tmp_path, monkeypatch):
    dest = tmp_path / "birefnet"
    dest.mkdir()
    (dest / "w.safetensors").write_text("x")
    (dest / "big.safetensors.incomplete").write_text("x")
    assert pv.is_ready("birefnet", str(dest)) is False
    calls = []

    def resume(row, d):
        calls.append(d)
        os.remove(os.path.join(d, "big.safetensors.incomplete"))

    monkeypatch.setattr(pv, "_download", resume)
    assert pv.prefetch("birefnet", root=str(tmp_path))["models"]["birefnet"]["status"] == "downloaded"
    assert calls == [str(dest)]


def test_token_is_never_reported(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "hf_SECRETVALUE123")

    def fail(row, dest):
        raise RuntimeError("401 Client Error for token hf_SECRETVALUE123: gated")

    monkeypatch.setattr(pv, "_download", fail)
    out = pv.prefetch("flux", root=str(tmp_path))
    assert out["ok"] is False
    assert "hf_SECRETVALUE123" not in json.dumps(out)
    captured = capsys.readouterr()
    assert "hf_SECRETVALUE123" not in captured.out + captured.err
    assert "HF_TOKEN" in out["models"]["flux-schnell"]["error"]


def test_prefetch_records_volume_status(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "_download", _fake_download)
    conn = FakeConn()
    db.bind(lambda: conn)
    try:
        pv.prefetch("whisper", root=str(tmp_path))
    finally:
        db.bind(None)
    rows = [p for s, p in conn.calls if s.startswith("INSERT INTO aimx_volume_status")]
    assert rows and rows[0][1] == "whisper-large-v3-turbo" and rows[0][5] == "downloaded"


def test_handler_prefetch_action(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "_download", _fake_download)
    out = handler.handler({"input": {"action": "prefetch", "models": ["birefnet"], "root": str(tmp_path)}})
    assert out["status"] == "succeeded" and out["models"]["birefnet"]["status"] == "downloaded"
    blocked = handler.execute("prefetch", {"models": ["flux-dev"], "root": str(tmp_path)})
    assert blocked["status"] == "failed" and "non-commercial" in blocked["error"]
    listing = handler.execute("volume_status", {"root": str(tmp_path)})
    assert listing["models"]["birefnet"]["ready"] is True


def test_cli_list(tmp_path, capsys):
    assert pv.main(["--list", "--root", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "qwen3.6-35b-a3b-awq" in text and "optional" in text
