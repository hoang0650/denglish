"""Contract for the AI Markets RunPod worker. No GPU and no network."""

from __future__ import annotations

import json
import pathlib

import handler
import pod_pull
from fakes import FakeRedis

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_health_lists_commercial_models():
    body = handler.health_body()
    assert body["service"] == "aimx-runpod-worker"
    licenses = {row["id"]: row["license"] for row in body["models"]}
    assert licenses["ResembleAI/chatterbox"] == "MIT"
    assert licenses["Tongyi-MAI/Z-Image-Turbo"] == "Apache-2.0"
    assert licenses["black-forest-labs/FLUX.1-schnell"] == "Apache-2.0"
    assert licenses["Wan-AI/Wan2.2-TI2V-5B-Diffusers"] == "Apache-2.0"
    assert licenses["THUDM/CogVideoX-2b"] == "Apache-2.0"
    assert licenses["ZhengPeng7/BiRefNet"] == "MIT"
    assert licenses["openai/whisper-large-v3-turbo"] == "MIT"
    assert licenses["PaddlePaddle/PaddleOCR-VL-1.5"] == "Apache-2.0"
    assert licenses["AITeamVN/Vietnamese_Embedding"] == "Apache-2.0"
    assert licenses["AITeamVN/Vietnamese_Reranker"] == "Apache-2.0"
    assert licenses["ACE-Step/Ace-Step1.5"] == "MIT"
    assert licenses["QuantTrio/Qwen3.6-35B-A3B-AWQ"] == "Apache-2.0"
    assert licenses["pnnbao-ump/VieNeu-TTS"] == "Apache-2.0"
    assert licenses["pnnbao-ump/VieNeu-TTS-0.3B"] == "Apache-2.0"
    assert licenses["neuphonic/neucodec"] == "Apache-2.0"
    assert not any("VieNeu" in name for name in body["blocked"])


def test_tasks_resolve_commercial_checkpoints():
    assert handler.execute("tts", {"text": "hello"})["model"] == "ResembleAI/chatterbox"
    assert handler.execute("clone", {"text": "hello"})["model"] == "ResembleAI/chatterbox"
    assert handler.execute("image", {"prompt": "a lamp"})["model"] == "Tongyi-MAI/Z-Image-Turbo"
    assert handler.execute("fast", {"prompt": "a walk"})["model"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert handler.execute("film", {"prompt": "a walk"})["model"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert handler.execute("bg_remove", {})["model"] == "ZhengPeng7/BiRefNet"
    assert handler.execute("stt", {})["model"] == "openai/whisper-large-v3-turbo"
    assert handler.execute("transcribe", {})["model"] == "openai/whisper-large-v3-turbo"
    assert handler.execute("ocr", {})["model"] == "PaddlePaddle/PaddleOCR-VL-1.5"
    assert handler.execute("embed", {"texts": ["xin chào"]})["model"] == "AITeamVN/Vietnamese_Embedding"
    assert handler.execute("rerank", {})["model"] == "AITeamVN/Vietnamese_Reranker"
    assert handler.execute("music", {})["model"] == "ACE-Step/Ace-Step1.5"
    assert handler.execute("llm", {"prompt": "hi"})["model"] == "QuantTrio/Qwen3.6-35B-A3B-AWQ"
    assert handler.execute("tts", {"text": "hi"})["license"] == "MIT"
    assert handler.execute("tts", {"text": "xin chào", "language": "vi"})["model"] == "pnnbao-ump/VieNeu-TTS"


def test_optional_models_are_selectable():
    assert handler.execute("image", {"model": "flux-schnell"})["model"] == "black-forest-labs/FLUX.1-schnell"
    assert handler.execute("image", {"model": "black-forest-labs/FLUX.1-schnell"})["model"] == "black-forest-labs/FLUX.1-schnell"
    assert handler.execute("fast", {"model": "cogvideox-2b"})["model"] == "THUDM/CogVideoX-2b"
    out = handler.execute("image", {"model": "cogvideox-2b"})
    assert out["status"] == "failed" and "does not serve image" in out["error"]


def test_tutor_actions_fail():
    for action in ("chat", "agent_turn", "generate_test", "grade_test"):
        out = handler.execute(action, {"text": "hello"})
        assert out["status"] == "failed"
        assert action in out["error"]


def test_blocked_and_unknown_models():
    for task, name in (
        ("tts", "meta-llama/Meta-Llama-3-70B"),
        ("image", "black-forest-labs/FLUX.1-dev"),
    ):
        try:
            handler.model_for(task, name)
            raised = False
        except ValueError:
            raised = True
        assert raised, name
    out = handler.execute("tts", {"model": "denglish-lora"})
    assert out["status"] == "failed"
    out = handler.execute("image", {"model": "black-forest-labs/FLUX.1-dev"})
    assert out["status"] == "failed" and "non-commercial" in out["error"]


def test_train_policy_hand_off():
    ok = handler.execute("train", {"policy": "act"})
    assert ok["status"] == "handed_off"
    assert ok["backend"] == "pod"
    assert ok["license"] == "Apache-2.0"
    assert ok["gpu_seconds"] == 0
    bad = handler.execute("train", {"policy": "pi0"})
    assert bad["status"] == "failed"
    assert "pi0" in bad["error"]


def test_empty_volume_is_an_error(monkeypatch):
    monkeypatch.setenv("AIMX_WORKER_SANDBOX", "0")
    out = handler.execute("image", {"prompt": "a cup"})
    assert out["status"] == "failed"
    assert "z-image-turbo" in out["error"]


def test_chatterbox_rejects_vietnamese(monkeypatch, tmp_path):
    monkeypatch.setenv("AIMX_WORKER_SANDBOX", "0")
    vol = tmp_path / "chatterbox"
    vol.mkdir()
    (vol / "t3.safetensors").write_text("x")
    monkeypatch.setitem(handler.COMMERCIAL_MODELS["chatterbox"], "volume", str(vol))
    monkeypatch.setattr(handler, "VOLUME", str(tmp_path))
    out = handler.execute("tts", {"text": "xin chào", "language": "vi", "model": "chatterbox"})
    assert out["status"] == "failed"
    assert "Vietnamese" in out["error"] and "vieneu" in out["error"]


def test_llm_sidecar_unreachable_is_a_clean_failure(monkeypatch):
    monkeypatch.setenv("AIMX_WORKER_SANDBOX", "0")
    monkeypatch.setenv("AIMX_VLLM_URL", "http://127.0.0.1:9/v1")
    out = handler.execute("llm", {"prompt": "hi"})
    assert out["status"] == "failed"
    assert "not reachable" in out["error"]


def test_music_stub_explains_setup(monkeypatch):
    monkeypatch.setenv("AIMX_WORKER_SANDBOX", "0")
    monkeypatch.delenv("AIMX_ACESTEP_URL", raising=False)
    out = handler.execute("music", {"prompt": "lofi"})
    assert out["status"] == "failed"
    assert "AIMX_ACESTEP_URL" in out["error"]


def test_pod_pull_leases_paid_video_before_free_tts():
    fake = FakeRedis()
    pod_pull.bind(fake)
    prefix = "aimarkets"
    free_id = "a" * 32
    paid_id = "b" * 32
    fake.set(
        f"{prefix}:aimx:job:{free_id}",
        json.dumps({"id": free_id, "kind": "tts", "status": "queued", "tier": "free", "score": 10, "payload": {"text": "hi"}}),
    )
    fake.zadd(f"{prefix}:aimx:q:free", {free_id: 10})
    fake.set(
        f"{prefix}:aimx:job:{paid_id}",
        json.dumps({"id": paid_id, "kind": "fast", "status": "queued", "tier": "paid", "score": 1, "payload": {"prompt": "go"}}),
    )
    fake.zadd(f"{prefix}:aimx:q:paid", {paid_id: 1})

    leased = pod_pull.lease()
    assert leased["id"] == paid_id
    assert pod_pull.lease() is None
    pod_pull.finish(paid_id, status="succeeded", gpu_seconds=1)
    assert pod_pull.step() == 1
    saved = json.loads(fake.get(f"{prefix}:aimx:job:{free_id}"))
    assert saved["status"] == "succeeded"
    assert saved["gpu_seconds"] == 0.01


def test_image_pin_and_ignore_env():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "torch==2.8.0" in dockerfile
    assert "torchaudio==2.8.0" in dockerfile
    assert "cu128" in dockerfile
    for line in dockerfile.splitlines():
        if line.strip().startswith("COPY"):
            for banned in ("tts_vieneu", "stt_phowhisper", "voice_assistant", "train.py", "llama", ".env"):
                assert banned not in line
    workflow = (ROOT / ".github" / "workflows" / "docker-build.yml").read_text()
    assert ":sha-${{ github.sha }}" in workflow
    assert ".env" in (ROOT / ".gitignore").read_text()
    assert ".env" in (ROOT / ".dockerignore").read_text()
    example = (ROOT / ".env.example").read_text()
    for line in example.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition("=")
        assert name not in {"REDIS_URL", "DATABASE_URL", "AIMX_DATABASE_URL", "HF_TOKEN", "AIMX_HTTP_TOKEN",
                            "AIMX_VLLM_API_KEY"} or value == "", f"{name} must not carry a value"


def test_deep_health_reports_store_checks(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("AIMX_REDIS_URL", raising=False)
    body = handler.handler({"input": {"action": "health", "deep": True}})
    assert body["checks"]["postgres"]["ok"] is False and body["checks"]["redis"]["ok"] is False
    assert "checks" not in handler.handler({"input": {"action": "health"}})


def test_chatterbox_style_clamps_and_ignores_junk():
    import handler
    assert handler.chatterbox_style(None) == {}
    assert handler.chatterbox_style({"cfg_weight": 0}) == {"cfg_weight": 0.0}
    got = handler.chatterbox_style({"cfg_weight": "5", "exaggeration": -1, "temperature": "abc"})
    assert got == {"cfg_weight": 1.0, "exaggeration": 0.25}
    assert handler.chatterbox_style({"cfg_weight": True, "exaggeration": float("nan")}) == {}
