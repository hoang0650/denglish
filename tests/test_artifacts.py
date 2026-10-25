"""Job outputs live only in Postgres + Redis, never on the model volume."""

from __future__ import annotations

import base64
import json
import os
import pathlib

import pytest

import artifacts
import catalog
import db
import handler
import media
import pod_pull
from fakes import FakeConn, FakeRedis

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = "aimarkets"


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    vol = tmp_path / "vol"
    vol.mkdir()
    monkeypatch.setattr(catalog, "VOLUME", str(vol))
    monkeypatch.setenv("AIMX_VOLUME", str(vol))
    monkeypatch.setenv("AIMX_SCRATCH_DIR", str(tmp_path / "scratch"))
    return tmp_path


@pytest.fixture
def stores(monkeypatch):
    conn = FakeConn()
    db.bind(lambda: conn)
    monkeypatch.setenv("AIMX_DB_MIGRATE", "0")
    fake = FakeRedis()
    pod_pull.bind(fake)
    yield conn, fake
    db.bind(None)
    pod_pull.bind(None)


def test_guard_refuses_volume_paths(scratch):
    vol = scratch / "vol"
    for bad in (str(vol), str(vol / "outputs" / "a.wav"), "/runpod-volume/outputs/x.png",
                str(vol / ".." / "vol" / "x.mp4")):
        with pytest.raises(artifacts.OutputPathError):
            artifacts.assert_not_volume(bad)
    assert artifacts.assert_not_volume(str(scratch / "scratch" / "a.wav"))


def test_symlink_into_volume_is_refused(scratch):
    link = scratch / "sneaky"
    os.symlink(scratch / "vol", link)
    with pytest.raises(artifacts.OutputPathError):
        artifacts.assert_not_volume(str(link / "a.wav"))


def test_scratch_dir_on_volume_is_refused(scratch, monkeypatch):
    monkeypatch.setenv("AIMX_SCRATCH_DIR", str(scratch / "vol" / "tmp"))
    with pytest.raises(artifacts.OutputPathError):
        artifacts.new_path("tts", "wav")
    with pytest.raises(artifacts.OutputPathError):
        artifacts.temp_file(".wav")


def test_tempdir_guard(scratch, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(scratch / "vol" / "tmp"))
    artifacts.guard_tempdir()
    assert os.environ["TMPDIR"] == "/tmp"
    assert not artifacts.under_volume(__import__("tempfile").gettempdir())


def test_outputs_go_to_scratch_not_volume(scratch):
    path = handler._out("tts", "wav")
    assert path.startswith(str(scratch / "scratch"))
    assert not artifacts.under_volume(path)
    artifacts.cleanup()


def test_collect_persists_to_postgres_and_redis_and_deletes(scratch, stores):
    conn, fake = stores
    path = artifacts.new_path("image", "png")
    pathlib.Path(path).write_bytes(b"\x89PNG fake")
    out = artifacts.collect({"artifact_url": path, "image_url": path}, job_id="j9", kind="image")
    assert not os.path.exists(path)
    aid = out["artifact_id"]
    assert out["artifact_url"] == out["image_url"] == f"aimx-artifact://{aid}"
    assert out["artifact_store"] == ["postgres", "redis"]
    assert base64.b64decode(out["artifact_base64"]) == b"\x89PNG fake"
    inserts = [c for c in conn.calls if c[0].startswith("INSERT INTO aimx_job_artifacts")]
    assert len(inserts) == 1 and inserts[0][1][0] == aid and inserts[0][1][1] == "j9"
    assert inserts[0][1][-1] == b"\x89PNG fake"
    stored = json.loads(fake.get(f"{P}:aimx:artifact:{aid}"))
    assert base64.b64decode(stored["data_base64"]) == b"\x89PNG fake"
    assert fake.ttl[f"{P}:aimx:artifact:{aid}"] == 7 * 24 * 3600


def test_collect_refuses_output_on_volume(scratch, stores):
    bad = scratch / "vol" / "outputs" / "a.wav"
    bad.parent.mkdir()
    bad.write_bytes(b"x")
    with pytest.raises(artifacts.OutputPathError):
        artifacts.collect({"audio_url": str(bad)})


def test_no_store_fails_and_deletes(scratch, monkeypatch):
    monkeypatch.setenv("AIMX_INLINE_ARTIFACT_MAX_BYTES", "0")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("AIMX_REDIS_URL", raising=False)
    path = artifacts.new_path("video", "mp4")
    pathlib.Path(path).write_bytes(b"video")
    with pytest.raises(RuntimeError):
        artifacts.collect({"video_url": path})
    assert not os.path.exists(path)


def test_run_cleans_scratch_inputs_and_outputs(scratch, stores, monkeypatch):
    seen = {}

    def fake_infer(task, row, payload):
        ref = media.resolve(payload, "audio", catalog.VOLUME, ".wav")
        dest = handler._out("tts", "wav")
        pathlib.Path(dest).write_bytes(b"RIFF")
        seen.update(ref=ref, dest=dest)
        return {"artifact_url": dest, "audio_url": dest}

    monkeypatch.setattr(handler, "_infer", fake_infer)
    row = {"id": "m", "license": "MIT", "key": "chatterbox", "volume": str(scratch / "vol")}
    out = handler._run("tts", row, {"audio_base64": base64.b64encode(b"ref").decode()}, job_id="j1")
    assert out["status"] == "succeeded" and out["artifact_url"].startswith("aimx-artifact://")
    assert not os.path.exists(seen["ref"]) and not os.path.exists(seen["dest"])
    assert not any((scratch / "vol").rglob("*"))


def test_run_fails_when_output_lands_on_volume(scratch, stores, monkeypatch):
    def bad_infer(task, row, payload):
        dest = scratch / "vol" / "x.png"
        dest.write_bytes(b"x")
        return {"image_url": str(dest)}

    monkeypatch.setattr(handler, "_infer", bad_infer)
    out = handler._run("image", {"id": "m", "license": "MIT", "key": "z", "volume": ""}, {})
    assert out["status"] == "failed" and "model volume" in out["error"]


def test_db_result_and_redis_row_never_carry_base64(stores):
    _conn, fake = stores
    assert "artifact_base64" not in db._RESULT_KEYS
    raw = db._result_json({"artifact_id": "a", "artifact_base64": "AAAA", "artifact_url": "aimx-artifact://a"})
    assert "AAAA" not in raw and json.loads(raw)["artifact_id"] == "a"
    fake.set(f"{P}:aimx:job:j2", json.dumps({"id": "j2", "kind": "tts", "status": "running", "startedAt": 1}))
    row = pod_pull.finish("j2", status="succeeded", output={"artifact_id": "a", "artifact_base64": "AAAA",
                                                              "artifact_url": "aimx-artifact://a"})
    assert "artifact_base64" not in row and row["artifact_id"] == "a"


def test_source_never_writes_under_volume():
    for name in ("handler.py", "media.py", "prefetch_volume.py", "pod_pull.py", "Dockerfile"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "/outputs" not in text, name
        assert "tempfile.mkstemp(" not in text or name == "artifacts.py", name


def test_migration_has_artifact_table():
    sql = (ROOT / "migrations" / "001_aimx_jobs.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS aimx_job_artifacts" in sql and "BYTEA" in sql
