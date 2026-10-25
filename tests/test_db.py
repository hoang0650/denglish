from __future__ import annotations

import os

import pytest

import db
import handler
from fakes import FakeConn


@pytest.fixture(autouse=True)
def reset():
    db.bind(None)
    yield
    db.bind(None)


def test_disabled_without_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("AIMX_DATABASE_URL", raising=False)
    monkeypatch.delenv("AIMX_DB_DISABLE", raising=False)
    assert db.enabled() is False
    assert db.job_started({"id": "x"}, runtime="pod") is False


def test_aimx_url_wins(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://a@h/one")
    monkeypatch.setenv("AIMX_DATABASE_URL", "postgresql://a@h/aimx_worker")
    assert db.database_url().endswith("/aimx_worker")


def test_errors_are_scrubbed(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://aimx_worker:S3cret@db.aimarkets.vn:5432/aimx_worker?sslmode=require")
    text = db._safe(RuntimeError("failed postgresql://aimx_worker:S3cret@db.aimarkets.vn:5432/aimx_worker?sslmode=require password=S3cret"))
    assert "S3cret" not in text


def test_failed_writes_never_raise_and_back_off(capsys):
    calls = []

    def factory():
        calls.append(1)
        raise OSError("connection refused for postgresql://u:pw@h/db")

    db.bind(factory)
    assert db.job_started({"id": "a", "kind": "tts"}, runtime="serverless") is False
    assert db.job_finished({"id": "a", "status": "failed"}, runtime="serverless") is False
    assert len(calls) == 1  # second call is inside the back-off window
    assert "pw" not in capsys.readouterr().out


def test_serverless_handler_records_job():
    conn = FakeConn()
    db.bind(lambda: conn)
    out = handler.handler({"id": "rp-123", "input": {"action": "image", "prompt": "a cat", "tier": "paid"}})
    assert out["status"] == "succeeded"
    inserts = [p for s, p in conn.calls if s.startswith("INSERT INTO aimx_jobs")]
    assert inserts[0][0] == "rp-123" and inserts[0][1] == "image" and inserts[0][4] == "paid"
    assert inserts[1][2] == "Tongyi-MAI/Z-Image-Turbo" and inserts[1][9] == "succeeded"


def test_db_failure_does_not_fail_job():
    db.bind(lambda: FakeConn(fail=True))
    out = handler.handler({"id": "rp-9", "input": {"action": "stt"}})
    assert out["status"] == "succeeded"


def test_result_json_caps_size(monkeypatch):
    monkeypatch.setenv("AIMX_DB_RESULT_MAX_BYTES", "100")
    raw = db._result_json({"text": "ok", "embeddings": [[0.5] * 100]})
    assert '"truncated": true' in raw and "embeddings" not in raw


def test_migration_file_matches_tables():
    sql = open(db._MIGRATION, encoding="utf-8").read()
    assert "CREATE TABLE IF NOT EXISTS aimx_jobs" in sql
    assert "CREATE TABLE IF NOT EXISTS aimx_volume_status" in sql
    for column in ("id", "kind", "model", "tier", "status", "gpu_seconds", "artifact_url", "error",
                   "created_at", "started_at", "finished_at"):
        assert f"    {column} " in sql


@pytest.mark.skipif(not os.environ.get("AIMX_TEST_DATABASE_URL"), reason="AIMX_TEST_DATABASE_URL not set")
def test_live_postgres_round_trip(monkeypatch):
    import uuid

    import psycopg

    url = os.environ["AIMX_TEST_DATABASE_URL"]
    monkeypatch.delenv("AIMX_DB_DISABLE", raising=False)
    monkeypatch.setenv("AIMX_DATABASE_URL", url)
    db.bind(None)
    assert db.migrate() is True
    jid = "live-" + uuid.uuid4().hex[:8]
    assert db.job_started({"id": jid, "kind": "embed", "tier": "free", "createdAt": 1700000000}, runtime="http")
    assert db.job_finished({"id": jid, "kind": "embed", "status": "succeeded", "gpu_seconds": 1.25,
                            "finishedAt": 1700000100}, runtime="http",
                           output={"model": "AITeamVN/Vietnamese_Embedding", "dim": 3, "embeddings": [[1, 2, 3]]})
    assert db.volume_status("/runpod-volume", "birefnet", {"repo": "ZhengPeng7/BiRefNet", "status": "cached"})
    with psycopg.connect(url) as conn:
        row = conn.execute("select status, gpu_seconds, model, result->>'dim', tier from aimx_jobs where id=%s", (jid,)).fetchone()
        assert row == ("succeeded", 1.25, "AITeamVN/Vietnamese_Embedding", "3", "free")
        conn.execute("delete from aimx_jobs where id=%s", (jid,))
        assert conn.execute("select status from aimx_volume_status where model_key='birefnet'").fetchone()[0] == "cached"
