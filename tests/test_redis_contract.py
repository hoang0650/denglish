"""Queue semantics must match providers/aimx_queue.py in ai-marketplace-ai."""

from __future__ import annotations

import json
import os

import pytest

import db
import pod_pull
from fakes import FakeConn, FakeRedis

P = "aimarkets"


def _job(fake, jid, kind, tier, score, **extra):
    row = {"id": jid, "kind": kind, "status": "queued", "tier": tier, "score": score,
           "priority": 0, "backend": "pod", "createdAt": 1.0, "payload": {"text": "hi"}, **extra}
    fake.set(f"{P}:aimx:job:{jid}", json.dumps(row))
    fake.zadd(f"{P}:aimx:q:{tier}", {jid: score})
    return row


@pytest.fixture()
def fake():
    f = FakeRedis()
    pod_pull.bind(f)
    db.bind(None)
    yield f
    pod_pull.bind(None)


def test_key_names_and_prefix(monkeypatch):
    assert pod_pull._lane_key("paid") == "aimarkets:aimx:q:paid"
    assert pod_pull._lane_key("free") == "aimarkets:aimx:q:free"
    assert pod_pull._job_key("x") == "aimarkets:aimx:job:x"
    assert pod_pull._running_key() == "aimarkets:aimx:running"
    monkeypatch.setenv("REDIS_KEY_PREFIX", "other")
    assert pod_pull._lane_key("paid") == "other:aimx:q:paid"
    assert pod_pull._ns("other:aimx:q:paid") == "other:aimx:q:paid"


def test_highest_score_first_and_head_not_skipped(fake):
    _job(fake, "low", "tts", "free", 5)
    _job(fake, "high", "tts", "free", 9)
    assert pod_pull.lease()["id"] == "high"
    # A video at the head of paid must wait for the running light job; nothing else jumps it.
    _job(fake, "vid", "fast", "paid", 1)
    assert pod_pull.lease() is None
    assert "low" in fake.zsets[f"{P}:aimx:q:free"]


def test_light_slots(fake, monkeypatch):
    monkeypatch.setenv("AIMX_POD_LIGHT_SLOTS", "2")
    for i in range(3):
        _job(fake, f"t{i}", "image", "free", 10 - i)
    assert pod_pull.lease()["id"] == "t0"
    assert pod_pull.lease()["id"] == "t1"
    assert pod_pull.lease() is None


def test_lease_and_finish_rows(fake):
    _job(fake, "j1", "tts", "paid", 3)
    row = pod_pull.lease()
    stored = json.loads(fake.get(f"{P}:aimx:job:j1"))
    assert stored["status"] == "running" and stored["position"] == 0 and stored["startedAt"] > 0
    assert fake.ttl[f"{P}:aimx:job:j1"] == 7 * 24 * 3600
    assert "j1" in fake.smembers(f"{P}:aimx:running")
    out = pod_pull.finish(row["id"], status="succeeded", gpu_seconds=2.5,
                          output={"artifact_url": "aimx-artifact://a1", "artifact_id": "a1", "model": "m", "license": "MIT",
                                  "text": "xin chào", "ignored": "x"})
    stored = json.loads(fake.get(f"{P}:aimx:job:j1"))
    assert stored["status"] == "succeeded" and stored["gpu_seconds"] == 2.5
    assert stored["artifact_url"] == "aimx-artifact://a1" and stored["artifact_id"] == "a1" and stored["license"] == "MIT"
    assert "payload" not in stored and "ignored" not in stored
    assert stored["text"] == "xin chào"
    assert "j1" not in fake.smembers(f"{P}:aimx:running")
    assert out["finishedAt"] >= stored["startedAt"]


def test_big_results_stay_out_of_redis(fake, monkeypatch):
    monkeypatch.setenv("REDIS_MAX_VALUE_BYTES", "2000")
    _job(fake, "e1", "tts", "free", 1)
    pod_pull.lease()
    pod_pull.finish("e1", status="succeeded", output={"embeddings": [[0.1] * 1000]})
    stored = json.loads(fake.get(f"{P}:aimx:job:e1"))
    assert "embeddings" not in stored and stored["result_truncated"] is True


def test_reap_uses_api_timeouts(fake, monkeypatch):
    assert pod_pull._timeout("tts") == 120
    assert pod_pull._timeout("image") == 180
    assert pod_pull._timeout("fast") == 600
    assert pod_pull._timeout("train") == 3600
    monkeypatch.setenv("AIMX_TIMEOUT_VIDEO", "1")
    assert pod_pull._timeout("film") == 30  # same floor as aimx_models.timeout_sec
    _job(fake, "old", "tts", "free", 1)
    pod_pull.lease()
    row = json.loads(fake.get(f"{P}:aimx:job:old"))
    row["startedAt"] = 1.0
    fake.set(f"{P}:aimx:job:old", json.dumps(row))
    assert pod_pull.reap() == 1
    assert json.loads(fake.get(f"{P}:aimx:job:old"))["error"] == "timeout"


def test_rediss_kwargs_and_masking(monkeypatch):
    monkeypatch.setenv("REDIS_TLS_CA_FILE", "/etc/ssl/aimx-ca.pem")
    kw = pod_pull.connect_kwargs("rediss://aimx-worker:pw@redis.aimarkets.vn:6380/0")
    assert kw["ssl_cert_reqs"] == "required" and kw["ssl_ca_certs"] == "/etc/ssl/aimx-ca.pem"
    assert kw["decode_responses"] is True
    assert "ssl_cert_reqs" not in pod_pull.connect_kwargs("redis://h:6379/0")
    assert pod_pull.safe_url("rediss://aimx-worker:secret@h:6380/0") == "rediss://***@h:6380/0"
    import redis

    client = redis.Redis.from_url("rediss://u:p@127.0.0.1:1/0", **kw)
    assert client.connection_pool.connection_class.__name__ == "SSLConnection"


def test_pod_jobs_are_recorded_in_postgres(fake):
    conn = FakeConn()
    db.bind(lambda: conn)
    try:
        _job(fake, "pg1", "tts", "paid", 2)
        assert pod_pull.step() == 1
    finally:
        db.bind(None)
    sqls = [sql for sql, _ in conn.calls]
    assert any(s.startswith("CREATE TABLE IF NOT EXISTS aimx_jobs") for s in sqls)
    inserts = [(s, p) for s, p in conn.calls if s.startswith("INSERT INTO aimx_jobs")]
    assert len(inserts) == 2
    started, finished = inserts[0][1], inserts[1][1]
    assert started[0] == "pg1" and started[1] == "tts" and started[4] == "paid" and started[7] == "pod"
    assert finished[9] == "succeeded" and finished[10] == 0.01 and finished[11] == "sandbox://aimx"


def test_db_outage_does_not_block_queue(fake):
    db.bind(lambda: FakeConn(fail=True))
    try:
        _job(fake, "pg2", "image", "free", 1)
        assert pod_pull.step() == 1
        assert json.loads(fake.get(f"{P}:aimx:job:pg2"))["status"] == "succeeded"
    finally:
        db.bind(None)


@pytest.mark.skipif(not os.environ.get("AIMX_TEST_REDIS_URL"), reason="AIMX_TEST_REDIS_URL not set")
def test_live_redis_round_trip(monkeypatch):
    import uuid

    monkeypatch.setenv("REDIS_URL", os.environ["AIMX_TEST_REDIS_URL"])
    monkeypatch.setenv("REDIS_KEY_PREFIX", f"aimxtest{uuid.uuid4().hex[:6]}")
    pod_pull.bind(None)
    client = pod_pull._redis()
    prefix = pod_pull.prefix()
    jid = uuid.uuid4().hex
    client.setex(f"{prefix}:aimx:job:{jid}", 60, json.dumps({"id": jid, "kind": "tts", "status": "queued",
                                                             "tier": "paid", "score": 5.0, "payload": {"text": "x"}}))
    client.zadd(f"{prefix}:aimx:q:paid", {jid: 5.0})
    try:
        assert pod_pull.step() == 1
        row = json.loads(client.get(f"{prefix}:aimx:job:{jid}"))
        assert row["status"] == "succeeded" and "payload" not in row
        assert client.zcard(f"{prefix}:aimx:q:paid") == 0
        assert not client.smembers(f"{prefix}:aimx:running")
        assert 0 < client.ttl(f"{prefix}:aimx:job:{jid}") <= 7 * 24 * 3600
    finally:
        for key in client.scan_iter(f"{prefix}:*"):
            client.delete(key)
        pod_pull.bind(None)
