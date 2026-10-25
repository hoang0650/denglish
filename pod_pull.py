"""Pull AIMX jobs from the Redis queue shared with ai-marketplace-ai.

Keys (REDIS_KEY_PREFIX defaults to aimarkets), identical to
providers/aimx_queue.py + cache/client.py in ai-marketplace-ai:

- zset  {prefix}:aimx:q:paid       member = job id, score = priority*1e6 - seq
- zset  {prefix}:aimx:q:free
- json  {prefix}:aimx:job:{id}     SETEX 7 days, compact JSON
- set   {prefix}:aimx:running

Lease: paid lane first; inside a lane the highest score wins (ZREVRANGE 0 0)
and the head job is never skipped. A video or train job uses the GPU alone;
image and TTS share AIMX_POD_LIGHT_SLOTS (default 3). finish() writes status,
finishedAt, gpu_seconds, error and the artifact keys, drops the payload, and
removes the id from the running set and both lanes. reap() fails jobs that ran
longer than the kind's timeout (AIMX_TIMEOUT_*; same defaults as the API).

Connection: REDIS_URL (redis:// or rediss://). For rediss, REDIS_TLS_CA_FILE
points at the CA bundle; certificates are verified unless
REDIS_TLS_INSECURE=1. Run this puller only when AIMX_RUNTIME=pod, and do not
also run `python -m aimx_pod_worker` against the same Redis.

Each leased and finished job is also written to Postgres (db.py); a DB error
never blocks the queue.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

import db
import handler

JOB_TTL = 7 * 24 * 3600
_client: Any = None
_ARTIFACT_KEYS = ("audio_url", "video_url", "image_url", "artifact_url", "model", "license", "stage")
# Additive keys so text/vector results reach the API through the job row.
_RESULT_KEYS = ("text", "embeddings", "dim", "scores", "ranking", "usage", "chunks", "language", "duration_sec")


def prefix() -> str:
    return (os.environ.get("REDIS_KEY_PREFIX") or "aimarkets").strip() or "aimarkets"


def _ns(key: str) -> str:
    key = str(key or "").lstrip(":")
    head = prefix()
    if key.startswith(f"{head}:"):
        return key
    return f"{head}:{key}"


def bind(client: Any) -> None:
    """Tests inject a Redis-shaped client. Production uses REDIS_URL."""
    global _client
    _client = client


def redis_url() -> str:
    return (os.environ.get("REDIS_URL") or "redis://127.0.0.1:6379/0").strip()


def safe_url(url: str) -> str:
    return re.sub(r"(rediss?://)[^@/]*@", r"\1***@", url)


def connect_kwargs(url: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "decode_responses": True,
        "socket_connect_timeout": float(os.environ.get("REDIS_CONNECT_TIMEOUT", "5") or "5"),
        "socket_timeout": float(os.environ.get("REDIS_SOCKET_TIMEOUT", "10") or "10"),
        "retry_on_timeout": True,
        "health_check_interval": 30,
    }
    if url.startswith("rediss://"):
        ca = (os.environ.get("REDIS_TLS_CA_FILE") or "").strip()
        if ca:
            kwargs["ssl_ca_certs"] = ca
        if os.environ.get("REDIS_TLS_INSECURE", "").strip().lower() in {"1", "true", "yes"}:
            kwargs["ssl_cert_reqs"] = None
            kwargs["ssl_check_hostname"] = False
        else:
            kwargs["ssl_cert_reqs"] = "required"
    return kwargs


def _redis() -> Any:
    global _client
    if _client is not None:
        return _client
    import redis

    url = redis_url()
    _client = redis.Redis.from_url(url, **connect_kwargs(url))
    return _client


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _lane_key(lane: str) -> str:
    return _ns(f"aimx:q:{lane}")


def _job_key(job_id: str) -> str:
    return _ns(f"aimx:job:{job_id}")


def _running_key() -> str:
    return _ns("aimx:running")


def _loads(raw: Any) -> Optional[dict]:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        body = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return body if isinstance(body, dict) else None


def job_row(job_id: str) -> Optional[dict]:
    return _loads(_redis().get(_job_key(job_id)))


def _max_value_bytes() -> int:
    try:
        return int(os.environ.get("REDIS_MAX_VALUE_BYTES", "1500000") or "1500000")
    except ValueError:
        return 1500000


def save(row: dict) -> None:
    payload = json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str)
    _redis().set(_job_key(str(row.get("id") or "")), payload, ex=JOB_TTL)


def _light_slots() -> int:
    try:
        return max(1, int(os.environ.get("AIMX_POD_LIGHT_SLOTS", "3") or "3"))
    except ValueError:
        return 3


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except ValueError:
        return default


def _timeout(kind: str) -> int:
    """Same table as aimx_models.timeout_sec in ai-marketplace-ai."""
    if kind == "tts":
        return max(5, _int_env("AIMX_TIMEOUT_TTS", 120))
    if kind == "image":
        return max(5, _int_env("AIMX_TIMEOUT_IMAGE", 180))
    if kind == "train":
        return max(30, _int_env("AIMX_TIMEOUT_TRAIN", 3600))
    return max(30, _int_env("AIMX_TIMEOUT_VIDEO", 600))


def running_jobs() -> list[dict]:
    client = _redis()
    members = client.smembers(_running_key()) or []
    rows = []
    for item in members:
        jid = _text(item)
        row = job_row(jid)
        if row and row.get("status") == "running":
            rows.append(row)
        else:
            client.srem(_running_key(), jid)
    return rows


def can_admit(kind: str, running: Optional[list[dict]] = None) -> bool:
    running = running_jobs() if running is None else running
    videos = [job for job in running if job.get("kind") in handler.VIDEO_KINDS or job.get("kind") in handler.TRAIN_KINDS]
    lights = [job for job in running if job.get("kind") in handler.LIGHT_KINDS]
    if kind in handler.VIDEO_KINDS or kind in handler.TRAIN_KINDS:
        return not videos and not lights
    return not videos and len(lights) < _light_slots()


def _peek(lane: str) -> Optional[str]:
    rows = _redis().zrevrange(_lane_key(lane), 0, 0) or []
    if not rows:
        return None
    return _text(rows[0])


def lease() -> Optional[dict]:
    client = _redis()
    running = running_jobs()
    for lane in ("paid", "free"):
        jid = _peek(lane)
        if not jid:
            continue
        row = job_row(jid)
        if not row:
            client.zrem(_lane_key(lane), jid)
            continue
        if not can_admit(str(row.get("kind") or ""), running):
            return None
        client.zrem(_lane_key(lane), jid)
        row["status"] = "running"
        row["startedAt"] = time.time()
        row["position"] = 0
        save(row)
        client.sadd(_running_key(), jid)
        db.job_started({k: v for k, v in row.items() if k != "payload"}, runtime="pod")
        return row
    return None


def finish(
    job_id: str,
    *,
    status: str,
    error: str = "",
    gpu_seconds: Any = None,
    output: Optional[dict] = None,
) -> dict:
    client = _redis()
    row = job_row(job_id) or {"id": job_id, "createdAt": time.time()}
    now = time.time()
    started = float(row.get("startedAt") or now)
    measured = round(max(0.0, now - started), 3) if row.get("startedAt") else 0.0
    row["status"] = status
    row["finishedAt"] = now
    row["gpu_seconds"] = measured if gpu_seconds is None else round(float(gpu_seconds), 3)
    if error:
        row["error"] = error
    if isinstance(output, dict):
        for key in _ARTIFACT_KEYS:
            if output.get(key):
                row[key] = output[key]
    row.pop("payload", None)
    if isinstance(output, dict):
        extra = {k: output[k] for k in _RESULT_KEYS if output.get(k) not in (None, "", [], {})}
        if extra:
            trial = dict(row, **extra)
            raw = json.dumps(trial, ensure_ascii=False, separators=(",", ":"), default=str)
            if len(raw.encode("utf-8")) <= _max_value_bytes():
                row = trial
            else:
                row["result_truncated"] = True
    save(row)
    client.srem(_running_key(), job_id)
    for lane in ("paid", "free"):
        client.zrem(_lane_key(lane), job_id)
    db.job_finished(row, runtime="pod", output=output)
    return row


def reap() -> int:
    now = time.time()
    n = 0
    for row in running_jobs():
        started = float(row.get("startedAt") or row.get("createdAt") or now)
        if now - started > _timeout(str(row.get("kind") or "tts")):
            finish(str(row["id"]), status="failed", error="timeout", gpu_seconds=round(now - started, 3))
            n += 1
    return n


def step() -> int:
    reap()
    job = lease()
    if not job:
        return 0
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    try:
        result = handler.execute(str(job.get("kind") or ""), payload)
    except Exception as exc:  # noqa: BLE001
        finish(str(job["id"]), status="failed", error=str(exc)[:300])
        return 1
    status = str(result.get("status") or "succeeded")
    if status == "handed_off":
        status = "succeeded"
    if status not in {"succeeded", "failed", "cancelled"}:
        status = "succeeded"
    finish(
        str(job["id"]),
        status=status,
        error=str(result.get("error") or ""),
        gpu_seconds=result.get("gpu_seconds"),
        output=result if isinstance(result, dict) else None,
    )
    return 1


def ping() -> bool:
    try:
        return bool(_redis().ping())
    except Exception as exc:  # noqa: BLE001
        print(f"[aimx.pod] redis {safe_url(redis_url())} not reachable: {type(exc).__name__}", flush=True)
        return False


def main() -> None:
    idle = float(os.environ.get("AIMX_POD_POLL_SEC", "0.5") or "0.5")
    if ping():
        print(f"[aimx.pod] redis {safe_url(redis_url())} prefix={prefix()} db={'on' if db.enabled() else 'off'}", flush=True)
    if db.enabled():
        db.migrate()
    while True:
        try:
            worked = step()
        except Exception as exc:  # noqa: BLE001
            print(f"[aimx.pod] step error: {type(exc).__name__}", flush=True)
            worked = 0
        if not worked:
            time.sleep(max(0.05, idle))
