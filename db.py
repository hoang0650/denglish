"""Postgres persistence for AIMX jobs (psycopg 3).

The worker writes one row per job into `aimx_jobs` (kind, model, tier, status,
gpu_seconds, artifact url, error, small result, timestamps), one row per
output file into `aimx_job_artifacts` (bytea; see artifacts.py), and one row
per model into `aimx_volume_status` after a prefetch. Job outputs live only in
Postgres and Redis, never on the model volume.

Connection: AIMX_DATABASE_URL, else AIMX_PG_DSN, else DATABASE_URL. TLS is libpq's: put
`?sslmode=verify-full&sslrootcert=/path/ca.crt` (or `sslmode=require`) in the
URL, or set AIMX_DB_SSLMODE / PGSSLROOTCERT. Nothing here ever raises into the
job: a failed write is logged without the URL or password and the job goes on.
After a failure the module backs off for AIMX_DB_RETRY_SEC (default 30).
"""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

_lock = threading.Lock()
_conn: Any = None
_factory: Optional[Callable[[], Any]] = None
_down_until = 0.0
_migrated = False

_MIGRATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "migrations", "001_aimx_jobs.sql")

_UPSERT_STARTED = """
INSERT INTO aimx_jobs (id, kind, model, license, tier, priority, backend, runtime, worker_id,
                       status, created_at, started_at, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (id) DO UPDATE SET
    kind = EXCLUDED.kind,
    model = COALESCE(EXCLUDED.model, aimx_jobs.model),
    license = COALESCE(EXCLUDED.license, aimx_jobs.license),
    tier = COALESCE(EXCLUDED.tier, aimx_jobs.tier),
    priority = COALESCE(EXCLUDED.priority, aimx_jobs.priority),
    backend = COALESCE(EXCLUDED.backend, aimx_jobs.backend),
    runtime = EXCLUDED.runtime,
    worker_id = EXCLUDED.worker_id,
    status = EXCLUDED.status,
    created_at = COALESCE(aimx_jobs.created_at, EXCLUDED.created_at),
    started_at = COALESCE(EXCLUDED.started_at, aimx_jobs.started_at),
    updated_at = now()
"""

_UPSERT_FINISHED = """
INSERT INTO aimx_jobs (id, kind, model, license, tier, priority, backend, runtime, worker_id,
                       status, gpu_seconds, artifact_url, error, result,
                       created_at, started_at, finished_at, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, now())
ON CONFLICT (id) DO UPDATE SET
    kind = COALESCE(NULLIF(EXCLUDED.kind, ''), aimx_jobs.kind),
    model = COALESCE(EXCLUDED.model, aimx_jobs.model),
    license = COALESCE(EXCLUDED.license, aimx_jobs.license),
    tier = COALESCE(EXCLUDED.tier, aimx_jobs.tier),
    priority = COALESCE(EXCLUDED.priority, aimx_jobs.priority),
    backend = COALESCE(EXCLUDED.backend, aimx_jobs.backend),
    runtime = EXCLUDED.runtime,
    worker_id = EXCLUDED.worker_id,
    status = EXCLUDED.status,
    gpu_seconds = EXCLUDED.gpu_seconds,
    artifact_url = COALESCE(EXCLUDED.artifact_url, aimx_jobs.artifact_url),
    error = EXCLUDED.error,
    result = COALESCE(EXCLUDED.result, aimx_jobs.result),
    created_at = COALESCE(aimx_jobs.created_at, EXCLUDED.created_at),
    started_at = COALESCE(aimx_jobs.started_at, EXCLUDED.started_at),
    finished_at = EXCLUDED.finished_at,
    updated_at = now()
"""

_INSERT_ARTIFACT = """
INSERT INTO aimx_job_artifacts (id, job_id, kind, mime, bytes, sha256, filename, data, created_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (id) DO NOTHING
"""

_UPSERT_VOLUME = """
INSERT INTO aimx_volume_status (volume, model_key, repo, license, path, status, approx_gb, error, worker_id, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (volume, model_key) DO UPDATE SET
    repo = EXCLUDED.repo, license = EXCLUDED.license, path = EXCLUDED.path,
    status = EXCLUDED.status, approx_gb = EXCLUDED.approx_gb, error = EXCLUDED.error,
    worker_id = EXCLUDED.worker_id, updated_at = now()
"""


def database_url() -> str:
    return (
        os.environ.get("AIMX_DATABASE_URL")
        or os.environ.get("AIMX_PG_DSN")
        or os.environ.get("DATABASE_URL")
        or ""
    ).strip()


def enabled() -> bool:
    if _factory is not None:
        return True
    if os.environ.get("AIMX_DB_DISABLE", "").strip().lower() in {"1", "true", "yes"}:
        return False
    return bool(database_url())


def worker_id() -> str:
    return (
        os.environ.get("AIMX_WORKER_ID")
        or os.environ.get("RUNPOD_POD_ID")
        or os.environ.get("RUNPOD_ENDPOINT_ID")
        or socket.gethostname()
    )[:120]


def bind(factory: Optional[Callable[[], Any]]) -> None:
    """Tests inject a connection factory. None restores DATABASE_URL."""
    global _factory, _conn, _down_until, _migrated
    with _lock:
        _factory = factory
        _conn = None
        _down_until = 0.0
        _migrated = False


def _retry_sec() -> float:
    try:
        return max(1.0, float(os.environ.get("AIMX_DB_RETRY_SEC", "30") or "30"))
    except ValueError:
        return 30.0


def _safe(exc: BaseException) -> str:
    """Error text without the connection string or any password."""
    text = f"{type(exc).__name__}: {exc}"
    url = database_url()
    if url:
        text = text.replace(url, "<DATABASE_URL>")
    text = re.sub(r"(postgres(?:ql)?://)[^@\s]*@", r"\1***@", text)
    text = re.sub(r"(?i)(password\s*=\s*)\S+", r"\1***", text)
    return text.replace("\n", " ")[:300]


def _log(msg: str) -> None:
    print(f"[aimx.db] {msg}", flush=True)


def _connect() -> Any:
    if _factory is not None:
        return _factory()
    import psycopg

    kwargs: dict[str, Any] = {
        "autocommit": True,
        "connect_timeout": int(os.environ.get("AIMX_DB_CONNECT_TIMEOUT", "5") or "5"),
        "application_name": "aimx-worker",
        "options": f"-c statement_timeout={int(os.environ.get('AIMX_DB_STATEMENT_TIMEOUT_MS', '5000') or '5000')}",
    }
    url = database_url()
    sslmode = (os.environ.get("AIMX_DB_SSLMODE") or "").strip()
    if sslmode and "sslmode=" not in url:
        kwargs["sslmode"] = sslmode
    return psycopg.connect(url, **kwargs)


def _migrate(conn: Any) -> None:
    global _migrated
    if _migrated or os.environ.get("AIMX_DB_MIGRATE", "1").strip().lower() in {"0", "false", "no"}:
        _migrated = True
        return
    with open(_MIGRATION, encoding="utf-8") as fh:
        sql = fh.read()
    statements = [s.strip() for s in re.sub(r"--[^\n]*", "", sql).split(";") if s.strip()]
    with conn.cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)
    _migrated = True


def _get() -> Any:
    global _conn, _down_until
    if not enabled():
        return None
    if time.time() < _down_until:
        return None
    if _conn is not None and not getattr(_conn, "closed", False):
        return _conn
    try:
        _conn = _connect()
        _migrate(_conn)
        return _conn
    except Exception as exc:  # noqa: BLE001
        _log(f"connect failed, retry in {_retry_sec():.0f}s: {_safe(exc)}")
        _drop()
        _down_until = time.time() + _retry_sec()
        return None


def _drop() -> None:
    global _conn
    try:
        if _conn is not None:
            _conn.close()
    except Exception:  # noqa: BLE001
        pass
    _conn = None


def _execute(sql: str, params: tuple, timeout_ms: Optional[int] = None) -> bool:
    global _down_until
    with _lock:
        conn = _get()
        if conn is None:
            return False
        try:
            with conn.cursor() as cur:
                if timeout_ms:
                    cur.execute(f"SET statement_timeout = {int(timeout_ms)}")
                cur.execute(sql, params)
                if timeout_ms:
                    cur.execute("RESET statement_timeout")
            return True
        except Exception as exc:  # noqa: BLE001
            _log(f"write failed: {_safe(exc)}")
            _drop()
            _down_until = time.time() + _retry_sec()
            return False


def migrate() -> bool:
    """Create the tables now. Returns False when the DB is off or unreachable."""
    global _migrated
    with _lock:
        _migrated = False
        return _get() is not None


def _ts(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _int(value: Any) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> Optional[float]:
    try:
        return None if value is None else round(float(value), 3)
    except (TypeError, ValueError):
        return None


def _str(value: Any, limit: int = 2000) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)[:limit]


_RESULT_KEYS = ("text", "language", "embeddings", "dim", "scores", "ranking", "usage", "stage",
                "audio_url", "video_url", "image_url", "artifact_url", "duration_sec", "chunks",
                "artifact_id", "artifact_mime", "artifact_bytes", "artifact_sha256", "artifact_store")


def _result_json(output: Optional[dict]) -> Optional[str]:
    if not isinstance(output, dict):
        return None
    body = {k: output[k] for k in _RESULT_KEYS if k in output and output[k] not in (None, "")}
    if not body:
        return None
    raw = json.dumps(body, ensure_ascii=False, default=str)
    limit = int(os.environ.get("AIMX_DB_RESULT_MAX_BYTES", "262144") or "262144")
    if len(raw.encode("utf-8")) > limit:
        slim = {k: v for k, v in body.items() if k not in {"embeddings", "chunks", "scores"}}
        slim["truncated"] = True
        raw = json.dumps(slim, ensure_ascii=False, default=str)
    return raw


def job_started(row: dict, *, runtime: str, model: str = "", license: str = "") -> bool:
    """Record a leased or started job. Never raises."""
    try:
        return _execute(_UPSERT_STARTED, (
            str(row.get("id") or ""),
            str(row.get("kind") or ""),
            _str(model or row.get("model"), 300),
            _str(license or row.get("license"), 100),
            _str(row.get("tier"), 40),
            _int(row.get("priority")),
            _str(row.get("backend"), 40),
            runtime,
            worker_id(),
            str(row.get("status") or "running"),
            _ts(row.get("createdAt")) or _ts(time.time()),
            _ts(row.get("startedAt")) or _ts(time.time()),
        ))
    except Exception as exc:  # noqa: BLE001
        _log(f"job_started skipped: {_safe(exc)}")
        return False


def job_finished(row: dict, *, runtime: str, output: Optional[dict] = None) -> bool:
    """Record the final state of a job. Never raises."""
    try:
        out = output if isinstance(output, dict) else {}
        artifact = (out.get("artifact_url") or out.get("video_url") or out.get("image_url")
                    or out.get("audio_url") or row.get("artifact_url") or row.get("video_url")
                    or row.get("image_url") or row.get("audio_url"))
        return _execute(_UPSERT_FINISHED, (
            str(row.get("id") or ""),
            str(row.get("kind") or ""),
            _str(out.get("model") or row.get("model"), 300),
            _str(out.get("license") or row.get("license"), 100),
            _str(row.get("tier"), 40),
            _int(row.get("priority")),
            _str(row.get("backend"), 40),
            runtime,
            worker_id(),
            str(row.get("status") or out.get("status") or "succeeded"),
            _float(row.get("gpu_seconds") if row.get("gpu_seconds") is not None else out.get("gpu_seconds")),
            _str(artifact, 2000),
            _str(row.get("error") or out.get("error"), 2000),
            _result_json(out),
            _ts(row.get("createdAt")),
            _ts(row.get("startedAt")),
            _ts(row.get("finishedAt")) or _ts(time.time()),
        ))
    except Exception as exc:  # noqa: BLE001
        _log(f"job_finished skipped: {_safe(exc)}")
        return False


def check() -> dict:
    """Round trip for health checks: SELECT 1 and read aimx_volume_status. Never raises."""
    if not enabled():
        return {"ok": False, "error": "disabled"}
    with _lock:
        conn = _get()
        if conn is None:
            return {"ok": False, "error": "unreachable"}
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.execute("SELECT count(*) FROM aimx_volume_status")
                row = cur.fetchone() if hasattr(cur, "fetchone") else None
            return {"ok": True, "volume_status_rows": int(row[0]) if row else None}
        except Exception as exc:  # noqa: BLE001
            _drop()
            return {"ok": False, "error": _safe(exc)[:200]}


def artifact_put(meta: dict, blob: bytes) -> bool:
    """Store one job output file (bytea). Never raises; False when the DB is off or fails."""
    try:
        timeout = int(os.environ.get("AIMX_DB_ARTIFACT_TIMEOUT_MS", "120000") or "120000")
        return _execute(_INSERT_ARTIFACT, (
            str(meta.get("artifact_id") or ""),
            _str(meta.get("job_id"), 200),
            _str(meta.get("kind"), 40),
            _str(meta.get("mime"), 100),
            _int(meta.get("bytes")),
            _str(meta.get("sha256"), 64),
            _str(meta.get("filename"), 300),
            bytes(blob),
        ), timeout_ms=timeout)
    except Exception as exc:  # noqa: BLE001
        _log(f"artifact_put skipped: {_safe(exc)}")
        return False


def volume_status(volume: str, key: str, row: dict) -> bool:
    """Record one model's prefetch result. Never raises."""
    try:
        return _execute(_UPSERT_VOLUME, (
            volume,
            key,
            _str(row.get("repo"), 300),
            _str(row.get("license"), 100),
            _str(row.get("path"), 500),
            str(row.get("status") or "unknown"),
            _float(row.get("approx_gb")),
            _str(row.get("error"), 2000),
            worker_id(),
        ))
    except Exception as exc:  # noqa: BLE001
        _log(f"volume_status skipped: {_safe(exc)}")
        return False
