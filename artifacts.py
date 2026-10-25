"""Job outputs: scratch on local disk, kept only in Postgres and Redis.

The RunPod network volume (AIMX_VOLUME, default /runpod-volume) holds model
weights and caches ONLY. No job input or output may be written there. Every
file a job produces goes to AIMX_SCRATCH_DIR (default /tmp/aimx-jobs, the
container's ephemeral disk), is persisted by `collect()`, then deleted.

Persistence:
- Postgres table `aimx_job_artifacts` (bytea), one row per artifact, keyed by
  artifact id with the job id. Capped by AIMX_DB_ARTIFACT_MAX_BYTES (100 MB).
- Redis `{prefix}:aimx:artifact:{id}` (base64 JSON, SETEX 7 days) when a Redis
  URL is configured and the blob fits REDIS_MAX_VALUE_BYTES.
- The job result carries `artifact_id`, `artifact_mime`, `artifact_bytes`,
  `artifact_sha256`, and `artifact_url` = `aimx-artifact://<id>`. The
  serverless response also inlines `artifact_base64` when the blob is at most
  AIMX_INLINE_ARTIFACT_MAX_BYTES (8 MB); that field never goes into a DB row.

If neither store accepts the blob, the job fails with an error instead of
leaving the file anywhere; the scratch file is deleted either way.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import tempfile
import threading
import time
import uuid
from typing import Any, Optional

import catalog

SCHEME = "aimx-artifact://"
ARTIFACT_TTL = 7 * 24 * 3600
URL_KEYS = ("artifact_url", "audio_url", "image_url", "video_url")
DEFAULT_SCRATCH = "/tmp/aimx-jobs"

_local = threading.local()


class OutputPathError(RuntimeError):
    """A job tried to write an input or output on the model volume."""


def _volume_roots() -> list[str]:
    roots = {os.path.realpath(catalog.VOLUME), os.path.realpath("/runpod-volume")}
    extra = (os.environ.get("AIMX_VOLUME") or "").strip()
    if extra:
        roots.add(os.path.realpath(extra))
    return sorted(r for r in roots if r and r != os.sep)


def under_volume(path: str) -> bool:
    real = os.path.realpath(str(path))
    return any(real == root or real.startswith(root + os.sep) for root in _volume_roots())


def assert_not_volume(path: str) -> str:
    """Refuse any job input/output path on the network volume. Returns the real path."""
    if under_volume(path):
        raise OutputPathError(f"job files must not be written to the model volume: {path}")
    return os.path.realpath(str(path))


def scratch_dir() -> str:
    path = (os.environ.get("AIMX_SCRATCH_DIR") or DEFAULT_SCRATCH).strip() or DEFAULT_SCRATCH
    assert_not_volume(path)
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path


def guard_tempdir() -> None:
    """Keep library temp files (tempfile, TMPDIR) off the volume."""
    for name in ("TMPDIR", "TEMP", "TMP"):
        value = os.environ.get(name)
        if value and under_volume(value):
            os.environ[name] = "/tmp"
    tempfile.tempdir = None
    if under_volume(tempfile.gettempdir()):
        tempfile.tempdir = "/tmp"


def _track(path: str) -> str:
    files = getattr(_local, "files", None)
    if files is None:
        files = _local.files = []
    files.append(path)
    return path


def new_path(name: str, ext: str) -> str:
    """A fresh output path on local scratch, tracked for cleanup."""
    base = scratch_dir()
    path = os.path.join(base, f"{name}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.{ext.lstrip('.')}")
    assert_not_volume(path)
    return _track(path)


def temp_file(suffix: str = "") -> tuple[int, str]:
    """mkstemp on local scratch for job inputs (downloads, decoded base64)."""
    fd, path = tempfile.mkstemp(suffix=suffix, dir=scratch_dir())
    assert_not_volume(path)
    _track(path)
    return fd, path


def _remove(path: str) -> None:
    try:
        if path and os.path.isfile(path) and not under_volume(path):
            os.remove(path)
    except OSError:
        pass


def cleanup() -> None:
    """Delete every scratch file this thread created for the current job."""
    files = getattr(_local, "files", None) or []
    _local.files = []
    for path in files:
        _remove(path)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except ValueError:
        return default


def _redis_put(artifact_id: str, meta: dict, blob: bytes) -> bool:
    import pod_pull  # lazy: pod_pull imports handler

    if pod_pull._client is None and not (os.environ.get("AIMX_REDIS_URL") or os.environ.get("REDIS_URL")):
        return False
    raw = json.dumps({**meta, "data_base64": base64.b64encode(blob).decode("ascii")}, separators=(",", ":"))
    if len(raw) > pod_pull._max_value_bytes():
        return False
    try:
        pod_pull._redis().set(pod_pull._ns(f"aimx:artifact:{artifact_id}"), raw, ex=ARTIFACT_TTL)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[aimx.artifacts] redis write failed: {type(exc).__name__}", flush=True)
        return False


def _local_file(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value or "://" in value:
        return None
    return value if os.path.isabs(value) else None


def persist(path: str, *, job_id: str = "", kind: str = "") -> dict:
    """Store one scratch file in Postgres + Redis, delete it, return result fields."""
    import db

    if under_volume(path):
        raise OutputPathError(f"job output on the model volume is not allowed: {path}")
    try:
        with open(path, "rb") as fh:
            blob = fh.read()
    finally:
        _remove(path)
    artifact_id = uuid.uuid4().hex
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    meta = {
        "artifact_id": artifact_id,
        "job_id": job_id or None,
        "kind": kind or None,
        "mime": mime,
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "filename": os.path.basename(path),
    }
    in_db = False
    if len(blob) <= _int_env("AIMX_DB_ARTIFACT_MAX_BYTES", 100 * 1024 * 1024):
        in_db = db.artifact_put(meta, blob)
    in_redis = _redis_put(artifact_id, meta, blob)
    out: dict[str, Any] = {
        "artifact_id": artifact_id,
        "artifact_mime": mime,
        "artifact_bytes": len(blob),
        "artifact_sha256": meta["sha256"],
        "artifact_store": [name for name, ok in (("postgres", in_db), ("redis", in_redis)) if ok],
        "artifact_url": SCHEME + artifact_id,
    }
    if len(blob) <= _int_env("AIMX_INLINE_ARTIFACT_MAX_BYTES", 8 * 1024 * 1024):
        out["artifact_base64"] = base64.b64encode(blob).decode("ascii")
    if not (in_db or in_redis or "artifact_base64" in out):
        raise RuntimeError("job output could not be stored in Postgres or Redis")
    return out


def collect(out: dict, *, job_id: str = "", kind: str = "") -> dict:
    """Replace local output paths in a result with persisted artifact fields."""
    if not isinstance(out, dict):
        return out
    seen: dict[str, dict] = {}
    for key in URL_KEYS:
        path = _local_file(out.get(key))
        if not path:
            continue
        if under_volume(path):
            raise OutputPathError(f"job output on the model volume is not allowed: {path}")
        if path not in seen:
            if not os.path.isfile(path):
                continue
            seen[path] = persist(path, job_id=job_id, kind=kind)
        stored = seen[path]
        out[key] = stored["artifact_url"]
        for field, value in stored.items():
            if field != "artifact_url":
                out.setdefault(field, value)
    return out
