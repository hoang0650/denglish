"""Input media for jobs: base64, an https URL, or a file already on the volume.

URLs are fetched with a size cap and only when every resolved address is
public (no loopback, private, link-local, or metadata addresses), so a job
cannot make the GPU box read internal services.
"""

from __future__ import annotations

import base64
import ipaddress
import os
import socket
import tempfile
import urllib.parse
import urllib.request
from typing import Optional

MAX_BYTES = int(os.environ.get("AIMX_MEDIA_MAX_BYTES", str(100 * 1024 * 1024)) or str(100 * 1024 * 1024))


class MediaError(ValueError):
    pass


# Same host rules as the old tts_vieneu.assert_safe_http_url.
_BLOCKED_HOSTS = {"localhost", "metadata", "metadata.google.internal", "metadata.internal"}
_BLOCKED_SUFFIXES = (".localhost", ".internal", ".local")


def _public_host(host: str) -> bool:
    host = str(host or "").strip(".").lower()
    if not host or host in _BLOCKED_HOSTS or host.endswith(_BLOCKED_SUFFIXES):
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        raw = str(info[4][0]).split("%")[0]
        if raw.startswith("::ffff:"):
            raw = raw[7:]
        addr = ipaddress.ip_address(raw)
        if (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast
                or addr.is_reserved or addr.is_unspecified):
            return False
    return bool(infos)


def _allow_http() -> bool:
    return os.environ.get("AIMX_MEDIA_ALLOW_HTTP", "").strip().lower() in {"1", "true", "yes"}


def assert_safe_url(url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    if parsed.scheme not in {"https"} and not (parsed.scheme == "http" and _allow_http()):
        raise MediaError("media URL must be https")
    if parsed.username or parsed.password:
        raise MediaError("media URL must not carry credentials")
    if not parsed.hostname or not _public_host(parsed.hostname):
        raise MediaError("media URL host is not public")
    return parsed


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Re-check every redirect hop so a public URL cannot bounce to an internal one."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        assert_safe_url(urllib.parse.urljoin(req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url: str, suffix: str = "") -> str:
    parsed = assert_safe_url(url)
    req = urllib.request.Request(parsed.geturl(), headers={"User-Agent": "aimx-worker/1"})
    fd, path = tempfile.mkstemp(suffix=suffix or os.path.splitext(parsed.path)[1][:8])
    total = 0
    opener = urllib.request.build_opener(_SafeRedirect)
    with opener.open(req, timeout=60) as resp, os.fdopen(fd, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_BYTES:
                raise MediaError("media is too large")
            out.write(chunk)
    return path


def from_base64(data: str, suffix: str = "") -> str:
    raw = str(data or "")
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    blob = base64.b64decode(raw, validate=False)
    if len(blob) > MAX_BYTES:
        raise MediaError("media is too large")
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as out:
        out.write(blob)
    return path


def local(path: str, volume: str) -> str:
    real = os.path.realpath(str(path))
    root = os.path.realpath(volume)
    if not real.startswith(root + os.sep) or not os.path.isfile(real):
        raise MediaError("local media must be a file on the volume")
    return real


def resolve(payload: dict, prefix: str, volume: str, suffix: str = "") -> Optional[str]:
    """Return a local file for `<prefix>_base64`, `<prefix>_url`, or `<prefix>_path`, else None."""
    b64 = payload.get(f"{prefix}_base64")
    if b64:
        return from_base64(str(b64), suffix)
    url = payload.get(f"{prefix}_url")
    if url:
        if str(url).startswith(("http://", "https://")):
            return fetch(str(url), suffix)
        return local(str(url), volume)
    path = payload.get(f"{prefix}_path")
    if path:
        return local(str(path), volume)
    return None
