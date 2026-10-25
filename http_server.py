"""Stdlib HTTP front for the AIMX worker.

GET  /health
POST /v1/tts  /v1/image  /v1/video/fast  /v1/video/talking  /v1/film  /v1/train
     /v1/bg_remove (/v1/image/remove-bg)  /v1/stt  /v1/ocr  /v1/embed  /v1/rerank
     /v1/music  /v1/llm  /v1/prefetch

AIMX_RUNTIME=http. The marketplace puller can POST here with
AIMX_POD_EXECUTOR=http and RUNPOD_POD_URL. When AIMX_HTTP_TOKEN is set every
POST must send `Authorization: Bearer <AIMX_HTTP_TOKEN>` (set the same value as
RUNPOD_POD_TOKEN on the puller). An `X-AIMX-Job-Id` header (or `job_id` in the
body) becomes the Postgres row id. Do not run this and AIMX_RUNTIME=pod together
against the same GPU.
"""

from __future__ import annotations

import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import handler

MAX_BODY = int(os.environ.get("AIMX_HTTP_MAX_BODY", str(150 * 1024 * 1024)) or str(150 * 1024 * 1024))


def _authorized(header: str) -> bool:
    token = (os.environ.get("AIMX_HTTP_TOKEN") or "").strip()
    if not token:
        return True
    given = (header or "").strip()
    if given.lower().startswith("bearer "):
        given = given[7:].strip()
    return hmac.compare_digest(given.encode(), token.encode())


class AimxHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send(200, handler.health_body())
            return
        self._send(404, {"status": "failed", "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        kind = handler._KIND_FROM_PATH.get(path)
        if not kind:
            self._send(404, {"status": "failed", "error": "not found"})
            return
        if not _authorized(self.headers.get("Authorization") or ""):
            self._send(401, {"status": "failed", "error": "unauthorized"})
            return
        length = int(self.headers.get("Content-Length") or "0")
        if length > MAX_BODY:
            self._send(413, {"status": "failed", "error": "body too large"})
            return
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except ValueError:
            self._send(400, {"status": "failed", "error": "invalid json"})
            return
        if not isinstance(payload, dict):
            self._send(400, {"status": "failed", "error": "json object required"})
            return
        meta = {"id": self.headers.get("X-AIMX-Job-Id") or payload.get("job_id"), "backend": "http"}
        result = handler.run_job(kind, payload, meta)
        code = 200 if result.get("status") != "failed" else 400
        self._send(code, result)


def main() -> None:
    port = int(os.environ.get("PORT", "8000") or "8000")
    host = os.environ.get("AIMX_HTTP_HOST", "0.0.0.0")
    server = ThreadingHTTPServer((host, port), AimxHandler)
    server.serve_forever()
