from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import http_server
import media


@pytest.fixture()
def server(monkeypatch):
    monkeypatch.setenv("AIMX_HTTP_TOKEN", "tok-123")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), http_server.AimxHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _post(url, body, token=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_health_open(server):
    with urllib.request.urlopen(server + "/health", timeout=5) as resp:
        assert json.loads(resp.read())["service"] == "aimx-runpod-worker"


def test_post_requires_token(server):
    assert _post(server + "/v1/stt", {})[0] == 401
    assert _post(server + "/v1/stt", {}, "wrong")[0] == 401
    code, body = _post(server + "/v1/stt", {}, "tok-123")
    assert code == 200 and body["model"] == "openai/whisper-large-v3-turbo"
    code, body = _post(server + "/v1/embed", {"texts": ["a"]}, "tok-123")
    assert code == 200 and body["model"] == "AITeamVN/Vietnamese_Embedding"


def test_media_blocks_internal_urls(tmp_path):
    for url in ("http://example.com/a.wav", "https://127.0.0.1/a.wav", "https://localhost/a.wav",
                "https://169.254.169.254/latest", "file:///etc/passwd"):
        with pytest.raises(media.MediaError):
            media.fetch(url)
    outside = tmp_path / "x.wav"
    outside.write_text("x")
    with pytest.raises(media.MediaError):
        media.local(str(outside), str(tmp_path / "vol"))


def test_media_base64(tmp_path):
    path = media.resolve({"audio_base64": "data:audio/wav;base64,aGVsbG8="}, "audio", str(tmp_path), ".wav")
    assert open(path, "rb").read() == b"hello"


def test_media_rejects_credentials_and_internal_names():
    for url in ("https://user:pw@example.com/a.wav", "https://metadata.google.internal/x",
                "https://foo.internal/a.wav", "https://printer.local/a.wav"):
        with pytest.raises(media.MediaError):
            media.assert_safe_url(url)


def test_redirect_to_internal_is_blocked():
    import urllib.request

    handler_ = media._SafeRedirect()
    req = urllib.request.Request("https://example.com/a.wav")
    with pytest.raises(media.MediaError):
        handler_.redirect_request(req, None, 302, "Found", {}, "http://127.0.0.1/secret")
