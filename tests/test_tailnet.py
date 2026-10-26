"""Tailnet entrypoint helpers: protocol probes and env aliases (no network)."""

import importlib.util
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import db
import pod_pull

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tailnet_wait", ROOT / "scripts" / "tailnet_wait.py")
tailnet_wait = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tailnet_wait)


def _server(reply: bytes | None):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)

    def run():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            with conn:
                conn.recv(64)
                if reply:
                    conn.sendall(reply)

    threading.Thread(target=run, daemon=True).start()
    return srv


def test_probe_needs_a_protocol_reply():
    redis_srv = _server(b"-NOAUTH Authentication required.\r\n")
    pg_srv = _server(b"N")
    dead_srv = _server(None)  # accepts, then closes: what socat does when the tailnet dial fails
    try:
        r_port = redis_srv.getsockname()[1]
        p_port = pg_srv.getsockname()[1]
        d_port = dead_srv.getsockname()[1]
        assert tailnet_wait.probe("127.0.0.1", r_port, timeout=1, kind="redis")
        assert tailnet_wait.probe("127.0.0.1", p_port, timeout=1, kind="postgres")
        assert not tailnet_wait.probe("127.0.0.1", d_port, timeout=1, kind="redis")
        assert not tailnet_wait.probe("127.0.0.1", d_port, timeout=1, kind="postgres")
    finally:
        for s in (redis_srv, pg_srv, dead_srv):
            s.close()


def test_wait_times_out_on_dead_forward():
    dead_srv = _server(None)
    try:
        port = dead_srv.getsockname()[1]
        assert not tailnet_wait.wait("127.0.0.1", [f"{port}:6379"], timeout=0.3, interval=0.05)
    finally:
        dead_srv.close()


def test_wait_accepts_local_remote_specs():
    pg_srv = _server(b"S")
    try:
        port = pg_srv.getsockname()[1]
        assert tailnet_wait.parse(f"{port}:5432") == (port, "postgres")
        assert tailnet_wait.parse("6379") == (6379, "redis")
        assert tailnet_wait.wait("127.0.0.1", [f"{port}:5432"], timeout=2, interval=0.05)
    finally:
        pg_srv.close()


def test_probe_closed_port_is_not_ready():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    assert tailnet_wait.probe("127.0.0.1", port, timeout=0.5) is False


def test_env_aliases(monkeypatch):
    for k in ("AIMX_DATABASE_URL", "DATABASE_URL", "AIMX_PG_DSN", "AIMX_REDIS_URL", "REDIS_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AIMX_PG_DSN", "postgresql://u:p@127.0.0.1:5432/aimx_worker")
    monkeypatch.setenv("DATABASE_URL", "postgresql://other")
    assert db.database_url().endswith("@127.0.0.1:5432/aimx_worker")
    monkeypatch.setenv("AIMX_DATABASE_URL", "postgresql://explicit")
    assert db.database_url() == "postgresql://explicit"
    monkeypatch.setenv("REDIS_URL", "redis://fallback")
    monkeypatch.setenv("AIMX_REDIS_URL", "redis://aimx-worker:p@127.0.0.1:6379/0")
    assert pod_pull.redis_url().startswith("redis://aimx-worker:")


def test_entrypoint_without_authkey_runs_command():
    env = {k: v for k, v in os.environ.items() if k not in ("TS_AUTHKEY", "TAILSCALE_AUTHKEY")}
    out = subprocess.run(
        ["bash", str(ROOT / "scripts" / "entrypoint.sh"), sys.executable, "-c", "import os;print(os.getcwd())"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == str(ROOT)
    assert "tailnet" not in out.stderr


def test_entrypoint_hides_authkey_from_app_when_tailscale_missing(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "TS_AUTHKEY": "tskey-test-not-real", "HOME": str(tmp_path)}
    out = subprocess.run(
        ["bash", str(ROOT / "scripts" / "entrypoint.sh"), sys.executable, "-c",
         "import os;print(os.environ.get('TS_AUTHKEY','<unset>'))"],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "<unset>"
    assert "tskey-test-not-real" not in out.stderr


def test_handler_has_literal_serverless_start():
    src = (ROOT / "handler.py").read_text(encoding="utf-8")
    assert 'runpod.serverless.start({"handler": handler' in src


# ---- auth-key expiry / rotation handling ------------------------------------

def _fake_tailscale_bin(tmp_path, up_stderr: str, up_rc: int = 1):
    """Fake tailscaled/tailscale/socat on PATH; `tailscale up` prints up_stderr."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "tailscaled").write_text(
        "#!/usr/bin/env python3\n"
        "import socket, sys, time\n"
        "sock = [a.split('=', 1)[1] for a in sys.argv if a.startswith('--socket=')][0]\n"
        "s = socket.socket(socket.AF_UNIX); s.bind(sock); time.sleep(3)\n"
    )
    (bindir / "tailscale").write_text(
        "#!/usr/bin/env bash\n"
        f"echo {up_stderr!r} >&2\n"
        f"exit {up_rc}\n"
    )
    (bindir / "socat").write_text("#!/usr/bin/env bash\nexit 0\n")
    for f in bindir.iterdir():
        f.chmod(0o755)
    return bindir


def _run_entrypoint(env):
    return subprocess.run(
        ["bash", str(ROOT / "scripts" / "entrypoint.sh"), sys.executable, "-c",
         "import os;print(os.environ.get('AIMX_TAILNET_STATUS'), os.environ.get('TS_AUTHKEY','<unset>'),"
         " os.environ.get('TAILSCALE_AUTHKEY','<unset>'))"],
        env=env, capture_output=True, text=True, timeout=60,
    )


def test_entrypoint_rejected_key_logs_clear_error_and_degrades(tmp_path):
    bindir = _fake_tailscale_bin(
        tmp_path, "backend error: invalid key: API key tskey-auth-kAbC123-SECRETPART not valid")
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "HOME": str(tmp_path), "TMPDIR": str(tmp_path),
           "TS_AUTHKEY": "tskey-auth-kAbC123-SECRETPART", "TS_SOCKET": str(tmp_path / "ts.sock"),
           "TS_LOG": str(tmp_path / "ts.log"), "AIMX_TS_WAIT_SEC": "5"}
    out = _run_entrypoint(env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["rejected", "<unset>", "<unset>"]
    assert "Tailscale auth key was rejected" in out.stderr
    assert "90 days" in out.stderr and "aimx_ts_authkey" in out.stderr and "TAILSCALE_AUTHKEY" in out.stderr
    assert "SECRETPART" not in out.stderr and "tskey-<redacted>" in out.stderr


def test_entrypoint_accepts_tailscale_authkey_alias(tmp_path):
    bindir = _fake_tailscale_bin(tmp_path, "some other failure")
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "HOME": str(tmp_path), "TMPDIR": str(tmp_path),
           "TAILSCALE_AUTHKEY": "tskey-auth-kAlias-SECRETALIAS", "TS_SOCKET": str(tmp_path / "ts.sock"),
           "TS_LOG": str(tmp_path / "ts.log"), "AIMX_TS_WAIT_SEC": "5"}
    out = _run_entrypoint(env)
    assert out.returncode == 0, out.stderr
    # The alias reached `tailscale up` (status is failed, not off) and both names are gone.
    assert out.stdout.split() == ["failed", "<unset>", "<unset>"]
    assert "SECRETALIAS" not in out.stderr


def test_entrypoint_missing_key_with_tailnet_dsn_logs_error(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
           "AIMX_PG_DSN": "postgresql://aimx_worker:pw@127.0.0.1:5432/aimx_worker"}
    out = _run_entrypoint(env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split()[0] == "missing"
    assert "TS_AUTHKEY / TAILSCALE_AUTHKEY is not set" in out.stderr
    assert "pw@" not in out.stderr


def test_entrypoint_unresolved_runpod_secret_is_not_used(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
           "TS_AUTHKEY": "{{ RUNPOD_SECRET_aimx_ts_authkey }}"}
    out = _run_entrypoint(env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["off", "<unset>", "<unset>"]
    assert "unresolved RunPod secret reference" in out.stderr


def test_health_reports_tailnet_status(monkeypatch):
    import handler
    monkeypatch.setenv("AIMX_TAILNET_STATUS", "rejected")
    assert handler.health_body()["tailnet"] == "rejected"
    monkeypatch.delenv("AIMX_TAILNET_STATUS")
    assert handler.health_body()["tailnet"] == "off"
