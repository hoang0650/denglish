"""Wait until local forwards to Postgres / Redis answer at the protocol level.

Usage: python tailnet_wait.py [--host 127.0.0.1] [--timeout 90] 5432 6379
       (an entry "15432:5432" probes local 15432 as Postgres)

A TCP accept alone proves nothing (socat accepts before the tailnet dial), so:
- 5432 (Postgres): send an SSLRequest, expect b"N" or b"S".
- 6379 (Redis): send PING, expect a RESP reply ("+PONG" or "-NOAUTH ...").
- any other port: the connection must stay open for 0.5 s without EOF.
Exit 0 when every port answered, 1 on timeout. Never prints credentials.
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time

SSL_REQUEST = struct.pack("!ii", 8, 80877103)


KINDS = {5432: "postgres", 6379: "redis"}


def probe(host: str, port: int, timeout: float = 3.0, kind: str | None = None) -> bool:
    kind = kind or KINDS.get(port, "tcp")
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            if kind == "postgres":
                sock.sendall(SSL_REQUEST)
                return sock.recv(1) in (b"N", b"S")
            if kind == "redis":
                sock.sendall(b"PING\r\n")
                data = sock.recv(64)
                return data[:1] in (b"+", b"-")
            sock.settimeout(0.5)
            try:
                return sock.recv(1) != b""
            except socket.timeout:
                return True
    except OSError:
        return False


def parse(spec: str | int) -> tuple[int, str]:
    """"5432" -> (5432, "postgres"); "15432:5432" -> (15432, "postgres")."""
    text = str(spec)
    local, _, remote = text.partition(":")
    return int(local), KINDS.get(int(remote or local), "tcp")


def wait(host: str, ports: list, timeout: float, interval: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    pending = [parse(p) for p in ports]
    while pending:
        pending = [(p, k) for p, k in pending if not probe(host, p, kind=k)]
        if not pending:
            return True
        if time.monotonic() >= deadline:
            print(f"[tailnet_wait] not ready: {[p for p, _ in pending]}", file=sys.stderr, flush=True)
            return False
        time.sleep(interval)
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("ports", nargs="*", default=["5432", "6379"])
    args = ap.parse_args(argv)
    return 0 if wait(args.host, args.ports, args.timeout) else 1


if __name__ == "__main__":
    sys.exit(main())
