#!/usr/bin/env bash
# Container entrypoint for the AIMX worker.
#
# Without TS_AUTHKEY: runs the worker directly (AIMX_DATABASE_URL / REDIS_URL
# point wherever you like).
#
# With TS_AUTHKEY (an ephemeral, pre-approved Tailscale auth key):
#   1. starts tailscaled in userspace mode (no /dev/net/tun, state in memory),
#   2. joins the tailnet as runpod-aimx-<RUNPOD_POD_ID>,
#   3. opens local forwards 127.0.0.1:<port> -> AIMX_TAILNET_HOST:<port>
#      for each port in AIMX_TAILNET_PORTS (default "5432 6379"; an entry
#      "15432:5432" listens on local 15432 and forwards to remote 5432),
#   4. waits until Postgres and Redis answer through the forwards,
#   5. removes TS_AUTHKEY from the environment and starts the worker.
# So the app uses postgresql://aimx_worker:...@127.0.0.1:5432/aimx_worker and
# redis://aimx-worker:...@127.0.0.1:6379/0. If the tailnet is not reachable
# within AIMX_TS_WAIT_SEC the worker still starts (Postgres/Redis are optional
# for serverless jobs) and logs a warning.
#
# Arguments, if any, replace the default command, e.g.
#   bash /app/scripts/entrypoint.sh python prefetch_volume.py --list
set -uo pipefail

log() { echo "[entrypoint] $*" >&2; }
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

start_tailnet() {
  local host="${AIMX_TAILNET_HOST:-100.96.240.104}"
  local ports="${AIMX_TAILNET_PORTS:-5432 6379}"
  local wait_sec="${AIMX_TS_WAIT_SEC:-90}"
  local sock="${TS_SOCKET:-/tmp/tailscaled.sock}"
  local id="${RUNPOD_POD_ID:-$(hostname)}"
  local name="${TS_HOSTNAME:-runpod-aimx-${id}}"
  name="$(printf '%s' "$name" | tr 'A-Z_.' 'a-z--' | tr -cd 'a-z0-9-' | cut -c1-63)"

  if ! command -v tailscaled >/dev/null || ! command -v socat >/dev/null; then
    log "TS_AUTHKEY set but tailscaled/socat missing in the image; skipping tailnet"
    return 1
  fi

  tailscaled --tun=userspace-networking --state=mem: --socket="$sock" \
    >"${TS_LOG:-/tmp/tailscaled.log}" 2>&1 &
  for _ in $(seq 1 100); do [[ -S "$sock" ]] && break; sleep 0.1; done

  # Hand the key over through a 0600 file so it never shows up in `ps`.
  local keyfile
  keyfile="$(mktemp)"
  chmod 600 "$keyfile"
  printf '%s' "$TS_AUTHKEY" >"$keyfile"
  if ! tailscale --socket="$sock" up --auth-key="file:$keyfile" --hostname="$name" \
       --accept-dns=false --timeout="${wait_sec}s" >/dev/null 2>&1; then
    rm -f "$keyfile"
    log "tailscale up failed (see ${TS_LOG:-/tmp/tailscaled.log}); continuing without tailnet"
    return 1
  fi
  rm -f "$keyfile"
  log "joined tailnet as $name ($(tailscale --socket="$sock" ip -4 2>/dev/null | head -1))"

  local spec lport rport
  for spec in $ports; do
    lport="${spec%%:*}"
    rport="${spec##*:}"
    socat "TCP-LISTEN:${lport},bind=127.0.0.1,fork,reuseaddr" \
      "EXEC:tailscale --socket=${sock} nc ${host} ${rport}" &
    log "forward 127.0.0.1:${lport} -> ${host}:${rport} (tailnet)"
  done

  if python3 "$APP_DIR/scripts/tailnet_wait.py" --timeout "$wait_sec" $ports; then
    log "tailnet forwards ready"
  else
    log "WARNING: tailnet forwards not answering after ${wait_sec}s; starting worker anyway"
  fi
}

if [[ -n "${TS_AUTHKEY:-}" ]]; then
  start_tailnet || true
fi
unset TS_AUTHKEY

cd "$APP_DIR"
if [[ $# -gt 0 ]]; then
  exec "$@"
fi
exec python3 -u "$APP_DIR/handler.py"
