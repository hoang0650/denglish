#!/usr/bin/env bash
# Container entrypoint for the AIMX worker.
#
# The key comes only from the environment: TS_AUTHKEY, or TAILSCALE_AUTHKEY as an
# alias (TS_AUTHKEY wins). On the RunPod endpoint it is the template env
# TS_AUTHKEY={{ RUNPOD_SECRET_aimx_ts_authkey }}; nothing is baked into the image.
# Tailscale auth keys expire after at most 90 days. Rotation is described in README.md
# ("Rotating the Tailscale auth key").
#
# Without a key: runs the worker directly (AIMX_DATABASE_URL / REDIS_URL point
# wherever you like). If those DSNs point at 127.0.0.1 (i.e. they expect the
# tailnet forwards) a clear error is logged and the worker still starts.
#
# With a key (an ephemeral, reusable, pre-approved Tailscale auth key):
#   1. starts tailscaled in userspace mode (no /dev/net/tun, state in memory),
#   2. joins the tailnet as runpod-aimx-<RUNPOD_POD_ID>,
#   3. opens local forwards 127.0.0.1:<port> -> AIMX_TAILNET_HOST:<port>
#      for each port in AIMX_TAILNET_PORTS (default "5432 6379"; an entry
#      "15432:5432" listens on local 15432 and forwards to remote 5432),
#   4. waits until Postgres and Redis answer through the forwards,
#   5. removes TS_AUTHKEY/TAILSCALE_AUTHKEY from the environment, exports
#      AIMX_TAILNET_STATUS (off|missing|ready|degraded|rejected|timeout|failed|
#      unavailable; reported by the health job) and starts the worker.
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
ROTATE_HINT="create a new reusable + ephemeral + pre-approved key in the Tailscale admin (Settings > Keys), then set it as TAILSCALE_AUTHKEY on the Dokploy app for ai.aimarkets.vn and redeploy (it syncs to the RunPod secret aimx_ts_authkey), or edit that RunPod secret directly. New workers pick it up on their next boot."
AIMX_TAILNET_STATUS=off

# Any DSN that targets the local tailnet forwards?
wants_tailnet() {
  local v
  for v in "${AIMX_DATABASE_URL:-}" "${AIMX_PG_DSN:-}" "${DATABASE_URL:-}" "${AIMX_REDIS_URL:-}" "${REDIS_URL:-}"; do
    case "$v" in *@127.0.0.1:*|*@localhost:*) return 0 ;; esac
  done
  return 1
}

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
    AIMX_TAILNET_STATUS=unavailable
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
  local errfile
  errfile="$(mktemp)"
  chmod 600 "$errfile"
  if ! tailscale --socket="$sock" up --auth-key="file:$keyfile" --hostname="$name" \
       --accept-dns=false --timeout="${wait_sec}s" >"$errfile" 2>&1; then
    rm -f "$keyfile"
    # Never echo the key: redact anything that looks like one before logging.
    local reason
    reason="$(sed -E 's/tskey-[A-Za-z0-9_-]+/tskey-<redacted>/g' "$errfile" | tr -s '\n' ' ' | cut -c1-300)"
    rm -f "$errfile"
    if printf '%s' "$reason" | grep -qiE 'invalid key|expired|not valid|does not exist|revoked|unauthori[sz]ed|key.*(deleted|disabled)'; then
      AIMX_TAILNET_STATUS=rejected
      log "ERROR: Tailscale auth key was rejected (expired, revoked or mistyped; auth keys live at most 90 days): ${reason}"
      log "ERROR: to fix, ${ROTATE_HINT}"
    elif printf '%s' "$reason" | grep -qiE 'timeout|timed out|deadline exceeded'; then
      AIMX_TAILNET_STATUS=timeout
      log "ERROR: tailscale up timed out after ${wait_sec}s (control plane unreachable or key needs device approval): ${reason}"
    else
      AIMX_TAILNET_STATUS=failed
      log "ERROR: tailscale up failed: ${reason:-no output} (see ${TS_LOG:-/tmp/tailscaled.log})"
    fi
    log "continuing without tailnet: Postgres job log / Redis queue over the tailnet are unavailable"
    return 1
  fi
  rm -f "$keyfile" "$errfile"
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
    AIMX_TAILNET_STATUS=ready
  else
    log "WARNING: tailnet forwards not answering after ${wait_sec}s; starting worker anyway"
    AIMX_TAILNET_STATUS=degraded
  fi
}

# TAILSCALE_AUTHKEY is accepted as an alias; TS_AUTHKEY wins when both are set.
TS_AUTHKEY="${TS_AUTHKEY:-${TAILSCALE_AUTHKEY:-}}"
# An unresolved RunPod secret reference (secret deleted/renamed) is not a key.
case "$TS_AUTHKEY" in *RUNPOD_SECRET_*) log "ERROR: TS_AUTHKEY is an unresolved RunPod secret reference; check the secret aimx_ts_authkey exists"; TS_AUTHKEY="" ;; esac
if [[ -n "${TS_AUTHKEY:-}" ]]; then
  start_tailnet || true
elif wants_tailnet; then
  AIMX_TAILNET_STATUS=missing
  log "ERROR: TS_AUTHKEY / TAILSCALE_AUTHKEY is not set, but the Postgres/Redis DSN points at 127.0.0.1 (tailnet forwards). Starting without tailnet; to fix, ${ROTATE_HINT}"
fi
unset TS_AUTHKEY TAILSCALE_AUTHKEY
export AIMX_TAILNET_STATUS

cd "$APP_DIR"
if [[ $# -gt 0 ]]; then
  exec "$@"
fi
exec python3 -u "$APP_DIR/handler.py"
