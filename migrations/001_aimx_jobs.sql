-- AI Markets RunPod worker tables. Idempotent: safe to run more than once.
-- db.py runs the same statements on first connect when AIMX_DB_MIGRATE=1 (default).

CREATE TABLE IF NOT EXISTS aimx_jobs (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL DEFAULT '',
    model         TEXT,
    license       TEXT,
    tier          TEXT,
    priority      INTEGER,
    backend       TEXT,
    runtime       TEXT,
    worker_id     TEXT,
    status        TEXT NOT NULL DEFAULT 'queued',
    gpu_seconds   DOUBLE PRECISION,
    artifact_url  TEXT,
    error         TEXT,
    result        JSONB,
    created_at    TIMESTAMPTZ,
    started_at    TIMESTAMPTZ,
    finished_at   TIMESTAMPTZ,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS aimx_jobs_status_created_idx ON aimx_jobs (status, created_at DESC);
CREATE INDEX IF NOT EXISTS aimx_jobs_kind_finished_idx ON aimx_jobs (kind, finished_at DESC);

CREATE TABLE IF NOT EXISTS aimx_volume_status (
    volume      TEXT NOT NULL,
    model_key   TEXT NOT NULL,
    repo        TEXT,
    license     TEXT,
    path        TEXT,
    status      TEXT NOT NULL,
    approx_gb   REAL,
    error       TEXT,
    worker_id   TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (volume, model_key)
);

-- Job output files. Outputs live only here and in Redis, never on the RunPod volume.
CREATE TABLE IF NOT EXISTS aimx_job_artifacts (
    id          TEXT PRIMARY KEY,
    job_id      TEXT,
    kind        TEXT,
    mime        TEXT,
    bytes       BIGINT,
    sha256      TEXT,
    filename    TEXT,
    data        BYTEA NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS aimx_job_artifacts_job_idx ON aimx_job_artifacts (job_id);
