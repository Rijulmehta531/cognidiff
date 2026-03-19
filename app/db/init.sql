CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


-- ── Repositories ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS repositories (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    full_name       TEXT NOT NULL UNIQUE,
    clone_url       TEXT NOT NULL,
    default_branch  TEXT NOT NULL DEFAULT 'main',
    last_indexed_at TIMESTAMPTZ,
    status          TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'indexing', 'indexed', 'failed')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ── Index Runs ────────────────────────────────────────────────────
-- One row per indexing attempt. Chunks link here, not to repositories
-- directly. A failed run leaves the previous completed run untouched.
CREATE TABLE IF NOT EXISTS index_runs (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    repo_id          UUID NOT NULL REFERENCES repositories(id)
                         ON DELETE CASCADE,
    commit_sha       TEXT NOT NULL,
    branch           TEXT NOT NULL DEFAULT 'main',
    status           TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed', 'superseded')),
    embedding_model  TEXT NOT NULL,
    chunking_version TEXT NOT NULL DEFAULT '1.0',
    parser_version   TEXT NOT NULL,
    files_processed  INTEGER NOT NULL DEFAULT 0,
    chunks_created   INTEGER NOT NULL DEFAULT 0,
    chunks_deleted   INTEGER NOT NULL DEFAULT 0,
    error_message    TEXT,
    started_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at     TIMESTAMPTZ
);

ALTER TABLE repositories
    ADD COLUMN IF NOT EXISTS active_index_run_id
    UUID REFERENCES index_runs(id) ON DELETE SET NULL;


-- ── Code Chunks ───────────────────────────────────────────────────
-- content_hash is SHA-256 of content — used to skip re-embedding
-- unchanged functions on reindex. Uniqueness scoped to run, not
-- globally, since the same function exists across multiple runs.
CREATE TABLE IF NOT EXISTS code_chunks (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    repo_id       UUID NOT NULL REFERENCES repositories(id)
                      ON DELETE CASCADE,
    index_run_id  UUID NOT NULL REFERENCES index_runs(id)
                      ON DELETE CASCADE,
    file_path     TEXT NOT NULL,
    chunk_type    TEXT NOT NULL
        CHECK (chunk_type IN ('function', 'class', 'method', 'module')),
    name          TEXT NOT NULL,
    parent_class  TEXT NOT NULL DEFAULT '',
    language      TEXT NOT NULL,
    content       TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    line_start    INTEGER,
    line_end      INTEGER,
    embedding     vector(768),
    metadata      JSONB NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ── PR Reviews ────────────────────────────────────────────────────
-- append-only — never update rows, new commit = new row
CREATE TABLE IF NOT EXISTS pr_reviews (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    repo_id         UUID NOT NULL REFERENCES repositories(id)
                        ON DELETE CASCADE,
    pr_number       INTEGER NOT NULL,
    pr_title        TEXT NOT NULL,
    commit_sha      TEXT NOT NULL,
    review_run      INTEGER NOT NULL DEFAULT 1
        CHECK (review_run > 0),
    review_decision TEXT NOT NULL
        CHECK (review_decision IN ('APPROVE', 'REQUEST_CHANGES', 'COMMENT')),
    review_summary  TEXT NOT NULL,
    comments_count  INTEGER NOT NULL DEFAULT 0,
    issues_found    INTEGER NOT NULL DEFAULT 0,
    issues_resolved INTEGER NOT NULL DEFAULT 0,
    issues_open     INTEGER NOT NULL DEFAULT 0,
    processing_ms   INTEGER
        CHECK (processing_ms IS NULL OR processing_ms >= 0),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ── Indexes ───────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS repositories_status_idx
    ON repositories (status);

CREATE INDEX IF NOT EXISTS index_runs_repo_idx
    ON index_runs (repo_id, started_at DESC);

-- only track runs that need attention
CREATE INDEX IF NOT EXISTS index_runs_status_idx
    ON index_runs (status)
    WHERE status IN ('running', 'failed');

-- no global ivfflat — queries are always repo-scoped.
-- sequential scan over one repo (~10k-30k rows) gives
-- better recall than post-filtering a global ANN index.
-- revisit if a single repo exceeds ~50k chunks.
CREATE INDEX IF NOT EXISTS code_chunks_repo_idx
    ON code_chunks (repo_id);

CREATE INDEX IF NOT EXISTS code_chunks_run_idx
    ON code_chunks (index_run_id);

CREATE INDEX IF NOT EXISTS code_chunks_language_idx
    ON code_chunks (language, chunk_type);

CREATE UNIQUE INDEX IF NOT EXISTS code_chunks_idempotency_idx
    ON code_chunks (index_run_id, file_path, content_hash);

CREATE INDEX IF NOT EXISTS pr_reviews_lookup_idx
    ON pr_reviews (repo_id, pr_number, created_at DESC);

CREATE INDEX IF NOT EXISTS pr_reviews_repo_idx
    ON pr_reviews (repo_id, created_at DESC);


-- ── Views ─────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW pr_reviews_latest AS
SELECT DISTINCT ON (repo_id, pr_number)
    *
FROM pr_reviews
ORDER BY repo_id, pr_number, created_at DESC;

CREATE OR REPLACE VIEW pr_review_progress AS
WITH first_runs AS (
    SELECT DISTINCT ON (repo_id, pr_number)
        repo_id,
        pr_number,
        pr_title,
        issues_found    AS initial_issues,
        review_decision AS initial_decision,
        created_at      AS first_reviewed_at
    FROM pr_reviews
    ORDER BY repo_id, pr_number, created_at ASC
),
latest_runs AS (
    SELECT DISTINCT ON (repo_id, pr_number)
        repo_id,
        pr_number,
        review_run      AS latest_run,
        issues_found    AS current_issues,
        issues_resolved,
        review_decision AS latest_decision,
        created_at      AS last_reviewed_at
    FROM pr_reviews
    ORDER BY repo_id, pr_number, created_at DESC
)
SELECT
    f.repo_id,
    f.pr_number,
    f.pr_title,
    l.latest_run                       AS total_runs,
    f.initial_issues,
    l.current_issues,
    f.initial_issues - l.current_issues AS issues_fixed,
    f.initial_decision,
    l.latest_decision,
    l.issues_resolved,
    f.first_reviewed_at,
    l.last_reviewed_at
FROM first_runs f
JOIN latest_runs l
    ON  f.repo_id    = l.repo_id
    AND f.pr_number  = l.pr_number;


-- ── Triggers ──────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER repositories_updated_at
    BEFORE UPDATE ON repositories
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();