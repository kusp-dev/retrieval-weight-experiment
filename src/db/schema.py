"""
Database schema for the retrieval weight learning experiment.

Single SQLite file (experiment_v3.db) with FTS5 for BM25 keyword search.
Schema separates static structure (tasks, skills, presets) from
dynamic state (episodes, feedback, bandit posteriors).
"""

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 4

SCHEMA_SQL = """
-- ============================================================
-- STATIC STRUCTURE (populated before experiment runs)
-- ============================================================

-- 300 tasks across 5 themes (60 per theme)
CREATE TABLE IF NOT EXISTS tasks (
    task_id             TEXT PRIMARY KEY,
    theme               TEXT NOT NULL,    -- e.g. "ML Fundamentals"
    title               TEXT NOT NULL,
    description         TEXT NOT NULL,
    difficulty          TEXT NOT NULL DEFAULT 'medium',  -- easy/medium/hard
    ground_truth_skills TEXT,             -- JSON array of skill IDs (primary first)
    expected_approach   TEXT,             -- how to combine skills for this task
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 205 skills across 10 domains
CREATE TABLE IF NOT EXISTS skills (
    skill_id    TEXT PRIMARY KEY,
    domain      TEXT NOT NULL,    -- research/creative/synthesis/evaluation/operational
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,    -- full skill text (indexed by FTS5)
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- FTS5 virtual table for BM25 keyword search over skills
CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
    skill_id UNINDEXED,
    title,
    content,
    content='skills',
    content_rowid='rowid'
);

-- Triggers to keep FTS5 in sync with skills table
CREATE TRIGGER IF NOT EXISTS skills_ai AFTER INSERT ON skills BEGIN
    INSERT INTO skills_fts(rowid, skill_id, title, content)
    VALUES (new.rowid, new.skill_id, new.title, new.content);
END;

CREATE TRIGGER IF NOT EXISTS skills_ad AFTER DELETE ON skills BEGIN
    INSERT INTO skills_fts(skills_fts, rowid, skill_id, title, content)
    VALUES ('delete', old.rowid, old.skill_id, old.title, old.content);
END;

CREATE TRIGGER IF NOT EXISTS skills_au AFTER UPDATE ON skills BEGIN
    INSERT INTO skills_fts(skills_fts, rowid, skill_id, title, content)
    VALUES ('delete', old.rowid, old.skill_id, old.title, old.content);
    INSERT INTO skills_fts(rowid, skill_id, title, content)
    VALUES (new.rowid, new.skill_id, new.title, new.content);
END;

-- Precomputed dense embeddings for skills (one per skill)
CREATE TABLE IF NOT EXISTS skill_embeddings (
    skill_id    TEXT PRIMARY KEY REFERENCES skills(skill_id),
    embedding   BLOB NOT NULL,   -- numpy float32 array, 1024 dims
    model       TEXT NOT NULL DEFAULT 'Qwen/Qwen3-Embedding-0.6B',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Weight presets: the discrete arms of the Thompson Sampling bandit
CREATE TABLE IF NOT EXISTS weight_presets (
    preset_id   TEXT PRIMARY KEY,
    w_recency   REAL NOT NULL,
    w_importance REAL NOT NULL,
    w_relevance REAL NOT NULL,
    CHECK (abs(w_recency + w_importance + w_relevance - 1.0) < 0.01)
);

-- ============================================================
-- DYNAMIC STATE (populated during experiment runs)
-- ============================================================

-- Each row = one task execution under one condition
CREATE TABLE IF NOT EXISTS episodes (
    episode_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id    INTEGER NOT NULL,  -- 1-4
    task_id         TEXT NOT NULL REFERENCES tasks(task_id),
    preset_id       TEXT REFERENCES weight_presets(preset_id),  -- NULL for control
    task_order      INTEGER NOT NULL,  -- sequential position within this condition
    prompt_sent     TEXT,              -- full prompt for reproducibility
    llm_response    TEXT,              -- raw LLM output
    success         INTEGER,           -- 1=success, 0=failure
    step_count      INTEGER,           -- LLM turns to complete (max 35)
    total_tokens    INTEGER,           -- total tokens consumed (input + output)
    input_tokens    INTEGER,           -- prompt/input tokens
    output_tokens   INTEGER,           -- completion/output tokens
    started_at      TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at    TEXT,
    duration_ms     INTEGER,
    UNIQUE(condition_id, task_id)
);

-- What was retrieved for each episode (top-k results)
CREATE TABLE IF NOT EXISTS retrieval_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id      INTEGER NOT NULL REFERENCES episodes(episode_id),
    skill_id        TEXT NOT NULL REFERENCES skills(skill_id),
    rank            INTEGER NOT NULL,  -- position in result list
    bm25_score      REAL,              -- raw BM25 score
    dense_score     REAL,              -- cosine similarity
    rrf_score       REAL,              -- fused RRF score
    recency_score   REAL,              -- dimension score before weighting
    importance_score REAL,             -- dimension score before weighting
    relevance_score REAL,              -- dimension score before weighting
    final_score     REAL,              -- w_rec*recency + w_imp*importance + w_rel*relevance
    is_ground_truth INTEGER DEFAULT 0, -- 1 if skill_id in task's ground_truth_skills
    UNIQUE(episode_id, rank)
);

-- Per-step API call log (captures multi-step conversations)
CREATE TABLE IF NOT EXISTS step_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id      INTEGER NOT NULL REFERENCES episodes(episode_id),
    step_number     INTEGER NOT NULL,  -- 1-indexed step within episode
    prompt_text     TEXT NOT NULL,      -- full prompt sent at this step
    response_text   TEXT NOT NULL,      -- LLM response at this step
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    is_final        INTEGER NOT NULL,   -- 1 if this was the last step
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(episode_id, step_number)
);

-- Dimension-specific feedback (conditions 2, 3, 4)
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id      INTEGER NOT NULL REFERENCES episodes(episode_id),
    -- Likert ratings (1-5) for conditions 2 and 3; NULL for condition 4
    rating_recency      INTEGER CHECK (rating_recency BETWEEN 1 AND 5),
    rating_importance   INTEGER CHECK (rating_importance BETWEEN 1 AND 5),
    rating_relevance    INTEGER CHECK (rating_relevance BETWEEN 1 AND 5),
    -- Natural language explanation (all feedback conditions)
    explanation     TEXT NOT NULL,
    -- For condition 4: ratings inferred from free-text
    inferred_recency    REAL,
    inferred_importance REAL,
    inferred_relevance  REAL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(episode_id)
);

-- Embedded explanations (condition 3 only)
CREATE TABLE IF NOT EXISTS feedback_embeddings (
    feedback_id     INTEGER PRIMARY KEY REFERENCES feedback(feedback_id),
    embedding       BLOB NOT NULL,   -- numpy float32 array, 1024 dims
    model           TEXT NOT NULL DEFAULT 'Qwen/Qwen3-Embedding-0.6B',
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Thompson Sampling state: Beta(alpha, beta) per preset per condition
CREATE TABLE IF NOT EXISTS bandit_state (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id    INTEGER NOT NULL,
    preset_id       TEXT NOT NULL REFERENCES weight_presets(preset_id),
    alpha           REAL NOT NULL DEFAULT 1.0,
    beta            REAL NOT NULL DEFAULT 1.0,
    pulls           INTEGER NOT NULL DEFAULT 0,
    total_reward    REAL NOT NULL DEFAULT 0.0,
    last_updated    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(condition_id, preset_id)
);

-- Per-condition skill usage history (for dynamic recency/importance)
CREATE TABLE IF NOT EXISTS skill_usage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id    INTEGER NOT NULL,
    skill_id        TEXT NOT NULL REFERENCES skills(skill_id),
    episode_id      INTEGER NOT NULL REFERENCES episodes(episode_id),
    task_success    INTEGER,           -- 1=success, 0=failure (for importance)
    used_at_order   INTEGER NOT NULL,  -- task_order when this skill was used
    UNIQUE(condition_id, skill_id, episode_id)
);

-- Reproducibility metadata (key-value store)
CREATE TABLE IF NOT EXISTS experiment_metadata (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);

-- ============================================================
-- METRICS & LOGGING
-- ============================================================

CREATE TABLE IF NOT EXISTS experiment_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    level       TEXT NOT NULL DEFAULT 'INFO',  -- INFO/WARN/ERROR
    component   TEXT NOT NULL,                 -- db/embeddings/search/bandit/runner
    message     TEXT NOT NULL,
    metadata    TEXT,  -- JSON blob for structured data
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_episodes_condition ON episodes(condition_id);
CREATE INDEX IF NOT EXISTS idx_episodes_task ON episodes(task_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_episode ON retrieval_results(episode_id);
CREATE INDEX IF NOT EXISTS idx_feedback_episode ON feedback(episode_id);
CREATE INDEX IF NOT EXISTS idx_bandit_condition ON bandit_state(condition_id);
CREATE INDEX IF NOT EXISTS idx_skill_usage_cond_skill ON skill_usage(condition_id, skill_id);
CREATE INDEX IF NOT EXISTS idx_skill_usage_episode ON skill_usage(episode_id);
CREATE INDEX IF NOT EXISTS idx_log_component ON experiment_log(component);
CREATE INDEX IF NOT EXISTS idx_log_level ON experiment_log(level);
CREATE INDEX IF NOT EXISTS idx_step_log_episode ON step_log(episode_id);
"""


def init_db(db_path: str | Path = "experiment_v3.db") -> sqlite3.Connection:
    """Initialize the experiment database with full schema.

    Returns a connection with WAL mode and foreign keys enabled.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Performance settings
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Apply schema
    conn.executescript(SCHEMA_SQL)

    # Record schema version if not already set
    existing = conn.execute("SELECT version FROM schema_version").fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (SCHEMA_VERSION,),
        )

    conn.commit()
    return conn


def seed_weight_presets(conn: sqlite3.Connection, presets: dict[str, list[float]]) -> None:
    """Insert weight presets from config into the database."""
    for preset_id, weights in presets.items():
        conn.execute(
            """INSERT OR REPLACE INTO weight_presets
               (preset_id, w_recency, w_importance, w_relevance)
               VALUES (?, ?, ?, ?)""",
            (preset_id, weights[0], weights[1], weights[2]),
        )
    conn.commit()


def init_bandit_state(
    conn: sqlite3.Connection,
    condition_ids: list[int],
    preset_ids: list[str],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> None:
    """Initialize Thompson Sampling state for all condition × preset combinations."""
    for cond_id in condition_ids:
        for preset_id in preset_ids:
            conn.execute(
                """INSERT OR IGNORE INTO bandit_state
                   (condition_id, preset_id, alpha, beta)
                   VALUES (?, ?, ?, ?)""",
                (cond_id, preset_id, prior_alpha, prior_beta),
            )
    conn.commit()


def log_event(
    conn: sqlite3.Connection,
    component: str,
    message: str,
    level: str = "INFO",
    metadata: str | None = None,
) -> None:
    """Log an event to the experiment_log table."""
    conn.execute(
        "INSERT INTO experiment_log (level, component, message, metadata) VALUES (?, ?, ?, ?)",
        (level, component, message, metadata),
    )
    conn.commit()
