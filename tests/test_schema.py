"""Tests for database schema initialization and operations."""

import sqlite3

import pytest

from src.db.schema import init_bandit_state, init_db, log_event, seed_weight_presets


@pytest.fixture
def db():
    """In-memory database for testing."""
    conn = init_db(":memory:")
    yield conn
    conn.close()


class TestSchemaInit:
    def test_creates_all_tables(self, db):
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        expected = {
            "tasks",
            "skills",
            "skill_embeddings",
            "weight_presets",
            "episodes",
            "retrieval_results",
            "feedback",
            "feedback_embeddings",
            "bandit_state",
            "experiment_log",
            "schema_version",
            "skill_usage",
            "experiment_metadata",
        }
        assert expected.issubset(tables)

    def test_creates_fts5_table(self, db):
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "skills_fts" in tables

    def test_schema_version_recorded(self, db):
        row = db.execute("SELECT version FROM schema_version").fetchone()
        assert row["version"] == 4

    def test_foreign_keys_enabled(self, db):
        result = db.execute("PRAGMA foreign_keys").fetchone()
        assert result[0] == 1

    def test_wal_mode(self, db):
        # In-memory databases don't support WAL, but the pragma shouldn't error
        result = db.execute("PRAGMA journal_mode").fetchone()
        assert result[0] in ("wal", "memory")


class TestWeightPresets:
    def test_seed_presets(self, db):
        presets = {
            "equal": [0.333, 0.333, 0.334],
            "relevance_heavy": [0.10, 0.20, 0.70],
        }
        seed_weight_presets(db, presets)

        rows = db.execute("SELECT * FROM weight_presets").fetchall()
        assert len(rows) == 2

    def test_preset_weights_sum_to_one(self, db):
        presets = {"test": [0.333, 0.333, 0.334]}
        seed_weight_presets(db, presets)

        row = db.execute("SELECT * FROM weight_presets WHERE preset_id = 'test'").fetchone()
        total = row["w_recency"] + row["w_importance"] + row["w_relevance"]
        assert abs(total - 1.0) < 0.01

    def test_preset_constraint_rejects_bad_weights(self, db):
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO weight_presets VALUES (?, ?, ?, ?)",
                ("bad", 0.5, 0.5, 0.5),  # sums to 1.5
            )


class TestBanditState:
    def test_init_bandit_state(self, db):
        presets = {"equal": [0.333, 0.333, 0.334], "heavy": [0.1, 0.2, 0.7]}
        seed_weight_presets(db, presets)
        init_bandit_state(db, [2, 3], ["equal", "heavy"])

        rows = db.execute("SELECT * FROM bandit_state").fetchall()
        assert len(rows) == 4  # 2 conditions × 2 presets

    def test_bandit_state_defaults(self, db):
        presets = {"equal": [0.333, 0.333, 0.334]}
        seed_weight_presets(db, presets)
        init_bandit_state(db, [2], ["equal"], prior_alpha=2.0, prior_beta=3.0)

        row = db.execute("SELECT * FROM bandit_state").fetchone()
        assert row["alpha"] == 2.0
        assert row["beta"] == 3.0
        assert row["pulls"] == 0


class TestFTS5Sync:
    def test_insert_syncs_to_fts(self, db):
        db.execute(
            "INSERT INTO skills (skill_id, domain, title, content) VALUES (?, ?, ?, ?)",
            ("s1", "research", "Neural Search", "Dense retrieval with transformers"),
        )
        db.commit()

        results = db.execute(
            "SELECT skill_id FROM skills_fts WHERE skills_fts MATCH 'transformers'"
        ).fetchall()
        assert len(results) == 1
        assert results[0]["skill_id"] == "s1"

    def test_update_syncs_to_fts(self, db):
        db.execute(
            "INSERT INTO skills (skill_id, domain, title, content) VALUES (?, ?, ?, ?)",
            ("s1", "research", "Test Skill", "original_unique_token_xyz content"),
        )
        db.commit()

        db.execute("UPDATE skills SET content = 'new content about bandits' WHERE skill_id = 's1'")
        db.commit()

        # original_unique_token_xyz should no longer be findable
        old = db.execute(
            "SELECT * FROM skills_fts WHERE skills_fts MATCH 'original_unique_token_xyz'"
        ).fetchall()
        new = db.execute("SELECT * FROM skills_fts WHERE skills_fts MATCH 'bandits'").fetchall()
        assert len(old) == 0
        assert len(new) == 1

    def test_delete_syncs_to_fts(self, db):
        db.execute(
            "INSERT INTO skills (skill_id, domain, title, content) VALUES (?, ?, ?, ?)",
            ("s1", "research", "Title", "unique_keyword_xyz"),
        )
        db.commit()

        db.execute("DELETE FROM skills WHERE skill_id = 's1'")
        db.commit()

        results = db.execute(
            "SELECT * FROM skills_fts WHERE skills_fts MATCH 'unique_keyword_xyz'"
        ).fetchall()
        assert len(results) == 0


class TestLogging:
    def test_log_event(self, db):
        log_event(db, "test", "something happened", metadata='{"key": "val"}')

        row = db.execute("SELECT * FROM experiment_log").fetchone()
        assert row["component"] == "test"
        assert row["message"] == "something happened"
        assert row["level"] == "INFO"
