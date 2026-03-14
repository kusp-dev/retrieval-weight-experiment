"""Tests for dynamic skill scoring (recency + importance)."""

import pytest

from src.db.schema import init_db
from src.experiment.scoring import (
    compute_importance,
    compute_recency,
    record_skill_usage,
)


@pytest.fixture
def db():
    """Database with schema and test skills/tasks."""
    conn = init_db(":memory:")

    # Insert test skills
    for i in range(1, 6):
        conn.execute(
            "INSERT INTO skills (skill_id, domain, title, content) VALUES (?, ?, ?, ?)",
            (f"s{i}", "research", f"Skill {i}", f"Content for skill {i}"),
        )

    # Insert a test task and episode
    conn.execute(
        "INSERT INTO tasks (task_id, theme, title, description) VALUES (?, ?, ?, ?)",
        ("t1", "ML", "Test Task", "Test task description"),
    )
    conn.execute(
        "INSERT INTO episodes (condition_id, task_id, preset_id, task_order) VALUES (?, ?, ?, ?)",
        (2, "t1", None, 1),
    )
    conn.commit()

    yield conn
    conn.close()


class TestComputeRecency:
    def test_never_used_returns_neutral(self, db):
        """Skills never used in a condition should return 0.5."""
        score = compute_recency(db, condition_id=2, skill_id="s1", current_task_order=10)
        assert score == 0.5

    def test_just_used_returns_high(self, db):
        """Skill used on the current task_order should return ~1.0."""
        # Record usage at task_order=10
        record_skill_usage(
            db, condition_id=2, skill_id="s1", episode_id=1, task_success=1, task_order=10
        )
        db.commit()

        score = compute_recency(
            db, condition_id=2, skill_id="s1", current_task_order=10, decay_window=50
        )
        assert score == pytest.approx(1.0)

    def test_decays_over_time(self, db):
        """Recency should decay linearly from 1.0 to 0.0 over the window."""
        record_skill_usage(
            db, condition_id=2, skill_id="s1", episode_id=1, task_success=1, task_order=1
        )
        db.commit()

        # 25 iterations later = 50% of window = 0.5
        score = compute_recency(
            db, condition_id=2, skill_id="s1", current_task_order=26, decay_window=50
        )
        assert score == pytest.approx(0.5)

    def test_clamps_at_zero(self, db):
        """Recency should be 0 when skill was used > decay_window ago."""
        record_skill_usage(
            db, condition_id=2, skill_id="s1", episode_id=1, task_success=1, task_order=1
        )
        db.commit()

        score = compute_recency(
            db, condition_id=2, skill_id="s1", current_task_order=100, decay_window=50
        )
        assert score == 0.0

    def test_conditions_independent(self, db):
        """Usage in condition 2 shouldn't affect recency in condition 3."""
        record_skill_usage(
            db, condition_id=2, skill_id="s1", episode_id=1, task_success=1, task_order=5
        )
        db.commit()

        # Condition 3 has no usage
        score = compute_recency(db, condition_id=3, skill_id="s1", current_task_order=6)
        assert score == 0.5  # never used in condition 3

    def test_most_recent_usage_matters(self, db):
        """Should use MAX(used_at_order), not first or average."""
        # Insert another episode for a second usage
        db.execute(
            "INSERT INTO tasks (task_id, theme, title, description) VALUES (?, ?, ?, ?)",
            ("t2", "ML", "Task 2", "Another task"),
        )
        db.execute(
            "INSERT INTO episodes"
            " (condition_id, task_id, preset_id, task_order)"
            " VALUES (?, ?, ?, ?)",
            (2, "t2", None, 20),
        )
        db.commit()
        episode_2_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

        record_skill_usage(
            db, condition_id=2, skill_id="s1", episode_id=1, task_success=1, task_order=5
        )
        record_skill_usage(
            db,
            condition_id=2,
            skill_id="s1",
            episode_id=episode_2_id,
            task_success=1,
            task_order=20,
        )
        db.commit()

        # At task_order=25 with decay_window=50, most recent=20 → (25-20)/50 = 0.1 → 1-0.1=0.9
        score = compute_recency(
            db, condition_id=2, skill_id="s1", current_task_order=25, decay_window=50
        )
        assert score == pytest.approx(0.9)


class TestComputeImportance:
    def test_never_used_returns_prior(self, db):
        """Laplace smoothing: (0+1)/(0+0+2) = 0.5."""
        score = compute_importance(db, condition_id=2, skill_id="s1")
        assert score == pytest.approx(0.5)

    def test_all_successes(self, db):
        """All successes: (3+1)/(3+0+2) = 0.8."""
        for i in range(3):
            db.execute(
                "INSERT INTO tasks (task_id, theme, title, description) VALUES (?, ?, ?, ?)",
                (f"t_imp_{i}", "ML", f"Task {i}", "Desc"),
            )
            db.execute(
                "INSERT INTO episodes"
                " (condition_id, task_id, preset_id, task_order)"
                " VALUES (?, ?, ?, ?)",
                (2, f"t_imp_{i}", None, i + 10),
            )
            db.commit()
            ep_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
            record_skill_usage(
                db,
                condition_id=2,
                skill_id="s1",
                episode_id=ep_id,
                task_success=1,
                task_order=i + 10,
            )
        db.commit()

        score = compute_importance(db, condition_id=2, skill_id="s1")
        assert score == pytest.approx(4 / 5)  # (3+1)/(3+0+2)

    def test_all_failures(self, db):
        """All failures: (0+1)/(0+3+2) = 0.2."""
        for i in range(3):
            db.execute(
                "INSERT INTO tasks (task_id, theme, title, description) VALUES (?, ?, ?, ?)",
                (f"t_fail_{i}", "ML", f"Task {i}", "Desc"),
            )
            db.execute(
                "INSERT INTO episodes"
                " (condition_id, task_id, preset_id, task_order)"
                " VALUES (?, ?, ?, ?)",
                (2, f"t_fail_{i}", None, i + 20),
            )
            db.commit()
            ep_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
            record_skill_usage(
                db,
                condition_id=2,
                skill_id="s1",
                episode_id=ep_id,
                task_success=0,
                task_order=i + 20,
            )
        db.commit()

        score = compute_importance(db, condition_id=2, skill_id="s1")
        assert score == pytest.approx(1 / 5)  # (0+1)/(0+3+2)

    def test_mixed_results(self, db):
        """2 success, 1 failure: (2+1)/(2+1+2) = 0.6."""
        results = [1, 1, 0]
        for i, success in enumerate(results):
            db.execute(
                "INSERT INTO tasks (task_id, theme, title, description) VALUES (?, ?, ?, ?)",
                (f"t_mix_{i}", "ML", f"Task {i}", "Desc"),
            )
            db.execute(
                "INSERT INTO episodes"
                " (condition_id, task_id, preset_id, task_order)"
                " VALUES (?, ?, ?, ?)",
                (2, f"t_mix_{i}", None, i + 30),
            )
            db.commit()
            ep_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
            record_skill_usage(
                db,
                condition_id=2,
                skill_id="s1",
                episode_id=ep_id,
                task_success=success,
                task_order=i + 30,
            )
        db.commit()

        score = compute_importance(db, condition_id=2, skill_id="s1")
        assert score == pytest.approx(3 / 5)


class TestRecordSkillUsage:
    def test_records_usage(self, db):
        record_skill_usage(
            db, condition_id=2, skill_id="s1", episode_id=1, task_success=1, task_order=1
        )
        db.commit()

        row = db.execute(
            "SELECT * FROM skill_usage WHERE condition_id=2 AND skill_id='s1'"
        ).fetchone()
        assert row is not None
        assert row["task_success"] == 1
        assert row["used_at_order"] == 1

    def test_replace_on_duplicate(self, db):
        """INSERT OR REPLACE should update on duplicate (condition, skill, episode)."""
        record_skill_usage(
            db, condition_id=2, skill_id="s1", episode_id=1, task_success=None, task_order=1
        )
        db.commit()

        record_skill_usage(
            db, condition_id=2, skill_id="s1", episode_id=1, task_success=1, task_order=1
        )
        db.commit()

        rows = db.execute(
            "SELECT * FROM skill_usage WHERE condition_id=2 AND skill_id='s1' AND episode_id=1"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["task_success"] == 1
