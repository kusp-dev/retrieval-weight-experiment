#!/usr/bin/env python3
"""
Database schema and integrity tests for experiment.db (schema v4).

Tests:
  1. Schema version is 4
  2. All expected tables exist
  3. retrieval_results has v4 columns
  4. episodes has v4 columns (input_tokens, output_tokens)
  5. step_log table schema
  6. Foreign key enforcement
  7. Unique constraints
  8. weight_presets has 12 presets with correct names
  9. Task data: 300 tasks, 5 themes, difficulty distribution
  10. Skill data: 205 skills, 10 domains, 205 embeddings
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "experiment_v3.db"

PASS = 0
FAIL = 0


def report(test_num: int, name: str, passed: bool, detail: str = ""):
    global PASS, FAIL
    status = "PASS" if passed else "FAIL"
    if passed:
        PASS += 1
    else:
        FAIL += 1
    suffix = f" -- {detail}" if detail else ""
    print(f"  [{status}] Test {test_num}: {name}{suffix}")


def get_conn() -> sqlite3.Connection:
    """Get a connection with foreign keys enabled."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def get_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Return column names for a table."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return [row["name"] for row in cursor.fetchall()]


def test_1_schema_version():
    """Verify schema version is 4."""
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if row is None:
            report(1, "Schema version is 4", False, "No rows in schema_version")
            return
        version = row["version"]
        report(1, "Schema version is 4", version == 4, f"got version={version}")
    finally:
        conn.close()


def test_2_all_tables_exist():
    """Verify ALL expected tables exist."""
    expected_tables = [
        "tasks",
        "skills",
        "skills_fts",
        "skill_embeddings",
        "episodes",
        "feedback",
        "feedback_embeddings",
        "retrieval_results",
        "bandit_state",
        "weight_presets",
        "experiment_metadata",
        "experiment_log",
        "skill_usage",
        "step_log",
        "schema_version",
    ]
    conn = get_conn()
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
        )
        existing = {row["name"] for row in cursor.fetchall()}
        missing = [t for t in expected_tables if t not in existing]
        passed = len(missing) == 0
        detail = f"missing: {missing}" if missing else f"all {len(expected_tables)} tables present"
        report(2, "All expected tables exist", passed, detail)
    finally:
        conn.close()


def test_3_retrieval_results_v4_columns():
    """Verify retrieval_results has v4 columns."""
    v4_columns = ["recency_score", "importance_score", "relevance_score", "is_ground_truth"]
    conn = get_conn()
    try:
        cols = get_table_columns(conn, "retrieval_results")
        missing = [c for c in v4_columns if c not in cols]
        passed = len(missing) == 0
        detail = f"missing: {missing}" if missing else f"all v4 columns present: {v4_columns}"
        report(3, "retrieval_results has v4 columns", passed, detail)
    finally:
        conn.close()


def test_4_episodes_v4_columns():
    """Verify episodes has input_tokens and output_tokens."""
    v4_columns = ["input_tokens", "output_tokens"]
    conn = get_conn()
    try:
        cols = get_table_columns(conn, "episodes")
        missing = [c for c in v4_columns if c not in cols]
        passed = len(missing) == 0
        detail = f"missing: {missing}" if missing else f"columns present: {v4_columns}"
        report(4, "episodes has v4 token columns", passed, detail)
    finally:
        conn.close()


def test_5_step_log_schema():
    """Verify step_log table schema."""
    expected_cols = [
        "episode_id",
        "step_number",
        "prompt_text",
        "response_text",
        "input_tokens",
        "output_tokens",
        "is_final",
        "created_at",
    ]
    conn = get_conn()
    try:
        cols = get_table_columns(conn, "step_log")
        missing = [c for c in expected_cols if c not in cols]
        passed = len(missing) == 0
        detail = (
            f"missing: {missing}"
            if missing
            else f"all {len(expected_cols)} columns present (+ id PK)"
        )
        report(5, "step_log table schema correct", passed, detail)
    finally:
        conn.close()


def test_6_foreign_key_enforcement():
    """Verify foreign key relationships are enforced."""
    conn = get_conn()
    # Verify FK pragma is on
    fk_status = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    if fk_status != 1:
        report(6, "Foreign key enforcement", False, "PRAGMA foreign_keys is OFF")
        conn.close()
        return

    tests_passed = 0
    tests_total = 3
    details = []

    # Test 1: Invalid task_id in episodes
    try:
        conn.execute(
            "INSERT INTO episodes (condition_id, task_id, task_order)"
            " VALUES (99, 'NONEXISTENT_TASK_999', 1)"
        )
        conn.commit()
        details.append("episodes FK on task_id NOT enforced")
        # Clean up
        conn.execute("DELETE FROM episodes WHERE task_id = 'NONEXISTENT_TASK_999'")
        conn.commit()
    except sqlite3.IntegrityError:
        tests_passed += 1
        conn.rollback()

    # Test 2: Invalid skill_id in skill_embeddings
    try:
        conn.execute(
            "INSERT INTO skill_embeddings"
            " (skill_id, embedding, model)"
            " VALUES ('NONEXISTENT_SKILL_999', X'00', 'test')"
        )
        conn.commit()
        details.append("skill_embeddings FK on skill_id NOT enforced")
        conn.execute("DELETE FROM skill_embeddings WHERE skill_id = 'NONEXISTENT_SKILL_999'")
        conn.commit()
    except sqlite3.IntegrityError:
        tests_passed += 1
        conn.rollback()

    # Test 3: Invalid preset_id in bandit_state
    try:
        conn.execute(
            "INSERT INTO bandit_state"
            " (condition_id, preset_id)"
            " VALUES (99, 'NONEXISTENT_PRESET_999')"
        )
        conn.commit()
        details.append("bandit_state FK on preset_id NOT enforced")
        conn.execute("DELETE FROM bandit_state WHERE preset_id = 'NONEXISTENT_PRESET_999'")
        conn.commit()
    except sqlite3.IntegrityError:
        tests_passed += 1
        conn.rollback()

    passed = tests_passed == tests_total
    if passed:
        detail = f"all {tests_total} FK violations correctly rejected"
    else:
        detail = f"{tests_passed}/{tests_total} passed; issues: {'; '.join(details)}"
    report(6, "Foreign key enforcement", passed, detail)
    conn.close()


def test_7_unique_constraints():
    """Verify unique constraints on step_log and episodes."""
    conn = get_conn()
    tests_passed = 0
    tests_total = 2
    details = []

    # We need a valid task_id and episode for testing
    task_row = conn.execute("SELECT task_id FROM tasks LIMIT 1").fetchone()
    if not task_row:
        report(7, "Unique constraints", False, "No tasks in DB to test against")
        conn.close()
        return

    task_id = task_row["task_id"]

    # Test 1: UNIQUE(episode_id, step_number) on step_log
    # Insert a temporary episode
    try:
        conn.execute(
            "INSERT INTO episodes (condition_id, task_id, task_order) VALUES (99, ?, 9999)",
            (task_id,),
        )
        conn.commit()
        ep_row = conn.execute(
            "SELECT episode_id FROM episodes"
            " WHERE condition_id=99 AND task_id=?"
            " AND task_order=9999",
            (task_id,),
        ).fetchone()
        ep_id = ep_row["episode_id"]

        # Insert first step
        conn.execute(
            "INSERT INTO step_log"
            " (episode_id, step_number,"
            " prompt_text, response_text, is_final)"
            " VALUES (?, 1, 'test', 'test', 0)",
            (ep_id,),
        )
        conn.commit()
        # Try duplicate
        try:
            conn.execute(
                "INSERT INTO step_log"
                " (episode_id, step_number,"
                " prompt_text, response_text, is_final)"
                " VALUES (?, 1, 'test2', 'test2', 0)",
                (ep_id,),
            )
            conn.commit()
            details.append("step_log UNIQUE(episode_id, step_number) NOT enforced")
        except sqlite3.IntegrityError:
            tests_passed += 1
            conn.rollback()

        # Clean up
        conn.execute("DELETE FROM step_log WHERE episode_id = ?", (ep_id,))
        conn.execute("DELETE FROM episodes WHERE episode_id = ?", (ep_id,))
        conn.commit()
    except sqlite3.IntegrityError as e:
        details.append(f"setup error for step_log test: {e}")
        conn.rollback()

    # Test 2: UNIQUE(condition_id, task_id) on episodes (but need unused combo)
    # Use condition_id=98 which shouldn't exist
    try:
        conn.execute(
            "INSERT INTO episodes (condition_id, task_id, task_order) VALUES (98, ?, 1)",
            (task_id,),
        )
        conn.commit()
        try:
            conn.execute(
                "INSERT INTO episodes (condition_id, task_id, task_order) VALUES (98, ?, 2)",
                (task_id,),
            )
            conn.commit()
            details.append("episodes UNIQUE(condition_id, task_id) NOT enforced")
            # Clean up both
            conn.execute("DELETE FROM episodes WHERE condition_id=98 AND task_id=?", (task_id,))
            conn.commit()
        except sqlite3.IntegrityError:
            tests_passed += 1
            conn.rollback()
            conn.execute("DELETE FROM episodes WHERE condition_id=98 AND task_id=?", (task_id,))
            conn.commit()
    except sqlite3.IntegrityError as e:
        # Might already exist if condition_id=98 was used before
        details.append(f"setup error for episodes uniqueness test: {e}")
        conn.rollback()

    passed = tests_passed == tests_total
    if passed:
        detail = f"all {tests_total} unique constraints enforced"
    else:
        detail = f"{tests_passed}/{tests_total} passed; issues: {'; '.join(details)}"
    report(7, "Unique constraints", passed, detail)
    conn.close()


def test_8_weight_presets():
    """Check that weight_presets has 12 presets with correct names."""
    expected_names = {
        "pure_relevance",
        "pure_recency",
        "pure_importance",
        "relevance_heavy",
        "recency_heavy",
        "importance_heavy",
        "recency_relevance",
        "importance_relevance",
        "recency_importance",
        "balanced",
        "diversity_seeker",
        "exploration",
    }
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT preset_id, w_recency, w_importance, w_relevance FROM weight_presets"
        ).fetchall()
        actual_names = {row["preset_id"] for row in rows}
        count = len(rows)

        issues = []
        if count != 12:
            issues.append(f"expected 12, got {count}")
        if actual_names != expected_names:
            missing = expected_names - actual_names
            extra = actual_names - expected_names
            if missing:
                issues.append(f"missing: {missing}")
            if extra:
                issues.append(f"extra: {extra}")

        # Check weights sum to ~1.0
        for row in rows:
            total = row["w_recency"] + row["w_importance"] + row["w_relevance"]
            if abs(total - 1.0) > 0.01:
                issues.append(f"{row['preset_id']} weights sum to {total:.3f}")

        passed = len(issues) == 0
        detail = (
            "; ".join(issues) if issues else "12 presets with correct names, all weights sum to 1.0"
        )
        report(8, "weight_presets has 12 correct presets", passed, detail)
    finally:
        conn.close()


def test_9_task_data():
    """Verify 300 tasks, 5 themes, proper difficulty distribution."""
    conn = get_conn()
    try:
        # Total count
        total = conn.execute("SELECT COUNT(*) as c FROM tasks").fetchone()["c"]

        # Theme distribution
        themes = conn.execute(
            "SELECT theme, COUNT(*) as c FROM tasks GROUP BY theme ORDER BY theme"
        ).fetchall()
        theme_names = [r["theme"] for r in themes]
        theme_counts = {r["theme"]: r["c"] for r in themes}

        # Difficulty distribution
        diffs = conn.execute(
            "SELECT difficulty, COUNT(*) as c FROM tasks GROUP BY difficulty ORDER BY difficulty"
        ).fetchall()
        diff_counts = {r["difficulty"]: r["c"] for r in diffs}

        issues = []
        if total != 300:
            issues.append(f"expected 300 tasks, got {total}")
        if len(themes) != 5:
            issues.append(f"expected 5 themes, got {len(themes)}: {theme_names}")

        # Check each theme has 60 tasks
        for theme, cnt in theme_counts.items():
            if cnt != 60:
                issues.append(f"theme '{theme}' has {cnt} tasks (expected 60)")

        # Check difficulty levels exist
        expected_diffs = {"easy", "medium", "hard"}
        actual_diffs = set(diff_counts.keys())
        if not expected_diffs.issubset(actual_diffs):
            issues.append(f"missing difficulties: {expected_diffs - actual_diffs}")

        passed = len(issues) == 0
        if passed:
            detail = f"300 tasks, 5 themes (60 each), difficulties: {dict(diff_counts)}"
        else:
            detail = "; ".join(issues)
        report(9, "Task data integrity", passed, detail)
    finally:
        conn.close()


def test_10_skill_data():
    """Verify 205 skills, 10 domains, 205 embeddings."""
    conn = get_conn()
    try:
        # Total skills
        total_skills = conn.execute("SELECT COUNT(*) as c FROM skills").fetchone()["c"]

        # Domains
        domains = conn.execute(
            "SELECT domain, COUNT(*) as c FROM skills GROUP BY domain ORDER BY domain"
        ).fetchall()
        domain_names = [r["domain"] for r in domains]
        domain_counts = {r["domain"]: r["c"] for r in domains}

        # Embeddings
        total_embeddings = conn.execute("SELECT COUNT(*) as c FROM skill_embeddings").fetchone()[
            "c"
        ]

        # Check all skills have embeddings
        orphan_skills = conn.execute(
            "SELECT COUNT(*) as c FROM skills s"
            " LEFT JOIN skill_embeddings se"
            " ON s.skill_id = se.skill_id"
            " WHERE se.skill_id IS NULL"
        ).fetchone()["c"]

        issues = []
        if total_skills != 205:
            issues.append(f"expected 205 skills, got {total_skills}")
        if len(domains) != 10:
            issues.append(f"expected 10 domains, got {len(domains)}: {domain_names}")
        if total_embeddings != 205:
            issues.append(f"expected 205 embeddings, got {total_embeddings}")
        if orphan_skills > 0:
            issues.append(f"{orphan_skills} skills without embeddings")

        # v3 has uneven domain sizes (base: 5-8 each, new: 35 each)
        # Just verify total adds up rather than checking per-domain counts
        domain_total = sum(domain_counts.values())
        if domain_total != total_skills:
            issues.append(
                f"domain counts sum to {domain_total}, but total skills is {total_skills}"
            )

        passed = len(issues) == 0
        if passed:
            detail = "205 skills, 10 domains, 205 embeddings, no orphans"
        else:
            detail = "; ".join(issues)
        report(10, "Skill data integrity", passed, detail)
    finally:
        conn.close()


def main():
    print(f"\nDatabase: {DB_PATH}")
    if not DB_PATH.exists():
        print(f"ERROR: Database file not found at {DB_PATH}")
        sys.exit(1)
    print(f"File size: {DB_PATH.stat().st_size:,} bytes")
    print()
    print("=" * 60)
    print("DATABASE SCHEMA & INTEGRITY TESTS")
    print("=" * 60)
    print()

    test_1_schema_version()
    test_2_all_tables_exist()
    test_3_retrieval_results_v4_columns()
    test_4_episodes_v4_columns()
    test_5_step_log_schema()
    test_6_foreign_key_enforcement()
    test_7_unique_constraints()
    test_8_weight_presets()
    test_9_task_data()
    test_10_skill_data()

    print()
    print("=" * 60)
    total = PASS + FAIL
    print(f"RESULTS: {PASS}/{total} passed, {FAIL}/{total} failed")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
