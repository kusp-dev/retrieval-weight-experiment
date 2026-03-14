#!/usr/bin/env python3
"""Load the curated skill library into the experiment database.

Reads data/skills/library.json and inserts into the `skills` table.
The `content` field is a concatenation of description, when_to_use,
steps, and expected_outcome — this is what FTS5 indexes for BM25 search.

Usage:
    python scripts/load_skills.py [--db experiment.db]
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.schema import init_db


def build_content(skill: dict) -> str:
    """Build the content field for FTS5 indexing.

    Combines all text fields into a single searchable blob.
    This is what BM25 and dense search operate on.
    """
    parts = [
        skill["description"],
        "",
        f"When to use: {skill['when_to_use']}",
        "",
        "Steps:",
    ]
    for step in skill["steps"]:
        parts.append(f"- {step}")

    parts.append("")
    parts.append(f"Expected outcome: {skill['expected_outcome']}")
    return "\n".join(parts)


def load_skills(
    db_path: str = "experiment_v3.db",
    library_path: str = "data/skills/library_v3.json",
):
    """Load skills from library.json into the database."""
    library = Path(library_path)
    if not library.exists():
        print(f"Error: {library_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(library) as f:
        skills = json.load(f)

    conn = init_db(db_path)

    loaded = 0
    skipped = 0
    for skill in skills:
        content = build_content(skill)
        try:
            conn.execute(
                """INSERT OR REPLACE INTO skills (skill_id, domain, title, content)
                   VALUES (?, ?, ?, ?)""",
                (skill["skill_id"], skill["domain"], skill["title"], content),
            )
            loaded += 1
        except Exception as e:
            print(f"Error loading {skill['skill_id']}: {e}", file=sys.stderr)
            skipped += 1

    conn.commit()

    # Verify
    count = conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
    fts_count = conn.execute("SELECT COUNT(*) FROM skills_fts").fetchone()[0]

    print(f"Loaded {loaded} skills ({skipped} skipped)")
    print(f"Database: {count} skills in table, {fts_count} in FTS5 index")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load skill library into experiment DB",
    )
    parser.add_argument(
        "--db",
        default="experiment_v3.db",
        help="Database path",
    )
    parser.add_argument(
        "--library",
        default="data/skills/library_v3.json",
        help="Library JSON path",
    )
    args = parser.parse_args()

    load_skills(args.db, args.library)
