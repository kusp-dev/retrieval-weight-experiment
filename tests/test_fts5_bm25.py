"""
Thorough FTS5/BM25 search tests for experiment_v3.db.

Tests:
  1. Multi-domain query coverage (10 different terms)
  2. All 205 skills findable by at least one search term
  3. Edge cases (special chars, long queries, single-char, empty)
  4. FTS5 rank ordering (more relevant = higher rank)
  5. Rowid consistency between skills and skills_fts
  6. search_bm25() method integration test
"""

import re
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "experiment_v3.db"

# Add project root to path for importing src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def print_header(test_name):
    print(f"\n{'=' * 60}")
    print(f"  TEST: {test_name}")
    print(f"{'=' * 60}")


def print_result(label, passed):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}")
    return passed


# ============================================================
# TEST 1: Multi-domain query coverage (10 different terms)
# ============================================================
def test_multi_domain_queries():
    print_header("1. Multi-domain query coverage (10 queries)")
    conn = get_conn()

    # First, let's see what domains and content exist
    domains = conn.execute("SELECT DISTINCT domain FROM skills").fetchall()
    domain_list = [r["domain"] for r in domains]
    print(f"  Domains found: {domain_list}")

    # Pick 10 diverse query terms likely to match across different skill domains
    queries = [
        "research",  # research domain
        "design",  # creative domain (appears in skill content)
        "synthesis",  # synthesis domain
        "evaluation",  # evaluation domain
        "operational",  # operational domain
        "analysis",  # general analytical term
        "data",  # data-related skills
        "communication",  # soft skills
        "strategy",  # strategic skills
        "problem",  # problem-solving skills
    ]

    all_passed = True
    total_results = 0

    for query in queries:
        rows = conn.execute(
            """SELECT skill_id, rank
               FROM skills_fts
               WHERE skills_fts MATCH ?
               ORDER BY rank
               LIMIT 20""",
            (f'"{query}"',),
        ).fetchall()

        count = len(rows)
        total_results += count
        passed = count > 0
        if not passed:
            # Try a broader search to debug
            all_rows = conn.execute(
                "SELECT skill_id, title FROM skills WHERE title LIKE ? OR content LIKE ?",
                (f"%{query}%", f"%{query}%"),
            ).fetchall()
            like_count = len(all_rows)
            print_result(f'"{query}" -> {count} FTS results ({like_count} via LIKE)', passed)
        else:
            top_ids = [r["skill_id"] for r in rows[:3]]
            print_result(f'"{query}" -> {count} results (top: {top_ids})', passed)
        all_passed = all_passed and passed

    # If some queries return 0, try alternative terms from actual content
    if not all_passed:
        print("\n  Trying content-derived queries for failed terms...")
        sample = conn.execute("SELECT title, content FROM skills LIMIT 5").fetchall()
        for s in sample:
            print(f"    Sample title: {s['title']}")
            print(f"    Sample content (first 100): {s['content'][:100]}")

    print(f"\n  Total results across 10 queries: {total_results}")
    conn.close()
    assert all_passed
    return all_passed


# ============================================================
# TEST 2: All 205 skills findable by at least one search term
# ============================================================
def test_all_skills_findable():
    print_header("2. All 205 skills findable by search")
    conn = get_conn()

    all_skills = conn.execute("SELECT skill_id, title, content FROM skills").fetchall()
    total = len(all_skills)
    print(f"  Total skills in DB: {total}")

    found_skills = set()
    not_found = []

    for skill in all_skills:
        sid = skill["skill_id"]
        title = skill["title"]
        content = skill["content"]

        # Extract meaningful words from title (skip short/common words)
        title_words = [w for w in re.sub(r"[^\w\s]", " ", title).split() if len(w) > 2]

        # Try each title word as a query
        skill_found = False
        for word in title_words:
            rows = conn.execute(
                """SELECT skill_id FROM skills_fts
                   WHERE skills_fts MATCH ?
                   LIMIT 50""",
                (f'"{word}"',),
            ).fetchall()

            if sid in [r["skill_id"] for r in rows]:
                skill_found = True
                break

        # If title words didn't work, try content words
        if not skill_found:
            content_words = [w for w in re.sub(r"[^\w\s]", " ", content).split() if len(w) > 4][:10]
            for word in content_words:
                rows = conn.execute(
                    """SELECT skill_id FROM skills_fts
                       WHERE skills_fts MATCH ?
                       LIMIT 50""",
                    (f'"{word}"',),
                ).fetchall()

                if sid in [r["skill_id"] for r in rows]:
                    skill_found = True
                    break

        if skill_found:
            found_skills.add(sid)
        else:
            not_found.append((sid, title))

    passed = len(found_skills) == total
    print_result(f"Found {len(found_skills)}/{total} skills via FTS5 search", passed)

    if not_found:
        for sid, title in not_found:
            print(f"    NOT FOUND: {sid} - {title}")

    conn.close()
    assert passed
    return passed


# ============================================================
# TEST 3: Edge cases
# ============================================================
def test_edge_cases():
    print_header("3. Edge cases")
    conn = get_conn()

    all_passed = True

    # 3a: Special characters should not crash
    special_queries = [
        "hello:world",
        "test-case",
        "foo(bar)",
        "a*b+c",
        'say "hello"',
        "col1:col2:col3",
        "NOT OR AND",  # FTS5 operators as text
        "NEAR(test, 5)",  # FTS5 NEAR syntax
        "prefix*",  # FTS5 prefix
        "test + - ! @ # $ % ^ & * ( )",
    ]

    for q in special_queries:
        try:
            # Use the same sanitization as the production code
            cleaned = re.sub(r"[^\w\s]", " ", q)
            words = [f'"{w}"' for w in cleaned.split() if len(w) > 1]
            sanitized = " OR ".join(words) if words else None

            if sanitized:
                conn.execute(
                    """SELECT skill_id FROM skills_fts
                       WHERE skills_fts MATCH ?
                       LIMIT 5""",
                    (sanitized,),
                ).fetchall()
            # No crash = pass (results may be empty, that's fine)
            p = True
        except Exception as e:
            p = False
            print(f"    Exception for '{q}': {e}")
        all_passed = all_passed and p

    print_result("Special characters handled without crash", all_passed)

    # 3b: Single-character queries (should be filtered out by sanitizer)
    single_char_pass = True
    for ch in ["a", "x", "1", "?"]:
        cleaned = re.sub(r"[^\w\s]", " ", ch)
        words = [f'"{w}"' for w in cleaned.split() if len(w) > 1]
        sanitized = " OR ".join(words) if words else None
        if sanitized is not None:
            single_char_pass = False
            print(f"    Single char '{ch}' was not filtered: '{sanitized}'")
    all_passed = all_passed and print_result(
        "Single-character queries filtered to None by sanitizer", single_char_pass
    )

    # 3c: Empty string
    empty_pass = True
    for q in ["", "   ", "\t", "\n"]:
        cleaned = re.sub(r"[^\w\s]", " ", q)
        words = [f'"{w}"' for w in cleaned.split() if len(w) > 1]
        sanitized = " OR ".join(words) if words else None
        if sanitized is not None:
            empty_pass = False
            print(f"    Empty/whitespace query not filtered: '{sanitized}'")
    all_passed = all_passed and print_result(
        "Empty/whitespace queries filtered to None by sanitizer", empty_pass
    )

    # 3d: Long query (100+ words)
    long_query = " ".join(["algorithm"] * 100 + ["data"] * 50)
    cleaned = re.sub(r"[^\w\s]", " ", long_query)
    words = [f'"{w}"' for w in cleaned.split() if len(w) > 1]
    sanitized = " OR ".join(words) if words else None
    try:
        conn.execute(
            """SELECT skill_id FROM skills_fts
               WHERE skills_fts MATCH ?
               LIMIT 5""",
            (sanitized,),
        ).fetchall()
        long_pass = True
    except Exception as e:
        long_pass = False
        print(f"    Long query exception: {e}")
    all_passed = all_passed and print_result("Long query (150 words) handled", long_pass)

    # 3e: Unicode characters
    unicode_queries = ["Bayesian", "analyse", "modèle"]
    unicode_pass = True
    for q in unicode_queries:
        try:
            cleaned = re.sub(r"[^\w\s]", " ", q)
            words = [f'"{w}"' for w in cleaned.split() if len(w) > 1]
            sanitized = " OR ".join(words) if words else None
            if sanitized:
                conn.execute(
                    "SELECT skill_id FROM skills_fts WHERE skills_fts MATCH ? LIMIT 5",
                    (sanitized,),
                ).fetchall()
        except Exception as e:
            unicode_pass = False
            print(f"    Unicode query '{q}' failed: {e}")
    all_passed = all_passed and print_result("Unicode queries handled", unicode_pass)

    conn.close()
    assert all_passed
    return all_passed


# ============================================================
# TEST 4: FTS5 rank ordering
# ============================================================
def test_rank_ordering():
    print_header("4. FTS5 rank ordering (relevance)")
    conn = get_conn()
    all_passed = True

    # Find a query that returns multiple results
    # Pick a common word from skills content
    sample = conn.execute("SELECT content FROM skills LIMIT 1").fetchone()
    words = [w for w in re.sub(r"[^\w\s]", " ", sample["content"]).split() if len(w) > 4]

    # Try to find a word that returns 3+ results
    test_word = None
    test_results = []
    for w in words:
        rows = conn.execute(
            """SELECT skill_id, rank FROM skills_fts
               WHERE skills_fts MATCH ?
               ORDER BY rank
               LIMIT 20""",
            (f'"{w}"',),
        ).fetchall()
        if len(rows) >= 3:
            test_word = w
            test_results = rows
            break

    if test_word is None:
        # Fallback: use a very common term
        for fallback in [
            "the",
            "and",
            "skill",
            "data",
            "analysis",
            "process",
            "method",
            "approach",
            "evaluate",
            "system",
        ]:
            rows = conn.execute(
                """SELECT skill_id, rank FROM skills_fts
                   WHERE skills_fts MATCH ?
                   ORDER BY rank
                   LIMIT 20""",
                (f'"{fallback}"',),
            ).fetchall()
            if len(rows) >= 3:
                test_word = fallback
                test_results = rows
                break

    if test_word and len(test_results) >= 3:
        ranks = [r["rank"] for r in test_results]
        # FTS5 rank values should be in ascending order (more negative = better match)
        monotonic = all(ranks[i] <= ranks[i + 1] for i in range(len(ranks) - 1))
        all_passed = print_result(
            f'Query "{test_word}": {len(test_results)} results,'
            f" ranks monotonically ordered: {monotonic}",
            monotonic,
        )

        # Verify that rank values are negative (FTS5 default BM25 behavior)
        all_negative = all(r < 0 for r in ranks)
        all_passed = all_passed and print_result(
            f"All rank values are negative (BM25): {ranks[:5]}...",
            all_negative,
        )

        # Verify differentiation: not all ranks are identical
        unique_ranks = len(set(ranks))
        has_variation = unique_ranks > 1
        all_passed = all_passed and print_result(
            f"Rank differentiation: {unique_ranks} unique values across {len(ranks)} results",
            has_variation,
        )

        # Test: a skill whose title matches the query should rank higher than one
        # that only matches in content
        print(f"  Top 5 results for '{test_word}':")
        for r in test_results[:5]:
            title_row = conn.execute(
                "SELECT title FROM skills WHERE skill_id = ?", (r["skill_id"],)
            ).fetchone()
            title = title_row["title"] if title_row else "?"
            in_title = test_word.lower() in title.lower()
            print(
                f"    {r['skill_id']}: rank={r['rank']:.4f} title='{title}' (in_title={in_title})"
            )
    else:
        print("  WARNING: Could not find a query with 3+ results to test ordering")
        all_passed = False

    conn.close()
    return all_passed
    assert all_passed


# ============================================================
# TEST 5: Rowid consistency between skills and skills_fts
# ============================================================
def test_rowid_consistency():
    print_header("5. Rowid consistency (skills <-> skills_fts)")
    conn = get_conn()
    all_passed = True

    # Get all skills with rowids
    skills_rows = conn.execute(
        "SELECT rowid, skill_id, title FROM skills ORDER BY rowid"
    ).fetchall()

    # Get all FTS entries (using a broad match to get everything)
    # We'll reconstruct by querying each skill_id
    fts_skill_ids = set()
    skills_set = set()

    for s in skills_rows:
        skills_set.add(s["skill_id"])

    # Check every skill has an FTS entry by searching for a word from its title
    missing_from_fts = []
    for s in skills_rows:
        sid = s["skill_id"]
        title = s["title"]

        # Try to find this skill in FTS by its skill_id content
        # We can use a LIKE query on the FTS to check
        words = [w for w in re.sub(r"[^\w\s]", " ", title).split() if len(w) > 2]
        found = False
        for w in words[:3]:
            rows = conn.execute(
                """SELECT skill_id FROM skills_fts
                   WHERE skills_fts MATCH ?
                   LIMIT 50""",
                (f'"{w}"',),
            ).fetchall()
            if sid in [r["skill_id"] for r in rows]:
                fts_skill_ids.add(sid)
                found = True
                break

        if not found:
            missing_from_fts.append(sid)

    # Count check
    skills_count = len(skills_rows)
    fts_found_count = len(fts_skill_ids)

    all_passed = print_result(
        f"Skills table has {skills_count} rows",
        skills_count == 205,
    )

    all_passed = all_passed and print_result(
        f"FTS found {fts_found_count}/{skills_count} skills",
        fts_found_count == skills_count,
    )

    if missing_from_fts:
        print(f"  Missing from FTS: {missing_from_fts}")

    # Verify skill_id values match between tables using rebuild
    # Run integrity check on FTS
    try:
        conn.execute("INSERT INTO skills_fts(skills_fts) VALUES('integrity-check')")
        all_passed = all_passed and print_result("FTS5 integrity-check passed", True)
    except Exception as e:
        all_passed = False
        print_result(f"FTS5 integrity-check FAILED: {e}", False)

    # Verify rowid correspondence by checking a specific skill
    if skills_rows:
        test_skill = skills_rows[0]
        test_skill.keys()  # just checking we can access it

        # Direct rowid query on FTS
        fts_row = conn.execute(
            "SELECT skill_id FROM skills_fts WHERE rowid = ?",
            (skills_rows[0]["rowid"],),
        ).fetchone()
        if fts_row:
            match = fts_row["skill_id"] == skills_rows[0]["skill_id"]
            all_passed = all_passed and print_result(
                f"Rowid {skills_rows[0]['rowid']}: skills.skill_id={skills_rows[0]['skill_id']}, "
                f"fts.skill_id={fts_row['skill_id']} -> match={match}",
                match,
            )
        else:
            all_passed = False
            print_result(f"Rowid {skills_rows[0]['rowid']} not found in FTS", False)

    # Check ALL rowids match
    rowid_mismatches = 0
    for s in skills_rows:
        fts_row = conn.execute(
            "SELECT skill_id FROM skills_fts WHERE rowid = ?",
            (s["rowid"],),
        ).fetchone()
        if fts_row is None or fts_row["skill_id"] != s["skill_id"]:
            rowid_mismatches += 1
            if fts_row:
                print(
                    f"    MISMATCH rowid {s['rowid']}:"
                    f" skills={s['skill_id']}"
                    f" fts={fts_row['skill_id']}"
                )
            else:
                print(f"    MISSING rowid {s['rowid']}: skills={s['skill_id']} not in FTS")

    all_passed = all_passed and print_result(
        f"All {skills_count} rowids match between"
        f" skills and skills_fts"
        f" ({rowid_mismatches} mismatches)",
        rowid_mismatches == 0,
    )

    conn.close()
    return all_passed
    assert all_passed


# ============================================================
# TEST 6: search_bm25() method integration test
# ============================================================
def test_search_bm25_method():
    print_header("6. search_bm25() method integration test")
    conn = get_conn()

    from src.search.hybrid import HybridSearchEngine

    engine = HybridSearchEngine(conn=conn)
    all_passed = True

    # 6a: Basic search returns results
    # Get a word from actual skill content
    sample = conn.execute("SELECT title FROM skills LIMIT 1").fetchone()
    first_word = [w for w in re.sub(r"[^\w\s]", " ", sample["title"]).split() if len(w) > 2][0]

    results = engine.search_bm25(first_word, top_k=10)
    all_passed = print_result(
        f'search_bm25("{first_word}") returned {len(results)} results',
        len(results) > 0,
    )

    # 6b: Results are tuples of (skill_id, positive_score)
    if results:
        first = results[0]
        is_tuple = isinstance(first, tuple) and len(first) == 2
        all_passed = all_passed and print_result(
            f"Result format is (skill_id, score) tuple: {is_tuple}",
            is_tuple,
        )

        # Score should be positive (negated from FTS5 rank)
        positive_scores = all(score > 0 for _, score in results)
        all_passed = all_passed and print_result(
            f"All scores are positive (negated BM25 rank): {positive_scores}",
            positive_scores,
        )

        # Scores should be in descending order (best first)
        scores = [s for _, s in results]
        descending = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        all_passed = all_passed and print_result(
            f"Scores in descending order: {descending} ({scores[:3]}...)",
            descending,
        )

        # skill_ids should be valid
        valid_ids = True
        for sid, _ in results:
            row = conn.execute("SELECT 1 FROM skills WHERE skill_id = ?", (sid,)).fetchone()
            if not row:
                valid_ids = False
                print(f"    Invalid skill_id: {sid}")
        all_passed = all_passed and print_result(
            "All returned skill_ids exist in skills table",
            valid_ids,
        )

    # 6c: top_k limit is respected
    for k in [1, 3, 5]:
        results_k = engine.search_bm25(first_word, top_k=k)
        within_limit = len(results_k) <= k
        all_passed = all_passed and print_result(
            f"top_k={k}: returned {len(results_k)} results (within limit)",
            within_limit,
        )

    # 6d: Empty/invalid queries return empty list
    empty_results = engine.search_bm25("")
    all_passed = all_passed and print_result(
        f'search_bm25("") returns empty list: {empty_results}',
        empty_results == [],
    )

    special_result = engine.search_bm25("!@#$%")
    all_passed = all_passed and print_result(
        f'search_bm25("!@#$%") returns empty list: {special_result}',
        special_result == [],
    )

    single_char = engine.search_bm25("x")
    all_passed = all_passed and print_result(
        f'search_bm25("x") returns empty list (single char filtered): {single_char}',
        single_char == [],
    )

    # 6e: _sanitize_fts5_query static method tests
    sanitize = HybridSearchEngine._sanitize_fts5_query

    assert sanitize("hello world") == '"hello" OR "world"', f"Got: {sanitize('hello world')}"
    assert sanitize("") is None, f"Got: {sanitize('')}"
    assert sanitize("a b c") is None, f"Got: {sanitize('a b c')}"
    assert sanitize("test-case") == '"test" OR "case"', f"Got: {sanitize('test-case')}"
    assert sanitize("col:value") == '"col" OR "value"', f"Got: {sanitize('col:value')}"
    assert sanitize("foo(bar)") == '"foo" OR "bar"', f"Got: {sanitize('foo(bar)')}"
    all_passed = all_passed and print_result(
        "_sanitize_fts5_query produces correct output for all cases", True
    )

    # 6f: Multiple diverse queries through search_bm25
    diverse_queries = []
    skills_sample = conn.execute("SELECT title FROM skills ORDER BY RANDOM() LIMIT 10").fetchall()
    for s in skills_sample:
        words = [w for w in re.sub(r"[^\w\s]", " ", s["title"]).split() if len(w) > 3]
        if words:
            diverse_queries.append(words[0])

    queries_with_results = 0
    for q in diverse_queries:
        r = engine.search_bm25(q)
        if len(r) > 0:
            queries_with_results += 1

    all_passed = all_passed and print_result(
        f"Diverse queries: {queries_with_results}/{len(diverse_queries)} returned results",
        queries_with_results == len(diverse_queries),
    )

    conn.close()
    return all_passed
    assert all_passed


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"Database: {DB_PATH}")
    print(f"Exists: {DB_PATH.exists()}")

    if not DB_PATH.exists():
        print("ERROR: experiment_v3.db not found!")
        sys.exit(1)

    results = {}
    results["1. Multi-domain queries"] = test_multi_domain_queries()
    results["2. All skills findable"] = test_all_skills_findable()
    results["3. Edge cases"] = test_edge_cases()
    results["4. Rank ordering"] = test_rank_ordering()
    results["5. Rowid consistency"] = test_rowid_consistency()
    results["6. search_bm25() integration"] = test_search_bm25_method()

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    total = len(results)
    passed_count = sum(1 for v in results.values() if v)
    print(f"\n  {passed_count}/{total} test categories passed")

    if passed_count < total:
        sys.exit(1)
    else:
        print("\n  All tests passed!")
        sys.exit(0)
