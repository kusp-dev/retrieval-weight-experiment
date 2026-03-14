"""Tests for feedback parsing (regex + separated-question anchor embedding)."""

import numpy as np
import pytest

from src.embeddings.embedder import vector_to_blob
from src.experiment.feedback_parser import (
    RUBRIC_LIKERT,
    RUBRIC_QUALITATIVE,
    SeparatedAnchorParser,
    extract_qualitative_sections,
    parse_likert_feedback,
    parse_qualitative_feedback,
)

# ── Likert Regex Parser (Conditions 2-3) ──


class TestParseLikertFeedback:
    def test_parses_standard_format(self):
        response = """Here is my analysis of the task...

[RETRIEVAL EFFECTIVENESS]
Recency: 3
Importance: 4
Relevance: 5

Brief explanation (2-3 sentences):
- Relevance was most valuable because the topic matched perfectly.
- Recency was least valuable because the task doesn't depend on time.
- Skills with more implementation detail would have been helpful.
[/RETRIEVAL EFFECTIVENESS]"""

        parsed = parse_likert_feedback(response)
        assert parsed is not None
        assert parsed.recency == pytest.approx(0.5)  # (3-1)/4
        assert parsed.importance == pytest.approx(0.75)  # (4-1)/4
        assert parsed.relevance == pytest.approx(1.0)  # (5-1)/4
        assert parsed.parse_method == "likert"
        assert parsed.composite_reward == pytest.approx((0.5 + 0.75 + 1.0) / 3.0)

    def test_parses_angle_bracket_format(self):
        response = """[RETRIEVAL EFFECTIVENESS]
Recency: <2>
Importance: <5>
Relevance: <3>
Explanation goes here.
[/RETRIEVAL EFFECTIVENESS]"""

        parsed = parse_likert_feedback(response)
        assert parsed is not None
        assert parsed.recency == pytest.approx(0.25)  # (2-1)/4
        assert parsed.importance == pytest.approx(1.0)  # (5-1)/4
        assert parsed.relevance == pytest.approx(0.5)  # (3-1)/4

    def test_parses_slash_format(self):
        response = """[RETRIEVAL EFFECTIVENESS]
Recency: 4/5
Importance: 2/5
Relevance: 5/5
The retrieved skills were good.
[/RETRIEVAL EFFECTIVENESS]"""

        parsed = parse_likert_feedback(response)
        assert parsed is not None
        assert parsed.recency == pytest.approx(0.75)
        assert parsed.importance == pytest.approx(0.25)
        assert parsed.relevance == pytest.approx(1.0)

    def test_returns_none_for_missing_block(self):
        response = "I completed the task. No feedback block here."
        assert parse_likert_feedback(response) is None

    def test_returns_none_for_missing_dimension(self):
        response = """[RETRIEVAL EFFECTIVENESS]
Recency: 3
Importance: 4
Some text but no Relevance rating.
[/RETRIEVAL EFFECTIVENESS]"""
        assert parse_likert_feedback(response) is None

    def test_returns_none_for_out_of_range(self):
        response = """[RETRIEVAL EFFECTIVENESS]
Recency: 0
Importance: 6
Relevance: 3
[/RETRIEVAL EFFECTIVENESS]"""
        assert parse_likert_feedback(response) is None

    def test_case_insensitive(self):
        response = """[retrieval effectiveness]
recency: 2
importance: 3
relevance: 4
Good results overall.
[/retrieval effectiveness]"""

        parsed = parse_likert_feedback(response)
        assert parsed is not None
        assert parsed.recency == pytest.approx(0.25)

    def test_explanation_extracted(self):
        response = """[RETRIEVAL EFFECTIVENESS]
Recency: 3
Importance: 4
Relevance: 5
Relevance was the most valuable dimension.
Recency was the least useful.
[/RETRIEVAL EFFECTIVENESS]"""

        parsed = parse_likert_feedback(response)
        assert parsed is not None
        assert "most valuable" in parsed.explanation

    def test_ratings_tuple(self):
        response = """[RETRIEVAL EFFECTIVENESS]
Recency: 1
Importance: 3
Relevance: 5
Text.
[/RETRIEVAL EFFECTIVENESS]"""

        parsed = parse_likert_feedback(response)
        assert parsed is not None
        assert parsed.ratings_tuple == pytest.approx((0.0, 0.5, 1.0))

    def test_all_ones_gives_zero_reward(self):
        response = """[RETRIEVAL EFFECTIVENESS]
Recency: 1
Importance: 1
Relevance: 1
Everything was terrible.
[/RETRIEVAL EFFECTIVENESS]"""

        parsed = parse_likert_feedback(response)
        assert parsed is not None
        assert parsed.composite_reward == pytest.approx(0.0)

    def test_all_fives_gives_one_reward(self):
        response = """[RETRIEVAL EFFECTIVENESS]
Recency: 5
Importance: 5
Relevance: 5
Everything was perfect.
[/RETRIEVAL EFFECTIVENESS]"""

        parsed = parse_likert_feedback(response)
        assert parsed is not None
        assert parsed.composite_reward == pytest.approx(1.0)


# ── Separated-Question Anchor Parser (Condition 4) ──


def _make_anchor_blobs(rng, dims=("recency", "importance", "relevance")):
    """Create well-separated random anchor blobs for testing."""
    anchor_blobs = {}
    for dim in dims:
        blobs = {}
        for level in ["low", "mid", "high"]:
            vec = rng.standard_normal(1024).astype(np.float32)
            vec /= np.linalg.norm(vec)
            blobs[level] = vector_to_blob(vec)
        anchor_blobs[dim] = blobs
    return anchor_blobs


class TestSeparatedAnchorParser:
    @pytest.fixture
    def parser_with_random_anchors(self):
        rng = np.random.default_rng(42)
        parser = SeparatedAnchorParser()
        parser.initialize_from_blobs(_make_anchor_blobs(rng))
        return parser

    def test_initialize_from_blobs(self, parser_with_random_anchors):
        assert parser_with_random_anchors._initialized

    def test_infer_rating_returns_float(self, parser_with_random_anchors):
        rng = np.random.default_rng(99)
        emb = rng.standard_normal(1024).astype(np.float32)
        emb /= np.linalg.norm(emb)

        rating = parser_with_random_anchors.infer_rating(emb, "recency")
        assert isinstance(rating, float)
        assert 0.0 <= rating <= 1.0

    def test_infer_all_ratings_returns_all_dims(self, parser_with_random_anchors):
        rng = np.random.default_rng(99)
        section_embs = {}
        for dim in ["recency", "importance", "relevance"]:
            emb = rng.standard_normal(1024).astype(np.float32)
            emb /= np.linalg.norm(emb)
            section_embs[dim] = emb

        ratings = parser_with_random_anchors.infer_all_ratings(section_embs)
        assert set(ratings.keys()) == {"recency", "importance", "relevance"}
        for val in ratings.values():
            assert 0.0 <= val <= 1.0

    def test_high_anchor_gives_high_rating(self, parser_with_random_anchors):
        high_vec = parser_with_random_anchors._anchor_embeddings["recency"]["high"]
        rating = parser_with_random_anchors.infer_rating(high_vec, "recency")
        assert rating > 0.7

    def test_low_anchor_gives_low_rating(self, parser_with_random_anchors):
        low_vec = parser_with_random_anchors._anchor_embeddings["recency"]["low"]
        rating = parser_with_random_anchors.infer_rating(low_vec, "recency")
        assert rating < 0.3

    def test_mid_anchor_gives_mid_rating(self, parser_with_random_anchors):
        mid_vec = parser_with_random_anchors._anchor_embeddings["recency"]["mid"]
        rating = parser_with_random_anchors.infer_rating(mid_vec, "recency")
        assert 0.2 <= rating <= 0.8

    def test_raises_if_not_initialized(self):
        parser = SeparatedAnchorParser()
        emb = np.zeros(1024, dtype=np.float32)
        with pytest.raises(RuntimeError, match="initialize"):
            parser.infer_rating(emb, "recency")

    def test_get_anchor_blobs_roundtrip(self, parser_with_random_anchors):
        blobs = parser_with_random_anchors.get_anchor_blobs()
        new_parser = SeparatedAnchorParser()
        new_parser.initialize_from_blobs(blobs)

        rng = np.random.default_rng(123)
        emb = rng.standard_normal(1024).astype(np.float32)
        emb /= np.linalg.norm(emb)

        r1 = parser_with_random_anchors.infer_rating(emb, "importance")
        r2 = new_parser.infer_rating(emb, "importance")
        assert r1 == pytest.approx(r2)

    def test_temperature_affects_sharpness(self, parser_with_random_anchors):
        """Lower temperature should produce more extreme ratings."""
        high_vec = parser_with_random_anchors._anchor_embeddings["recency"]["high"]
        rating_sharp = parser_with_random_anchors.infer_rating(
            high_vec, "recency", temperature=0.01
        )
        rating_soft = parser_with_random_anchors.infer_rating(high_vec, "recency", temperature=0.5)
        # Sharp temperature should give rating closer to 1.0
        assert rating_sharp >= rating_soft


# ── Section Extraction ──


class TestExtractQualitativeSections:
    def test_extracts_all_sections(self):
        response = """Some output...
[RETRIEVAL EFFECTIVENESS]

[RECENCY EVALUATION]
Freshness was very important for this task. I needed the most recent skills.
[/RECENCY EVALUATION]

[IMPORTANCE EVALUATION]
Track record mattered somewhat. Proven skills helped but weren't essential.
[/IMPORTANCE EVALUATION]

[RELEVANCE EVALUATION]
Topic match was critical. Only skills about this exact subject were useful.
[/RELEVANCE EVALUATION]

[/RETRIEVAL EFFECTIVENESS]"""

        sections = extract_qualitative_sections(response)
        assert sections is not None
        assert len(sections) == 3
        assert "freshness" in sections["recency"].lower()
        assert "track record" in sections["importance"].lower()
        assert "topic match" in sections["relevance"].lower()

    def test_returns_none_for_missing_section(self):
        response = """[RETRIEVAL EFFECTIVENESS]
[RECENCY EVALUATION]
Some text here.
[/RECENCY EVALUATION]
[IMPORTANCE EVALUATION]
More text.
[/IMPORTANCE EVALUATION]
[/RETRIEVAL EFFECTIVENESS]"""
        # Missing RELEVANCE EVALUATION
        assert extract_qualitative_sections(response) is None

    def test_returns_none_for_too_short_section(self):
        response = """[RECENCY EVALUATION]
Too short.
[/RECENCY EVALUATION]
[IMPORTANCE EVALUATION]
This section has enough text to pass the minimum length check.
[/IMPORTANCE EVALUATION]
[RELEVANCE EVALUATION]
This section also has enough text to pass the length requirement.
[/RELEVANCE EVALUATION]"""
        assert extract_qualitative_sections(response) is None


# ── Full Qualitative Parse ──


class TestParseQualitativeFeedback:
    @pytest.fixture
    def parser(self):
        rng = np.random.default_rng(42)
        parser = SeparatedAnchorParser()
        parser.initialize_from_blobs(_make_anchor_blobs(rng))
        return parser

    def test_parses_full_qualitative_response(self, parser):
        response = """Task output goes here...

[RETRIEVAL EFFECTIVENESS]

[RECENCY EVALUATION]
Freshness was very important for this task. I needed the most recently
practiced skills because my workflow has been evolving rapidly.
[/RECENCY EVALUATION]

[IMPORTANCE EVALUATION]
Track record mattered somewhat. Having proven, reliable skills gave me
some confidence, but it wasn't the deciding factor for this task.
[/IMPORTANCE EVALUATION]

[RELEVANCE EVALUATION]
Topic match was absolutely critical for this task. Only skills that were
directly about the same subject area provided useful context.
[/RELEVANCE EVALUATION]

[/RETRIEVAL EFFECTIVENESS]"""

        rng = np.random.default_rng(55)
        section_embeddings = {}
        for dim in ["recency", "importance", "relevance"]:
            emb = rng.standard_normal(1024).astype(np.float32)
            emb /= np.linalg.norm(emb)
            section_embeddings[dim] = emb

        parsed = parse_qualitative_feedback(response, section_embeddings, parser)
        assert parsed is not None
        assert parsed.parse_method == "separated_anchor"
        assert 0.0 <= parsed.recency <= 1.0
        assert 0.0 <= parsed.importance <= 1.0
        assert 0.0 <= parsed.relevance <= 1.0
        assert 0.0 <= parsed.composite_reward <= 1.0
        assert "[RECENCY]" in parsed.explanation

    def test_returns_none_for_missing_sections(self, parser):
        response = "No sections here at all."
        section_embeddings = {
            dim: np.zeros(1024, dtype=np.float32) for dim in ["recency", "importance", "relevance"]
        }
        assert parse_qualitative_feedback(response, section_embeddings, parser) is None

    def test_returns_none_for_missing_embedding(self, parser):
        response = """[RECENCY EVALUATION]
Freshness was important for this particular task.
[/RECENCY EVALUATION]
[IMPORTANCE EVALUATION]
Track record mattered for this particular task.
[/IMPORTANCE EVALUATION]
[RELEVANCE EVALUATION]
Topic match was essential for this particular task.
[/RELEVANCE EVALUATION]"""

        # Missing relevance embedding
        section_embeddings = {
            "recency": np.zeros(1024, dtype=np.float32),
            "importance": np.zeros(1024, dtype=np.float32),
        }
        assert parse_qualitative_feedback(response, section_embeddings, parser) is None


# ── Rubric Constants ──


class TestRubricConstants:
    def test_likert_rubric_contains_rating_fields(self):
        assert "Recency:" in RUBRIC_LIKERT
        assert "Importance:" in RUBRIC_LIKERT
        assert "Relevance:" in RUBRIC_LIKERT
        assert "1-5" in RUBRIC_LIKERT

    def test_qualitative_rubric_has_separated_sections(self):
        assert "RECENCY EVALUATION" in RUBRIC_QUALITATIVE
        assert "IMPORTANCE EVALUATION" in RUBRIC_QUALITATIVE
        assert "RELEVANCE EVALUATION" in RUBRIC_QUALITATIVE

    def test_qualitative_rubric_asks_dimension_specific_questions(self):
        assert "freshness" in RUBRIC_QUALITATIVE.lower()
        assert "track record" in RUBRIC_QUALITATIVE.lower()
        assert "subject matter" in RUBRIC_QUALITATIVE.lower()
