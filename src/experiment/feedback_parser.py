"""
Feedback parsing for all experimental conditions.

Two parsers:
  1. Regex parser (Conditions 2-3): Extract Likert ratings from [RETRIEVAL EFFECTIVENESS] blocks
  2. Separated-question anchor parser (Condition 4): 3 per-dimension sections, each compared
     to 3 anchors (low/mid/high) via softmax-weighted cosine similarity

Design ref: SYSTEM_DESIGN.md §4
"""

import re
from dataclasses import dataclass

import numpy as np

from src.embeddings.embedder import blob_to_vector, cosine_similarity, vector_to_blob

# ── Rubric text (appended to agent prompts) ──

RUBRIC_LIKERT = """RETRIEVAL FEEDBACK — MANDATORY (if skills were provided above):
[RETRIEVAL EFFECTIVENESS]
You were provided with skills retrieved using three scoring dimensions:
- RECENCY: How recent/fresh the skill is
- IMPORTANCE: How important it was previously flagged
- RELEVANCE: How well it semantically matches your current task

Rate how useful EACH dimension was for helping you complete this specific task.
You MUST use exactly this format (one rating per line, number only):

Recency: <single number 1-5, where 1=irrelevant to task, 5=essential>
Importance: <single number 1-5>
Relevance: <single number 1-5>

Brief explanation (2-3 sentences):
- Which dimension was most valuable and why?
- Which dimension was least valuable and why?
- What KIND of skills would have been even more helpful?
[/RETRIEVAL EFFECTIVENESS]"""

RUBRIC_QUALITATIVE = """RETRIEVAL FEEDBACK — MANDATORY (if skills were provided above):
[RETRIEVAL EFFECTIVENESS]
Evaluate how useful each retrieval dimension was for this specific task.
You MUST respond inside each tagged section below with 2-3 sentences.
Keep the [TAG] and [/TAG] markers exactly as shown.

[RECENCY EVALUATION]
How important was the freshness/timing of the retrieved skills for this task?
Were recently-used skills more helpful than older ones, or did timing not matter?
[/RECENCY EVALUATION]

[IMPORTANCE EVALUATION]
How important was the proven track record of the retrieved skills for this task?
Did you need battle-tested, reliable skills, or would untested ones have worked just as well?
[/IMPORTANCE EVALUATION]

[RELEVANCE EVALUATION]
How important was the subject matter match of the retrieved skills for this task?
Were topically aligned skills essential, or could off-topic skills have helped equally?
[/RELEVANCE EVALUATION]
[/RETRIEVAL EFFECTIVENESS]"""


# ── Data classes ──


@dataclass
class ParsedFeedback:
    """Result of parsing feedback from an LLM response."""

    recency: float  # Normalized 0.0-1.0
    importance: float  # Normalized 0.0-1.0
    relevance: float  # Normalized 0.0-1.0
    explanation: str  # Natural language text
    raw_block: str  # Full [RETRIEVAL EFFECTIVENESS] block
    parse_method: str  # "likert" or "separated_anchor"
    composite_reward: float  # (r + i + v) / 3 — bandit reward signal

    @property
    def ratings_tuple(self) -> tuple[float, float, float]:
        return (self.recency, self.importance, self.relevance)


# ══════════════════════════════════════════════════════════════
# 1. REGEX PARSER (Conditions 2 and 3)
# ══════════════════════════════════════════════════════════════


def _extract_feedback_block(response: str) -> str | None:
    """Extract the retrieval effectiveness section from the response.

    Tries strict tag match first, then falls back to locating a section
    by heading or keyword (MiniMax M2.5 often uses markdown headings
    instead of reproducing the [RETRIEVAL EFFECTIVENESS] tags).
    """
    # Strict: tagged block
    match = re.search(
        r"\[RETRIEVAL EFFECTIVENESS\](.*?)\[/RETRIEVAL EFFECTIVENESS\]",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1)

    # Lenient: everything after a "retrieval effectiveness/feedback" heading/mention
    match = re.search(
        r"(?:#{1,3}\s*)?retrieval\s+(?:effectiveness|feedback).*?\n(.*)",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1)

    # Last resort: scan the last 1500 chars for all three dimension ratings
    # (catches garbled headings like "RETRIEVAL FELEMENT")
    tail = response[-1500:]
    has_all = all(
        re.search(rf"{dim}[:\s]+\d", tail, re.IGNORECASE)
        for dim in ["recency", "importance", "relevance"]
    )
    if has_all:
        return tail

    return None


def _extract_dimension_rating(text: str, dimension: str) -> int | None:
    """Extract a 1-5 rating for a dimension, supporting multiple formats.

    Handles: "Recency: 4", "Recency: <4>", "Recency: 4/5",
    "| **Recency** | 4/5 |", "**Recency:** 4", "Recency — 3/5"
    """
    # Strip markdown bold markers for cleaner matching
    cleaned = text.replace("*", "")
    patterns = [
        # "Recency: 4", "Recency: <4>", "Recency: 4/5"
        rf"{dimension}[:\s]+<?(\d)\s*[>/]?\s*5?",
        # "| Recency | 4/5 |" or "| Recency | 4 |"
        rf"\|\s*{dimension}\s*\|\s*(\d)\s*/?\s*5?\s*\|",
        # "Recency — 4/5" or "Recency - 4"
        rf"{dimension}\s*[\-–—]\s*(\d)\s*/?\s*5?",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return score
    return None


def parse_likert_feedback(response: str) -> ParsedFeedback | None:
    """
    Extract dimension ratings from the retrieval effectiveness section.

    Tries strict [RETRIEVAL EFFECTIVENESS] tags first, then falls back to
    detecting ratings from markdown-formatted sections (common with MiniMax M2.5).

    Returns None if parsing fails. Failed parses do NOT update the bandit.
    Expected parse rate: >65% (TIER 0 achieved 64.8% with messier prompts).
    """
    block = _extract_feedback_block(response)

    # If block extraction failed, try scanning the full response as fallback
    if block is None:
        all_found = all(
            _extract_dimension_rating(response, d) is not None
            for d in ["recency", "importance", "relevance"]
        )
        if all_found:
            block = response
        else:
            return None

    dimensions: dict[str, float | None] = {}

    for dim in ["recency", "importance", "relevance"]:
        score = _extract_dimension_rating(block, dim)
        if score is not None:
            dimensions[dim] = (score - 1) / 4.0  # Normalize 1-5 → 0.0-1.0
        else:
            dimensions[dim] = None

    # Strict: all three must parse
    if any(v is None for v in dimensions.values()):
        return None

    # Extract explanation text (lines that aren't dimension ratings)
    explanation_lines = []
    for line in block.split("\n"):
        stripped = line.strip()
        if stripped and not any(
            re.search(rf"\b{d}\b.*\d\s*/?\s*5?", stripped, re.IGNORECASE)
            for d in ["recency", "importance", "relevance"]
        ):
            explanation_lines.append(stripped)

    rec = dimensions["recency"]
    imp = dimensions["importance"]
    rel = dimensions["relevance"]
    assert rec is not None and imp is not None and rel is not None  # guarded above
    composite = (rec + imp + rel) / 3.0

    return ParsedFeedback(
        recency=rec,
        importance=imp,
        relevance=rel,
        explanation="\n".join(explanation_lines).strip(),
        raw_block=block,
        parse_method="likert",
        composite_reward=composite,
    )


# ══════════════════════════════════════════════════════════════
# 2. SEPARATED-QUESTION ANCHOR PARSER (Condition 4)
# ══════════════════════════════════════════════════════════════

# ── 3 anchors per dimension (low / mid / high) ──
# Each set uses identical sentence structure; only dimension content differs.
# Anchors are compared ONLY to their own dimension's feedback section,
# eliminating cross-dimension crosstalk entirely.
#
# Calibrated 2026-03-08, anchors redesigned 2026-03-09 (qualitatively different
# scenarios instead of degree variations). Extended calibration: low/mid/high
# means = 0.18/0.43/0.83 across 9 synthetic texts per dimension.
# Spread 0.57-0.72 (vs 0.03-0.07 with old degree-based anchors).

ANCHOR_TEXTS = {
    "recency": {
        "low": (
            "I already knew the current state of the art for this task. "
            "The retrieved skills, whether old or new, were redundant — "
            "freshness made no difference because the fundamentals haven't changed."
        ),
        "mid": (
            "I used a standard approach that has been around for a while. "
            "Recently used skills gave me useful context for incremental "
            "improvements, but I could have solved it with older knowledge alone."
        ),
        "high": (
            "This task required cutting-edge techniques that have only "
            "emerged recently. Only the most recently practiced skills had "
            "methods I wasn't already familiar with. Older skills were obsolete."
        ),
    },
    "importance": {
        "low": (
            "I solved this task from first principles and experimentation. "
            "Whether the skills had been successfully applied before didn't "
            "affect my approach — I would have tried any method regardless of track record."
        ),
        "mid": (
            "Proven skills gave me more confidence in my approach, but I "
            "would have still explored unproven alternatives if needed. "
            "Track record was a useful signal but not decisive."
        ),
        "high": (
            "This task was too high-stakes to experiment with unproven methods. "
            "I relied exclusively on skills with demonstrated success because "
            "failure would have been costly. Untested approaches were unacceptable."
        ),
    },
    "relevance": {
        "low": (
            "I needed creative cross-domain thinking for this task. "
            "Skills from unrelated fields sparked novel ideas just as much "
            "as on-topic skills. Semantic match to the task description was irrelevant."
        ),
        "mid": (
            "I needed a mix of deep domain knowledge and lateral thinking. "
            "On-topic skills provided essential structure, but insights from "
            "adjacent fields also contributed meaningfully to my solution."
        ),
        "high": (
            "This task demanded deep, specialized domain expertise. Only skills "
            "directly about the exact topic were useful — off-topic skills "
            "were noise that distracted from the precise technical requirements."
        ),
    },
}

# Section tag names used in the qualitative rubric
_SECTION_TAGS = {
    "recency": "RECENCY EVALUATION",
    "importance": "IMPORTANCE EVALUATION",
    "relevance": "RELEVANCE EVALUATION",
}

# Softmax temperature for anchor scoring. Lower = more decisive.
# t=0.05 gave best average correlation (0.907) in calibration.
SOFTMAX_TEMPERATURE = 0.05


class SeparatedAnchorParser:
    """Infer per-dimension ratings from separated qualitative feedback sections.

    The Condition 4 rubric asks 3 separate questions (one per dimension).
    Each response section is embedded independently and compared to 3 anchors
    (low/mid/high) for that dimension only. Softmax-weighted interpolation
    produces a 0.0-1.0 rating.

    This eliminates cross-dimension crosstalk because each embedding is only
    compared to its own dimension's anchors.
    """

    def __init__(self, embedder=None):
        self._embedder = embedder
        self._anchor_embeddings: dict[str, dict[str, np.ndarray]] = {}
        self._initialized = False

    def initialize(self, embedder=None) -> None:
        """Pre-compute anchor embeddings. Call once before experiment starts."""
        if embedder is not None:
            self._embedder = embedder
        if self._embedder is None:
            raise ValueError("Embedder required for anchor initialization")

        for dim in ["recency", "importance", "relevance"]:
            self._anchor_embeddings[dim] = {
                level: self._embedder.embed(ANCHOR_TEXTS[dim][level])
                for level in ["low", "mid", "high"]
            }
        self._initialized = True

    def initialize_from_blobs(self, anchor_blobs: dict[str, dict[str, bytes]]) -> None:
        """Initialize from pre-computed anchor embedding blobs (for testing/caching)."""
        for dim in ["recency", "importance", "relevance"]:
            self._anchor_embeddings[dim] = {
                level: blob_to_vector(anchor_blobs[dim][level]) for level in ["low", "mid", "high"]
            }
        self._initialized = True

    def infer_rating(
        self,
        feedback_embedding: np.ndarray,
        dimension: str,
        temperature: float = SOFTMAX_TEMPERATURE,
    ) -> float:
        """
        Infer a single dimension's rating from its section's embedding.

        Uses softmax-weighted interpolation over 3 anchors:
          low=0.0, mid=0.5, high=1.0

        Args:
            feedback_embedding: Embedding of the dimension-specific feedback section.
            dimension: One of 'recency', 'importance', 'relevance'.
            temperature: Softmax temperature. Lower = sharper discrimination.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before inferring ratings")

        anchors = self._anchor_embeddings[dimension]
        sims = {
            level: cosine_similarity(feedback_embedding, anchors[level])
            for level in ["low", "mid", "high"]
        }

        # Softmax over similarities
        max_sim = max(sims.values())
        exp_sims = {k: np.exp((v - max_sim) / temperature) for k, v in sims.items()}
        total = sum(exp_sims.values())
        weights = {k: v / total for k, v in exp_sims.items()}

        rating = weights["low"] * 0.0 + weights["mid"] * 0.5 + weights["high"] * 1.0
        return float(max(0.0, min(1.0, rating)))

    def infer_all_ratings(
        self,
        section_embeddings: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Infer ratings for all dimensions from their section embeddings."""
        return {
            dim: self.infer_rating(section_embeddings[dim], dim)
            for dim in ["recency", "importance", "relevance"]
        }

    def get_anchor_blobs(self) -> dict[str, dict[str, bytes]]:
        """Export anchor embeddings as blobs for caching."""
        if not self._initialized:
            raise RuntimeError("Call initialize() first")
        return {
            dim: {
                level: vector_to_blob(self._anchor_embeddings[dim][level])
                for level in ["low", "mid", "high"]
            }
            for dim in ["recency", "importance", "relevance"]
        }


def _extract_section(response: str, tag: str) -> str | None:
    """Extract text for a tagged section.

    Tries [TAG]...[/TAG] first. If the closing tag is missing (common with
    MiniMax M2.5), extracts text from [TAG] to the next opening tag or end.
    """
    # Strict: matched opening and closing tags
    pattern = rf"\[{re.escape(tag)}\](.*?)\[/{re.escape(tag)}\]"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
        return text if len(text) >= 15 else None

    # Lenient: opening tag to next opening tag (or end of response)
    pattern = rf"\[{re.escape(tag)}\](.*?)(?=\[(?:RECENCY|IMPORTANCE|RELEVANCE|/RETRIEVAL)|$)"
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
        return text if len(text) >= 15 else None

    return None


def parse_qualitative_feedback(
    response: str,
    section_embeddings: dict[str, np.ndarray],
    anchor_parser: SeparatedAnchorParser,
) -> ParsedFeedback | None:
    """
    Parse Condition 4 qualitative feedback from separated sections.

    Expects 3 tagged sections in the response ([RECENCY EVALUATION], etc.).
    Each section's embedding is compared to that dimension's 3 anchors.

    Args:
        response: Full LLM response text.
        section_embeddings: Pre-computed embeddings for each section's text,
            keyed by dimension name ('recency', 'importance', 'relevance').
        anchor_parser: Initialized SeparatedAnchorParser.
    """
    # Verify all sections exist in the response
    sections = {}
    for dim, tag in _SECTION_TAGS.items():
        text = _extract_section(response, tag)
        if text is None:
            return None
        sections[dim] = text

    # Verify we have embeddings for all dimensions
    for dim in ["recency", "importance", "relevance"]:
        if dim not in section_embeddings:
            return None

    ratings = anchor_parser.infer_all_ratings(section_embeddings)
    composite = (ratings["recency"] + ratings["importance"] + ratings["relevance"]) / 3.0

    # Combine all section texts as the explanation
    explanation = "\n\n".join(
        f"[{dim.upper()}] {sections[dim]}" for dim in ["recency", "importance", "relevance"]
    )

    # Extract the full outer block for raw_block
    outer_match = re.search(
        r"\[RETRIEVAL EFFECTIVENESS\](.*?)\[/RETRIEVAL EFFECTIVENESS\]",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    raw_block = outer_match.group(1).strip() if outer_match else explanation

    return ParsedFeedback(
        recency=ratings["recency"],
        importance=ratings["importance"],
        relevance=ratings["relevance"],
        explanation=explanation,
        raw_block=raw_block,
        parse_method="separated_anchor",
        composite_reward=composite,
    )


def extract_qualitative_sections(response: str) -> dict[str, str] | None:
    """Extract the 3 dimension-specific sections from a Condition 4 response.

    Returns dict keyed by dimension name, or None if any section is missing.
    Used by the runner to embed each section before calling parse_qualitative_feedback.
    """
    sections = {}
    for dim, tag in _SECTION_TAGS.items():
        text = _extract_section(response, tag)
        if text is None:
            return None
        sections[dim] = text
    return sections
