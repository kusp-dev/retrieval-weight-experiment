"""Tests for the embedder module.

Tests serialization helpers, cosine similarity, and database operations
using synthetic vectors. The Embedder class methods use mocked
SentenceTransformer to avoid downloading models during testing.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.db.schema import init_db
from src.embeddings.embedder import (
    Embedder,
    blob_to_vector,
    cosine_similarity,
    cosine_similarity_batch,
    load_all_skill_embeddings,
    store_feedback_embedding,
    store_skill_embedding,
    vector_to_blob,
)

RNG = np.random.default_rng(42)
DIMS = 1024


def _random_unit_vector(dims: int = DIMS) -> np.ndarray:
    """Return a normalized random float32 vector."""
    v = RNG.standard_normal(dims).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


# ── Fixtures ──


@pytest.fixture
def db():
    """In-memory database with full schema."""
    conn = init_db(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def db_with_skills(db):
    """Database with 3 skills and embeddings."""
    skills = [
        ("s1", "research", "Neural Retrieval", "Dense retrieval with transformers"),
        ("s2", "research", "BM25 Basics", "Term frequency scoring"),
        ("s3", "synthesis", "RAG Pipeline", "Retrieval augmented generation"),
    ]
    vecs = {}
    for sid, domain, title, content in skills:
        db.execute(
            "INSERT INTO skills (skill_id, domain, title, content) VALUES (?, ?, ?, ?)",
            (sid, domain, title, content),
        )
        vec = _random_unit_vector()
        vecs[sid] = vec
        store_skill_embedding(db, sid, vec)
    return db, vecs


# ── Serialization helpers ──


class TestVectorSerialization:
    def test_roundtrip(self):
        vec = _random_unit_vector()
        blob = vector_to_blob(vec)
        recovered = blob_to_vector(blob, DIMS)
        np.testing.assert_array_equal(vec, recovered)

    def test_blob_is_bytes(self):
        vec = _random_unit_vector()
        blob = vector_to_blob(vec)
        assert isinstance(blob, bytes)

    def test_blob_length_matches_dimensions(self):
        vec = _random_unit_vector(512)
        blob = vector_to_blob(vec)
        # float32 = 4 bytes per element
        assert len(blob) == 512 * 4

    def test_float64_input_cast_to_float32(self):
        vec = RNG.standard_normal(DIMS).astype(np.float64)
        blob = vector_to_blob(vec)
        recovered = blob_to_vector(blob, DIMS)
        assert recovered.dtype == np.float32

    def test_blob_to_vector_returns_writable_copy(self):
        """blob_to_vector should return a writable array (not a read-only buffer view)."""
        vec = _random_unit_vector()
        blob = vector_to_blob(vec)
        recovered = blob_to_vector(blob, DIMS)
        # This should NOT raise — the array is a copy
        recovered[0] = 999.0
        assert recovered[0] == 999.0


# ── Cosine similarity ──


class TestCosineSimilarity:
    def test_identical_vectors(self):
        vec = _random_unit_vector()
        assert cosine_similarity(vec, vec) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        a = np.array([1, 0, 0], dtype=np.float32)
        b = np.array([0, 1, 0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-7)

    def test_opposite_vectors(self):
        a = np.array([1, 0], dtype=np.float32)
        b = np.array([-1, 0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-7)

    def test_returns_float(self):
        a = _random_unit_vector()
        b = _random_unit_vector()
        result = cosine_similarity(a, b)
        assert isinstance(result, float)


class TestCosineSimilarityBatch:
    def test_shape(self):
        query = _random_unit_vector()
        matrix = np.array([_random_unit_vector() for _ in range(5)])
        result = cosine_similarity_batch(query, matrix)
        assert result.shape == (5,)

    def test_self_similarity_in_batch(self):
        query = _random_unit_vector()
        matrix = np.array([query, _random_unit_vector(), _random_unit_vector()])
        result = cosine_similarity_batch(query, matrix)
        assert result[0] == pytest.approx(1.0, abs=1e-5)

    def test_single_row_matrix(self):
        query = _random_unit_vector()
        matrix = query.reshape(1, -1)
        result = cosine_similarity_batch(query, matrix)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(1.0, abs=1e-5)


# ── Database operations ──


class TestStoreSkillEmbedding:
    def test_stores_and_retrieves(self, db):
        vec = _random_unit_vector()
        db.execute(
            "INSERT INTO skills (skill_id, domain, title, content) VALUES (?, ?, ?, ?)",
            ("s1", "test", "Test Skill", "Test content"),
        )
        store_skill_embedding(db, "s1", vec)

        row = db.execute(
            "SELECT embedding, model FROM skill_embeddings WHERE skill_id = ?", ("s1",)
        ).fetchone()
        assert row is not None
        recovered = blob_to_vector(row["embedding"], DIMS)
        np.testing.assert_array_equal(vec, recovered)
        assert row["model"] == "Qwen/Qwen3-Embedding-0.6B"

    def test_custom_model_name(self, db):
        vec = _random_unit_vector()
        db.execute(
            "INSERT INTO skills (skill_id, domain, title, content) VALUES (?, ?, ?, ?)",
            ("s1", "test", "Test", "Content"),
        )
        store_skill_embedding(db, "s1", vec, model="custom-model")
        row = db.execute(
            "SELECT model FROM skill_embeddings WHERE skill_id = ?", ("s1",)
        ).fetchone()
        assert row["model"] == "custom-model"

    def test_upsert_replaces_existing(self, db):
        db.execute(
            "INSERT INTO skills (skill_id, domain, title, content) VALUES (?, ?, ?, ?)",
            ("s1", "test", "Test", "Content"),
        )
        vec1 = _random_unit_vector()
        vec2 = _random_unit_vector()
        store_skill_embedding(db, "s1", vec1)
        store_skill_embedding(db, "s1", vec2)

        row = db.execute(
            "SELECT embedding FROM skill_embeddings WHERE skill_id = ?", ("s1",)
        ).fetchone()
        recovered = blob_to_vector(row["embedding"], DIMS)
        np.testing.assert_array_equal(vec2, recovered)


class TestLoadAllSkillEmbeddings:
    def test_empty_database(self, db):
        skill_ids, matrix = load_all_skill_embeddings(db)
        assert skill_ids == []
        assert matrix.shape == (0, DIMS)
        assert matrix.dtype == np.float32

    def test_loads_multiple_skills(self, db_with_skills):
        db, vecs = db_with_skills
        skill_ids, matrix = load_all_skill_embeddings(db)
        assert len(skill_ids) == 3
        assert matrix.shape == (3, DIMS)
        # Sorted by skill_id
        assert skill_ids == ["s1", "s2", "s3"]

    def test_vectors_match_stored(self, db_with_skills):
        db, vecs = db_with_skills
        skill_ids, matrix = load_all_skill_embeddings(db)
        for i, sid in enumerate(skill_ids):
            np.testing.assert_array_equal(matrix[i], vecs[sid])

    def test_custom_dimensions_param(self, db):
        """The dimensions parameter affects only the empty-case shape."""
        skill_ids, matrix = load_all_skill_embeddings(db, dimensions=512)
        assert matrix.shape == (0, 512)


class TestStoreFeedbackEmbedding:
    def _insert_feedback_row(self, db):
        """Helper: insert task -> episode -> feedback, return feedback_id."""
        db.execute(
            "INSERT INTO tasks (task_id, theme, title, description) VALUES (?, ?, ?, ?)",
            ("t1", "test", "Test Task", "A test task"),
        )
        db.execute(
            """INSERT INTO episodes (condition_id, task_id, task_order)
               VALUES (?, ?, ?)""",
            (3, "t1", 1),
        )
        episode_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.execute(
            """INSERT INTO feedback (episode_id, rating_recency, rating_importance,
               rating_relevance, explanation)
               VALUES (?, ?, ?, ?, ?)""",
            (episode_id, 4, 3, 5, "Good response"),
        )
        return db.execute("SELECT last_insert_rowid()").fetchone()[0]

    def test_stores_feedback_embedding(self, db):
        feedback_id = self._insert_feedback_row(db)

        vec = _random_unit_vector()
        store_feedback_embedding(db, feedback_id, vec)

        row = db.execute(
            "SELECT embedding, model FROM feedback_embeddings WHERE feedback_id = ?",
            (feedback_id,),
        ).fetchone()
        assert row is not None
        recovered = blob_to_vector(row["embedding"], DIMS)
        np.testing.assert_array_equal(vec, recovered)
        assert row["model"] == "Qwen/Qwen3-Embedding-0.6B"

    def test_custom_model_name(self, db):
        feedback_id = self._insert_feedback_row(db)

        vec = _random_unit_vector()
        store_feedback_embedding(db, feedback_id, vec, model="custom-model")

        row = db.execute(
            "SELECT model FROM feedback_embeddings WHERE feedback_id = ?",
            (feedback_id,),
        ).fetchone()
        assert row["model"] == "custom-model"


# ── Embedder class (mocked model) ──


class TestEmbedderInit:
    def test_default_config(self):
        embedder = Embedder()
        assert embedder.model_name == "Qwen/Qwen3-Embedding-0.6B"
        assert embedder.dimensions == 1024
        assert embedder.device is None
        assert embedder._model is None

    def test_custom_config(self):
        embedder = Embedder(model_name="custom/model", dimensions=512, device="cpu")
        assert embedder.model_name == "custom/model"
        assert embedder.dimensions == 512
        assert embedder.device == "cpu"


class TestEmbedderLoadModel:
    @patch("src.embeddings.embedder.SentenceTransformer", create=True)
    def test_lazy_loads_on_first_call(self, mock_st_cls):
        """_load_model imports and instantiates SentenceTransformer."""
        mock_model = MagicMock()
        mock_st_cls.return_value = mock_model

        with patch(
            "src.embeddings.embedder.SentenceTransformer",
            new=mock_st_cls,
            create=True,
        ):
            embedder = Embedder(model_name="test/model", dimensions=768, device="cpu")
            # Patch the import inside _load_model
            with patch.dict(
                "sys.modules",
                {"sentence_transformers": MagicMock(SentenceTransformer=mock_st_cls)},
            ):
                embedder._load_model()

        assert embedder._model is mock_model

    def test_does_not_reload_if_already_loaded(self):
        embedder = Embedder()
        sentinel = MagicMock()
        embedder._model = sentinel
        embedder._load_model()
        # Model should remain the same object — no reload
        assert embedder._model is sentinel


class TestEmbedderEmbed:
    def test_embed_single_text(self):
        embedder = Embedder(dimensions=DIMS)
        mock_model = MagicMock()
        expected = _random_unit_vector()
        mock_model.encode.return_value = expected.astype(np.float64)
        embedder._model = mock_model

        result = embedder.embed("test text")

        mock_model.encode.assert_called_once_with("test text", normalize_embeddings=True)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_embed_empty_string(self):
        embedder = Embedder(dimensions=DIMS)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(DIMS, dtype=np.float32)
        embedder._model = mock_model

        result = embedder.embed("")
        mock_model.encode.assert_called_once_with("", normalize_embeddings=True)
        assert result.shape == (DIMS,)


class TestEmbedderEmbedBatch:
    def test_embed_batch(self):
        embedder = Embedder(dimensions=DIMS)
        mock_model = MagicMock()
        expected = np.array([_random_unit_vector() for _ in range(3)], dtype=np.float64)
        mock_model.encode.return_value = expected
        embedder._model = mock_model

        texts = ["text1", "text2", "text3"]
        result = embedder.embed_batch(texts, batch_size=16)

        mock_model.encode.assert_called_once_with(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,  # len(texts) <= 10
        )
        assert result.dtype == np.float32
        assert result.shape == (3, DIMS)

    def test_progress_bar_shown_for_large_batch(self):
        embedder = Embedder(dimensions=DIMS)
        mock_model = MagicMock()
        texts = [f"text{i}" for i in range(15)]
        mock_model.encode.return_value = np.zeros((15, DIMS), dtype=np.float32)
        embedder._model = mock_model

        embedder.embed_batch(texts)

        mock_model.encode.assert_called_once_with(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,  # len(texts) > 10
        )

    def test_embed_batch_single_item(self):
        embedder = Embedder(dimensions=DIMS)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((1, DIMS), dtype=np.float32)
        embedder._model = mock_model

        result = embedder.embed_batch(["single"])
        assert result.shape == (1, DIMS)

    def test_embed_batch_default_batch_size(self):
        embedder = Embedder(dimensions=DIMS)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((2, DIMS), dtype=np.float32)
        embedder._model = mock_model

        embedder.embed_batch(["a", "b"])

        # Default batch_size is 32
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs["batch_size"] == 32
