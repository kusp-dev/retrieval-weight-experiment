"""
Embedding service using Qwen3-Embedding-0.6B via sentence-transformers.

70.7 MTEB English, 1024 dimensions, 32k context window.
Vectors stored as float32 blobs in SQLite — no vector DB needed for ~200 skills.
"""

import sqlite3
from typing import Any, Optional

import numpy as np


class Embedder:
    """Wraps Qwen3-Embedding-0.6B for embedding text into dense vectors.

    Lazy-loads the model on first use to avoid slow imports during testing.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        dimensions: int = 1024,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.dimensions = dimensions
        self.device = device
        self._model: Any = None

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            truncate_dim=self.dimensions,
        )

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns float32 array of shape (dimensions,)."""
        self._load_model()
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed multiple texts. Returns float32 array of shape (n, dimensions)."""
        self._load_model()
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
        )
        return vecs.astype(np.float32)


# ── Serialization helpers for SQLite BLOB storage ──


def vector_to_blob(vec: np.ndarray) -> bytes:
    """Convert a numpy vector to bytes for SQLite BLOB storage."""
    return vec.astype(np.float32).tobytes()


def blob_to_vector(blob: bytes, dimensions: int = 1024) -> np.ndarray:
    """Convert a SQLite BLOB back to a numpy vector."""
    return np.frombuffer(blob, dtype=np.float32).copy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Assumes normalized vectors."""
    return float(np.dot(a, b))


def cosine_similarity_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and a matrix of vectors.

    Args:
        query: shape (dimensions,)
        matrix: shape (n, dimensions)

    Returns:
        shape (n,) similarity scores
    """
    return matrix @ query


# ── Database operations ──


def store_skill_embedding(
    conn: sqlite3.Connection,
    skill_id: str,
    embedding: np.ndarray,
    model: str = "Qwen/Qwen3-Embedding-0.6B",
) -> None:
    """Store a skill's embedding in the database."""
    conn.execute(
        """INSERT OR REPLACE INTO skill_embeddings (skill_id, embedding, model)
           VALUES (?, ?, ?)""",
        (skill_id, vector_to_blob(embedding), model),
    )
    conn.commit()


def load_all_skill_embeddings(
    conn: sqlite3.Connection,
    dimensions: int = 1024,
) -> tuple[list[str], np.ndarray]:
    """Load all skill embeddings from the database.

    Returns:
        (skill_ids, embedding_matrix) where embedding_matrix has shape (n, dimensions)
    """
    rows = conn.execute(
        "SELECT skill_id, embedding FROM skill_embeddings ORDER BY skill_id"
    ).fetchall()

    if not rows:
        return [], np.empty((0, dimensions), dtype=np.float32)

    skill_ids = [row["skill_id"] for row in rows]
    vectors = np.array(
        [blob_to_vector(row["embedding"], dimensions) for row in rows],
        dtype=np.float32,
    )
    return skill_ids, vectors


def store_feedback_embedding(
    conn: sqlite3.Connection,
    feedback_id: int,
    embedding: np.ndarray,
    model: str = "Qwen/Qwen3-Embedding-0.6B",
) -> None:
    """Store a feedback explanation's embedding (condition 3 only)."""
    conn.execute(
        """INSERT OR REPLACE INTO feedback_embeddings (feedback_id, embedding, model)
           VALUES (?, ?, ?)""",
        (feedback_id, vector_to_blob(embedding), model),
    )
    conn.commit()
