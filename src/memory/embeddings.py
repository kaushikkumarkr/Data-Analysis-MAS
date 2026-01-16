"""Local embedding generation for semantic memory.

Provides embeddings using sentence-transformers or MLX,
with a TF-IDF fallback for testing without models.
"""

import hashlib
from typing import Any

import numpy as np

from src.utils.logging import get_logger

logger = get_logger("memory.embeddings")

# Embedding dimension for MiniLM-L6-v2
EMBEDDING_DIM = 384


class LocalEmbeddings:
    """Local embedding generator using sentence-transformers.

    Uses all-MiniLM-L6-v2 by default (~80MB) for efficient
    local embedding generation.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        """Initialize embeddings.

        Args:
            model_name: HuggingFace model name for embeddings.
            device: Device to use (None for auto-detect, 'mps' for Mac).
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        logger.info(f"LocalEmbeddings initialized with: {model_name}")

    def _ensure_loaded(self) -> None:
        """Load the model if not already loaded."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                )
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise RuntimeError(f"Failed to load embedding model: {e}") from e

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        self._ensure_loaded()

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return EMBEDDING_DIM


class SimpleEmbeddings:
    """Simple TF-IDF-like embeddings for testing without ML models.

    Uses a hash-based approach to generate consistent embeddings
    without requiring any external models.
    """

    def __init__(self, dimension: int = EMBEDDING_DIM) -> None:
        """Initialize simple embeddings.

        Args:
            dimension: Embedding dimension.
        """
        self._dimension = dimension
        logger.info(f"SimpleEmbeddings initialized with dimension: {dimension}")

    def _text_to_hash_vector(self, text: str) -> list[float]:
        """Convert text to a deterministic hash-based vector.

        Args:
            text: Input text.

        Returns:
            Normalized vector.
        """
        # Create multiple hashes for the text
        text_lower = text.lower().strip()
        words = text_lower.split()

        # Initialize vector
        vector = np.zeros(self._dimension, dtype=np.float32)

        # Hash each word and accumulate
        for i, word in enumerate(words):
            # Create position-aware hash
            hash_input = f"{word}_{i % 10}"
            hash_bytes = hashlib.sha256(hash_input.encode()).digest()

            # Convert hash to indices and increment
            for j in range(0, len(hash_bytes), 2):
                idx = (hash_bytes[j] * 256 + hash_bytes[j + 1]) % self._dimension
                vector[idx] += 1.0 / (i + 1)  # Weight by position

        # Also hash the full text for context
        full_hash = hashlib.sha256(text_lower.encode()).digest()
        for j in range(0, len(full_hash), 2):
            idx = (full_hash[j] * 256 + full_hash[j + 1]) % self._dimension
            vector[idx] += 0.5

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.tolist()

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        return self._text_to_hash_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts.

        Returns:
            List of embedding vectors.
        """
        return [self.embed(text) for text in texts]

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension


def create_embeddings(use_simple: bool = False, **kwargs) -> LocalEmbeddings | SimpleEmbeddings:
    """Factory function to create embeddings.

    Args:
        use_simple: If True, use SimpleEmbeddings for testing.
        **kwargs: Additional arguments for the embeddings class.

    Returns:
        Embeddings instance.
    """
    if use_simple:
        return SimpleEmbeddings(**kwargs)
    return LocalEmbeddings(**kwargs)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        Cosine similarity score (-1 to 1).
    """
    a = np.array(v1)
    b = np.array(v2)

    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))
