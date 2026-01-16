"""Semantic Memory module for DataVault.

Provides vector-based semantic memory using pgvector
for storing and retrieving context across sessions.
"""

from src.memory.embeddings import (
    EMBEDDING_DIM,
    LocalEmbeddings,
    SimpleEmbeddings,
    cosine_similarity,
    create_embeddings,
)
from src.memory.mem0_layer import (
    Mem0Layer,
    MemoryStats,
    create_mem0_layer,
)
from src.memory.semantic import (
    Memory,
    MemoryType,
    SemanticMemory,
    create_semantic_memory,
)
from src.memory.vector_store import (
    InMemoryVectorStore,
    MemoryRecord,
    VectorStore,
)

__all__ = [
    # Embeddings
    "EMBEDDING_DIM",
    "LocalEmbeddings",
    "SimpleEmbeddings",
    "create_embeddings",
    "cosine_similarity",
    # Vector Store
    "VectorStore",
    "InMemoryVectorStore",
    "MemoryRecord",
    # Semantic Memory
    "SemanticMemory",
    "Memory",
    "MemoryType",
    "create_semantic_memory",
    # Mem0 Layer
    "Mem0Layer",
    "MemoryStats",
    "create_mem0_layer",
]
