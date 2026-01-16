"""High-level semantic memory interface.

Combines embeddings and vector store for a simple
remember/recall interface.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.memory.embeddings import LocalEmbeddings, SimpleEmbeddings, create_embeddings
from src.memory.vector_store import (
    InMemoryVectorStore,
    MemoryRecord,
    VectorStore,
)
from src.utils.logging import get_logger

logger = get_logger("memory.semantic")


class MemoryType(str, Enum):
    """Types of memories that can be stored."""

    FACT = "fact"  # Business facts, data insights
    PREFERENCE = "preference"  # User preferences
    SCHEMA = "schema"  # Table schemas, data structure
    QUERY = "query"  # Successful query patterns
    ERROR = "error"  # Error patterns to avoid
    CONTEXT = "context"  # Conversation context


@dataclass
class Memory:
    """A semantic memory with content and metadata."""

    content: str
    memory_type: MemoryType
    user_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] | None = None
    id: int | None = None
    similarity: float | None = None

    @classmethod
    def from_record(cls, record: MemoryRecord) -> "Memory":
        """Create from MemoryRecord.

        Args:
            record: MemoryRecord from vector store.

        Returns:
            Memory instance.
        """
        return cls(
            id=record.id,
            content=record.content,
            memory_type=MemoryType(record.memory_type),
            user_id=record.user_id,
            session_id=record.session_id,
            metadata=record.metadata,
            similarity=record.similarity,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "similarity": self.similarity,
        }


class SemanticMemory:
    """High-level semantic memory interface.

    Provides a simple remember/recall interface for storing
    and retrieving memories based on semantic similarity.
    """

    def __init__(
        self,
        embeddings: LocalEmbeddings | SimpleEmbeddings | None = None,
        vector_store: VectorStore | InMemoryVectorStore | None = None,
        use_simple: bool = False,
    ) -> None:
        """Initialize semantic memory.

        Args:
            embeddings: Embedding generator. Created if not provided.
            vector_store: Vector store. Created if not provided.
            use_simple: Use simple/in-memory implementations for testing.
        """
        if embeddings is None:
            embeddings = create_embeddings(use_simple=use_simple)
        self.embeddings = embeddings

        if vector_store is None:
            vector_store = InMemoryVectorStore() if use_simple else VectorStore()
        self.vector_store = vector_store

        logger.info(
            f"SemanticMemory initialized with "
            f"{type(embeddings).__name__} and {type(vector_store).__name__}"
        )

    async def close(self) -> None:
        """Close resources."""
        await self.vector_store.close()

    async def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Store a memory.

        Args:
            content: Text content to remember.
            memory_type: Type of memory.
            user_id: Optional user ID.
            session_id: Optional session ID.
            metadata: Optional metadata.

        Returns:
            Stored Memory object with ID.
        """
        logger.debug(f"Remembering: {content[:50]}...")

        # Generate embedding
        embedding = self.embeddings.embed(content)

        # Store in vector store
        record_id = await self.vector_store.add(
            content=content,
            embedding=embedding,
            memory_type=memory_type.value,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
        )

        logger.info(f"Stored memory id={record_id} type={memory_type.value}")

        return Memory(
            id=record_id,
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
        )

    async def recall(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.5,
        memory_type: MemoryType | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> list[Memory]:
        """Retrieve relevant memories.

        Args:
            query: Query text to find similar memories.
            limit: Maximum number of memories to return.
            min_similarity: Minimum similarity threshold (0-1).
            memory_type: Filter by memory type.
            user_id: Filter by user ID.
            session_id: Filter by session ID.

        Returns:
            List of relevant memories sorted by similarity.
        """
        logger.debug(f"Recalling for: {query[:50]}...")

        # Generate query embedding
        embedding = self.embeddings.embed(query)

        # Search vector store
        records = await self.vector_store.search(
            query_embedding=embedding,
            limit=limit,
            min_similarity=min_similarity,
            memory_type=memory_type.value if memory_type else None,
            user_id=user_id,
            session_id=session_id,
        )

        memories = [Memory.from_record(r) for r in records]
        logger.info(f"Recalled {len(memories)} memories for query")

        return memories

    async def forget(self, memory_id: int) -> bool:
        """Delete a memory.

        Args:
            memory_id: ID of memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        result = await self.vector_store.delete(memory_id)
        if result:
            logger.info(f"Forgot memory id={memory_id}")
        return result

    async def get(self, memory_id: int) -> Memory | None:
        """Get a specific memory by ID.

        Args:
            memory_id: Memory ID.

        Returns:
            Memory or None if not found.
        """
        record = await self.vector_store.get(memory_id)
        if record is None:
            return None
        return Memory.from_record(record)

    async def list_memories(
        self,
        memory_type: MemoryType | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """List all memories with optional filters.

        Args:
            memory_type: Filter by type.
            user_id: Filter by user.
            session_id: Filter by session.
            limit: Maximum results.

        Returns:
            List of memories.
        """
        records = await self.vector_store.list_all(
            memory_type=memory_type.value if memory_type else None,
            user_id=user_id,
            session_id=session_id,
            limit=limit,
        )
        return [Memory.from_record(r) for r in records]

    async def count(self) -> int:
        """Get total number of memories.

        Returns:
            Count of memories.
        """
        return await self.vector_store.count()


def create_semantic_memory(
    use_simple: bool = False,
    connection_string: str | None = None,
) -> SemanticMemory:
    """Factory function to create semantic memory.

    Args:
        use_simple: Use simple/in-memory implementations.
        connection_string: PostgreSQL connection string.

    Returns:
        Configured SemanticMemory instance.
    """
    if use_simple:
        return SemanticMemory(
            embeddings=SimpleEmbeddings(),
            vector_store=InMemoryVectorStore(),
            use_simple=True,
        )

    embeddings = LocalEmbeddings()
    vector_store = VectorStore(connection_string=connection_string)

    return SemanticMemory(
        embeddings=embeddings,
        vector_store=vector_store,
    )
