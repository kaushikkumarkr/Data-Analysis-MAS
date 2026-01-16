"""Vector store using PostgreSQL with pgvector extension.

Provides CRUD operations for storing and searching embeddings
with semantic similarity.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger("memory.vector_store")


@dataclass
class MemoryRecord:
    """A memory record stored in the vector store."""

    id: int | None = None
    content: str = ""
    embedding: list[float] = field(default_factory=list)
    memory_type: str = "fact"
    user_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    similarity: float | None = None  # Set during search

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "similarity": self.similarity,
        }


class VectorStore:
    """PostgreSQL pgvector-based vector store.

    Stores embeddings and enables semantic similarity search.
    """

    def __init__(
        self,
        connection_string: str | None = None,
        table_name: str = "memories",
    ) -> None:
        """Initialize vector store.

        Args:
            connection_string: PostgreSQL connection URL.
            table_name: Table name for memories.
        """
        config = get_config()
        self.connection_string = connection_string or config.postgres.connection_string
        self.table_name = table_name
        self._pool = None
        logger.info(f"VectorStore initialized for table: {table_name}")

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=1,
                max_size=10,
            )
            logger.info("Connection pool created")
        return self._pool

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Connection pool closed")

    async def add(
        self,
        content: str,
        embedding: list[float],
        memory_type: str = "fact",
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a memory to the store.

        Args:
            content: Text content of the memory.
            embedding: Embedding vector.
            memory_type: Type of memory (fact, preference, schema, etc.).
            user_id: Optional user ID for filtering.
            session_id: Optional session ID for filtering.
            metadata: Optional metadata dictionary.

        Returns:
            ID of the inserted record.
        """
        pool = await self._get_pool()

        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        metadata_json = json.dumps(metadata or {})

        query = f"""
            INSERT INTO {self.table_name} 
            (content, embedding, memory_type, user_id, session_id, metadata)
            VALUES ($1, $2::vector, $3, $4, $5, $6::jsonb)
            RETURNING id
        """

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                content,
                embedding_str,
                memory_type,
                user_id,
                session_id,
                metadata_json,
            )
            record_id = row["id"]
            logger.debug(f"Added memory with id={record_id}")
            return record_id

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        min_similarity: float = 0.5,
        memory_type: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> list[MemoryRecord]:
        """Search for similar memories.

        Args:
            query_embedding: Query embedding vector.
            limit: Maximum number of results.
            min_similarity: Minimum similarity threshold (0-1).
            memory_type: Filter by memory type.
            user_id: Filter by user ID.
            session_id: Filter by session ID.

        Returns:
            List of similar memory records.
        """
        pool = await self._get_pool()

        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

        # Build query with optional filters
        conditions = []
        params = [embedding_str, limit]
        param_idx = 3

        if memory_type:
            conditions.append(f"memory_type = ${param_idx}")
            params.append(memory_type)
            param_idx += 1

        if user_id:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        if session_id:
            conditions.append(f"session_id = ${param_idx}")
            params.append(session_id)
            param_idx += 1

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT 
                id, content, memory_type, user_id, session_id, 
                metadata, created_at, updated_at,
                1 - (embedding <=> $1::vector) as similarity
            FROM {self.table_name}
            {where_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

            results = []
            for row in rows:
                similarity = float(row["similarity"])
                if similarity >= min_similarity:
                    results.append(
                        MemoryRecord(
                            id=row["id"],
                            content=row["content"],
                            memory_type=row["memory_type"],
                            user_id=row["user_id"],
                            session_id=row["session_id"],
                            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                            created_at=row["created_at"],
                            updated_at=row["updated_at"],
                            similarity=similarity,
                        )
                    )

            logger.debug(f"Search returned {len(results)} results")
            return results

    async def get(self, record_id: int) -> MemoryRecord | None:
        """Get a memory by ID.

        Args:
            record_id: Memory ID.

        Returns:
            MemoryRecord or None if not found.
        """
        pool = await self._get_pool()

        query = f"""
            SELECT id, content, memory_type, user_id, session_id, 
                   metadata, created_at, updated_at
            FROM {self.table_name}
            WHERE id = $1
        """

        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, record_id)
            if row is None:
                return None

            return MemoryRecord(
                id=row["id"],
                content=row["content"],
                memory_type=row["memory_type"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    async def delete(self, record_id: int) -> bool:
        """Delete a memory by ID.

        Args:
            record_id: Memory ID.

        Returns:
            True if deleted, False if not found.
        """
        pool = await self._get_pool()

        query = f"DELETE FROM {self.table_name} WHERE id = $1"

        async with pool.acquire() as conn:
            result = await conn.execute(query, record_id)
            deleted = result.endswith("1")
            if deleted:
                logger.debug(f"Deleted memory id={record_id}")
            return deleted

    async def list_all(
        self,
        memory_type: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryRecord]:
        """List all memories with optional filters.

        Args:
            memory_type: Filter by memory type.
            user_id: Filter by user ID.
            session_id: Filter by session ID.
            limit: Maximum results.

        Returns:
            List of memory records.
        """
        pool = await self._get_pool()

        conditions = []
        params = []
        param_idx = 1

        if memory_type:
            conditions.append(f"memory_type = ${param_idx}")
            params.append(memory_type)
            param_idx += 1

        if user_id:
            conditions.append(f"user_id = ${param_idx}")
            params.append(user_id)
            param_idx += 1

        if session_id:
            conditions.append(f"session_id = ${param_idx}")
            params.append(session_id)
            param_idx += 1

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        params.append(limit)
        limit_param = f"${param_idx}"

        query = f"""
            SELECT id, content, memory_type, user_id, session_id, 
                   metadata, created_at, updated_at
            FROM {self.table_name}
            {where_clause}
            ORDER BY created_at DESC
            LIMIT {limit_param}
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

            return [
                MemoryRecord(
                    id=row["id"],
                    content=row["content"],
                    memory_type=row["memory_type"],
                    user_id=row["user_id"],
                    session_id=row["session_id"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

    async def count(self) -> int:
        """Get total number of memories.

        Returns:
            Count of memories.
        """
        pool = await self._get_pool()

        query = f"SELECT COUNT(*) FROM {self.table_name}"

        async with pool.acquire() as conn:
            row = await conn.fetchrow(query)
            return row[0]


class InMemoryVectorStore:
    """In-memory vector store for testing without PostgreSQL.

    Stores embeddings in memory and uses numpy for similarity search.
    """

    def __init__(self) -> None:
        """Initialize in-memory store."""
        self._memories: dict[int, MemoryRecord] = {}
        self._embeddings: dict[int, list[float]] = {}
        self._next_id = 1
        logger.info("InMemoryVectorStore initialized")

    async def close(self) -> None:
        """No-op for in-memory store."""
        pass

    async def add(
        self,
        content: str,
        embedding: list[float],
        memory_type: str = "fact",
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a memory."""
        import numpy as np

        record_id = self._next_id
        self._next_id += 1

        now = datetime.utcnow()
        self._memories[record_id] = MemoryRecord(
            id=record_id,
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        self._embeddings[record_id] = embedding

        return record_id

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        min_similarity: float = 0.5,
        memory_type: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> list[MemoryRecord]:
        """Search for similar memories."""
        import numpy as np

        query_vec = np.array(query_embedding)
        results = []

        for record_id, embedding in self._embeddings.items():
            record = self._memories[record_id]

            # Apply filters
            if memory_type and record.memory_type != memory_type:
                continue
            if user_id and record.user_id != user_id:
                continue
            if session_id and record.session_id != session_id:
                continue

            # Compute cosine similarity
            emb_vec = np.array(embedding)
            similarity = float(np.dot(query_vec, emb_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(emb_vec)
            ))

            if similarity >= min_similarity:
                record_copy = MemoryRecord(
                    id=record.id,
                    content=record.content,
                    memory_type=record.memory_type,
                    user_id=record.user_id,
                    session_id=record.session_id,
                    metadata=record.metadata,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    similarity=similarity,
                )
                results.append(record_copy)

        # Sort by similarity
        results.sort(key=lambda x: x.similarity or 0, reverse=True)
        return results[:limit]

    async def get(self, record_id: int) -> MemoryRecord | None:
        """Get a memory by ID."""
        return self._memories.get(record_id)

    async def delete(self, record_id: int) -> bool:
        """Delete a memory."""
        if record_id in self._memories:
            del self._memories[record_id]
            del self._embeddings[record_id]
            return True
        return False

    async def list_all(
        self,
        memory_type: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryRecord]:
        """List all memories."""
        results = []
        for record in self._memories.values():
            if memory_type and record.memory_type != memory_type:
                continue
            if user_id and record.user_id != user_id:
                continue
            if session_id and record.session_id != session_id:
                continue
            results.append(record)

        results.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
        return results[:limit]

    async def count(self) -> int:
        """Get count."""
        return len(self._memories)
