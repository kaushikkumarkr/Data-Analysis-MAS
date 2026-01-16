"""Mem0-style business logic layer for structured memory management.

Provides session-based and user-based memory management with
automatic memory extraction from agent outputs.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from src.memory.semantic import Memory, MemoryType, SemanticMemory, create_semantic_memory
from src.utils.logging import get_logger

logger = get_logger("memory.mem0_layer")


@dataclass
class MemoryStats:
    """Statistics about stored memories."""

    total: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    by_user: dict[str, int] = field(default_factory=dict)
    by_session: dict[str, int] = field(default_factory=dict)


class Mem0Layer:
    """Mem0-style memory management layer.

    Provides structured memory management with:
    - Session-based memory (per conversation)
    - User-based memory (across sessions)
    - Automatic memory extraction from agent outputs
    - Memory consolidation and summarization
    """

    def __init__(
        self,
        semantic_memory: SemanticMemory | None = None,
        user_id: str | None = None,
        use_simple: bool = False,
    ) -> None:
        """Initialize Mem0 layer.

        Args:
            semantic_memory: Underlying semantic memory. Created if not provided.
            user_id: Default user ID for this instance.
            use_simple: Use simple/in-memory implementations.
        """
        if semantic_memory is None:
            semantic_memory = create_semantic_memory(use_simple=use_simple)
        self.memory = semantic_memory
        self.user_id = user_id
        self._current_session: str | None = None
        logger.info(f"Mem0Layer initialized for user={user_id}")

    async def close(self) -> None:
        """Close resources."""
        await self.memory.close()

    def start_session(self) -> str:
        """Start a new memory session.

        Returns:
            Session ID.
        """
        self._current_session = str(uuid4())
        logger.info(f"Started session: {self._current_session}")
        return self._current_session

    def end_session(self) -> None:
        """End the current session."""
        logger.info(f"Ended session: {self._current_session}")
        self._current_session = None

    @property
    def session_id(self) -> str | None:
        """Get current session ID."""
        return self._current_session

    async def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        metadata: dict[str, Any] | None = None,
        session_scoped: bool = True,
    ) -> Memory:
        """Add a memory.

        Args:
            content: Text content.
            memory_type: Type of memory.
            metadata: Optional metadata.
            session_scoped: If True, memory is session-scoped.

        Returns:
            Stored memory.
        """
        session_id = self._current_session if session_scoped else None

        return await self.memory.remember(
            content=content,
            memory_type=memory_type,
            user_id=self.user_id,
            session_id=session_id,
            metadata=metadata,
        )

    async def search_memories(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.5,
        memory_type: MemoryType | None = None,
        include_all_sessions: bool = False,
    ) -> list[Memory]:
        """Search for relevant memories.

        Args:
            query: Query text.
            limit: Maximum results.
            min_similarity: Minimum similarity.
            memory_type: Filter by type.
            include_all_sessions: Include memories from all sessions.

        Returns:
            List of relevant memories.
        """
        session_id = None if include_all_sessions else self._current_session

        return await self.memory.recall(
            query=query,
            limit=limit,
            min_similarity=min_similarity,
            memory_type=memory_type,
            user_id=self.user_id,
            session_id=session_id,
        )

    async def get_context_for_query(
        self,
        query: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Get memory context for an agent query.

        Retrieves relevant memories from all sessions to provide
        context for the current query.

        Args:
            query: User query.
            limit: Maximum memories to include.

        Returns:
            List of memory dictionaries for agent context.
        """
        memories = await self.search_memories(
            query=query,
            limit=limit,
            min_similarity=0.4,  # Lower threshold for context
            include_all_sessions=True,
        )

        return [m.to_dict() for m in memories]

    async def extract_and_store_facts(
        self,
        text: str,
        source: str = "agent",
    ) -> list[Memory]:
        """Extract and store facts from text.

        Extracts key facts, insights, and learnings from
        agent output or user input and stores them as memories.

        Args:
            text: Text to extract facts from.
            source: Source of the text.

        Returns:
            List of stored memories.
        """
        # Simple fact extraction patterns
        facts = self._extract_facts(text)

        stored = []
        for fact in facts:
            memory = await self.add_memory(
                content=fact,
                memory_type=MemoryType.FACT,
                metadata={"source": source, "extracted": True},
                session_scoped=False,  # Facts persist across sessions
            )
            stored.append(memory)

        logger.info(f"Extracted and stored {len(stored)} facts from {source}")
        return stored

    def _extract_facts(self, text: str) -> list[str]:
        """Extract facts from text.

        Args:
            text: Text to analyze.

        Returns:
            List of extracted facts.
        """
        facts = []

        # Extract sentences with key indicators
        indicators = [
            r"the (?:total|average|sum|count) (?:is|was|are|were) [\d,.$]+",
            r"(?:there are|there were|found) \d+ (?:rows?|records?|transactions?|items?)",
            r"(?:highest|lowest|most|least) .+ (?:is|was|are|were) .+",
            r"(?:revenue|sales|profit|amount) .+ (?:is|was) [\d,.$]+",
            r"(?:increased|decreased|grew|dropped) by [\d,.%]+",
        ]

        for pattern in indicators:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean up and store
                fact = match.strip()
                if len(fact) > 10:  # Minimum length for a fact
                    facts.append(fact)

        # Extract sentences containing numbers (likely data insights)
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            sentence = sentence.strip()
            # Check if contains numbers and is reasonably sized
            if (
                re.search(r'\d+', sentence)
                and 20 < len(sentence) < 200
                and sentence not in facts
            ):
                facts.append(sentence)

        # Deduplicate while preserving order
        seen = set()
        unique_facts = []
        for fact in facts:
            fact_lower = fact.lower()
            if fact_lower not in seen:
                seen.add(fact_lower)
                unique_facts.append(fact)

        return unique_facts[:10]  # Limit to 10 facts

    async def store_query_pattern(
        self,
        query: str,
        sql: str,
        success: bool,
        result_summary: str | None = None,
    ) -> Memory:
        """Store a successful query pattern.

        Args:
            query: Natural language query.
            sql: Generated SQL.
            success: Whether query succeeded.
            result_summary: Optional summary of results.

        Returns:
            Stored memory.
        """
        memory_type = MemoryType.QUERY if success else MemoryType.ERROR

        content = f"Query: {query}\nSQL: {sql}"
        if result_summary:
            content += f"\nResult: {result_summary}"

        return await self.add_memory(
            content=content,
            memory_type=memory_type,
            metadata={
                "query": query,
                "sql": sql,
                "success": success,
            },
            session_scoped=False,  # Query patterns persist
        )

    async def store_schema_info(
        self,
        table_name: str,
        schema_info: dict[str, Any],
    ) -> Memory:
        """Store table schema information.

        Args:
            table_name: Name of table.
            schema_info: Schema details.

        Returns:
            Stored memory.
        """
        columns = schema_info.get("columns", [])
        col_desc = ", ".join(f"{c['name']} ({c['type']})" for c in columns)
        row_count = schema_info.get("row_count", "unknown")

        content = f"Table {table_name} has {row_count} rows with columns: {col_desc}"

        return await self.add_memory(
            content=content,
            memory_type=MemoryType.SCHEMA,
            metadata={"table": table_name, "schema": schema_info},
            session_scoped=False,  # Schema info persists
        )

    async def store_preference(
        self,
        preference: str,
        category: str = "general",
    ) -> Memory:
        """Store a user preference.

        Args:
            preference: Preference description.
            category: Preference category.

        Returns:
            Stored memory.
        """
        return await self.add_memory(
            content=preference,
            memory_type=MemoryType.PREFERENCE,
            metadata={"category": category},
            session_scoped=False,  # Preferences persist
        )

    async def get_stats(self) -> MemoryStats:
        """Get memory statistics.

        Returns:
            Memory statistics.
        """
        all_memories = await self.memory.list_memories(
            user_id=self.user_id,
            limit=1000,
        )

        stats = MemoryStats(total=len(all_memories))

        for memory in all_memories:
            # Count by type
            type_key = memory.memory_type.value
            stats.by_type[type_key] = stats.by_type.get(type_key, 0) + 1

            # Count by user
            user_key = memory.user_id or "anonymous"
            stats.by_user[user_key] = stats.by_user.get(user_key, 0) + 1

            # Count by session
            session_key = memory.session_id or "global"
            stats.by_session[session_key] = stats.by_session.get(session_key, 0) + 1

        return stats

    async def clear_session_memories(self) -> int:
        """Clear all memories for current session.

        Returns:
            Number of memories deleted.
        """
        if not self._current_session:
            return 0

        memories = await self.memory.list_memories(
            session_id=self._current_session,
        )

        count = 0
        for memory in memories:
            if memory.id:
                await self.memory.forget(memory.id)
                count += 1

        logger.info(f"Cleared {count} session memories")
        return count


def create_mem0_layer(
    user_id: str | None = None,
    use_simple: bool = False,
    connection_string: str | None = None,
) -> Mem0Layer:
    """Factory function to create Mem0 layer.

    Args:
        user_id: User ID.
        use_simple: Use simple/in-memory implementations.
        connection_string: PostgreSQL connection string.

    Returns:
        Configured Mem0Layer instance.
    """
    memory = create_semantic_memory(
        use_simple=use_simple,
        connection_string=connection_string,
    )
    return Mem0Layer(
        semantic_memory=memory,
        user_id=user_id,
        use_simple=use_simple,
    )
