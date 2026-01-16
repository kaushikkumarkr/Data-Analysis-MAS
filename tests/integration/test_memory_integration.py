"""Integration tests for memory persistence.

Tests semantic memory with in-memory store.
"""

import pytest
import asyncio

from src.memory import (
    SemanticMemory,
    MemoryType,
    SimpleEmbeddings,
    InMemoryVectorStore,
    create_mem0_layer,
)


class TestMemoryPersistence:
    """Test memory persistence functionality."""

    @pytest.mark.asyncio
    async def test_remember_and_recall(self) -> None:
        """Test basic remember and recall."""
        memory = SemanticMemory(
            embeddings=SimpleEmbeddings(),
            vector_store=InMemoryVectorStore(),
        )

        # Remember some facts
        await memory.remember(
            "The sales table has 1000 rows",
            MemoryType.FACT,
        )
        await memory.remember(
            "Revenue increased by 20% in Q4",
            MemoryType.FACT,
        )

        # Recall
        results = await memory.recall(
            "How many rows in sales?",
            limit=5,
            min_similarity=0.0,
        )

        assert len(results) >= 1
        await memory.close()

    @pytest.mark.asyncio
    async def test_memory_types_filtering(self) -> None:
        """Test filtering by memory type."""
        memory = SemanticMemory(
            embeddings=SimpleEmbeddings(),
            vector_store=InMemoryVectorStore(),
        )

        await memory.remember("User prefers dark mode", MemoryType.PREFERENCE)
        await memory.remember("Sales grew 10%", MemoryType.FACT)
        await memory.remember("SELECT * FROM sales", MemoryType.QUERY)

        # Filter by type
        preferences = await memory.list_memories(memory_type=MemoryType.PREFERENCE)
        facts = await memory.list_memories(memory_type=MemoryType.FACT)

        assert len(preferences) == 1
        assert len(facts) == 1

        await memory.close()

    @pytest.mark.asyncio
    async def test_forget_memory(self) -> None:
        """Test forgetting memories."""
        memory = SemanticMemory(
            embeddings=SimpleEmbeddings(),
            vector_store=InMemoryVectorStore(),
        )

        stored = await memory.remember("Temporary fact", MemoryType.FACT)
        assert await memory.count() == 1

        await memory.forget(stored.id)
        assert await memory.count() == 0

        await memory.close()


class TestMem0LayerIntegration:
    """Test Mem0 layer integration."""

    @pytest.mark.asyncio
    async def test_session_scoped_memories(self) -> None:
        """Test session-scoped memories."""
        mem0 = create_mem0_layer(user_id="test_user", use_simple=True)

        # Start session
        session_id = mem0.start_session()

        # Add session-scoped memory
        await mem0.add_memory(
            "Current query is about sales",
            MemoryType.CONTEXT,
            session_scoped=True,
        )

        # Add global memory
        await mem0.add_memory(
            "User prefers bar charts",
            MemoryType.PREFERENCE,
            session_scoped=False,
        )

        # Get stats
        stats = await mem0.get_stats()
        assert stats.total == 2

        # Clear session memories
        cleared = await mem0.clear_session_memories()
        assert cleared == 1

        # Global memory should remain
        assert await mem0.memory.count() == 1

        await mem0.close()

    @pytest.mark.asyncio
    async def test_query_pattern_storage(self) -> None:
        """Test storing query patterns."""
        mem0 = create_mem0_layer(user_id="analyst", use_simple=True)

        # Store successful query
        await mem0.store_query_pattern(
            query="Show sales by region",
            sql="SELECT region, SUM(amount) FROM sales GROUP BY region",
            success=True,
            result_summary="3 regions found",
        )

        # Store failed query
        await mem0.store_query_pattern(
            query="Invalid query",
            sql="SELCT * FORM sales",  # typo
            success=False,
        )

        stats = await mem0.get_stats()
        assert stats.by_type.get("query", 0) == 1
        assert stats.by_type.get("error", 0) == 1

        await mem0.close()

    @pytest.mark.asyncio
    async def test_schema_storage(self) -> None:
        """Test storing schema information."""
        mem0 = create_mem0_layer(use_simple=True)

        await mem0.store_schema_info(
            "customers",
            {
                "columns": [
                    {"name": "id", "type": "INTEGER"},
                    {"name": "name", "type": "VARCHAR"},
                    {"name": "email", "type": "VARCHAR"},
                ],
                "row_count": 500,
            },
        )

        memories = await mem0.memory.list_memories(memory_type=MemoryType.SCHEMA)
        assert len(memories) == 1
        assert "customers" in memories[0].content

        await mem0.close()

    @pytest.mark.asyncio
    async def test_fact_extraction(self) -> None:
        """Test automatic fact extraction."""
        mem0 = create_mem0_layer(use_simple=True)

        text = """
        Analysis complete. The total revenue was $1,234,567 in Q4.
        There were 5,432 transactions processed.
        The highest sale was $15,000 from customer #123.
        Revenue increased by 25% compared to Q3.
        """

        facts = await mem0.extract_and_store_facts(text, source="analyst")

        # Should extract multiple facts
        assert len(facts) > 0

        await mem0.close()


class TestMemoryContextRetrieval:
    """Test memory context retrieval for agents."""

    @pytest.mark.asyncio
    async def test_context_for_query(self) -> None:
        """Test getting context for agent queries."""
        mem0 = create_mem0_layer(user_id="user1", use_simple=True)

        # Store some context
        await mem0.store_schema_info(
            "orders",
            {"columns": [{"name": "id", "type": "INT"}], "row_count": 100},
        )

        await mem0.store_query_pattern(
            query="Count orders",
            sql="SELECT COUNT(*) FROM orders",
            success=True,
        )

        # Get context
        context = await mem0.get_context_for_query("How many orders?")

        assert isinstance(context, list)

        await mem0.close()
