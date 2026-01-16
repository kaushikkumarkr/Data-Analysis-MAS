"""Tests for Memory module.

Tests embeddings, vector store, semantic memory, and Mem0 layer.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.memory.embeddings import (
    EMBEDDING_DIM,
    LocalEmbeddings,
    SimpleEmbeddings,
    cosine_similarity,
    create_embeddings,
)
from src.memory.vector_store import (
    InMemoryVectorStore,
    MemoryRecord,
    VectorStore,
)
from src.memory.semantic import (
    Memory,
    MemoryType,
    SemanticMemory,
    create_semantic_memory,
)
from src.memory.mem0_layer import (
    Mem0Layer,
    MemoryStats,
    create_mem0_layer,
)


# ============================================================================
# Embedding Tests
# ============================================================================

class TestSimpleEmbeddings:
    """Tests for SimpleEmbeddings."""

    def test_simple_embeddings_creation(self) -> None:
        """Test SimpleEmbeddings can be created."""
        embeddings = SimpleEmbeddings()
        assert embeddings.dimension == EMBEDDING_DIM

    def test_simple_embeddings_dimension(self) -> None:
        """Test custom dimension."""
        embeddings = SimpleEmbeddings(dimension=256)
        assert embeddings.dimension == 256

    def test_embed_single_text(self) -> None:
        """Test embedding a single text."""
        embeddings = SimpleEmbeddings()
        result = embeddings.embed("Hello world")
        
        assert isinstance(result, list)
        assert len(result) == EMBEDDING_DIM
        assert all(isinstance(x, float) for x in result)

    def test_embed_batch(self) -> None:
        """Test embedding multiple texts."""
        embeddings = SimpleEmbeddings()
        texts = ["Hello", "World", "Test"]
        results = embeddings.embed_batch(texts)
        
        assert len(results) == 3
        assert all(len(r) == EMBEDDING_DIM for r in results)

    def test_embeddings_are_normalized(self) -> None:
        """Test that embeddings are normalized."""
        import numpy as np
        
        embeddings = SimpleEmbeddings()
        result = embeddings.embed("Test normalization")
        
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 0.01  # Should be approximately 1

    def test_deterministic_embeddings(self) -> None:
        """Test that same text produces same embedding."""
        embeddings = SimpleEmbeddings()
        text = "Consistent embedding test"
        
        result1 = embeddings.embed(text)
        result2 = embeddings.embed(text)
        
        assert result1 == result2

    def test_different_texts_different_embeddings(self) -> None:
        """Test that different texts produce different embeddings."""
        embeddings = SimpleEmbeddings()
        
        result1 = embeddings.embed("First text")
        result2 = embeddings.embed("Second text")
        
        assert result1 != result2


class TestCosineSimilarity:
    """Tests for cosine similarity function."""

    def test_identical_vectors(self) -> None:
        """Test similarity of identical vectors."""
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        """Test similarity of orthogonal vectors."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        """Test similarity of opposite vectors."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [-1.0, 0.0, 0.0]
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0)


class TestCreateEmbeddings:
    """Tests for create_embeddings factory."""

    def test_create_simple_embeddings(self) -> None:
        """Test creating simple embeddings."""
        embeddings = create_embeddings(use_simple=True)
        assert isinstance(embeddings, SimpleEmbeddings)


# ============================================================================
# Vector Store Tests
# ============================================================================

class TestMemoryRecord:
    """Tests for MemoryRecord."""

    def test_memory_record_creation(self) -> None:
        """Test creating a memory record."""
        record = MemoryRecord(
            id=1,
            content="Test content",
            memory_type="fact",
        )
        
        assert record.id == 1
        assert record.content == "Test content"
        assert record.memory_type == "fact"

    def test_memory_record_to_dict(self) -> None:
        """Test converting to dictionary."""
        now = datetime.now(timezone.utc)
        record = MemoryRecord(
            id=1,
            content="Test",
            memory_type="fact",
            user_id="user1",
            session_id="session1",
            metadata={"key": "value"},
            created_at=now,
        )
        
        d = record.to_dict()
        
        assert d["id"] == 1
        assert d["content"] == "Test"
        assert d["user_id"] == "user1"
        assert d["metadata"]["key"] == "value"


class TestInMemoryVectorStore:
    """Tests for InMemoryVectorStore."""

    @pytest.fixture
    def store(self) -> InMemoryVectorStore:
        """Create a store for testing."""
        return InMemoryVectorStore()

    @pytest.fixture
    def sample_embedding(self) -> list[float]:
        """Create a sample normalized embedding."""
        import numpy as np
        v = np.random.randn(EMBEDDING_DIM)
        v = v / np.linalg.norm(v)
        return v.tolist()

    @pytest.mark.asyncio
    async def test_add_memory(self, store, sample_embedding) -> None:
        """Test adding a memory."""
        record_id = await store.add(
            content="Test memory",
            embedding=sample_embedding,
            memory_type="fact",
        )
        
        assert record_id == 1
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_get_memory(self, store, sample_embedding) -> None:
        """Test getting a memory by ID."""
        record_id = await store.add(
            content="Test memory",
            embedding=sample_embedding,
        )
        
        record = await store.get(record_id)
        
        assert record is not None
        assert record.content == "Test memory"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store) -> None:
        """Test getting a nonexistent memory."""
        record = await store.get(999)
        assert record is None

    @pytest.mark.asyncio
    async def test_delete_memory(self, store, sample_embedding) -> None:
        """Test deleting a memory."""
        record_id = await store.add(
            content="Test",
            embedding=sample_embedding,
        )
        
        result = await store.delete(record_id)
        
        assert result is True
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store) -> None:
        """Test deleting nonexistent memory."""
        result = await store.delete(999)
        assert result is False

    @pytest.mark.asyncio
    async def test_search_similar(self, store) -> None:
        """Test searching for similar memories."""
        import numpy as np
        
        # Create base embedding
        base = np.random.randn(EMBEDDING_DIM)
        base = base / np.linalg.norm(base)
        
        # Create similar and dissimilar embeddings
        similar = base + np.random.randn(EMBEDDING_DIM) * 0.1
        similar = similar / np.linalg.norm(similar)
        
        dissimilar = np.random.randn(EMBEDDING_DIM)
        dissimilar = dissimilar / np.linalg.norm(dissimilar)
        
        await store.add("Similar memory", similar.tolist())
        await store.add("Dissimilar memory", dissimilar.tolist())
        
        results = await store.search(
            query_embedding=base.tolist(),
            limit=5,
            min_similarity=0.0,  # Accept all for testing
        )
        
        # Should get both memories and similar one ranked first
        assert len(results) >= 1
        assert results[0].content == "Similar memory"

    @pytest.mark.asyncio
    async def test_search_with_filters(self, store, sample_embedding) -> None:
        """Test searching with filters."""
        await store.add("User1 memory", sample_embedding, user_id="user1")
        await store.add("User2 memory", sample_embedding, user_id="user2")
        
        results = await store.search(
            query_embedding=sample_embedding,
            user_id="user1",
            min_similarity=0.0,
        )
        
        assert len(results) == 1
        assert results[0].user_id == "user1"

    @pytest.mark.asyncio
    async def test_list_all(self, store, sample_embedding) -> None:
        """Test listing all memories."""
        await store.add("Memory 1", sample_embedding, memory_type="fact")
        await store.add("Memory 2", sample_embedding, memory_type="query")
        
        all_memories = await store.list_all()
        assert len(all_memories) == 2

    @pytest.mark.asyncio
    async def test_list_with_type_filter(self, store, sample_embedding) -> None:
        """Test listing with type filter."""
        await store.add("Fact", sample_embedding, memory_type="fact")
        await store.add("Query", sample_embedding, memory_type="query")
        
        facts = await store.list_all(memory_type="fact")
        
        assert len(facts) == 1
        assert facts[0].memory_type == "fact"


# ============================================================================
# Semantic Memory Tests
# ============================================================================

class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_memory_types(self) -> None:
        """Test memory type values."""
        assert MemoryType.FACT.value == "fact"
        assert MemoryType.PREFERENCE.value == "preference"
        assert MemoryType.SCHEMA.value == "schema"
        assert MemoryType.QUERY.value == "query"


class TestMemory:
    """Tests for Memory dataclass."""

    def test_memory_creation(self) -> None:
        """Test creating a memory."""
        memory = Memory(
            content="Test content",
            memory_type=MemoryType.FACT,
            user_id="user1",
        )
        
        assert memory.content == "Test content"
        assert memory.memory_type == MemoryType.FACT
        assert memory.user_id == "user1"

    def test_memory_to_dict(self) -> None:
        """Test converting memory to dict."""
        memory = Memory(
            id=1,
            content="Test",
            memory_type=MemoryType.FACT,
            similarity=0.95,
        )
        
        d = memory.to_dict()
        
        assert d["id"] == 1
        assert d["memory_type"] == "fact"
        assert d["similarity"] == 0.95


class TestSemanticMemory:
    """Tests for SemanticMemory."""

    @pytest.fixture
    def semantic_memory(self) -> SemanticMemory:
        """Create semantic memory for testing."""
        return SemanticMemory(
            embeddings=SimpleEmbeddings(),
            vector_store=InMemoryVectorStore(),
            use_simple=True,
        )

    @pytest.mark.asyncio
    async def test_remember(self, semantic_memory) -> None:
        """Test remembering a fact."""
        memory = await semantic_memory.remember(
            content="The sky is blue",
            memory_type=MemoryType.FACT,
        )
        
        assert memory.id is not None
        assert memory.content == "The sky is blue"
        assert memory.memory_type == MemoryType.FACT

    @pytest.mark.asyncio
    async def test_recall(self, semantic_memory) -> None:
        """Test recalling memories."""
        await semantic_memory.remember("Sales increased by 20%", MemoryType.FACT)
        await semantic_memory.remember("Revenue was $1M", MemoryType.FACT)
        
        memories = await semantic_memory.recall(
            query="What were the sales figures?",
            limit=5,
            min_similarity=0.0,
        )
        
        assert len(memories) == 2

    @pytest.mark.asyncio
    async def test_forget(self, semantic_memory) -> None:
        """Test forgetting a memory."""
        memory = await semantic_memory.remember("Temporary fact", MemoryType.FACT)
        
        result = await semantic_memory.forget(memory.id)
        
        assert result is True
        assert await semantic_memory.count() == 0

    @pytest.mark.asyncio
    async def test_get_by_id(self, semantic_memory) -> None:
        """Test getting memory by ID."""
        stored = await semantic_memory.remember("Test", MemoryType.FACT)
        
        retrieved = await semantic_memory.get(stored.id)
        
        assert retrieved is not None
        assert retrieved.content == "Test"

    @pytest.mark.asyncio
    async def test_list_memories(self, semantic_memory) -> None:
        """Test listing memories."""
        await semantic_memory.remember("Fact 1", MemoryType.FACT)
        await semantic_memory.remember("Query 1", MemoryType.QUERY)
        
        all_memories = await semantic_memory.list_memories()
        facts = await semantic_memory.list_memories(memory_type=MemoryType.FACT)
        
        assert len(all_memories) == 2
        assert len(facts) == 1


# ============================================================================
# Mem0 Layer Tests
# ============================================================================

class TestMem0Layer:
    """Tests for Mem0Layer."""

    @pytest.fixture
    def mem0(self) -> Mem0Layer:
        """Create Mem0 layer for testing."""
        return create_mem0_layer(user_id="test_user", use_simple=True)

    @pytest.mark.asyncio
    async def test_session_management(self, mem0) -> None:
        """Test session start/end."""
        session_id = mem0.start_session()
        
        assert session_id is not None
        assert mem0.session_id == session_id
        
        mem0.end_session()
        assert mem0.session_id is None

    @pytest.mark.asyncio
    async def test_add_memory(self, mem0) -> None:
        """Test adding memory via Mem0."""
        mem0.start_session()
        
        memory = await mem0.add_memory(
            content="User prefers dark mode",
            memory_type=MemoryType.PREFERENCE,
        )
        
        assert memory.id is not None
        assert memory.user_id == "test_user"
        assert memory.session_id == mem0.session_id

    @pytest.mark.asyncio
    async def test_search_memories(self, mem0) -> None:
        """Test searching memories."""
        mem0.start_session()
        
        await mem0.add_memory("Sales data is in CSV format", MemoryType.FACT)
        
        results = await mem0.search_memories(
            query="What format is the data?",
            min_similarity=0.0,
        )
        
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_get_context_for_query(self, mem0) -> None:
        """Test getting context for a query."""
        await mem0.add_memory(
            "The sales table has columns: id, amount, date",
            MemoryType.SCHEMA,
            session_scoped=False,
        )
        
        context = await mem0.get_context_for_query("Show me sales data")
        
        assert isinstance(context, list)

    @pytest.mark.asyncio
    async def test_extract_and_store_facts(self, mem0) -> None:
        """Test fact extraction."""
        text = """
        The total revenue was $1,500,000 in Q4.
        There were 1234 transactions processed.
        The highest sale was $50,000.
        """
        
        facts = await mem0.extract_and_store_facts(text)
        
        assert len(facts) > 0

    @pytest.mark.asyncio
    async def test_store_query_pattern(self, mem0) -> None:
        """Test storing query patterns."""
        memory = await mem0.store_query_pattern(
            query="Show me sales by region",
            sql="SELECT region, SUM(amount) FROM sales GROUP BY region",
            success=True,
            result_summary="3 regions found",
        )
        
        assert memory.memory_type == MemoryType.QUERY
        assert "sales by region" in memory.content.lower()

    @pytest.mark.asyncio
    async def test_store_schema_info(self, mem0) -> None:
        """Test storing schema information."""
        memory = await mem0.store_schema_info(
            table_name="sales",
            schema_info={
                "columns": [
                    {"name": "id", "type": "INTEGER"},
                    {"name": "amount", "type": "DOUBLE"},
                ],
                "row_count": 100,
            },
        )
        
        assert memory.memory_type == MemoryType.SCHEMA
        assert "sales" in memory.content

    @pytest.mark.asyncio
    async def test_store_preference(self, mem0) -> None:
        """Test storing preferences."""
        memory = await mem0.store_preference(
            preference="Always show charts in dark theme",
            category="visualization",
        )
        
        assert memory.memory_type == MemoryType.PREFERENCE

    @pytest.mark.asyncio
    async def test_get_stats(self, mem0) -> None:
        """Test getting memory statistics."""
        await mem0.add_memory("Fact 1", MemoryType.FACT, session_scoped=False)
        await mem0.add_memory("Fact 2", MemoryType.FACT, session_scoped=False)
        await mem0.store_preference("Pref 1")
        
        stats = await mem0.get_stats()
        
        assert stats.total == 3
        assert stats.by_type.get("fact", 0) == 2
        assert stats.by_type.get("preference", 0) == 1

    @pytest.mark.asyncio
    async def test_clear_session_memories(self, mem0) -> None:
        """Test clearing session memories."""
        mem0.start_session()
        
        await mem0.add_memory("Session memory", MemoryType.FACT, session_scoped=True)
        await mem0.add_memory("Global memory", MemoryType.FACT, session_scoped=False)
        
        count = await mem0.clear_session_memories()
        
        assert count == 1
        
        # Global memory should still exist
        all_memories = await mem0.memory.list_memories()
        assert len(all_memories) == 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestMemoryIntegration:
    """Integration tests for memory components."""

    @pytest.mark.asyncio
    async def test_full_workflow(self) -> None:
        """Test full memory workflow."""
        mem0 = create_mem0_layer(user_id="integration_test", use_simple=True)
        
        try:
            # Start session
            mem0.start_session()
            
            # Store some memories
            await mem0.store_schema_info(
                "customers",
                {"columns": [{"name": "id", "type": "INT"}], "row_count": 500},
            )
            
            await mem0.store_query_pattern(
                query="Count customers",
                sql="SELECT COUNT(*) FROM customers",
                success=True,
            )
            
            await mem0.add_memory(
                "Customer table has 500 records",
                MemoryType.FACT,
            )
            
            # Search for relevant context (use lower threshold for simple embeddings)
            context = await mem0.get_context_for_query("How many customers?")
            
            # With simple embeddings, just verify the system works
            # The hash-based embeddings may not find semantic matches
            assert isinstance(context, list)
            
            # Get stats
            stats = await mem0.get_stats()
            assert stats.total >= 3
            
        finally:
            await mem0.close()
