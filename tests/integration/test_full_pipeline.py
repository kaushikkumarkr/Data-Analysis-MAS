"""Integration tests for the full DataVault pipeline.

Tests end-to-end workflow from data loading through agent execution.
"""

import pytest
from pathlib import Path

from src.mcp.client import create_client
from src.agents.state import TaskType
from src.agents.graph import DataVaultGraph


@pytest.fixture
def sample_data_path() -> str:
    """Get path to sample data."""
    return str(Path(__file__).parent.parent.parent / "data" / "sample" / "sales_data.csv")


class TestFullPipeline:
    """Integration tests for full pipeline."""

    def test_load_and_list_tables(self, sample_data_path: str) -> None:
        """Test loading data and listing tables."""
        with create_client() as client:
            # Load data
            result = client.load_dataset(sample_data_path, "test_sales")
            assert result.success
            assert result.data["rows_loaded"] == 50

            # List tables
            tables_result = client.list_tables()
            assert tables_result.success
            assert "test_sales" in tables_result.data["tables"]

            # Describe table
            schema = client.describe_table("test_sales")
            assert schema.success
            assert len(schema.data["columns"]) > 0

    def test_execute_sql_query(self, sample_data_path: str) -> None:
        """Test SQL query execution."""
        with create_client() as client:
            client.load_dataset(sample_data_path, "sales")

            # Execute query
            result = client.execute_sql("SELECT COUNT(*) as cnt FROM sales")

            assert result.success
            assert result.data["row_count"] == 1
            assert result.data["rows"][0]["cnt"] == 50

    def test_aggregation_query(self, sample_data_path: str) -> None:
        """Test aggregation query."""
        with create_client() as client:
            client.load_dataset(sample_data_path, "sales")

            result = client.execute_sql("""
                SELECT region, SUM(total_amount) as total
                FROM sales
                GROUP BY region
                ORDER BY total DESC
            """)

            assert result.success
            assert result.data["row_count"] == 3  # 3 regions

    def test_statistics_calculation(self, sample_data_path: str) -> None:
        """Test statistics calculation."""
        with create_client() as client:
            client.load_dataset(sample_data_path, "sales")

            stats = client.get_statistics("sales", "total_amount")

            assert stats.success
            assert stats.data["count"] == 50
            assert stats.data["min"] > 0
            assert stats.data["max"] > stats.data["min"]
            assert stats.data["mean"] > 0


class TestAgentStateCreation:
    """Test agent state creation."""

    def test_task_type_enum(self) -> None:
        """Test TaskType enum values."""
        assert TaskType.CLEAN.value == "clean"
        assert TaskType.ANALYZE.value == "analyze"
        assert TaskType.VISUALIZE.value == "visualize"


class TestGraphInitialization:
    """Test graph initialization."""

    def test_graph_creation(self, sample_data_path: str) -> None:
        """Test creating the graph."""
        with create_client() as client:
            graph = DataVaultGraph(client)
            assert graph.graph is not None

    def test_graph_with_backend_param(self, sample_data_path: str) -> None:
        """Test graph with explicit backend."""
        with create_client() as client:
            # Should not raise even without MLX
            graph = DataVaultGraph(client, backend="ollama")
            assert graph is not None


class TestDataQualityChecks:
    """Test data quality functionality."""

    def test_null_detection(self, sample_data_path: str) -> None:
        """Test detecting null values."""
        with create_client() as client:
            client.load_dataset(sample_data_path, "sales")

            # Check for nulls
            result = client.execute_sql("""
                SELECT 
                    SUM(CASE WHEN product_name IS NULL THEN 1 ELSE 0 END) as null_count
                FROM sales
            """)

            assert result.success
            # Sample data should have no nulls
            assert result.data["rows"][0]["null_count"] == 0

    def test_duplicate_detection(self, sample_data_path: str) -> None:
        """Test detecting duplicates."""
        with create_client() as client:
            client.load_dataset(sample_data_path, "sales")

            result = client.execute_sql("""
                SELECT transaction_id, COUNT(*) as cnt
                FROM sales
                GROUP BY transaction_id
                HAVING COUNT(*) > 1
            """)

            assert result.success
            # Should have no duplicates on transaction_id
            assert result.data["row_count"] == 0


class TestMultipleDatasets:
    """Test working with multiple datasets."""

    def test_load_multiple(self, sample_data_path: str) -> None:
        """Test loading multiple datasets."""
        with create_client() as client:
            # Load same file with different names
            client.load_dataset(sample_data_path, "sales_2024")
            client.load_dataset(sample_data_path, "sales_2023")

            tables = client.list_tables()

            assert tables.success
            assert "sales_2024" in tables.data["tables"]
            assert "sales_2023" in tables.data["tables"]

    def test_cross_table_query(self, sample_data_path: str) -> None:
        """Test querying across tables."""
        with create_client() as client:
            client.load_dataset(sample_data_path, "current")
            client.load_dataset(sample_data_path, "previous")

            result = client.execute_sql("""
                SELECT 
                    (SELECT COUNT(*) FROM current) as current_count,
                    (SELECT COUNT(*) FROM previous) as previous_count
            """)

            assert result.success
            assert result.data["rows"][0]["current_count"] == 50
            assert result.data["rows"][0]["previous_count"] == 50
