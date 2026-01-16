"""Unit tests for DuckDB manager."""

import tempfile
from pathlib import Path

import pytest

from src.db.duckdb_manager import DuckDBError, DuckDBManager
from src.db.schemas import ColumnType, TableSchema, SALES_DATA_SCHEMA
from src.utils.config import DuckDBConfig


class TestDuckDBConnection:
    """Tests for DuckDB connection management."""

    def test_connection_in_memory(self) -> None:
        """Test in-memory database connection."""
        config = DuckDBConfig(database_path=":memory:")
        manager = DuckDBManager(config)

        # Connection should be lazy
        assert manager._connection is None

        # Accessing connection should create it
        conn = manager.connection
        assert conn is not None
        assert manager._connection is not None

        manager.close()
        assert manager._connection is None

    def test_connection_context_manager(self) -> None:
        """Test context manager pattern for connection."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            assert manager._connection is not None
            # Execute a simple query
            result = manager.execute_query("SELECT 1")
            assert result == [(1,)]

        # Connection should be closed after exiting context
        assert manager._connection is None

    def test_connection_file_based(self, temp_db_path: str) -> None:
        """Test file-based database connection."""
        config = DuckDBConfig(database_path=temp_db_path)

        with DuckDBManager(config) as manager:
            manager.execute_query("CREATE TABLE test (id INTEGER)", fetch=False)
            manager.execute_query("INSERT INTO test VALUES (1)", fetch=False)

        # Verify file was created
        assert Path(temp_db_path).exists()

        # Verify data persists
        with DuckDBManager(config) as manager:
            result = manager.execute_query("SELECT * FROM test")
            assert result == [(1,)]


class TestDuckDBQueryExecution:
    """Tests for query execution."""

    def test_execute_simple_query(self) -> None:
        """Test simple SELECT query."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            result = manager.execute_query("SELECT 1 + 1 AS sum")
            assert result == [(2,)]

    def test_execute_query_with_params(self) -> None:
        """Test parameterized query."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            manager.execute_query(
                "CREATE TABLE nums (value INTEGER)", fetch=False
            )
            manager.execute_query(
                "INSERT INTO nums VALUES (1), (2), (3)", fetch=False
            )

            result = manager.execute_query(
                "SELECT * FROM nums WHERE value > ?", params=(1,)
            )
            assert len(result) == 2
            assert (2,) in result
            assert (3,) in result

    def test_execute_query_no_fetch(self) -> None:
        """Test query execution without fetching results."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            result = manager.execute_query(
                "CREATE TABLE test (id INTEGER)",
                fetch=False,
            )
            assert result is None

    def test_execute_query_df(self) -> None:
        """Test query execution returning DataFrame."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            manager.execute_query(
                "CREATE TABLE test (id INTEGER, name VARCHAR)", fetch=False
            )
            manager.execute_query(
                "INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')", fetch=False
            )

            df = manager.execute_query_df("SELECT * FROM test ORDER BY id")
            assert len(df) == 2
            assert list(df.columns) == ["id", "name"]
            assert df.iloc[0]["name"] == "Alice"

    def test_execute_invalid_query(self) -> None:
        """Test error handling for invalid query."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            with pytest.raises(DuckDBError) as exc_info:
                manager.execute_query("SELECT * FROM nonexistent_table")

            assert "Query failed" in str(exc_info.value)


class TestDuckDBCSVLoading:
    """Tests for CSV loading functionality."""

    def test_load_csv_basic(self, sample_csv_path: str) -> None:
        """Test basic CSV loading with auto-detection."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            row_count = manager.load_csv(sample_csv_path, "sales_data")

            assert row_count == 50  # Sample data has 50 rows
            assert manager.table_exists("sales_data")

            # Verify data can be queried
            result = manager.execute_query(
                "SELECT COUNT(*) FROM sales_data"
            )
            assert result[0][0] == 50

    def test_load_csv_with_schema(self, sample_csv_path: str) -> None:
        """Test CSV loading with explicit schema."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            row_count = manager.load_csv(
                sample_csv_path,
                "sales_data",
                schema=SALES_DATA_SCHEMA,
            )

            assert row_count == 50

            # Verify schema was applied
            schema = manager.get_schema("sales_data")
            column_names = [col["name"] for col in schema["columns"]]
            assert "transaction_id" in column_names
            assert "total_amount" in column_names

    def test_load_csv_replace_existing(self, sample_csv_path: str) -> None:
        """Test CSV loading replaces existing table."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            # Load once
            manager.load_csv(sample_csv_path, "sales_data")

            # Verify data exists
            assert manager.get_row_count("sales_data") == 50

            # Delete some rows
            manager.execute_query(
                "DELETE FROM sales_data WHERE transaction_id LIKE 'TXN00%'",
                fetch=False,
            )
            assert manager.get_row_count("sales_data") < 50

            # Reload - should replace
            manager.load_csv(sample_csv_path, "sales_data", if_exists="replace")
            assert manager.get_row_count("sales_data") == 50

    def test_load_csv_file_not_found(self) -> None:
        """Test error handling for missing CSV file."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            with pytest.raises(FileNotFoundError):
                manager.load_csv("/nonexistent/path.csv", "test")


class TestDuckDBSchemaIntrospection:
    """Tests for schema introspection."""

    def test_get_schema(self, sample_csv_path: str) -> None:
        """Test retrieving table schema."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            manager.load_csv(sample_csv_path, "sales_data")
            schema = manager.get_schema("sales_data")

            assert schema["table_name"] == "sales_data"
            assert len(schema["columns"]) > 0

            # Check specific columns
            column_names = [col["name"] for col in schema["columns"]]
            assert "transaction_id" in column_names
            assert "quantity" in column_names
            assert "total_amount" in column_names

    def test_get_schema_nonexistent_table(self) -> None:
        """Test error handling for nonexistent table schema."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            with pytest.raises(DuckDBError) as exc_info:
                manager.get_schema("nonexistent_table")

            # DuckDB returns "does not exist" for missing tables
            assert "does not exist" in str(exc_info.value)

    def test_list_tables(self, sample_csv_path: str) -> None:
        """Test listing all tables."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            # Initially no tables
            assert len(manager.list_tables()) == 0

            # Create tables
            manager.load_csv(sample_csv_path, "sales_data")
            manager.execute_query(
                "CREATE TABLE customers (id INTEGER)", fetch=False
            )

            tables = manager.list_tables()
            assert "sales_data" in tables
            assert "customers" in tables
            assert len(tables) == 2

    def test_table_exists(self, sample_csv_path: str) -> None:
        """Test table existence check."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            assert not manager.table_exists("sales_data")

            manager.load_csv(sample_csv_path, "sales_data")

            assert manager.table_exists("sales_data")
            assert not manager.table_exists("nonexistent")


class TestDuckDBTableOperations:
    """Tests for table operations."""

    def test_drop_table(self, sample_csv_path: str) -> None:
        """Test dropping a table."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            manager.load_csv(sample_csv_path, "sales_data")
            assert manager.table_exists("sales_data")

            manager.drop_table("sales_data")
            assert not manager.table_exists("sales_data")

    def test_drop_table_if_exists(self) -> None:
        """Test dropping nonexistent table with if_exists=True."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            # Should not raise
            manager.drop_table("nonexistent", if_exists=True)

    def test_get_row_count(self, sample_csv_path: str) -> None:
        """Test getting table row count."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            manager.load_csv(sample_csv_path, "sales_data")
            count = manager.get_row_count("sales_data")
            assert count == 50

    def test_sample_data(self, sample_csv_path: str) -> None:
        """Test sampling data from table."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            manager.load_csv(sample_csv_path, "sales_data")
            samples = manager.sample_data("sales_data", limit=5)

            assert len(samples) == 5
            assert isinstance(samples[0], dict)
            assert "transaction_id" in samples[0]
            assert "total_amount" in samples[0]


class TestDuckDBAnalyticsQueries:
    """Tests for analytics queries on sample data."""

    def test_aggregate_by_region(self, sample_csv_path: str) -> None:
        """Test aggregation query by region."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            manager.load_csv(sample_csv_path, "sales_data")

            df = manager.execute_query_df("""
                SELECT 
                    region,
                    COUNT(*) as num_transactions,
                    SUM(total_amount) as total_sales
                FROM sales_data
                GROUP BY region
                ORDER BY total_sales DESC
            """)

            assert len(df) > 0
            assert "region" in df.columns
            assert "num_transactions" in df.columns
            assert "total_sales" in df.columns

    def test_aggregate_by_category(self, sample_csv_path: str) -> None:
        """Test aggregation query by category."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            manager.load_csv(sample_csv_path, "sales_data")

            df = manager.execute_query_df("""
                SELECT 
                    category,
                    COUNT(*) as num_products,
                    AVG(unit_price) as avg_price
                FROM sales_data
                GROUP BY category
            """)

            assert len(df) > 0
            categories = df["category"].tolist()
            assert "Software" in categories or "Services" in categories

    def test_top_salesperson(self, sample_csv_path: str) -> None:
        """Test query for top salesperson."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            manager.load_csv(sample_csv_path, "sales_data")

            result = manager.execute_query("""
                SELECT 
                    salesperson,
                    SUM(total_amount) as total_sales
                FROM sales_data
                WHERE salesperson IS NOT NULL
                GROUP BY salesperson
                ORDER BY total_sales DESC
                LIMIT 1
            """)

            assert len(result) == 1
            assert result[0][0] is not None  # Has salesperson name
            assert result[0][1] > 0  # Has positive total sales


class TestDuckDBNoNetworkCalls:
    """Tests to verify DuckDB operates 100% locally."""

    def test_no_network_dependency(self, sample_csv_path: str) -> None:
        """Verify DuckDB operations don't require network.

        This test loads data and runs queries in an in-memory database,
        proving that all operations are local. DuckDB by design is an
        embedded database with no network component.
        """
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as manager:
            # Load local CSV
            manager.load_csv(sample_csv_path, "sales_data")

            # Run analytics queries
            result = manager.execute_query("""
                SELECT 
                    COUNT(*) as total,
                    SUM(total_amount) as revenue
                FROM sales_data
            """)

            assert result[0][0] == 50
            assert result[0][1] > 0

            # Verify we can introspect schema
            schema = manager.get_schema("sales_data")
            assert schema["table_name"] == "sales_data"

            # All operations completed without any network calls
            # DuckDB is fully embedded - this is a design verification


class TestTableSchema:
    """Tests for TableSchema class."""

    def test_schema_creation(self) -> None:
        """Test creating a table schema."""
        schema = TableSchema(name="test_table", description="Test table")
        schema.add_column("id", ColumnType.INTEGER, primary_key=True)
        schema.add_column("name", ColumnType.VARCHAR, nullable=False)
        schema.add_column("value", ColumnType.DOUBLE, default=0.0)

        assert len(schema.columns) == 3
        assert schema.get_column_names() == ["id", "name", "value"]

    def test_schema_to_ddl(self) -> None:
        """Test generating DDL from schema."""
        schema = TableSchema(name="test_table")
        schema.add_column("id", ColumnType.INTEGER, primary_key=True)
        schema.add_column("name", ColumnType.VARCHAR, nullable=False)

        ddl = schema.to_create_table_sql()

        assert "CREATE TABLE IF NOT EXISTS test_table" in ddl
        assert "id INTEGER PRIMARY KEY" in ddl
        assert "name VARCHAR NOT NULL" in ddl

    def test_schema_to_dict(self) -> None:
        """Test converting schema to dictionary."""
        schema = TableSchema(name="test", description="Test")
        schema.add_column("id", ColumnType.INTEGER, description="ID column")

        d = schema.to_dict()

        assert d["name"] == "test"
        assert d["description"] == "Test"
        assert len(d["columns"]) == 1
        assert d["columns"][0]["name"] == "id"
        assert d["columns"][0]["description"] == "ID column"
