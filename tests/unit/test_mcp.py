"""Unit tests for MCP tools and server."""

import json
from pathlib import Path

import pytest

from src.db.duckdb_manager import DuckDBManager
from src.mcp.tools import DuckDBTools, ToolResult
from src.mcp.client import MCPClient, ToolCall, create_client
from src.mcp.server import DuckDBMCPServer, create_server
from src.utils.config import DuckDBConfig


class TestToolResult:
    """Tests for ToolResult class."""

    def test_success_result(self) -> None:
        """Test successful result serialization."""
        result = ToolResult(success=True, data={"count": 10})
        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["success"] is True
        assert parsed["data"]["count"] == 10

    def test_error_result(self) -> None:
        """Test error result serialization."""
        result = ToolResult(success=False, error="Something went wrong")
        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["success"] is False
        assert parsed["error"] == "Something went wrong"


class TestDuckDBTools:
    """Tests for DuckDBTools."""

    def test_tools_initialization(self) -> None:
        """Test tools can be initialized."""
        config = DuckDBConfig(database_path=":memory:")
        db_manager = DuckDBManager(config)

        with db_manager:
            tools = DuckDBTools(db_manager)
            assert tools.db is db_manager

    def test_execute_sql_success(self, sample_csv_path: str) -> None:
        """Test successful SQL execution."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as db:
            tools = DuckDBTools(db)

            # Load data first
            db.load_csv(sample_csv_path, "sales_data")

            # Execute query
            result = tools.execute_sql("SELECT COUNT(*) as cnt FROM sales_data")

            assert result.success is True
            assert result.data["row_count"] == 1
            assert result.data["rows"][0]["cnt"] == 50

    def test_execute_sql_invalid_query(self) -> None:
        """Test SQL execution with invalid query."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as db:
            tools = DuckDBTools(db)
            result = tools.execute_sql("SELECT * FROM nonexistent")

            assert result.success is False
            assert "error" in result.error.lower() or "not exist" in result.error.lower()

    def test_list_tables_empty(self) -> None:
        """Test listing tables when empty."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as db:
            tools = DuckDBTools(db)
            result = tools.list_tables()

            assert result.success is True
            assert result.data["count"] == 0
            assert result.data["tables"] == []

    def test_list_tables_with_data(self, sample_csv_path: str) -> None:
        """Test listing tables after loading data."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as db:
            tools = DuckDBTools(db)

            # Load data
            db.load_csv(sample_csv_path, "sales_data")
            db.execute_query("CREATE TABLE customers (id INTEGER)", fetch=False)

            result = tools.list_tables()

            assert result.success is True
            assert result.data["count"] == 2
            assert "sales_data" in result.data["tables"]
            assert "customers" in result.data["tables"]

    def test_describe_table(self, sample_csv_path: str) -> None:
        """Test describing a table's schema."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as db:
            tools = DuckDBTools(db)
            db.load_csv(sample_csv_path, "sales_data")

            result = tools.describe_table("sales_data")

            assert result.success is True
            assert result.data["table_name"] == "sales_data"
            assert len(result.data["columns"]) > 0
            assert result.data["row_count"] == 50
            assert "sample_data" in result.data

    def test_describe_table_nonexistent(self) -> None:
        """Test describing a nonexistent table."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as db:
            tools = DuckDBTools(db)
            result = tools.describe_table("nonexistent_table")

            assert result.success is False
            assert "not exist" in result.error.lower() or "error" in result.error.lower()

    def test_load_dataset(self, sample_csv_path: str) -> None:
        """Test loading a CSV dataset."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as db:
            tools = DuckDBTools(db)
            result = tools.load_dataset(sample_csv_path, "sales_data")

            assert result.success is True
            assert result.data["table_name"] == "sales_data"
            assert result.data["rows_loaded"] == 50
            assert len(result.data["columns"]) > 0

    def test_load_dataset_file_not_found(self) -> None:
        """Test loading a nonexistent file."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as db:
            tools = DuckDBTools(db)
            result = tools.load_dataset("/nonexistent/path.csv", "test")

            assert result.success is False
            assert "not found" in result.error.lower()

    def test_get_statistics_numeric(self, sample_csv_path: str) -> None:
        """Test getting statistics for a numeric column."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as db:
            tools = DuckDBTools(db)
            db.load_csv(sample_csv_path, "sales_data")

            result = tools.get_statistics("sales_data", "total_amount")

            assert result.success is True
            assert result.data["column_name"] == "total_amount"
            assert "count" in result.data
            assert "min" in result.data
            assert "max" in result.data
            assert "mean" in result.data

    def test_get_statistics_categorical(self, sample_csv_path: str) -> None:
        """Test getting statistics for a categorical column."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as db:
            tools = DuckDBTools(db)
            db.load_csv(sample_csv_path, "sales_data")

            result = tools.get_statistics("sales_data", "region")

            assert result.success is True
            assert result.data["column_name"] == "region"
            assert "unique_count" in result.data
            assert "top_values" in result.data

    def test_get_statistics_nonexistent_column(self, sample_csv_path: str) -> None:
        """Test getting statistics for a nonexistent column."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as db:
            tools = DuckDBTools(db)
            db.load_csv(sample_csv_path, "sales_data")

            result = tools.get_statistics("sales_data", "nonexistent_column")

            assert result.success is False
            assert "not found" in result.error.lower()

    def test_get_tool_definitions(self) -> None:
        """Test getting tool definitions for MCP."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as db:
            tools = DuckDBTools(db)
            definitions = tools.get_tool_definitions()

            assert len(definitions) >= 4
            tool_names = [t["name"] for t in definitions]
            assert "execute_sql" in tool_names
            assert "list_tables" in tool_names
            assert "describe_table" in tool_names
            assert "load_dataset" in tool_names


class TestMCPClient:
    """Tests for MCPClient."""

    def test_client_initialization(self) -> None:
        """Test client can be initialized."""
        with create_client() as client:
            assert client.db is not None
            assert client.tools is not None

    def test_client_with_existing_manager(self) -> None:
        """Test client with existing DuckDB manager."""
        config = DuckDBConfig(database_path=":memory:")

        with DuckDBManager(config) as db:
            client = MCPClient(db_manager=db)
            assert client.db is db
            # Should not close since we don't own it
            client.close()
            assert db._connection is not None

    def test_call_tool_execute_sql(self, sample_csv_path: str) -> None:
        """Test calling execute_sql tool."""
        with create_client() as client:
            client.db.load_csv(sample_csv_path, "sales_data")

            result = client.call_tool("execute_sql", {"query": "SELECT COUNT(*) as cnt FROM sales_data"})

            assert result.success is True
            assert result.data["rows"][0]["cnt"] == 50

    def test_call_tool_list_tables(self, sample_csv_path: str) -> None:
        """Test calling list_tables tool."""
        with create_client() as client:
            client.db.load_csv(sample_csv_path, "sales_data")

            result = client.call_tool("list_tables", {})

            assert result.success is True
            assert "sales_data" in result.data["tables"]

    def test_call_tool_unknown(self) -> None:
        """Test calling unknown tool."""
        with create_client() as client:
            result = client.call_tool("unknown_tool", {})

            assert result.success is False
            assert "Unknown tool" in result.error

    def test_execute_tool_call_object(self, sample_csv_path: str) -> None:
        """Test executing a ToolCall object."""
        with create_client() as client:
            client.db.load_csv(sample_csv_path, "sales_data")

            tool_call = ToolCall(name="list_tables", arguments={})
            result = client.execute(tool_call)

            assert result.success is True
            assert "sales_data" in result.data["tables"]

    def test_convenience_methods(self, sample_csv_path: str) -> None:
        """Test convenience methods on client."""
        with create_client() as client:
            # Load dataset
            result = client.load_dataset(sample_csv_path, "sales_data")
            assert result.success is True

            # List tables
            result = client.list_tables()
            assert "sales_data" in result.data["tables"]

            # Describe table
            result = client.describe_table("sales_data")
            assert result.data["row_count"] == 50

            # Execute SQL
            result = client.execute_sql("SELECT SUM(total_amount) as total FROM sales_data")
            assert result.success is True
            assert result.data["rows"][0]["total"] > 0

            # Get statistics
            result = client.get_statistics("sales_data", "quantity")
            assert result.success is True

    def test_get_available_tools(self) -> None:
        """Test getting available tools from client."""
        with create_client() as client:
            tools = client.get_available_tools()

            assert len(tools) >= 4
            tool_names = [t["name"] for t in tools]
            assert "execute_sql" in tool_names


class TestDuckDBMCPServer:
    """Tests for DuckDBMCPServer."""

    def test_server_initialization(self) -> None:
        """Test server can be initialized."""
        server = create_server()
        try:
            assert server.name == "datavault-duckdb"
            assert server.tools is not None
            assert server.db_manager is not None
        finally:
            server.close()

    def test_server_tool_dispatch(self, sample_csv_path: str) -> None:
        """Test server dispatches tools correctly."""
        config = DuckDBConfig(database_path=":memory:")
        server = DuckDBMCPServer(db_config=config)

        try:
            # Load data
            server.db_manager.load_csv(sample_csv_path, "sales_data")

            # Test execute_sql dispatch
            result = server._dispatch_tool("execute_sql", {"query": "SELECT 1"})
            assert result.success is True

            # Test list_tables dispatch
            result = server._dispatch_tool("list_tables", {})
            assert result.success is True
            assert "sales_data" in result.data["tables"]

            # Test describe_table dispatch
            result = server._dispatch_tool("describe_table", {"table_name": "sales_data"})
            assert result.success is True

            # Test unknown tool
            result = server._dispatch_tool("unknown", {})
            assert result.success is False
        finally:
            server.close()

    def test_server_with_custom_config(self, temp_db_path: str) -> None:
        """Test server with custom database path."""
        config = DuckDBConfig(database_path=temp_db_path)
        server = DuckDBMCPServer(
            db_config=config,
            name="test-server",
            version="0.1.0",
        )

        try:
            assert server.name == "test-server"
            assert server.version == "0.1.0"

            # Create a table and verify persistence
            server.db_manager.execute_query("CREATE TABLE test (id INTEGER)", fetch=False)
            result = server._dispatch_tool("list_tables", {})
            assert "test" in result.data["tables"]
        finally:
            server.close()

        # Verify data persisted
        assert Path(temp_db_path).exists()


class TestMCPIntegration:
    """Integration tests for MCP components."""

    def test_full_workflow(self, sample_csv_path: str) -> None:
        """Test a complete workflow using MCP client."""
        with create_client() as client:
            # Step 1: Load data
            result = client.load_dataset(sample_csv_path, "sales")
            assert result.success is True
            assert result.data["rows_loaded"] == 50

            # Step 2: List tables to discover data
            result = client.list_tables()
            assert "sales" in result.data["tables"]

            # Step 3: Describe table to understand schema
            result = client.describe_table("sales")
            column_names = [c["name"] for c in result.data["columns"]]
            assert "total_amount" in column_names
            assert "region" in column_names

            # Step 4: Execute analytics query
            result = client.execute_sql("""
                SELECT 
                    region,
                    COUNT(*) as transactions,
                    SUM(total_amount) as revenue
                FROM sales
                GROUP BY region
                ORDER BY revenue DESC
            """)
            assert result.success is True
            assert result.data["row_count"] > 0

            # Verify regions are present
            regions = [row["region"] for row in result.data["rows"]]
            assert len(regions) > 0

            # Step 5: Get statistics on key column
            result = client.get_statistics("sales", "total_amount")
            assert result.success is True
            assert result.data["min"] > 0
            assert result.data["max"] > result.data["min"]

    def test_error_handling(self) -> None:
        """Test error handling across MCP components."""
        with create_client() as client:
            # Invalid SQL
            result = client.execute_sql("SELECT FROM WHERE")
            assert result.success is False

            # Nonexistent table
            result = client.describe_table("nonexistent")
            assert result.success is False

            # Nonexistent file
            result = client.load_dataset("/no/such/file.csv", "test")
            assert result.success is False
