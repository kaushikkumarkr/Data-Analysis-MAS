"""MCP Client wrapper for agent integration.

Provides a high-level interface for agents to call MCP tools
without dealing with the low-level protocol details.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

from src.mcp.tools import DuckDBTools, ToolResult
from src.db.duckdb_manager import DuckDBManager
from src.utils.config import DuckDBConfig, get_config
from src.utils.logging import get_logger

logger = get_logger("mcp.client")


@dataclass
class ToolCall:
    """Represents a tool call request.

    Attributes:
        name: Tool name.
        arguments: Tool arguments as dictionary.
    """

    name: str
    arguments: dict


class MCPClient:
    """Client wrapper for MCP tools.

    Provides a simple interface for agents to call DuckDB tools.
    This is a direct client that doesn't go through the full MCP
    protocol - useful for in-process agent integration.

    For cross-process communication, use the full MCP server/client.
    """

    def __init__(
        self,
        db_config: DuckDBConfig | None = None,
        db_manager: DuckDBManager | None = None,
    ) -> None:
        """Initialize the MCP client.

        Args:
            db_config: Optional DuckDB configuration.
            db_manager: Optional existing DuckDB manager to reuse.
        """
        if db_manager:
            self._db_manager = db_manager
            self._owns_manager = False
        else:
            config = db_config or get_config().duckdb
            self._db_manager = DuckDBManager(config)
            self._owns_manager = True

        self.tools = DuckDBTools(self._db_manager)
        logger.info("MCPClient initialized")

    @property
    def db(self) -> DuckDBManager:
        """Get the underlying DuckDB manager.

        Returns:
            DuckDBManager instance.
        """
        return self._db_manager

    def call_tool(self, name: str, arguments: dict | None = None) -> ToolResult:
        """Call a tool by name with arguments.

        Args:
            name: Name of the tool to call.
            arguments: Optional tool arguments.

        Returns:
            ToolResult from the tool execution.
        """
        arguments = arguments or {}
        logger.debug(f"Calling tool: {name} with {arguments}")

        if name == "execute_sql":
            return self.tools.execute_sql(arguments.get("query", ""))
        elif name == "list_tables":
            return self.tools.list_tables()
        elif name == "describe_table":
            return self.tools.describe_table(arguments.get("table_name", ""))
        elif name == "load_dataset":
            return self.tools.load_dataset(
                arguments.get("file_path", ""),
                arguments.get("table_name", ""),
            )
        elif name == "get_statistics":
            return self.tools.get_statistics(
                arguments.get("table_name", ""),
                arguments.get("column_name", ""),
            )
        else:
            return ToolResult(success=False, error=f"Unknown tool: {name}")

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call object.

        Args:
            tool_call: ToolCall object with name and arguments.

        Returns:
            ToolResult from the tool execution.
        """
        return self.call_tool(tool_call.name, tool_call.arguments)

    def execute_sql(self, query: str) -> ToolResult:
        """Convenience method to execute SQL.

        Args:
            query: SQL query string.

        Returns:
            ToolResult with query results.
        """
        return self.tools.execute_sql(query)

    def list_tables(self) -> ToolResult:
        """Convenience method to list tables.

        Returns:
            ToolResult with table list.
        """
        return self.tools.list_tables()

    def describe_table(self, table_name: str) -> ToolResult:
        """Convenience method to describe a table.

        Args:
            table_name: Name of the table.

        Returns:
            ToolResult with table schema.
        """
        return self.tools.describe_table(table_name)

    def load_dataset(self, file_path: str, table_name: str) -> ToolResult:
        """Convenience method to load a dataset.

        Args:
            file_path: Path to CSV file.
            table_name: Target table name.

        Returns:
            ToolResult with load statistics.
        """
        return self.tools.load_dataset(file_path, table_name)

    def get_statistics(self, table_name: str, column_name: str) -> ToolResult:
        """Convenience method to get column statistics.

        Args:
            table_name: Name of the table.
            column_name: Name of the column.

        Returns:
            ToolResult with column statistics.
        """
        return self.tools.get_statistics(table_name, column_name)

    def get_available_tools(self) -> list[dict]:
        """Get list of available tool definitions.

        Returns:
            List of tool definition dictionaries.
        """
        return self.tools.get_tool_definitions()

    def close(self) -> None:
        """Close the client and cleanup resources."""
        if self._owns_manager:
            self._db_manager.close()
        logger.info("MCPClient closed")

    def __enter__(self) -> "MCPClient":
        """Enter context manager.

        Returns:
            Self.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and cleanup."""
        self.close()


def create_client(db_path: str | None = None) -> MCPClient:
    """Factory function to create an MCP client.

    Args:
        db_path: Optional database path. Uses :memory: if not provided.

    Returns:
        Configured MCPClient instance.
    """
    config = DuckDBConfig(database_path=db_path or ":memory:")
    return MCPClient(db_config=config)
