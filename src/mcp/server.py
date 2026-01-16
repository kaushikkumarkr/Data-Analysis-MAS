"""MCP Server for DuckDB operations.

This module implements a Model Context Protocol (MCP) server that exposes
DuckDB operations as tools for LLM agents. It uses stdio transport for
local communication, ensuring zero data exfiltration.
"""

import asyncio
import json
from typing import Any, Sequence

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
    Tool,
)

from src.db.duckdb_manager import DuckDBManager
from src.mcp.tools import DuckDBTools, ToolResult
from src.utils.config import DuckDBConfig, get_config
from src.utils.logging import get_logger

logger = get_logger("mcp.server")


class DuckDBMCPServer:
    """MCP Server exposing DuckDB operations.

    This server implements the Model Context Protocol to allow
    LLM agents to interact with DuckDB through standardized tools.
    All operations are local with no network calls.

    Attributes:
        server: The MCP server instance.
        tools: DuckDB tools collection.
        db_manager: DuckDB connection manager.
    """

    def __init__(
        self,
        db_config: DuckDBConfig | None = None,
        name: str = "datavault-duckdb",
        version: str = "1.0.0",
    ) -> None:
        """Initialize the MCP server.

        Args:
            db_config: Optional DuckDB configuration.
            name: Server name for identification.
            version: Server version string.
        """
        self.name = name
        self.version = version

        # Initialize DuckDB manager
        self.db_config = db_config or get_config().duckdb
        self.db_manager = DuckDBManager(self.db_config)

        # Initialize tools
        self.tools = DuckDBTools(self.db_manager)

        # Create MCP server
        self.server = Server(name)

        # Register handlers
        self._register_handlers()

        logger.info(f"DuckDBMCPServer initialized: {name} v{version}")

    def _register_handlers(self) -> None:
        """Register MCP request handlers."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """Handle list_tools request.

            Returns:
                List of available tools.
            """
            logger.debug("Handling list_tools request")
            tool_definitions = self.tools.get_tool_definitions()

            return [
                Tool(
                    name=td["name"],
                    description=td["description"],
                    inputSchema=td["inputSchema"],
                )
                for td in tool_definitions
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
            """Handle call_tool request.

            Args:
                name: Name of the tool to call.
                arguments: Tool arguments.

            Returns:
                Sequence of text content with results.
            """
            logger.info(f"Handling call_tool: {name}")
            logger.debug(f"Arguments: {arguments}")

            result = self._dispatch_tool(name, arguments)

            return [
                TextContent(
                    type="text",
                    text=result.to_json(),
                )
            ]

    def _dispatch_tool(self, name: str, arguments: dict) -> ToolResult:
        """Dispatch a tool call to the appropriate handler.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            ToolResult from the tool execution.
        """
        try:
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
                logger.warning(f"Unknown tool: {name}")
                return ToolResult(success=False, error=f"Unknown tool: {name}")
        except Exception as e:
            logger.error(f"Tool dispatch error: {e}")
            return ToolResult(success=False, error=str(e))

    async def run_stdio(self) -> None:
        """Run the server with stdio transport.

        This is the main entry point for running the MCP server.
        Uses stdio for local communication with zero network exposure.
        """
        logger.info("Starting MCP server with stdio transport")

        # Ensure database connection is ready
        _ = self.db_manager.connection

        init_options = InitializationOptions(
            server_name=self.name,
            server_version=self.version,
            capabilities=self.server.get_capabilities(
                notification_options=None,
                experimental_capabilities={},
            ),
        )

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                init_options,
            )

    def close(self) -> None:
        """Close the server and cleanup resources."""
        logger.info("Closing MCP server")
        self.db_manager.close()


def create_server(db_path: str | None = None) -> DuckDBMCPServer:
    """Factory function to create an MCP server.

    Args:
        db_path: Optional database path. Uses :memory: if not provided.

    Returns:
        Configured DuckDBMCPServer instance.
    """
    config = DuckDBConfig(database_path=db_path or ":memory:")
    return DuckDBMCPServer(db_config=config)


async def main() -> None:
    """Main entry point for running the MCP server."""
    import sys

    # Get database path from command line if provided
    db_path = sys.argv[1] if len(sys.argv) > 1 else None

    server = create_server(db_path)
    try:
        await server.run_stdio()
    finally:
        server.close()


if __name__ == "__main__":
    asyncio.run(main())
