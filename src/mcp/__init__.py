"""MCP (Model Context Protocol) integration for tool exposure.

This package provides:
- DuckDBTools: Collection of tools for DuckDB operations
- DuckDBMCPServer: MCP server exposing tools via stdio
- MCPClient: Client wrapper for agent integration
"""

from src.mcp.tools import DuckDBTools, ToolResult
from src.mcp.client import MCPClient, ToolCall, create_client
from src.mcp.server import DuckDBMCPServer, create_server

__all__ = [
    "DuckDBTools",
    "ToolResult",
    "MCPClient",
    "ToolCall",
    "create_client",
    "DuckDBMCPServer",
    "create_server",
]
