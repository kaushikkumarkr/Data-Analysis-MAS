"""MCP Tools for DuckDB operations.

Defines the tools that will be exposed via the Model Context Protocol
for LLM agents to interact with DuckDB.
"""

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.db.duckdb_manager import DuckDBError, DuckDBManager
from src.utils.config import DuckDBConfig
from src.utils.logging import get_logger

logger = get_logger("mcp.tools")


@dataclass
class ToolResult:
    """Result from a tool execution.

    Attributes:
        success: Whether the tool executed successfully.
        data: The result data if successful.
        error: Error message if failed.
    """

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None

    def to_json(self) -> str:
        """Convert result to JSON string.

        Returns:
            JSON representation of the result.
        """
        if self.success:
            return json.dumps({"success": True, "data": self.data})
        return json.dumps({"success": False, "error": self.error})


class DuckDBTools:
    """Collection of DuckDB tools for MCP exposure.

    Provides SQL execution, table management, and data loading
    capabilities through a standardized interface.
    """

    def __init__(self, db_manager: Optional[DuckDBManager] = None) -> None:
        """Initialize DuckDB tools.

        Args:
            db_manager: Optional DuckDB manager instance.
                       Creates a new one with default config if not provided.
        """
        self._db_manager = db_manager
        logger.info("DuckDBTools initialized")

    @property
    def db(self) -> DuckDBManager:
        """Get or create the DuckDB manager.

        Returns:
            Active DuckDBManager instance.
        """
        if self._db_manager is None:
            self._db_manager = DuckDBManager()
        return self._db_manager

    def set_db_manager(self, manager: DuckDBManager) -> None:
        """Set the DuckDB manager.

        Args:
            manager: DuckDBManager instance to use.
        """
        self._db_manager = manager

    def execute_sql(self, query: str) -> ToolResult:
        """Execute a SQL query and return results.

        This tool allows LLMs to run arbitrary SQL queries against
        the DuckDB database. Results are returned as a list of
        dictionaries for easy consumption.

        Args:
            query: SQL query string to execute.

        Returns:
            ToolResult with query results or error.

        Example:
            >>> tools.execute_sql("SELECT * FROM sales LIMIT 5")
            ToolResult(success=True, data=[...])
        """
        try:
            logger.info(f"Executing SQL: {query[:100]}...")

            # Execute and get DataFrame for better serialization
            df = self.db.execute_query_df(query)

            # Convert to list of dicts for JSON serialization
            results = df.to_dict(orient="records")

            # Handle numpy types that aren't JSON serializable
            serializable_results = self._make_serializable(results)

            logger.info(f"Query returned {len(serializable_results)} rows")
            return ToolResult(
                success=True,
                data={
                    "rows": serializable_results,
                    "row_count": len(serializable_results),
                    "columns": list(df.columns),
                },
            )
        except DuckDBError as e:
            logger.error(f"SQL execution failed: {e}")
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return ToolResult(success=False, error=f"Unexpected error: {e}")

    def list_tables(self) -> ToolResult:
        """List all tables in the database.

        Returns the names of all tables in the main schema,
        allowing LLMs to understand what data is available.

        Returns:
            ToolResult with list of table names or error.

        Example:
            >>> tools.list_tables()
            ToolResult(success=True, data={"tables": ["sales", "customers"]})
        """
        try:
            tables = self.db.list_tables()
            logger.info(f"Listed {len(tables)} tables")
            return ToolResult(
                success=True,
                data={"tables": tables, "count": len(tables)},
            )
        except DuckDBError as e:
            logger.error(f"Failed to list tables: {e}")
            return ToolResult(success=False, error=str(e))

    def describe_table(self, table_name: str) -> ToolResult:
        """Get the schema of a table.

        Returns detailed information about a table's columns,
        including names, types, nullability, and constraints.

        Args:
            table_name: Name of the table to describe.

        Returns:
            ToolResult with table schema or error.

        Example:
            >>> tools.describe_table("sales")
            ToolResult(success=True, data={"table_name": "sales", "columns": [...]})
        """
        try:
            schema = self.db.get_schema(table_name)

            # Add sample data for context
            try:
                samples = self.db.sample_data(table_name, limit=3)
                schema["sample_data"] = self._make_serializable(samples)
            except Exception:
                schema["sample_data"] = []

            # Add row count
            schema["row_count"] = self.db.get_row_count(table_name)

            logger.info(f"Described table: {table_name}")
            return ToolResult(success=True, data=schema)
        except DuckDBError as e:
            logger.error(f"Failed to describe table: {e}")
            return ToolResult(success=False, error=str(e))

    def load_dataset(self, file_path: str, table_name: str) -> ToolResult:
        """Load a CSV file into a table.

        Creates or replaces a table with data from a CSV file.
        The schema is automatically inferred from the CSV.

        Args:
            file_path: Path to the CSV file.
            table_name: Name for the target table.

        Returns:
            ToolResult with loading statistics or error.

        Example:
            >>> tools.load_dataset("/data/sales.csv", "sales")
            ToolResult(success=True, data={"rows_loaded": 1000, ...})
        """
        try:
            row_count = self.db.load_csv(file_path, table_name)

            # Get schema of loaded table
            schema = self.db.get_schema(table_name)

            logger.info(f"Loaded {row_count} rows into {table_name}")
            return ToolResult(
                success=True,
                data={
                    "table_name": table_name,
                    "rows_loaded": row_count,
                    "columns": [col["name"] for col in schema["columns"]],
                    "column_count": len(schema["columns"]),
                },
            )
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}")
            return ToolResult(success=False, error=str(e))
        except DuckDBError as e:
            logger.error(f"Failed to load dataset: {e}")
            return ToolResult(success=False, error=str(e))

    def get_statistics(self, table_name: str, column_name: str) -> ToolResult:
        """Get statistics for a column.

        Calculates descriptive statistics for numeric columns
        or value counts for categorical columns.

        Args:
            table_name: Name of the table.
            column_name: Name of the column.

        Returns:
            ToolResult with column statistics or error.
        """
        try:
            # Check if column exists
            schema = self.db.get_schema(table_name)
            column_info = None
            for col in schema["columns"]:
                if col["name"] == column_name:
                    column_info = col
                    break

            if column_info is None:
                return ToolResult(
                    success=False,
                    error=f"Column '{column_name}' not found in table '{table_name}'",
                )

            # Get statistics based on column type
            col_type = column_info["type"].upper()

            if any(t in col_type for t in ["INT", "DOUBLE", "FLOAT", "DECIMAL"]):
                # Numeric statistics
                stats_query = f"""
                    SELECT
                        COUNT(*) as count,
                        COUNT({column_name}) as non_null_count,
                        MIN({column_name}) as min,
                        MAX({column_name}) as max,
                        AVG({column_name}) as mean,
                        STDDEV({column_name}) as std,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {column_name}) as median
                    FROM {table_name}
                """
                df = self.db.execute_query_df(stats_query)
                stats = df.to_dict(orient="records")[0]
            else:
                # Categorical statistics
                stats_query = f"""
                    SELECT
                        COUNT(*) as count,
                        COUNT(DISTINCT {column_name}) as unique_count,
                        COUNT(*) - COUNT({column_name}) as null_count
                    FROM {table_name}
                """
                df = self.db.execute_query_df(stats_query)
                stats = df.to_dict(orient="records")[0]

                # Add top values
                top_query = f"""
                    SELECT {column_name}, COUNT(*) as frequency
                    FROM {table_name}
                    GROUP BY {column_name}
                    ORDER BY frequency DESC
                    LIMIT 10
                """
                top_df = self.db.execute_query_df(top_query)
                stats["top_values"] = top_df.to_dict(orient="records")

            stats = self._make_serializable([stats])[0]
            stats["column_name"] = column_name
            stats["column_type"] = col_type

            logger.info(f"Calculated statistics for {table_name}.{column_name}")
            return ToolResult(success=True, data=stats)
        except DuckDBError as e:
            logger.error(f"Failed to get statistics: {e}")
            return ToolResult(success=False, error=str(e))

    def _make_serializable(self, data: list[dict]) -> list[dict]:
        """Convert data to JSON-serializable format.

        Handles numpy types, dates, and other non-standard types.

        Args:
            data: List of dictionaries to convert.

        Returns:
            JSON-serializable list of dictionaries.
        """
        import numpy as np
        import pandas as pd

        result = []
        for row in data:
            clean_row = {}
            for key, value in row.items():
                if isinstance(value, (np.integer, np.floating)):
                    clean_row[key] = float(value) if np.isfinite(value) else None
                elif isinstance(value, np.ndarray):
                    clean_row[key] = value.tolist()
                elif isinstance(value, (pd.Timestamp, np.datetime64)):
                    clean_row[key] = str(value)
                elif isinstance(value, list):
                    # Handle lists (from to_dict on array columns)
                    clean_row[key] = [
                        self._make_serializable([{"v": v}])[0]["v"] if isinstance(v, (dict, np.ndarray)) else v
                        for v in value
                    ]
                elif value is None:
                    clean_row[key] = None
                elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    clean_row[key] = None
                else:
                    try:
                        # Try scalar isna check
                        if pd.isna(value):
                            clean_row[key] = None
                        else:
                            clean_row[key] = value
                    except (ValueError, TypeError):
                        # Value is not scalar, just use it
                        clean_row[key] = value
            result.append(clean_row)
        return result

    def get_tool_definitions(self) -> list[dict]:
        """Get MCP tool definitions for all available tools.

        Returns:
            List of tool definition dictionaries.
        """
        return [
            {
                "name": "execute_sql",
                "description": (
                    "Execute a SQL query against the DuckDB database. "
                    "Returns query results as a list of row dictionaries. "
                    "Use this for SELECT queries, aggregations, joins, etc."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "list_tables",
                "description": (
                    "List all tables in the database. "
                    "Use this to understand what data is available."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "describe_table",
                "description": (
                    "Get detailed schema information for a table including "
                    "column names, types, and sample data. "
                    "Use this to understand table structure before writing queries."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table to describe",
                        }
                    },
                    "required": ["table_name"],
                },
            },
            {
                "name": "load_dataset",
                "description": (
                    "Load a CSV file into a database table. "
                    "Creates or replaces the table with data from the file."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the CSV file",
                        },
                        "table_name": {
                            "type": "string",
                            "description": "Name for the target table",
                        },
                    },
                    "required": ["file_path", "table_name"],
                },
            },
            {
                "name": "get_statistics",
                "description": (
                    "Get descriptive statistics for a column. "
                    "For numeric columns: count, min, max, mean, std, median. "
                    "For categorical columns: unique count, top values."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table",
                        },
                        "column_name": {
                            "type": "string",
                            "description": "Name of the column",
                        },
                    },
                    "required": ["table_name", "column_name"],
                },
            },
        ]
