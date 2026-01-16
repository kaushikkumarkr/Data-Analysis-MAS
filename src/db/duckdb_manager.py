"""DuckDB connection manager with context manager pattern."""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

import duckdb
import pandas as pd

from src.db.schemas import TableSchema
from src.utils.config import DuckDBConfig, get_config
from src.utils.logging import get_logger

logger = get_logger("duckdb")


class DuckDBError(Exception):
    """Custom exception for DuckDB operations."""

    pass


class DuckDBManager:
    """Manager for DuckDB database operations.

    Provides a high-level interface for DuckDB operations with
    connection pooling and context manager support.

    Attributes:
        config: DuckDB configuration.
        connection: Current database connection.
    """

    def __init__(self, config: Optional[DuckDBConfig] = None) -> None:
        """Initialize DuckDBManager.

        Args:
            config: Optional DuckDB configuration. Uses default from env if not provided.
        """
        self.config = config or get_config().duckdb
        self._connection: Optional[duckdb.DuckDBPyConnection] = None
        logger.info(f"DuckDBManager initialized with database: {self.config.database_path}")

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection.

        Returns:
            Active DuckDB connection.

        Raises:
            DuckDBError: If connection cannot be established.
        """
        if self._connection is None:
            self._connect()
        return self._connection

    def _connect(self) -> None:
        """Establish database connection.

        Raises:
            DuckDBError: If connection fails.
        """
        try:
            # Create parent directories if using file-based database
            if self.config.database_path != ":memory:":
                db_path = Path(self.config.database_path)
                db_path.parent.mkdir(parents=True, exist_ok=True)

            self._connection = duckdb.connect(
                database=self.config.database_path,
                read_only=self.config.read_only,
            )

            # Configure connection settings
            self._connection.execute(f"SET threads = {self.config.threads}")

            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise DuckDBError(f"Connection failed: {e}") from e

    def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            logger.info("Database connection closed")

    def __enter__(self) -> "DuckDBManager":
        """Enter context manager.

        Returns:
            Self with active connection.
        """
        self._connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and close connection."""
        self.close()

    @contextmanager
    def transaction(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Context manager for transaction handling.

        Yields:
            Database connection within transaction.

        Raises:
            DuckDBError: If transaction fails.
        """
        try:
            self.connection.begin()
            yield self.connection
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Transaction failed: {e}")
            raise DuckDBError(f"Transaction failed: {e}") from e

    def execute_query(
        self,
        query: str,
        params: Optional[tuple[Any, ...]] = None,
        fetch: bool = True,
    ) -> Optional[list[tuple[Any, ...]]]:
        """Execute a SQL query.

        Args:
            query: SQL query string.
            params: Optional query parameters.
            fetch: Whether to fetch results.

        Returns:
            Query results if fetch is True, None otherwise.

        Raises:
            DuckDBError: If query execution fails.
        """
        try:
            logger.debug(f"Executing query: {query[:100]}...")

            if params:
                result = self.connection.execute(query, params)
            else:
                result = self.connection.execute(query)

            if fetch:
                return result.fetchall()
            return None
        except duckdb.Error as e:
            logger.error(f"Query execution failed: {e}")
            raise DuckDBError(f"Query failed: {e}") from e

    def execute_query_df(
        self,
        query: str,
        params: Optional[tuple[Any, ...]] = None,
    ) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame.

        Args:
            query: SQL query string.
            params: Optional query parameters.

        Returns:
            Query results as pandas DataFrame.

        Raises:
            DuckDBError: If query execution fails.
        """
        try:
            if params:
                result = self.connection.execute(query, params)
            else:
                result = self.connection.execute(query)
            return result.fetchdf()
        except duckdb.Error as e:
            logger.error(f"Query execution failed: {e}")
            raise DuckDBError(f"Query failed: {e}") from e

    def load_csv(
        self,
        file_path: str,
        table_name: str,
        schema: Optional[TableSchema] = None,
        if_exists: str = "replace",
    ) -> int:
        """Load a CSV file into a DuckDB table.

        Args:
            file_path: Path to the CSV file.
            table_name: Target table name.
            schema: Optional table schema for explicit typing.
            if_exists: Behavior when table exists ('replace', 'append', 'fail').

        Returns:
            Number of rows loaded.

        Raises:
            DuckDBError: If loading fails.
            FileNotFoundError: If CSV file doesn't exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        try:
            # Drop existing table if replacing
            if if_exists == "replace":
                self.connection.execute(f"DROP TABLE IF EXISTS {table_name}")

            # Create table from schema or infer from CSV
            if schema:
                self.connection.execute(schema.to_create_table_sql())
                # Insert data from CSV
                self.connection.execute(
                    f"INSERT INTO {table_name} SELECT * FROM read_csv_auto('{file_path}')"
                )
            else:
                # Create table directly from CSV with auto-detection
                self.connection.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}')"
                )

            # Get row count
            result = self.connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            row_count = result[0] if result else 0

            logger.info(f"Loaded {row_count} rows from {file_path} into {table_name}")
            return row_count
        except duckdb.Error as e:
            logger.error(f"Failed to load CSV: {e}")
            raise DuckDBError(f"CSV loading failed: {e}") from e

    def get_schema(self, table_name: str) -> dict[str, Any]:
        """Get schema information for a table.

        Args:
            table_name: Name of the table.

        Returns:
            Dictionary with column names and types.

        Raises:
            DuckDBError: If table doesn't exist or query fails.
        """
        try:
            # Use PRAGMA to get table info
            result = self.connection.execute(
                f"PRAGMA table_info('{table_name}')"
            ).fetchall()

            if not result:
                raise DuckDBError(f"Table '{table_name}' not found")

            schema = {
                "table_name": table_name,
                "columns": [
                    {
                        "cid": row[0],
                        "name": row[1],
                        "type": row[2],
                        "notnull": bool(row[3]),
                        "default": row[4],
                        "primary_key": bool(row[5]),
                    }
                    for row in result
                ],
            }

            logger.debug(f"Retrieved schema for table: {table_name}")
            return schema
        except duckdb.Error as e:
            logger.error(f"Failed to get schema: {e}")
            raise DuckDBError(f"Schema retrieval failed: {e}") from e

    def list_tables(self) -> list[str]:
        """List all tables in the database.

        Returns:
            List of table names.

        Raises:
            DuckDBError: If query fails.
        """
        try:
            result = self.connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()

            tables = [row[0] for row in result]
            logger.debug(f"Found {len(tables)} tables")
            return tables
        except duckdb.Error as e:
            logger.error(f"Failed to list tables: {e}")
            raise DuckDBError(f"Table listing failed: {e}") from e

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists.

        Args:
            table_name: Name of the table.

        Returns:
            True if table exists, False otherwise.
        """
        return table_name in self.list_tables()

    def drop_table(self, table_name: str, if_exists: bool = True) -> None:
        """Drop a table.

        Args:
            table_name: Name of the table to drop.
            if_exists: If True, don't raise error if table doesn't exist.

        Raises:
            DuckDBError: If drop fails.
        """
        try:
            modifier = "IF EXISTS " if if_exists else ""
            self.connection.execute(f"DROP TABLE {modifier}{table_name}")
            logger.info(f"Dropped table: {table_name}")
        except duckdb.Error as e:
            logger.error(f"Failed to drop table: {e}")
            raise DuckDBError(f"Table drop failed: {e}") from e

    def get_row_count(self, table_name: str) -> int:
        """Get the number of rows in a table.

        Args:
            table_name: Name of the table.

        Returns:
            Number of rows.

        Raises:
            DuckDBError: If query fails.
        """
        try:
            result = self.connection.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()
            return result[0] if result else 0
        except duckdb.Error as e:
            logger.error(f"Failed to get row count: {e}")
            raise DuckDBError(f"Row count failed: {e}") from e

    def sample_data(
        self,
        table_name: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get sample data from a table.

        Args:
            table_name: Name of the table.
            limit: Maximum number of rows to return.

        Returns:
            List of dictionaries representing rows.

        Raises:
            DuckDBError: If query fails.
        """
        try:
            df = self.execute_query_df(
                f"SELECT * FROM {table_name} LIMIT {limit}"
            )
            return df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Failed to sample data: {e}")
            raise DuckDBError(f"Data sampling failed: {e}") from e
