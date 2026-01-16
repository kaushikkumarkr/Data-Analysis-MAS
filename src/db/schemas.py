"""Table schema definitions for DuckDB."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ColumnType(str, Enum):
    """Supported DuckDB column types."""

    INTEGER = "INTEGER"
    BIGINT = "BIGINT"
    DOUBLE = "DOUBLE"
    FLOAT = "FLOAT"
    DECIMAL = "DECIMAL"
    VARCHAR = "VARCHAR"
    TEXT = "TEXT"
    BOOLEAN = "BOOLEAN"
    DATE = "DATE"
    TIMESTAMP = "TIMESTAMP"
    TIME = "TIME"
    BLOB = "BLOB"
    UUID = "UUID"
    JSON = "JSON"


@dataclass
class Column:
    """Column definition for a table schema."""

    name: str
    column_type: ColumnType
    nullable: bool = True
    primary_key: bool = False
    default: Optional[Any] = None
    description: Optional[str] = None

    def to_ddl(self) -> str:
        """Generate DDL fragment for this column.

        Returns:
            SQL DDL string for column definition.
        """
        parts = [self.name, self.column_type.value]

        if self.primary_key:
            parts.append("PRIMARY KEY")
        if not self.nullable:
            parts.append("NOT NULL")
        if self.default is not None:
            if isinstance(self.default, str):
                parts.append(f"DEFAULT '{self.default}'")
            else:
                parts.append(f"DEFAULT {self.default}")

        return " ".join(parts)


@dataclass
class TableSchema:
    """Table schema definition."""

    name: str
    columns: list[Column] = field(default_factory=list)
    description: Optional[str] = None

    def add_column(
        self,
        name: str,
        column_type: ColumnType,
        nullable: bool = True,
        primary_key: bool = False,
        default: Optional[Any] = None,
        description: Optional[str] = None,
    ) -> "TableSchema":
        """Add a column to the schema.

        Args:
            name: Column name.
            column_type: Column data type.
            nullable: Whether column allows NULL values.
            primary_key: Whether column is primary key.
            default: Default value for column.
            description: Human-readable column description.

        Returns:
            Self for method chaining.
        """
        self.columns.append(
            Column(
                name=name,
                column_type=column_type,
                nullable=nullable,
                primary_key=primary_key,
                default=default,
                description=description,
            )
        )
        return self

    def to_create_table_sql(self) -> str:
        """Generate CREATE TABLE SQL statement.

        Returns:
            SQL CREATE TABLE statement.
        """
        column_defs = ",\n    ".join(col.to_ddl() for col in self.columns)
        return f"CREATE TABLE IF NOT EXISTS {self.name} (\n    {column_defs}\n)"

    def get_column_names(self) -> list[str]:
        """Get list of column names.

        Returns:
            List of column names.
        """
        return [col.name for col in self.columns]

    def to_dict(self) -> dict[str, Any]:
        """Convert schema to dictionary representation.

        Returns:
            Dictionary with schema details.
        """
        return {
            "name": self.name,
            "description": self.description,
            "columns": [
                {
                    "name": col.name,
                    "type": col.column_type.value,
                    "nullable": col.nullable,
                    "primary_key": col.primary_key,
                    "description": col.description,
                }
                for col in self.columns
            ],
        }


# Pre-defined schemas for sample data
SALES_DATA_SCHEMA = TableSchema(
    name="sales_data",
    description="Enterprise sales transactions data",
).add_column(
    "transaction_id", ColumnType.VARCHAR, nullable=False, primary_key=True,
    description="Unique transaction identifier"
).add_column(
    "date", ColumnType.DATE, nullable=False,
    description="Transaction date"
).add_column(
    "customer_id", ColumnType.VARCHAR, nullable=False,
    description="Customer identifier"
).add_column(
    "product_id", ColumnType.VARCHAR, nullable=False,
    description="Product identifier"
).add_column(
    "product_name", ColumnType.VARCHAR, nullable=False,
    description="Product name"
).add_column(
    "category", ColumnType.VARCHAR, nullable=False,
    description="Product category"
).add_column(
    "quantity", ColumnType.INTEGER, nullable=False,
    description="Number of units sold"
).add_column(
    "unit_price", ColumnType.DOUBLE, nullable=False,
    description="Price per unit"
).add_column(
    "total_amount", ColumnType.DOUBLE, nullable=False,
    description="Total transaction amount"
).add_column(
    "region", ColumnType.VARCHAR, nullable=False,
    description="Sales region"
).add_column(
    "salesperson", ColumnType.VARCHAR, nullable=True,
    description="Salesperson name"
)


CUSTOMERS_SCHEMA = TableSchema(
    name="customers",
    description="Customer master data",
).add_column(
    "customer_id", ColumnType.VARCHAR, nullable=False, primary_key=True,
    description="Unique customer identifier"
).add_column(
    "company_name", ColumnType.VARCHAR, nullable=False,
    description="Company name"
).add_column(
    "contact_name", ColumnType.VARCHAR, nullable=True,
    description="Primary contact name"
).add_column(
    "email", ColumnType.VARCHAR, nullable=True,
    description="Contact email"
).add_column(
    "segment", ColumnType.VARCHAR, nullable=False,
    description="Customer segment (Enterprise, SMB, Startup)"
).add_column(
    "created_date", ColumnType.DATE, nullable=False,
    description="Customer account creation date"
)


PRODUCTS_SCHEMA = TableSchema(
    name="products",
    description="Product catalog",
).add_column(
    "product_id", ColumnType.VARCHAR, nullable=False, primary_key=True,
    description="Unique product identifier"
).add_column(
    "product_name", ColumnType.VARCHAR, nullable=False,
    description="Product name"
).add_column(
    "category", ColumnType.VARCHAR, nullable=False,
    description="Product category"
).add_column(
    "unit_price", ColumnType.DOUBLE, nullable=False,
    description="Standard unit price"
).add_column(
    "cost", ColumnType.DOUBLE, nullable=False,
    description="Unit cost"
).add_column(
    "active", ColumnType.BOOLEAN, nullable=False, default=True,
    description="Whether product is currently active"
)


# Registry of all predefined schemas
SCHEMA_REGISTRY: dict[str, TableSchema] = {
    "sales_data": SALES_DATA_SCHEMA,
    "customers": CUSTOMERS_SCHEMA,
    "products": PRODUCTS_SCHEMA,
}


def get_schema(table_name: str) -> Optional[TableSchema]:
    """Get a predefined schema by table name.

    Args:
        table_name: Name of the table.

    Returns:
        TableSchema if found, None otherwise.
    """
    return SCHEMA_REGISTRY.get(table_name)
