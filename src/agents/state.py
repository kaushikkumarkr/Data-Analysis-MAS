"""Agent state schema for LangGraph workflow.

This module defines the shared state that flows through the
multi-agent workflow. Each agent can read and modify this state.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence, TypedDict

from langchain_core.messages import AnyMessage


class TaskType(str, Enum):
    """Types of tasks the agents can perform."""

    CLEAN = "clean"
    ANALYZE = "analyze"
    VISUALIZE = "visualize"
    UNKNOWN = "unknown"


class AgentRole(str, Enum):
    """Roles of the agents in the workflow."""

    ROUTER = "router"
    CLEANER = "cleaner"
    ANALYST = "analyst"
    VISUALIZER = "visualizer"


class AgentState(TypedDict):
    """State that flows through the LangGraph workflow.

    This TypedDict defines the structure of data passed between
    agents in the multi-agent workflow.

    Attributes:
        messages: Conversation history with user and agents.
        current_task: Current task being processed.
        task_type: Classified type of task.
        data_context: Contextual information about available data.
        sql_results: Results from SQL queries.
        visualizations: Paths to generated visualization files.
        errors: Any errors encountered during processing.
        next_agent: The next agent to route to.
        iteration_count: Number of iterations to prevent infinite loops.
    """

    messages: Sequence[AnyMessage]
    current_task: str
    task_type: TaskType
    data_context: dict[str, Any]
    sql_results: list[dict[str, Any]]
    visualizations: list[str]
    errors: list[str]
    next_agent: Optional[str]
    iteration_count: int


def create_initial_state(
    user_message: str,
    data_context: Optional[dict[str, Any]] = None,
) -> AgentState:
    """Create an initial agent state from a user message.

    Args:
        user_message: The user's request.
        data_context: Optional context about available data.

    Returns:
        Initial AgentState ready for processing.
    """
    from langchain_core.messages import HumanMessage

    return AgentState(
        messages=[HumanMessage(content=user_message)],
        current_task=user_message,
        task_type=TaskType.UNKNOWN,
        data_context=data_context or {},
        sql_results=[],
        visualizations=[],
        errors=[],
        next_agent=None,
        iteration_count=0,
    )


@dataclass
class SQLResult:
    """Result of a SQL query execution.

    Attributes:
        query: The SQL query that was executed.
        success: Whether the query succeeded.
        rows: Query result rows.
        row_count: Number of rows returned.
        columns: Column names.
        error: Error message if failed.
    """

    query: str
    success: bool
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    columns: list[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all attributes.
        """
        return {
            "query": self.query,
            "success": self.success,
            "rows": self.rows,
            "row_count": self.row_count,
            "columns": self.columns,
            "error": self.error,
        }


@dataclass
class DataContext:
    """Context about available data in the database.

    Attributes:
        tables: List of available table names.
        schemas: Dictionary of table schemas.
        sample_data: Sample data from each table.
        statistics: Column statistics for key columns.
    """

    tables: list[str] = field(default_factory=list)
    schemas: dict[str, dict[str, Any]] = field(default_factory=dict)
    sample_data: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    statistics: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_context_string(self) -> str:
        """Generate a context string for LLM prompts.

        Returns:
            Human-readable string describing available data.
        """
        if not self.tables:
            return "No tables available in the database."

        lines = ["Available tables:"]
        for table in self.tables:
            if table in self.schemas:
                schema = self.schemas[table]
                columns = [f"{col['name']} ({col['type']})" for col in schema.get("columns", [])]
                lines.append(f"\n• {table}:")
                lines.append(f"  Columns: {', '.join(columns)}")
                if schema.get("row_count"):
                    lines.append(f"  Row count: {schema['row_count']}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all context data.
        """
        return {
            "tables": self.tables,
            "schemas": self.schemas,
            "sample_data": self.sample_data,
            "statistics": self.statistics,
        }


def add_error_to_state(state: AgentState, error: str) -> AgentState:
    """Add an error to the state.

    Args:
        state: Current agent state.
        error: Error message to add.

    Returns:
        Updated state with error added.
    """
    errors = list(state.get("errors", []))
    errors.append(error)
    return {**state, "errors": errors}


def add_sql_result_to_state(state: AgentState, result: SQLResult) -> AgentState:
    """Add a SQL result to the state.

    Args:
        state: Current agent state.
        result: SQL result to add.

    Returns:
        Updated state with result added.
    """
    sql_results = list(state.get("sql_results", []))
    sql_results.append(result.to_dict())
    return {**state, "sql_results": sql_results}


def add_visualization_to_state(state: AgentState, path: str) -> AgentState:
    """Add a visualization path to the state.

    Args:
        state: Current agent state.
        path: Path to visualization file.

    Returns:
        Updated state with visualization added.
    """
    visualizations = list(state.get("visualizations", []))
    visualizations.append(path)
    return {**state, "visualizations": visualizations}
