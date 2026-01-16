"""Unit tests for agents.

These tests verify agent functionality using mocked LLM responses
to avoid requiring Ollama to be running.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.state import (
    AgentState,
    TaskType,
    AgentRole,
    SQLResult,
    DataContext,
    create_initial_state,
    add_error_to_state,
    add_sql_result_to_state,
    add_visualization_to_state,
)
from src.mcp.client import MCPClient, create_client
from src.mcp.tools import ToolResult
from src.utils.config import DuckDBConfig


class TestAgentState:
    """Tests for AgentState and related utilities."""

    def test_create_initial_state(self) -> None:
        """Test creating initial state from user message."""
        state = create_initial_state("Show me sales by region")

        assert "Show me sales by region" in state["current_task"]
        assert len(state["messages"]) == 1
        assert state["task_type"] == TaskType.UNKNOWN
        assert state["sql_results"] == []
        assert state["visualizations"] == []
        assert state["errors"] == []

    def test_create_initial_state_with_context(self) -> None:
        """Test creating initial state with data context."""
        context = {"tables": ["sales"], "schemas": {}}
        state = create_initial_state("Analyze sales", context)

        assert state["data_context"] == context

    def test_add_error_to_state(self) -> None:
        """Test adding an error to state."""
        state = create_initial_state("Test")
        updated = add_error_to_state(state, "Test error")

        assert "Test error" in updated["errors"]
        assert len(updated["errors"]) == 1

    def test_add_sql_result_to_state(self) -> None:
        """Test adding SQL result to state."""
        state = create_initial_state("Test")
        result = SQLResult(
            query="SELECT 1",
            success=True,
            rows=[{"value": 1}],
            row_count=1,
            columns=["value"],
        )

        updated = add_sql_result_to_state(state, result)

        assert len(updated["sql_results"]) == 1
        assert updated["sql_results"][0]["success"] is True

    def test_add_visualization_to_state(self) -> None:
        """Test adding visualization path to state."""
        state = create_initial_state("Test")
        updated = add_visualization_to_state(state, "/path/to/chart.png")

        assert "/path/to/chart.png" in updated["visualizations"]


class TestSQLResult:
    """Tests for SQLResult dataclass."""

    def test_sql_result_creation(self) -> None:
        """Test creating SQLResult."""
        result = SQLResult(
            query="SELECT * FROM test",
            success=True,
            rows=[{"a": 1}, {"a": 2}],
            row_count=2,
            columns=["a"],
        )

        assert result.query == "SELECT * FROM test"
        assert result.success is True
        assert len(result.rows) == 2

    def test_sql_result_to_dict(self) -> None:
        """Test converting SQLResult to dict."""
        result = SQLResult(
            query="SELECT 1",
            success=True,
            row_count=1,
        )

        d = result.to_dict()
        assert d["query"] == "SELECT 1"
        assert d["success"] is True
        assert d["row_count"] == 1

    def test_sql_result_error(self) -> None:
        """Test SQLResult with error."""
        result = SQLResult(
            query="INVALID SQL",
            success=False,
            error="Syntax error",
        )

        assert result.success is False
        assert result.error == "Syntax error"


class TestDataContext:
    """Tests for DataContext dataclass."""

    def test_data_context_creation(self) -> None:
        """Test creating DataContext."""
        context = DataContext(
            tables=["sales", "customers"],
            schemas={"sales": {"columns": [{"name": "id", "type": "INTEGER"}]}},
        )

        assert "sales" in context.tables
        assert "customers" in context.tables

    def test_data_context_to_string_empty(self) -> None:
        """Test context string when empty."""
        context = DataContext()
        s = context.to_context_string()

        assert "No tables available" in s

    def test_data_context_to_string_with_data(self) -> None:
        """Test context string with data."""
        context = DataContext(
            tables=["sales"],
            schemas={
                "sales": {
                    "row_count": 100,
                    "columns": [
                        {"name": "id", "type": "INTEGER"},
                        {"name": "amount", "type": "DOUBLE"},
                    ],
                }
            },
        )

        s = context.to_context_string()
        assert "sales" in s
        assert "id (INTEGER)" in s
        assert "100" in s

    def test_data_context_to_dict(self) -> None:
        """Test converting DataContext to dict."""
        context = DataContext(tables=["test"])
        d = context.to_dict()

        assert d["tables"] == ["test"]
        assert "schemas" in d


class TestTaskType:
    """Tests for TaskType enum."""

    def test_task_types(self) -> None:
        """Test TaskType values."""
        assert TaskType.CLEAN.value == "clean"
        assert TaskType.ANALYZE.value == "analyze"
        assert TaskType.VISUALIZE.value == "visualize"
        assert TaskType.UNKNOWN.value == "unknown"


class TestAgentRole:
    """Tests for AgentRole enum."""

    def test_agent_roles(self) -> None:
        """Test AgentRole values."""
        assert AgentRole.ROUTER.value == "router"
        assert AgentRole.CLEANER.value == "cleaner"
        assert AgentRole.ANALYST.value == "analyst"
        assert AgentRole.VISUALIZER.value == "visualizer"


class TestCleanerAgent:
    """Tests for CleanerAgent."""

    def test_cleaner_extract_sql(self, sample_csv_path: str) -> None:
        """Test SQL extraction from various response formats."""
        from src.agents.nodes.cleaner import CleanerAgent

        with create_client() as client:
            agent = CleanerAgent(client)

            # Test JSON format
            json_response = '{"sql_queries": ["SELECT 1", "SELECT 2"]}'
            queries = agent._extract_sql_from_response(json_response)
            assert len(queries) == 2

            # Test code block format
            code_response = "```sql\nSELECT * FROM test\n```"
            queries = agent._extract_sql_from_response(code_response)
            assert len(queries) == 1

            # Test plain SQL
            plain_response = "You should run: SELECT COUNT(*) FROM test;"
            queries = agent._extract_sql_from_response(plain_response)
            assert len(queries) >= 1

    def test_cleaner_build_context(self, sample_csv_path: str) -> None:
        """Test context building."""
        from src.agents.nodes.cleaner import CleanerAgent

        with create_client() as client:
            agent = CleanerAgent(client)

            state = create_initial_state(
                "Clean the data",
                data_context={
                    "tables": ["sales"],
                    "schemas": {"sales": {"columns": [{"name": "id", "type": "INT"}]}},
                },
            )

            context = agent._build_context(state)
            assert "sales" in context
            assert "id" in context


class TestAnalystAgent:
    """Tests for AnalystAgent."""

    def test_analyst_extract_sql(self) -> None:
        """Test SQL extraction from various response formats."""
        from src.agents.nodes.analyst import AnalystAgent

        with create_client() as client:
            agent = AnalystAgent(client)

            # Test JSON format
            json_response = '{"sql_query": "SELECT COUNT(*) FROM sales"}'
            query = agent._extract_sql_from_response(json_response)
            assert query is not None
            assert "SELECT" in query

            # Test code block format
            code_response = "```sql\nSELECT SUM(amount) FROM sales\n```"
            query = agent._extract_sql_from_response(code_response)
            assert query is not None
            assert "SUM" in query

    def test_analyst_build_context(self, sample_csv_path: str) -> None:
        """Test context building."""
        from src.agents.nodes.analyst import AnalystAgent

        with create_client() as client:
            agent = AnalystAgent(client)

            state = create_initial_state(
                "Analyze sales",
                data_context={
                    "tables": ["sales"],
                    "schemas": {
                        "sales": {
                            "row_count": 100,
                            "columns": [{"name": "amount", "type": "DOUBLE"}],
                        }
                    },
                },
            )

            context = agent._build_context(state)
            assert "sales" in context
            assert "amount" in context


class TestVisualizerAgent:
    """Tests for VisualizerAgent."""

    def test_visualizer_extract_code(self, tmp_path) -> None:
        """Test code extraction from various response formats."""
        from src.agents.nodes.visualizer import VisualizerAgent

        with create_client() as client:
            agent = VisualizerAgent(client, output_dir=str(tmp_path))

            # Test JSON format
            json_response = '{"python_code": "import matplotlib.pyplot as plt\\nplt.plot([1,2,3])", "title": "Test Chart"}'
            code, title = agent._extract_code_from_response(json_response)
            assert code is not None
            assert "matplotlib" in code
            assert title == "test_chart" or "chart" in title.lower()

            # Test code block format
            code_response = "```python\nimport matplotlib.pyplot as plt\nplt.bar([1,2], [3,4])\n```"
            code, _ = agent._extract_code_from_response(code_response)
            assert code is not None
            assert "plt" in code

    def test_visualizer_get_latest_results(self) -> None:
        """Test getting latest SQL results."""
        from src.agents.nodes.visualizer import VisualizerAgent

        with create_client() as client:
            agent = VisualizerAgent(client)

            state = create_initial_state("Visualize data")
            state["sql_results"] = [
                {"success": True, "rows": [{"a": 1}]},
                {"success": True, "rows": [{"b": 2}]},
            ]

            results = agent._get_latest_results(state)
            assert results == [{"b": 2}]  # Should get most recent


class TestDataVaultGraph:
    """Tests for DataVaultGraph."""

    def test_graph_creation(self) -> None:
        """Test graph can be created."""
        from src.agents.graph import DataVaultGraph

        with create_client() as client:
            graph = DataVaultGraph(client)
            assert graph.graph is not None
            assert graph.cleaner is not None
            assert graph.analyst is not None
            assert graph.visualizer is not None

    def test_graph_gather_context(self, sample_csv_path: str) -> None:
        """Test gathering data context."""
        from src.agents.graph import DataVaultGraph

        with create_client() as client:
            # Load sample data
            client.load_dataset(sample_csv_path, "sales_data")

            graph = DataVaultGraph(client)
            context = graph._gather_data_context()

            assert "sales_data" in context.tables
            assert "sales_data" in context.schemas

    def test_select_agent_routing(self) -> None:
        """Test agent selection logic."""
        from src.agents.graph import DataVaultGraph

        with create_client() as client:
            graph = DataVaultGraph(client)

            # Test cleaner routing
            state = create_initial_state("Clean data")
            state["next_agent"] = "cleaner"
            assert graph._select_agent(state) == "cleaner"

            # Test analyst routing
            state["next_agent"] = "analyst"
            assert graph._select_agent(state) == "analyst"

            # Test visualizer routing
            state["next_agent"] = "visualizer"
            assert graph._select_agent(state) == "visualizer"


class TestCreateGraph:
    """Tests for graph factory function."""

    def test_create_datavault_graph(self) -> None:
        """Test factory function creates graph."""
        from src.agents.graph import create_datavault_graph

        graph = create_datavault_graph()
        assert graph is not None
        assert graph.mcp_client is not None

        # Cleanup
        graph.mcp_client.close()


class TestAgentIntegration:
    """Integration tests for agents (mocked LLM)."""

    @patch("src.agents.nodes.analyst.create_chat_model")
    def test_analyst_with_mock_llm(self, mock_create_llm, sample_csv_path: str) -> None:
        """Test analyst with mocked LLM response."""
        from src.agents.nodes.analyst import AnalystAgent

        # Setup mock
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"sql_query": "SELECT COUNT(*) as cnt FROM sales_data"}'

        # Create async mock for ainvoke
        async_mock = AsyncMock(return_value=mock_response)
        mock_llm.ainvoke = async_mock
        mock_create_llm.return_value = mock_llm

        with create_client() as client:
            # Load test data
            client.load_dataset(sample_csv_path, "sales_data")

            agent = AnalystAgent(client)

            state = create_initial_state(
                "How many rows?",
                data_context={
                    "tables": ["sales_data"],
                    "schemas": {"sales_data": {"columns": []}},
                },
            )

            import asyncio
            result = asyncio.run(agent.process(state))

            # Should have SQL results
            assert len(result["sql_results"]) > 0
            # The query should have been executed successfully
            assert result["sql_results"][0]["success"] is True

    @patch("src.agents.nodes.cleaner.create_chat_model")
    def test_cleaner_with_mock_llm(self, mock_create_llm, sample_csv_path: str) -> None:
        """Test cleaner with mocked LLM response."""
        from src.agents.nodes.cleaner import CleanerAgent

        # Setup mock
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"sql_queries": ["SELECT DISTINCT * FROM sales_data"]}'

        async_mock = AsyncMock(return_value=mock_response)
        mock_llm.ainvoke = async_mock
        mock_create_llm.return_value = mock_llm

        with create_client() as client:
            client.load_dataset(sample_csv_path, "sales_data")

            agent = CleanerAgent(client)

            state = create_initial_state(
                "Remove duplicates",
                data_context={"tables": ["sales_data"], "schemas": {}},
            )

            import asyncio
            result = asyncio.run(agent.process(state))

            assert len(result["sql_results"]) > 0
