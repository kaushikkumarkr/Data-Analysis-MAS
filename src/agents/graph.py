"""LangGraph workflow for multi-agent orchestration.

This module implements the StateGraph that orchestrates the
Data Cleaning, Analysis, and Visualization agents.
"""

import re
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from src.agents.nodes.analyst import AnalystAgent
from src.agents.nodes.cleaner import CleanerAgent
from src.agents.nodes.visualizer import VisualizerAgent
from src.agents.state import AgentState, DataContext, TaskType, create_initial_state
from src.mcp.client import MCPClient
from src.utils.config import get_config
from src.utils.llm_factory import create_chat_model
from src.utils.logging import get_logger

logger = get_logger("agents.graph")

# Router prompt for classifying tasks
ROUTER_PROMPT = """You are a task router for a data analytics system.
Classify the user's request into one of these categories:

1. CLEAN - Data quality tasks like removing nulls, fixing types, deduplication
2. ANALYZE - Analytics queries like aggregations, summaries, trends, statistics
3. VISUALIZE - Creating charts or graphs from data
4. UNKNOWN - Cannot determine the task type

User request: {task}

Respond with ONLY one word: CLEAN, ANALYZE, VISUALIZE, or UNKNOWN"""


class DataVaultGraph:
    """LangGraph-based multi-agent workflow.

    Orchestrates the Data Cleaning, Analysis, and Visualization agents
    using a shared state and conditional routing.
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        backend: str | None = None,
        model_name: str | None = None,
        output_dir: str | None = None,
    ) -> None:
        """Initialize the graph.

        Args:
            mcp_client: MCP client for database operations.
            backend: LLM backend ('mlx' or 'ollama'). Auto-detected if None.
            model_name: Optional override for LLM model name.
            output_dir: Optional output directory for visualizations.
        """
        self.mcp_client = mcp_client
        config = get_config()

        # Use LLM factory for automatic backend selection
        self.router_llm = create_chat_model(
            backend=backend,
            model_name=model_name,
            temperature=0.0,
        )

        # Initialize agents with same backend
        self.cleaner = CleanerAgent(mcp_client, backend, model_name)
        self.analyst = AnalystAgent(mcp_client, backend, model_name)
        self.visualizer = VisualizerAgent(mcp_client, output_dir, backend, model_name)

        # Build the graph
        self.graph = self._build_graph()

        backend_name = backend or ("mlx" if config.mlx.enabled else "ollama")
        model = model_name or (config.mlx.model if config.mlx.enabled else config.ollama.model)
        logger.info(f"DataVaultGraph initialized with {backend_name}: {model}")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph.

        Returns:
            Compiled StateGraph.
        """
        # Create state graph
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("router", self._route_task)
        graph.add_node("cleaner", self._run_cleaner)
        graph.add_node("analyst", self._run_analyst)
        graph.add_node("visualizer", self._run_visualizer)
        graph.add_node("output", self._generate_output)

        # Set entry point
        graph.set_entry_point("router")

        # Add conditional edges from router
        graph.add_conditional_edges(
            "router",
            self._select_agent,
            {
                "cleaner": "cleaner",
                "analyst": "analyst",
                "visualizer": "visualizer",
                "output": "output",
            },
        )

        # Connect agents to output
        graph.add_edge("cleaner", "output")
        graph.add_edge("analyst", "output")
        graph.add_edge("visualizer", "output")

        # Output goes to END
        graph.add_edge("output", END)

        return graph.compile()

    def _route_task(self, state: AgentState) -> AgentState:
        """Route the task to the appropriate agent.

        Args:
            state: Current agent state.

        Returns:
            Updated state with task type.
        """
        logger.info("Routing task...")

        task = state.get("current_task", "")

        # Use LLM to classify task
        try:
            prompt = ROUTER_PROMPT.format(task=task)
            response = self.router_llm.invoke(prompt)
            classification = response.content.strip().upper()

            # Parse classification
            if "CLEAN" in classification:
                task_type = TaskType.CLEAN
                next_agent = "cleaner"
            elif "ANALYZ" in classification or "ANALYZE" in classification:
                task_type = TaskType.ANALYZE
                next_agent = "analyst"
            elif "VISUAL" in classification:
                task_type = TaskType.VISUALIZE
                next_agent = "visualizer"
            else:
                task_type = TaskType.UNKNOWN
                next_agent = "output"

            logger.info(f"Task classified as: {task_type.value}, routing to: {next_agent}")

            return {
                **state,
                "task_type": task_type,
                "next_agent": next_agent,
                "iteration_count": state.get("iteration_count", 0) + 1,
            }

        except Exception as e:
            logger.error(f"Routing failed: {e}")
            return {
                **state,
                "task_type": TaskType.UNKNOWN,
                "next_agent": "output",
                "errors": [*state.get("errors", []), f"Routing failed: {str(e)}"],
            }

    def _select_agent(self, state: AgentState) -> str:
        """Select which agent to route to.

        Args:
            state: Current agent state.

        Returns:
            Name of the next node.
        """
        return state.get("next_agent", "output")

    def _run_cleaner(self, state: AgentState) -> AgentState:
        """Run the cleaner agent.

        Args:
            state: Current agent state.

        Returns:
            Updated state from cleaner.
        """
        logger.info("Running cleaner agent...")
        return self.cleaner(state)

    def _run_analyst(self, state: AgentState) -> AgentState:
        """Run the analyst agent.

        Args:
            state: Current agent state.

        Returns:
            Updated state from analyst.
        """
        logger.info("Running analyst agent...")
        return self.analyst(state)

    def _run_visualizer(self, state: AgentState) -> AgentState:
        """Run the visualizer agent.

        Args:
            state: Current agent state.

        Returns:
            Updated state from visualizer.
        """
        logger.info("Running visualizer agent...")
        return self.visualizer(state)

    def _generate_output(self, state: AgentState) -> AgentState:
        """Generate final output message.

        Args:
            state: Current agent state.

        Returns:
            State with final output message.
        """
        logger.info("Generating output...")

        output_parts = []

        # Summarize what was done
        task_type = state.get("task_type", TaskType.UNKNOWN)
        output_parts.append(f"Task Type: {task_type.value}")

        # Add SQL results summary
        sql_results = state.get("sql_results", [])
        if sql_results:
            output_parts.append(f"\nSQL Results: {len(sql_results)} queries executed")
            for i, result in enumerate(sql_results):
                if result.get("success"):
                    output_parts.append(f"  Query {i+1}: {result.get('row_count', 0)} rows")
                else:
                    output_parts.append(f"  Query {i+1}: FAILED - {result.get('error', 'Unknown')}")

        # Add visualization paths
        visualizations = state.get("visualizations", [])
        if visualizations:
            output_parts.append(f"\nVisualizations: {len(visualizations)} charts created")
            for path in visualizations:
                output_parts.append(f"  - {path}")

        # Add errors
        errors = state.get("errors", [])
        if errors:
            output_parts.append(f"\nErrors: {len(errors)}")
            for error in errors:
                output_parts.append(f"  - {error}")

        output_message = "\n".join(output_parts)

        return {
            **state,
            "messages": [*state["messages"], AIMessage(content=output_message)],
        }

    def _gather_data_context(self) -> DataContext:
        """Gather context about available data.

        Returns:
            DataContext with table information.
        """
        context = DataContext()

        # List tables
        tables_result = self.mcp_client.list_tables()
        if tables_result.success:
            context.tables = tables_result.data.get("tables", [])

            # Get schema for each table
            for table in context.tables:
                schema_result = self.mcp_client.describe_table(table)
                if schema_result.success:
                    context.schemas[table] = schema_result.data

                    # Get sample data
                    sample_result = self.mcp_client.execute_sql(
                        f"SELECT * FROM {table} LIMIT 3"
                    )
                    if sample_result.success:
                        context.sample_data[table] = sample_result.data.get("rows", [])

        return context

    def run(
        self,
        user_message: str,
        include_context: bool = True,
    ) -> AgentState:
        """Run the workflow with a user message.

        Args:
            user_message: The user's request.
            include_context: Whether to gather data context.

        Returns:
            Final agent state after processing.
        """
        logger.info(f"Running workflow for: {user_message[:100]}...")

        # Gather data context
        data_context = {}
        if include_context:
            context = self._gather_data_context()
            data_context = context.to_dict()

        # Create initial state
        initial_state = create_initial_state(user_message, data_context)

        # Run the graph
        result = self.graph.invoke(initial_state)

        logger.info("Workflow completed")
        return result

    async def arun(
        self,
        user_message: str,
        include_context: bool = True,
    ) -> AgentState:
        """Async version of run.

        Args:
            user_message: The user's request.
            include_context: Whether to gather data context.

        Returns:
            Final agent state after processing.
        """
        logger.info(f"Running async workflow for: {user_message[:100]}...")

        # Gather data context
        data_context = {}
        if include_context:
            context = self._gather_data_context()
            data_context = context.to_dict()

        # Create initial state
        initial_state = create_initial_state(user_message, data_context)

        # Run the graph
        result = await self.graph.ainvoke(initial_state)

        logger.info("Async workflow completed")
        return result


def create_datavault_graph(
    db_path: str | None = None,
    backend: str | None = None,
    model_name: str | None = None,
    output_dir: str | None = None,
) -> DataVaultGraph:
    """Factory function to create a DataVault graph.

    Args:
        db_path: Optional database path.
        backend: LLM backend ('mlx' or 'ollama'). Auto-detected if None.
        model_name: Optional LLM model name.
        output_dir: Optional output directory.

    Returns:
        Configured DataVaultGraph instance.
    """
    from src.mcp.client import create_client

    mcp_client = create_client(db_path)
    return DataVaultGraph(mcp_client, backend, model_name, output_dir)
