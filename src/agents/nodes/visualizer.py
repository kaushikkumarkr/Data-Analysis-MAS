"""Visualization Agent node.

This agent handles visualization tasks including:
- Generating charts from SQL results
- Creating matplotlib/plotly visualizations
- Saving charts to files
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from src.agents.state import AgentState, TaskType
from src.mcp.client import MCPClient
from src.utils.config import get_config
from src.utils.llm_factory import create_chat_model
from src.utils.logging import get_logger

logger = get_logger("agents.visualizer")

VISUALIZER_SYSTEM_PROMPT = """You are a Data Visualization Agent specialized in creating charts.
Your role is to analyze SQL query results and generate Python code for visualizations.

Available capabilities:
1. Bar charts for comparisons
2. Line charts for trends
3. Pie charts for proportions
4. Scatter plots for correlations
5. Histograms for distributions
6. Heatmaps for matrices

When given data to visualize:
1. Understand what story the data tells
2. Choose the most appropriate chart type
3. Generate clean, well-labeled matplotlib code
4. Include proper titles, labels, and legends

IMPORTANT:
- Use matplotlib.pyplot (import as plt)
- Set figure size appropriately
- Use readable fonts and colors
- Save the figure with plt.savefig()
- Call plt.close() after saving

Respond with ONLY a valid JSON object:
{{
    "chart_type": "bar|line|pie|scatter|histogram",
    "title": "Chart title",
    "description": "What this visualization shows",
    "python_code": "Complete matplotlib code as a single string"
}}

The python_code should be complete and runnable, including imports.
Use the placeholder {{SAVE_PATH}} for the file path where the chart will be saved.
"""


class VisualizerAgent:
    """Visualization Agent for creating charts from data.

    This agent uses an LLM to analyze query results and generate
    matplotlib code for visualizations.
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        output_dir: str | None = None,
        backend: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize the visualizer agent.

        Args:
            mcp_client: MCP client for database operations.
            output_dir: Directory to save visualizations.
            backend: LLM backend ('mlx' or 'ollama'). Auto-detected if None.
            model_name: Optional override for LLM model name.
        """
        self.mcp_client = mcp_client
        self.output_dir = Path(output_dir or "./output/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        config = get_config()

        # Use LLM factory for automatic backend selection
        self.llm = create_chat_model(
            backend=backend,
            model_name=model_name,
            temperature=0.2,
        )

        backend_name = backend or ("mlx" if config.mlx.enabled else "ollama")
        model = model_name or (config.mlx.model if config.mlx.enabled else config.ollama.model)
        logger.info(f"VisualizerAgent initialized with {backend_name}: {model}")

    def _get_latest_results(self, state: AgentState) -> list[dict[str, Any]]:
        """Get the latest SQL results from state.

        Args:
            state: Current agent state.

        Returns:
            List of result rows or empty list.
        """
        sql_results = state.get("sql_results", [])
        if not sql_results:
            return []

        # Get the most recent successful result
        for result in reversed(sql_results):
            if result.get("success") and result.get("rows"):
                return result["rows"]

        return []

    def _build_context(self, state: AgentState) -> str:
        """Build context string from state for the LLM.

        Args:
            state: Current agent state.

        Returns:
            Context string for the prompt.
        """
        context_parts = []

        # Add recent SQL results
        sql_results = state.get("sql_results", [])
        if sql_results:
            context_parts.append("QUERY RESULTS TO VISUALIZE:")
            for result in sql_results[-2:]:
                if result.get("success"):
                    context_parts.append(f"\nQuery: {result['query'][:150]}...")
                    context_parts.append(f"Columns: {result.get('columns', [])}")
                    context_parts.append(f"Row count: {result.get('row_count', 0)}")

                    rows = result.get("rows", [])
                    if rows:
                        context_parts.append("Data (first 10 rows):")
                        for row in rows[:10]:
                            context_parts.append(f"  {row}")

        return "\n".join(context_parts)

    def _extract_code_from_response(self, response: str) -> tuple[str | None, str]:
        """Extract Python code and chart title from LLM response.

        Args:
            response: Raw LLM response text.

        Returns:
            Tuple of (python_code, chart_title).
        """
        title = "chart"

        # Try parsing as JSON first
        try:
            cleaned = re.sub(r'```json\s*', '', response)
            cleaned = re.sub(r'```\s*', '', cleaned)

            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if "python_code" in data:
                    title = data.get("title", "chart").replace(" ", "_").lower()
                    return data["python_code"], title
        except json.JSONDecodeError:
            pass

        # Fallback: extract Python from code blocks
        code_blocks = re.findall(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0], title

        return None, title

    def _execute_visualization_code(
        self,
        code: str,
        data: list[dict[str, Any]],
        title: str,
    ) -> str | None:
        """Execute the visualization code safely.

        Args:
            code: Python code to execute.
            data: Data to visualize.
            title: Chart title for filename.

        Returns:
            Path to saved file or None if failed.
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = re.sub(r'[^a-z0-9_]', '', title.lower()[:30])
        filename = f"{safe_title}_{timestamp}.png"
        save_path = self.output_dir / filename

        # Replace placeholder in code
        code = code.replace("{SAVE_PATH}", str(save_path))
        code = code.replace("{{SAVE_PATH}}", str(save_path))

        # Add data to execution context
        exec_globals = {
            "data": data,
            "plt": plt,
            "save_path": str(save_path),
        }

        # Try to add common imports
        try:
            import pandas as pd
            exec_globals["pd"] = pd
        except ImportError:
            pass

        try:
            # Execute the code
            exec(code, exec_globals)
            plt.close('all')

            if save_path.exists():
                logger.info(f"Saved visualization to: {save_path}")
                return str(save_path)
            else:
                logger.warning("Visualization code executed but file not saved")
                return None

        except Exception as e:
            logger.error(f"Visualization code execution failed: {e}")
            plt.close('all')
            return None

    async def process(self, state: AgentState) -> AgentState:
        """Process the visualization task.

        Args:
            state: Current agent state.

        Returns:
            Updated agent state with visualization paths.
        """
        logger.info("VisualizerAgent processing task")

        task = state.get("current_task", "")
        context = self._build_context(state)
        data = self._get_latest_results(state)

        if not data:
            logger.warning("No data available for visualization")
            return {
                **state,
                "errors": [*state.get("errors", []), "No data available for visualization"],
            }

        prompt = f"""{VISUALIZER_SYSTEM_PROMPT}

VISUALIZATION REQUEST: {task}

{context}

Generate the matplotlib code to create this visualization.
Remember to use {{{{SAVE_PATH}}}} as the file path for saving."""

        try:
            # Get LLM response
            response = await self.llm.ainvoke(prompt)
            response_text = response.content

            logger.debug(f"LLM response: {response_text[:200]}...")

            # Extract code
            code, title = self._extract_code_from_response(response_text)

            if not code:
                logger.warning("No visualization code extracted from response")
                return {
                    **state,
                    "messages": [*state["messages"], AIMessage(content=response_text)],
                    "errors": [*state.get("errors", []), "No visualization code generated"],
                }

            # Execute visualization
            viz_path = self._execute_visualization_code(code, data, title)

            visualizations = list(state.get("visualizations", []))
            if viz_path:
                visualizations.append(viz_path)

            return {
                **state,
                "messages": [*state["messages"], AIMessage(content=response_text)],
                "visualizations": visualizations,
                "task_type": TaskType.VISUALIZE,
            }

        except Exception as e:
            logger.error(f"VisualizerAgent error: {e}")
            return {
                **state,
                "errors": [*state.get("errors", []), f"Visualization failed: {str(e)}"],
            }

    def __call__(self, state: AgentState) -> AgentState:
        """Synchronous wrapper for LangGraph.

        Args:
            state: Current agent state.

        Returns:
            Updated agent state.
        """
        import asyncio

        return asyncio.run(self.process(state))


def create_visualizer_node(
    mcp_client: MCPClient,
    output_dir: str | None = None,
) -> callable:
    """Factory function to create a visualizer node for LangGraph.

    Args:
        mcp_client: MCP client for database operations.
        output_dir: Optional output directory for visualizations.

    Returns:
        Callable node function.
    """
    agent = VisualizerAgent(mcp_client, output_dir)
    return agent
