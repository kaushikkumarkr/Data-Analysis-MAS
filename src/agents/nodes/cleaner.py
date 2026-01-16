"""Data Cleaning Agent node.

This agent handles data quality tasks including:
- Null value handling
- Type conversion
- Deduplication
- Data standardization
"""

from typing import Any

from langchain_core.messages import AIMessage

from src.agents.state import AgentState, SQLResult, TaskType
from src.mcp.client import MCPClient
from src.utils.config import get_config
from src.utils.llm_factory import create_chat_model
from src.utils.logging import get_logger

logger = get_logger("agents.cleaner")

CLEANER_SYSTEM_PROMPT = """You are a Data Cleaning Agent specialized in data quality tasks.
Your role is to analyze data quality issues and generate SQL queries to fix them.

Available capabilities:
1. Identify and handle NULL values (fill with defaults, remove rows, etc.)
2. Convert data types (strings to dates, numbers, etc.)
3. Remove duplicate records
4. Standardize text values (trim, lowercase, etc.)
5. Handle outliers and invalid values

When given a cleaning task:
1. First, analyze the table schema and sample data to understand the data
2. Identify specific quality issues
3. Generate a SQL query or series of queries to fix the issues
4. Explain what the query does

IMPORTANT:
- Always use valid SQL syntax for DuckDB
- Be cautious with destructive operations (DELETE, UPDATE)
- Prefer creating new cleaned columns/tables over modifying originals
- Document your approach

Respond with a JSON object containing:
{{
    "analysis": "Description of data quality issues found",
    "sql_queries": ["list", "of", "sql", "queries"],
    "explanation": "What the queries accomplish"
}}
"""


class CleanerAgent:
    """Data Cleaning Agent for handling data quality tasks.

    This agent uses an LLM to analyze data quality issues and
    generate SQL queries to clean and transform data.
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        backend: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize the cleaner agent.

        Args:
            mcp_client: MCP client for database operations.
            backend: LLM backend ('mlx' or 'ollama'). Auto-detected if None.
            model_name: Optional override for LLM model name.
        """
        self.mcp_client = mcp_client
        config = get_config()

        # Use LLM factory for automatic backend selection
        self.llm = create_chat_model(
            backend=backend,
            model_name=model_name,
            temperature=0.1,
        )

        backend_name = backend or ("mlx" if config.mlx.enabled else "ollama")
        model = model_name or (config.mlx.model if config.mlx.enabled else config.ollama.model)
        logger.info(f"CleanerAgent initialized with {backend_name}: {model}")

    def _build_context(self, state: AgentState) -> str:
        """Build context string from state for the LLM.

        Args:
            state: Current agent state.

        Returns:
            Context string for the prompt.
        """
        context_parts = []

        # Add data context if available
        data_context = state.get("data_context", {})
        if data_context:
            tables = data_context.get("tables", [])
            schemas = data_context.get("schemas", {})

            if tables:
                context_parts.append("Available Tables:")
                for table in tables:
                    if table in schemas:
                        schema = schemas[table]
                        cols = [f"{c['name']} ({c['type']})" for c in schema.get("columns", [])]
                        context_parts.append(f"  • {table}: {', '.join(cols)}")

            sample_data = data_context.get("sample_data", {})
            if sample_data:
                context_parts.append("\nSample Data (first 3 rows):")
                for table, samples in sample_data.items():
                    if samples:
                        context_parts.append(f"  {table}:")
                        for row in samples[:3]:
                            context_parts.append(f"    {row}")

        return "\n".join(context_parts)

    def _extract_sql_from_response(self, response: str) -> list[str]:
        """Extract SQL queries from LLM response.

        Args:
            response: Raw LLM response text.

        Returns:
            List of SQL query strings.
        """
        import json
        import re

        queries = []

        # Try parsing as JSON first
        try:
            # Find JSON object in response
            json_match = re.search(r'\{[^{}]*"sql_queries"[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if "sql_queries" in data:
                    return data["sql_queries"]
        except json.JSONDecodeError:
            pass

        # Fallback: extract SQL from code blocks
        sql_blocks = re.findall(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if sql_blocks:
            queries.extend(sql_blocks)

        # Fallback: look for SELECT/UPDATE/DELETE/CREATE statements
        if not queries:
            statements = re.findall(
                r'((?:SELECT|UPDATE|DELETE|CREATE|INSERT|ALTER)[^;]+;?)',
                response,
                re.IGNORECASE | re.DOTALL
            )
            queries.extend(statements)

        return queries

    async def process(self, state: AgentState) -> AgentState:
        """Process the cleaning task.

        Args:
            state: Current agent state.

        Returns:
            Updated agent state with cleaning results.
        """
        logger.info("CleanerAgent processing task")

        task = state.get("current_task", "")
        context = self._build_context(state)

        # Build prompt
        prompt = f"""{CLEANER_SYSTEM_PROMPT}

Current Task: {task}

{context}

Please analyze the data and provide cleaning SQL queries."""

        try:
            # Get LLM response
            response = await self.llm.ainvoke(prompt)
            response_text = response.content

            logger.debug(f"LLM response: {response_text[:200]}...")

            # Extract SQL queries
            queries = self._extract_sql_from_response(response_text)

            if not queries:
                logger.warning("No SQL queries extracted from response")
                return {
                    **state,
                    "messages": [*state["messages"], AIMessage(content=response_text)],
                    "errors": [*state.get("errors", []), "No cleaning queries generated"],
                }

            # Execute queries
            sql_results = list(state.get("sql_results", []))
            for query in queries:
                logger.info(f"Executing cleaning query: {query[:100]}...")
                result = self.mcp_client.execute_sql(query)

                sql_result = SQLResult(
                    query=query,
                    success=result.success,
                    rows=result.data.get("rows", []) if result.success else [],
                    row_count=result.data.get("row_count", 0) if result.success else 0,
                    columns=result.data.get("columns", []) if result.success else [],
                    error=result.error if not result.success else None,
                )
                sql_results.append(sql_result.to_dict())

            return {
                **state,
                "messages": [*state["messages"], AIMessage(content=response_text)],
                "sql_results": sql_results,
                "task_type": TaskType.CLEAN,
            }

        except Exception as e:
            logger.error(f"CleanerAgent error: {e}")
            return {
                **state,
                "errors": [*state.get("errors", []), f"Cleaning failed: {str(e)}"],
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


def create_cleaner_node(mcp_client: MCPClient) -> callable:
    """Factory function to create a cleaner node for LangGraph.

    Args:
        mcp_client: MCP client for database operations.

    Returns:
        Callable node function.
    """
    agent = CleanerAgent(mcp_client)
    return agent
