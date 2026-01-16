"""Analysis Agent node.

This agent handles analytical tasks including:
- Aggregations and summaries
- Statistical analysis
- Trend detection
- Complex joins and filtering
"""

import json
import re
from typing import Any

from langchain_core.messages import AIMessage

from src.agents.state import AgentState, SQLResult, TaskType
from src.mcp.client import MCPClient
from src.utils.config import get_config
from src.utils.llm_factory import create_chat_model
from src.utils.logging import get_logger

logger = get_logger("agents.analyst")

ANALYST_SYSTEM_PROMPT = """You are a Data Analysis Agent specialized in SQL analytics.
Your role is to understand analytical questions and generate SQL queries to answer them.

Available capabilities:
1. Generate aggregation queries (COUNT, SUM, AVG, MIN, MAX, etc.)
2. Group data by dimensions (GROUP BY)
3. Filter data (WHERE clauses)
4. Join tables when needed
5. Calculate percentages and ratios
6. Identify trends and patterns
7. Create window functions for advanced analytics

When given an analysis task:
1. Understand what the user wants to know
2. Identify which tables and columns are relevant
3. Generate an efficient SQL query
4. Explain your approach and what the results mean

IMPORTANT:
- Use valid DuckDB SQL syntax
- Optimize queries for performance
- Handle NULL values appropriately
- Use meaningful aliases for columns
- Limit results for large datasets

Respond with ONLY a valid JSON object (no markdown formatting):
{{
    "understanding": "What the user wants to know",
    "approach": "How you'll analyze this",
    "sql_query": "The SQL query to execute",
    "interpretation_guide": "How to interpret the results"
}}
"""


class AnalystAgent:
    """Analysis Agent for handling analytical SQL queries.

    This agent uses an LLM to understand analytical questions and
    generate appropriate SQL queries for DuckDB.
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        backend: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize the analyst agent.

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
        logger.info(f"AnalystAgent initialized with {backend_name}: {model}")

    def _build_context(self, state: AgentState) -> str:
        """Build context string from state for the LLM.

        Args:
            state: Current agent state.

        Returns:
            Context string for the prompt.
        """
        context_parts = []

        data_context = state.get("data_context", {})
        if data_context:
            tables = data_context.get("tables", [])
            schemas = data_context.get("schemas", {})

            if tables:
                context_parts.append("DATABASE SCHEMA:")
                for table in tables:
                    if table in schemas:
                        schema = schemas[table]
                        context_parts.append(f"\nTable: {table}")
                        context_parts.append(f"  Row count: {schema.get('row_count', 'unknown')}")
                        context_parts.append("  Columns:")
                        for col in schema.get("columns", []):
                            context_parts.append(f"    - {col['name']}: {col['type']}")

            sample_data = data_context.get("sample_data", {})
            if sample_data:
                context_parts.append("\nSAMPLE DATA:")
                for table, samples in sample_data.items():
                    if samples:
                        context_parts.append(f"\n{table} (first row):")
                        context_parts.append(f"  {samples[0]}")

        # Add previous SQL results for context
        sql_results = state.get("sql_results", [])
        if sql_results:
            context_parts.append("\nPREVIOUS QUERY RESULTS:")
            for result in sql_results[-2:]:  # Last 2 results
                if result.get("success"):
                    context_parts.append(f"  Query: {result['query'][:100]}...")
                    context_parts.append(f"  Rows: {result.get('row_count', 0)}")

        return "\n".join(context_parts)

    def _extract_sql_from_response(self, response: str) -> str | None:
        """Extract SQL query from LLM response.

        Args:
            response: Raw LLM response text.

        Returns:
            SQL query string or None.
        """
        # Try parsing as JSON first
        try:
            # Remove markdown code blocks if present
            cleaned = re.sub(r'```json\s*', '', response)
            cleaned = re.sub(r'```\s*', '', cleaned)
            cleaned = cleaned.strip()

            # Find JSON object
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if "sql_query" in data:
                    return data["sql_query"]
        except json.JSONDecodeError:
            pass

        # Fallback: extract SQL from code blocks
        sql_blocks = re.findall(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        if sql_blocks:
            return sql_blocks[0]

        # Fallback: look for SELECT statements
        select_match = re.search(
            r'(SELECT\s+.*?(?:;|$))',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if select_match:
            return select_match.group(1).rstrip(';') + ';'

        return None

    async def process(self, state: AgentState) -> AgentState:
        """Process the analysis task.

        Args:
            state: Current agent state.

        Returns:
            Updated agent state with analysis results.
        """
        logger.info("AnalystAgent processing task")

        task = state.get("current_task", "")
        context = self._build_context(state)

        prompt = f"""{ANALYST_SYSTEM_PROMPT}

USER QUESTION: {task}

{context}

Generate the SQL query to answer this question."""

        try:
            # Get LLM response
            response = await self.llm.ainvoke(prompt)
            response_text = response.content

            logger.debug(f"LLM response: {response_text[:200]}...")

            # Extract SQL query
            query = self._extract_sql_from_response(response_text)

            if not query:
                logger.warning("No SQL query extracted from response")
                return {
                    **state,
                    "messages": [*state["messages"], AIMessage(content=response_text)],
                    "errors": [*state.get("errors", []), "No analysis query generated"],
                }

            # Execute query
            logger.info(f"Executing analysis query: {query[:100]}...")
            result = self.mcp_client.execute_sql(query)

            sql_result = SQLResult(
                query=query,
                success=result.success,
                rows=result.data.get("rows", []) if result.success else [],
                row_count=result.data.get("row_count", 0) if result.success else 0,
                columns=result.data.get("columns", []) if result.success else [],
                error=result.error if not result.success else None,
            )

            sql_results = list(state.get("sql_results", []))
            sql_results.append(sql_result.to_dict())

            return {
                **state,
                "messages": [*state["messages"], AIMessage(content=response_text)],
                "sql_results": sql_results,
                "task_type": TaskType.ANALYZE,
            }

        except Exception as e:
            logger.error(f"AnalystAgent error: {e}")
            return {
                **state,
                "errors": [*state.get("errors", []), f"Analysis failed: {str(e)}"],
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


def create_analyst_node(mcp_client: MCPClient) -> callable:
    """Factory function to create an analyst node for LangGraph.

    Args:
        mcp_client: MCP client for database operations.

    Returns:
        Callable node function.
    """
    agent = AnalystAgent(mcp_client)
    return agent
