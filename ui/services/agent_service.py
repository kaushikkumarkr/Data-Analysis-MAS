"""Agent Service - Wrapper for LangGraph agents in Streamlit with Langfuse tracing."""

import streamlit as st
from typing import Any, Generator
from dataclasses import dataclass
import time
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mcp.client import MCPClient, create_client
from src.agents.state import AgentState, TaskType, create_initial_state
from src.utils.llm_factory import create_chat_model, get_available_backends
from src.utils.config import get_config
from src.utils.logging import get_logger

# Langfuse tracing
try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    observe = lambda *args, **kwargs: lambda f: f  # No-op decorator

logger = get_logger("ui.agent_service")


@dataclass
class AgentResponse:
    """Response from agent execution."""
    success: bool
    sql: str | None = None
    results: list[dict] | None = None
    understanding: str | None = None
    approach: str | None = None
    interpretation: str | None = None
    error: str | None = None
    elapsed_time: float = 0.0


class AgentService:
    """Service for running LLM agents from Streamlit."""
    
    def __init__(self, mcp_client: MCPClient, backend: str = "auto"):
        """Initialize the agent service.
        
        Args:
            mcp_client: MCP client for database operations.
            backend: LLM backend (auto, mlx, ollama).
        """
        self.mcp_client = mcp_client
        self.backend = backend
        self._llm = None
        
    @property
    def llm(self):
        """Get or create the LLM."""
        if self._llm is None:
            available = get_available_backends()
            if not available:
                raise RuntimeError("No LLM backend available. Install mlx-lm or langchain-ollama.")
            
            backend = self.backend if self.backend != "auto" else available[0]
            self._llm = create_chat_model(backend=backend, temperature=0.1)
            logger.info(f"LLM initialized with backend: {backend}")
        return self._llm
    
    @observe(name="generate_sql")
    def generate_sql(self, query: str, table: str, schema: dict) -> AgentResponse:
        """Generate SQL from natural language using LLM.
        
        Args:
            query: Natural language query.
            table: Table name to query.
            schema: Table schema.
            
        Returns:
            AgentResponse with generated SQL and reasoning.
        """
        # Add Langfuse trace metadata
        if LANGFUSE_AVAILABLE:
            try:
                langfuse_context.update_current_trace(
                    name="DataVault Query",
                    user_id="streamlit_user",
                    metadata={"table": table, "query": query},
                )
            except Exception:
                pass  # Langfuse not configured
        
        start_time = time.time()
        
        # Build context
        columns = schema.get("columns", [])
        column_info = "\n".join([f"  - {c['name']} ({c['type']})" for c in columns])
        
        prompt = f"""You are a SQL expert. Generate a DuckDB SQL query for this request.

Table: {table}
Columns:
{column_info}

User Request: {query}

Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{{
    "understanding": "What the user wants to know",
    "approach": "Your analysis approach",
    "sql": "SELECT ... FROM {table} ...",
    "interpretation": "How to read the results"
}}"""

        try:
            from langchain_core.messages import HumanMessage
            
            # Call LLM
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            import json
            import re
            
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                # Fallback: try parsing entire content
                parsed = json.loads(content)
            
            sql = parsed.get("sql", "").strip()
            
            # Execute the generated SQL
            result = self.mcp_client.execute_sql(sql)
            
            elapsed = time.time() - start_time
            
            if result.success:
                return AgentResponse(
                    success=True,
                    sql=sql,
                    results=result.data.get("rows", []),
                    understanding=parsed.get("understanding"),
                    approach=parsed.get("approach"),
                    interpretation=parsed.get("interpretation"),
                    elapsed_time=elapsed,
                )
            else:
                return AgentResponse(
                    success=False,
                    sql=sql,
                    error=result.error,
                    understanding=parsed.get("understanding"),
                    approach=parsed.get("approach"),
                    elapsed_time=elapsed,
                )
                
        except json.JSONDecodeError as e:
            elapsed = time.time() - start_time
            return AgentResponse(
                success=False,
                error=f"Failed to parse LLM response: {str(e)}\nRaw: {content[:200]}",
                elapsed_time=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Agent error: {e}")
            return AgentResponse(
                success=False,
                error=str(e),
                elapsed_time=elapsed,
            )


@st.cache_resource
def get_agent_service(_mcp_client: MCPClient, backend: str = "auto") -> AgentService:
    """Get or create cached agent service.
    
    Args:
        _mcp_client: MCP client (underscore prefix for Streamlit caching).
        backend: LLM backend.
        
    Returns:
        AgentService instance.
    """
    return AgentService(_mcp_client, backend)
