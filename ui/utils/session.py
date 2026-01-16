"""Session state management for Streamlit."""

import streamlit as st
from typing import Any


def init_session_state() -> None:
    """Initialize all session state variables."""
    defaults = {
        # Data state
        "tables": [],  # List of loaded table info dicts
        "mcp_client": None,
        
        # Chat state
        "messages": [],  # Chat history
        "query_history": [],  # All queries run
        
        # Memory state
        "memories": [],  # Semantic memories
        
        # Settings
        "llm_backend": "auto",
        "model_name": None,
        "langfuse_enabled": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_state(key: str, default: Any = None) -> Any:
    """Get a session state value.
    
    Args:
        key: State key.
        default: Default value if not found.
        
    Returns:
        State value.
    """
    return st.session_state.get(key, default)


def set_state(key: str, value: Any) -> None:
    """Set a session state value.
    
    Args:
        key: State key.
        value: Value to set.
    """
    st.session_state[key] = value


def add_table(name: str, rows: int, cols: int) -> None:
    """Add a table to the loaded tables list.
    
    Args:
        name: Table name.
        rows: Number of rows.
        cols: Number of columns.
    """
    # Check if already exists
    for table in st.session_state.tables:
        if table["name"] == name:
            table["rows"] = rows
            table["cols"] = cols
            return
    
    st.session_state.tables.append({
        "name": name,
        "rows": rows,
        "cols": cols,
    })


def add_message(role: str, content: str, sql: str | None = None) -> None:
    """Add a message to chat history.
    
    Args:
        role: Message role (user/assistant).
        content: Message content.
        sql: Optional SQL query.
    """
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "sql": sql,
    })


def add_query(query: str, sql: str, success: bool, result_count: int = 0) -> None:
    """Add a query to history.
    
    Args:
        query: Natural language query.
        sql: Generated SQL.
        success: Whether query succeeded.
        result_count: Number of results.
    """
    st.session_state.query_history.append({
        "query": query,
        "sql": sql,
        "success": success,
        "result_count": result_count,
    })


def clear_chat() -> None:
    """Clear chat history."""
    st.session_state.messages = []


def clear_all() -> None:
    """Clear all session state."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()
