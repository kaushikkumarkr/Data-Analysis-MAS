"""Query Page - Fixed chat-based natural language interface."""

import streamlit as st
import sys
import time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.components.sidebar import render_sidebar
from ui.utils.session import init_session_state, add_message, add_query
from src.mcp.client import create_client


st.set_page_config(
    page_title="Query - DataVault",
    page_icon="🔍",
    layout="wide",
)

init_session_state()
render_sidebar()


# SQL Templates
SQL_TEMPLATES = {
    "📊 Count rows": "SELECT COUNT(*) as total FROM {table}",
    "👀 Preview data": "SELECT * FROM {table} LIMIT 10",
    "📈 Group by count": "SELECT {column}, COUNT(*) as count FROM {table} GROUP BY {column} ORDER BY count DESC",
    "💰 Sum values": "SELECT SUM({numeric_col}) as total FROM {table}",
    "📉 Average values": "SELECT AVG({numeric_col}) as average FROM {table}",
    "🔝 Top N by value": "SELECT * FROM {table} ORDER BY {numeric_col} DESC LIMIT 10",
    "🔍 Filter rows": "SELECT * FROM {table} WHERE {column} = 'value' LIMIT 100",
}


def generate_sql(prompt: str, table: str, columns: list, numeric_cols: list) -> str:
    """Generate SQL from natural language prompt."""
    prompt_lower = prompt.lower()
    
    # Helper to find matching column
    def find_column(cols: list, keywords: list = None) -> str:
        # First try exact match in prompt
        for col in cols:
            if col.lower() in prompt_lower:
                return col
        # Then try keyword match
        if keywords:
            for col in cols:
                for kw in keywords:
                    if kw in col.lower():
                        return col
        # Default to first
        return cols[0] if cols else None
    
    # Count patterns
    if any(word in prompt_lower for word in ["count", "how many", "number of"]):
        if any(word in prompt_lower for word in ["by", "group", "each", "per"]):
            group_col = find_column(columns)
            return f"SELECT {group_col}, COUNT(*) as count FROM {table} GROUP BY {group_col} ORDER BY count DESC LIMIT 15"
        return f"SELECT COUNT(*) as total FROM {table}"
    
    # Sum patterns
    if any(word in prompt_lower for word in ["sum", "total"]) and numeric_cols:
        value_col = find_column(numeric_cols, ["amount", "total", "price", "value", "revenue", "sales"])
        if any(word in prompt_lower for word in ["by", "per", "each", "group"]):
            # Find grouping column (non-numeric)
            non_numeric = [c for c in columns if c not in numeric_cols]
            group_col = find_column(non_numeric) if non_numeric else columns[0]
            return f"SELECT {group_col}, SUM({value_col}) as total FROM {table} GROUP BY {group_col} ORDER BY total DESC LIMIT 15"
        return f"SELECT SUM({value_col}) as total FROM {table}"
    
    # Average patterns
    if any(word in prompt_lower for word in ["average", "avg", "mean"]) and numeric_cols:
        value_col = find_column(numeric_cols)
        return f"SELECT AVG({value_col}) as average, MIN({value_col}) as min, MAX({value_col}) as max FROM {table}"
    
    # Top/best patterns
    if any(word in prompt_lower for word in ["top", "best", "highest", "largest"]) and numeric_cols:
        value_col = find_column(numeric_cols, ["amount", "total", "price", "value"])
        return f"SELECT * FROM {table} ORDER BY {value_col} DESC LIMIT 10"
    
    # Show/list patterns
    if any(word in prompt_lower for word in ["show", "list", "display", "see", "view", "first"]):
        return f"SELECT * FROM {table} LIMIT 10"
    
    # Unique/distinct patterns
    if any(word in prompt_lower for word in ["unique", "distinct", "different"]):
        col = find_column(columns)
        return f"SELECT {col}, COUNT(*) as count FROM {table} GROUP BY {col} ORDER BY count DESC LIMIT 20"
    
    # Default
    return f"SELECT * FROM {table} LIMIT 10"


st.markdown("## 🔍 Query Your Data")

# Check if connected
if not st.session_state.get("tables"):
    st.warning("⚠️ No data loaded. Upload a CSV file or load sample data from the sidebar.")
    st.stop()

# Initialize client if needed
if st.session_state.mcp_client is None:
    st.session_state.mcp_client = create_client()

# Mode tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📝 SQL Editor", "📋 Templates"])

with tab1:
    st.caption("Ask questions in natural language")
    
    # Display chat history
    for msg in st.session_state.messages[-8:]:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("sql"):
                with st.expander("📝 SQL", expanded=False):
                    st.code(msg["sql"], language="sql")
    
    # Chat input
    if prompt := st.chat_input("Ask about your data..."):
        add_message("user", prompt)
        
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🤖"):
            start_time = time.time()
            
            try:
                tables = [t["name"] for t in st.session_state.tables]
                table = tables[0]
                
                # Get schema
                schema = st.session_state.mcp_client.describe_table(table)
                columns = [c["name"] for c in schema.data.get("columns", [])]
                numeric_cols = [c["name"] for c in schema.data.get("columns", []) 
                               if any(t in c["type"].upper() for t in ["INT", "FLOAT", "DOUBLE", "DECIMAL"])]
                
                # Generate SQL
                sql = generate_sql(prompt, table, columns, numeric_cols)
                
                # Execute
                result = st.session_state.mcp_client.execute_sql(sql)
                elapsed = time.time() - start_time
                
                if result.success:
                    rows = result.data.get("rows", [])
                    
                    # Show SQL
                    with st.expander("📝 Generated SQL", expanded=True):
                        st.code(sql, language="sql")
                        st.caption(f"⏱️ {elapsed:.2f}s")
                    
                    if rows:
                        df = pd.DataFrame(rows)
                        
                        # Results
                        st.dataframe(df, use_container_width=True, height=300)
                        st.caption(f"✅ {len(rows)} rows returned")
                        
                        # Export
                        csv = df.to_csv(index=False)
                        st.download_button("📥 Download CSV", csv, "results.csv", "text/csv")
                        
                        add_message("assistant", f"Found {len(rows)} results.", sql)
                        add_query(prompt, sql, True, len(rows))
                    else:
                        st.info("Query returned no results.")
                        add_message("assistant", "No results found.", sql)
                        add_query(prompt, sql, True, 0)
                else:
                    st.error(f"Query failed: {result.error}")
                    add_message("assistant", f"Error: {result.error}", sql)
                    add_query(prompt, sql, False, 0)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                add_message("assistant", f"Error: {str(e)}")

with tab2:
    st.caption("Write and execute SQL directly")
    
    tables = [t["name"] for t in st.session_state.tables]
    default_sql = f"SELECT * FROM {tables[0]} LIMIT 10" if tables else ""
    
    sql_query = st.text_area("SQL Query", value=default_sql, height=150, label_visibility="collapsed")
    
    if st.button("▶️ Run Query", type="primary"):
        if sql_query:
            start_time = time.time()
            result = st.session_state.mcp_client.execute_sql(sql_query)
            elapsed = time.time() - start_time
            
            if result.success:
                rows = result.data.get("rows", [])
                st.success(f"✅ {len(rows)} rows in {elapsed:.2f}s")
                
                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)
                    
                    csv = df.to_csv(index=False)
                    st.download_button("📥 Download CSV", csv, "sql_results.csv", "text/csv")
                
                add_query("SQL Editor", sql_query, True, len(rows))
            else:
                st.error(f"❌ {result.error}")
                add_query("SQL Editor", sql_query, False, 0)

with tab3:
    st.caption("Common SQL query templates")
    
    tables = [t["name"] for t in st.session_state.tables]
    if not tables:
        st.warning("Load data first")
        st.stop()
    
    table = st.selectbox("Table", tables, key="template_table")
    
    schema = st.session_state.mcp_client.describe_table(table)
    columns = [c["name"] for c in schema.data.get("columns", [])]
    numeric_cols = [c["name"] for c in schema.data.get("columns", []) 
                   if any(t in c["type"].upper() for t in ["INT", "FLOAT", "DOUBLE", "DECIMAL"])]
    
    st.divider()
    
    for template_name, template_sql in SQL_TEMPLATES.items():
        with st.expander(template_name):
            sql = template_sql.replace("{table}", table)
            
            if "{column}" in sql and columns:
                col = st.selectbox("Column", columns, key=f"col_{template_name}")
                sql = sql.replace("{column}", col)
            
            if "{numeric_col}" in sql:
                if numeric_cols:
                    num_col = st.selectbox("Numeric column", numeric_cols, key=f"num_{template_name}")
                    sql = sql.replace("{numeric_col}", num_col)
                else:
                    st.warning("No numeric columns")
                    continue
            
            st.code(sql, language="sql")
            
            if st.button("▶️ Run", key=f"run_{template_name}"):
                result = st.session_state.mcp_client.execute_sql(sql)
                if result.success:
                    st.dataframe(result.data.get("rows", []), use_container_width=True)
                else:
                    st.error(result.error)
