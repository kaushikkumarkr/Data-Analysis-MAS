"""Query Page - LLM-powered natural language interface."""

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


# Pattern-based SQL generation (fallback)
def generate_sql_pattern(prompt: str, table: str, columns: list, numeric_cols: list) -> str:
    """Generate SQL using pattern matching (fast fallback)."""
    prompt_lower = prompt.lower()
    
    def find_column(cols, keywords=None):
        for col in cols:
            if col.lower() in prompt_lower:
                return col
        if keywords:
            for col in cols:
                for kw in keywords:
                    if kw in col.lower():
                        return col
        return cols[0] if cols else None
    
    if any(w in prompt_lower for w in ["count", "how many"]):
        if any(w in prompt_lower for w in ["by", "group", "each"]):
            col = find_column(columns)
            return f"SELECT {col}, COUNT(*) as count FROM {table} GROUP BY {col} ORDER BY count DESC LIMIT 15"
        return f"SELECT COUNT(*) as total FROM {table}"
    
    if any(w in prompt_lower for w in ["sum", "total"]) and numeric_cols:
        val_col = find_column(numeric_cols, ["amount", "total", "price", "value"])
        if any(w in prompt_lower for w in ["by", "per", "each"]):
            non_num = [c for c in columns if c not in numeric_cols]
            grp_col = find_column(non_num) if non_num else columns[0]
            return f"SELECT {grp_col}, SUM({val_col}) as total FROM {table} GROUP BY {grp_col} ORDER BY total DESC LIMIT 15"
        return f"SELECT SUM({val_col}) as total FROM {table}"
    
    if any(w in prompt_lower for w in ["average", "avg", "mean"]) and numeric_cols:
        col = find_column(numeric_cols)
        return f"SELECT AVG({col}) as average, MIN({col}) as min, MAX({col}) as max FROM {table}"
    
    if any(w in prompt_lower for w in ["top", "best", "highest"]) and numeric_cols:
        col = find_column(numeric_cols, ["amount", "total", "price"])
        return f"SELECT * FROM {table} ORDER BY {col} DESC LIMIT 10"
    
    return f"SELECT * FROM {table} LIMIT 10"


st.markdown("## 🔍 Query Your Data")

# Check if connected
if not st.session_state.get("tables"):
    st.warning("⚠️ No data loaded. Upload a CSV file or load sample data from the sidebar.")
    st.stop()

# Initialize client
if st.session_state.mcp_client is None:
    st.session_state.mcp_client = create_client()

# AI Mode toggle
col1, col2 = st.columns([3, 1])
with col2:
    ai_mode = st.toggle("🤖 AI Mode", value=False, help="Use LLM for SQL generation (slower but smarter)")

if ai_mode:
    st.caption("🤖 **AI Mode ON** - Using LLM for intelligent SQL generation")
else:
    st.caption("⚡ **Fast Mode** - Using pattern matching (instant)")

st.divider()

# Tabs
tab1, tab2 = st.tabs(["💬 Chat", "📝 SQL Editor"])

with tab1:
    # Display chat history
    for msg in st.session_state.messages[-6:]:
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
            
            tables = [t["name"] for t in st.session_state.tables]
            table = tables[0]
            
            # Get schema
            schema_result = st.session_state.mcp_client.describe_table(table)
            schema = schema_result.data
            columns = [c["name"] for c in schema.get("columns", [])]
            numeric_cols = [c["name"] for c in schema.get("columns", []) 
                           if any(t in c["type"].upper() for t in ["INT", "FLOAT", "DOUBLE", "DECIMAL"])]
            
            if ai_mode:
                # LLM-powered generation
                with st.status("🤖 AI Agent thinking...", expanded=True) as status:
                    st.write("📖 Understanding your query...")
                    
                    try:
                        from ui.services.agent_service import AgentService
                        
                        st.write("🧠 Generating SQL with LLM...")
                        agent = AgentService(st.session_state.mcp_client)
                        response = agent.generate_sql(prompt, table, schema)
                        
                        elapsed = response.elapsed_time
                        
                        if response.success:
                            status.update(label="✅ Query complete!", state="complete")
                            
                            # Show reasoning
                            st.markdown("### 🧠 Agent Reasoning")
                            
                            if response.understanding:
                                st.markdown(f"**Understanding:** {response.understanding}")
                            if response.approach:
                                st.markdown(f"**Approach:** {response.approach}")
                            
                            st.divider()
                            
                            # Show SQL
                            st.markdown("### 📝 Generated SQL")
                            st.code(response.sql, language="sql")
                            st.caption(f"⏱️ Generated in {elapsed:.2f}s")
                            
                            # Show results
                            if response.results:
                                st.markdown("### 📊 Results")
                                df = pd.DataFrame(response.results)
                                st.dataframe(df, use_container_width=True)
                                st.caption(f"✅ {len(response.results)} rows")
                                
                                # Interpretation
                                if response.interpretation:
                                    st.info(f"💡 **Interpretation:** {response.interpretation}")
                                
                                # Export
                                csv = df.to_csv(index=False)
                                st.download_button("📥 Download CSV", csv, "results.csv", "text/csv")
                                
                                add_message("assistant", f"Found {len(response.results)} results.", response.sql)
                                add_query(prompt, response.sql, True, len(response.results))
                            else:
                                st.info("Query returned no results.")
                                add_message("assistant", "No results found.", response.sql)
                        else:
                            status.update(label="❌ Error", state="error")
                            st.error(f"Error: {response.error}")
                            if response.sql:
                                st.code(response.sql, language="sql")
                            add_message("assistant", f"Error: {response.error}", response.sql)
                            
                    except Exception as e:
                        status.update(label="❌ Error", state="error")
                        st.error(f"Agent error: {str(e)}")
                        st.info("💡 Tip: Make sure Ollama is running or MLX is installed")
                        add_message("assistant", f"Error: {str(e)}")
            else:
                # Pattern-based (fast)
                try:
                    sql = generate_sql_pattern(prompt, table, columns, numeric_cols)
                    result = st.session_state.mcp_client.execute_sql(sql)
                    elapsed = time.time() - start_time
                    
                    if result.success:
                        with st.expander("📝 Generated SQL", expanded=True):
                            st.code(sql, language="sql")
                            st.caption(f"⏱️ {elapsed:.2f}s (pattern matching)")
                        
                        rows = result.data.get("rows", [])
                        if rows:
                            df = pd.DataFrame(rows)
                            st.dataframe(df, use_container_width=True, height=300)
                            st.caption(f"✅ {len(rows)} rows")
                            
                            csv = df.to_csv(index=False)
                            st.download_button("📥 Download CSV", csv, "results.csv", "text/csv")
                            
                            add_message("assistant", f"Found {len(rows)} results.", sql)
                            add_query(prompt, sql, True, len(rows))
                        else:
                            st.info("Query returned no results.")
                            add_message("assistant", "No results found.", sql)
                    else:
                        st.error(f"Query failed: {result.error}")
                        add_message("assistant", f"Error: {result.error}", sql)
                        
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

# Info box about AI Mode
with st.expander("ℹ️ About AI Mode"):
    st.markdown("""
    ### 🤖 AI Mode
    When enabled, your queries are processed by a Large Language Model (LLM):
    - **MLX** (Apple Silicon) or **Ollama** (any platform)
    - Shows agent reasoning and approach
    - Handles complex queries better
    - Slower (~2-5 seconds per query)
    
    ### ⚡ Fast Mode
    Uses pattern matching:
    - Instant responses
    - Works offline
    - Good for simple queries
    
    **Tip:** Start with Fast Mode, switch to AI Mode for complex analysis!
    """)
