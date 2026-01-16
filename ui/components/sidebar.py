"""Sidebar component for DataVault UI."""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mcp.client import create_client
from ui.utils.session import add_table


def render_sidebar() -> None:
    """Render the sidebar with upload and table info."""
    with st.sidebar:
        st.markdown("## 🏛️ DataVault")
        st.caption("Privacy-Preserving Analytics")
        
        st.divider()
        
        # File upload section
        st.markdown("### 📤 Upload Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file to analyze",
            label_visibility="collapsed",
        )
        
        if uploaded_file is not None:
            # Get table name from filename
            default_name = Path(uploaded_file.name).stem
            table_name = st.text_input(
                "Table name",
                value=default_name,
                help="Name for the table in DuckDB",
            )
            
            if st.button("📥 Load Data", use_container_width=True):
                with st.spinner("Loading data..."):
                    try:
                        # Save temp file
                        temp_path = Path(f"/tmp/{uploaded_file.name}")
                        temp_path.write_bytes(uploaded_file.getvalue())
                        
                        # Initialize client if needed
                        if st.session_state.mcp_client is None:
                            st.session_state.mcp_client = create_client()
                        
                        # Load into DuckDB
                        result = st.session_state.mcp_client.load_dataset(
                            str(temp_path), 
                            table_name
                        )
                        
                        if result.success:
                            # Get column count
                            schema = st.session_state.mcp_client.describe_table(table_name)
                            col_count = len(schema.data.get("columns", [])) if schema.success else 0
                            
                            add_table(
                                table_name, 
                                result.data["rows_loaded"],
                                col_count
                            )
                            st.success(f"✅ Loaded {result.data['rows_loaded']:,} rows")
                            st.rerun()
                        else:
                            st.error(f"Failed: {result.error}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        st.divider()
        
        # Loaded tables section
        st.markdown("### 📊 Loaded Tables")
        
        if st.session_state.get("tables"):
            for table in st.session_state.tables:
                with st.container():
                    st.markdown(f"**{table['name']}**")
                    st.caption(f"{table['rows']:,} rows × {table['cols']} cols")
        else:
            st.caption("No tables loaded yet")
        
        st.divider()
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True, help="Clear all data"):
                if st.session_state.mcp_client:
                    st.session_state.mcp_client.close()
                    st.session_state.mcp_client = None
                st.session_state.tables = []
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            # Sample data dropdown
            sample_datasets = {
                "📊 Sales": ("sales_data.csv", "sales"),
                "👥 Customers": ("customers.csv", "customers"),
                "📦 Products": ("products.csv", "products"),
            }
            
            selected_sample = st.selectbox(
                "Sample Data",
                ["Select..."] + list(sample_datasets.keys()),
                label_visibility="collapsed",
            )
            
            if selected_sample != "Select...":
                filename, table_name = sample_datasets[selected_sample]
                sample_path = Path(__file__).parent.parent.parent / "data" / "sample" / filename
                
                if sample_path.exists():
                    if st.session_state.mcp_client is None:
                        st.session_state.mcp_client = create_client()
                    
                    result = st.session_state.mcp_client.load_dataset(str(sample_path), table_name)
                    if result.success:
                        schema = st.session_state.mcp_client.describe_table(table_name)
                        col_count = len(schema.data.get("columns", [])) if schema.success else 0
                        add_table(table_name, result.data["rows_loaded"], col_count)
                        st.success(f"✅ Loaded {table_name}")
                        st.rerun()
        
        st.divider()
        
        # Connection status
        st.markdown("### 🔌 Status")
        
        if st.session_state.mcp_client:
            st.markdown("🟢 Connected to DuckDB")
        else:
            st.markdown("🔴 Not connected")
        
        # Langfuse status
        if st.session_state.get("langfuse_enabled"):
            st.markdown("🟢 Langfuse tracing")
        else:
            st.markdown("⚪ Langfuse disabled")
