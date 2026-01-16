"""DataVault - Enhanced Main Streamlit Application."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.utils.session import init_session_state
from ui.components.sidebar import render_sidebar


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="DataVault",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # Enhanced CSS
    st.markdown("""
    <style>
        /* Hide default Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Gradient header */
        .main-header {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #7C3AED 0%, #EC4899 50%, #06B6D4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        
        .subtitle {
            text-align: center;
            color: #94A3B8;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        
        /* Feature cards */
        .feature-card {
            background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            border: 1px solid #475569;
            margin-bottom: 1rem;
        }
        
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        /* Metric cards */
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px solid #475569;
        }
        
        div[data-testid="stMetric"] label {
            color: #94A3B8 !important;
        }
        
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #F8FAFC !important;
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 0.75rem;
            border: none;
            background: linear-gradient(135deg, #7C3AED 0%, #8B5CF6 100%);
            color: white;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #8B5CF6 0%, #A78BFA 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            border-radius: 0.5rem;
            background-color: #1E293B;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border-radius: 0.5rem;
        }
        
        /* Chat styling */
        .stChatMessage {
            border-radius: 1rem;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    render_sidebar()

    # Hero section
    st.markdown('<p class="main-header">🏛️ DataVault</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Privacy-Preserving Data Analytics with Natural Language</p>', unsafe_allow_html=True)

    # Check if any tables are loaded
    if not st.session_state.get("tables", []):
        # Welcome screen
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔒 Privacy First")
            st.markdown("""
            All data processing happens locally 
            using DuckDB. Your data never leaves 
            your machine.
            """)
        
        with col2:
            st.markdown("### 💬 Natural Language")
            st.markdown("""
            Ask questions in plain English. 
            Our AI translates your intent 
            into SQL queries.
            """)
        
        with col3:
            st.markdown("### 📊 Visual Insights")
            st.markdown("""
            Auto-generate charts and dashboards.
            Export results to CSV for further 
            analysis.
            """)
        
        st.divider()
        
        # Quick start
        st.markdown("### 🚀 Get Started")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Upload Your Data")
            st.markdown("Drag & drop a CSV file in the sidebar to begin analyzing.")
            
        with col2:
            st.markdown("#### Or Try Sample Data")
            st.markdown("Click a sample dataset in the sidebar to explore the features.")
        
        st.divider()
        
        # Features grid
        st.markdown("### ✨ Features")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("#### 🔍 Query")
            st.caption("Natural language & SQL")
        
        with col2:
            st.markdown("#### 📊 Explorer")
            st.caption("Schema & data profiling")
        
        with col3:
            st.markdown("#### 📈 Dashboard")
            st.caption("Interactive charts")
        
        with col4:
            st.markdown("#### 🧠 Memory")
            st.caption("Query history")

    else:
        # Dashboard view when data is loaded
        st.divider()
        
        # Summary metrics
        total_tables = len(st.session_state.tables)
        total_rows = sum(t["rows"] for t in st.session_state.tables)
        total_cols = sum(t["cols"] for t in st.session_state.tables)
        total_queries = len(st.session_state.get("query_history", []))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Tables", total_tables)
        with col2:
            st.metric("📝 Total Rows", f"{total_rows:,}")
        with col3:
            st.metric("📋 Total Columns", total_cols)
        with col4:
            st.metric("🔍 Queries Run", total_queries)
        
        st.divider()
        
        # Loaded tables
        st.markdown("### 📊 Loaded Tables")
        
        for table in st.session_state.tables:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{table['name']}**")
            with col2:
                st.caption(f"{table['rows']:,} rows")
            with col3:
                st.caption(f"{table['cols']} columns")
        
        st.divider()
        
        # Quick actions
        st.markdown("### 🚀 Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔍 New Query", use_container_width=True):
                st.switch_page("pages/1_🔍_Query.py")
        
        with col2:
            if st.button("📊 Explore Data", use_container_width=True):
                st.switch_page("pages/2_📊_Explorer.py")
        
        with col3:
            if st.button("📈 Dashboard", use_container_width=True):
                st.switch_page("pages/3_📈_Dashboard.py")
        
        with col4:
            if st.button("📁 More Datasets", use_container_width=True):
                st.switch_page("pages/6_📁_Datasets.py")
        
        # Recent queries
        if st.session_state.get("query_history"):
            st.divider()
            st.markdown("### 📜 Recent Queries")
            
            for query in reversed(st.session_state.query_history[-3:]):
                with st.expander(f"{'✅' if query['success'] else '❌'} {query['query'][:50]}..."):
                    st.code(query['sql'], language="sql")
                    st.caption(f"Results: {query['result_count']} rows")


if __name__ == "__main__":
    main()
