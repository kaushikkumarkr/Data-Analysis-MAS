"""Memory Page - Query history and semantic memory."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.components.sidebar import render_sidebar
from ui.utils.session import init_session_state


st.set_page_config(
    page_title="Memory - DataVault",
    page_icon="🧠",
    layout="wide",
)

init_session_state()
render_sidebar()


st.markdown("## 🧠 Memory & History")
st.caption("View query history and stored memories")

# Tabs for different memory types
tab1, tab2 = st.tabs(["📝 Query History", "💭 Semantic Memory"])

with tab1:
    st.markdown("### Recent Queries")
    
    if st.session_state.get("query_history"):
        for i, query in enumerate(reversed(st.session_state.query_history[-20:])):
            with st.expander(
                f"{'✅' if query['success'] else '❌'} {query['query'][:50]}...", 
                expanded=False
            ):
                st.markdown(f"**Query:** {query['query']}")
                st.code(query['sql'], language="sql")
                st.caption(f"Results: {query['result_count']} rows")
    else:
        st.info("No queries yet. Go to the Query page to start exploring your data!")
    
    if st.session_state.get("query_history"):
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.query_history = []
            st.rerun()

with tab2:
    st.markdown("### Semantic Memory")
    st.caption("Facts, preferences, and context remembered across sessions")
    
    # Memory stats
    col1, col2, col3 = st.columns(3)
    
    memories = st.session_state.get("memories", [])
    
    with col1:
        st.metric("Total Memories", len(memories))
    with col2:
        facts = sum(1 for m in memories if m.get("type") == "fact")
        st.metric("Facts", facts)
    with col3:
        prefs = sum(1 for m in memories if m.get("type") == "preference")
        st.metric("Preferences", prefs)
    
    st.divider()
    
    if memories:
        for memory in memories:
            st.markdown(f"- **{memory.get('type', 'unknown').title()}:** {memory.get('content', '')}")
    else:
        st.info("No memories stored yet. Memory is built as you interact with DataVault.")
    
    st.divider()
    
    # Manual memory input
    st.markdown("### ➕ Add Memory")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        new_memory = st.text_input("Enter a fact or preference", placeholder="e.g., I prefer bar charts for sales data")
    with col2:
        memory_type = st.selectbox("Type", ["fact", "preference", "context"])
    
    if st.button("💾 Save Memory"):
        if new_memory:
            st.session_state.memories.append({
                "type": memory_type,
                "content": new_memory,
            })
            st.success("Memory saved!")
            st.rerun()
