"""Explorer Page - Enhanced data schema and profiling."""

import streamlit as st
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.components.sidebar import render_sidebar
from ui.utils.session import init_session_state
from ui.components.profiler import render_data_profile, render_column_stats, render_quality_report
from ui.components.charts import create_heatmap
from src.mcp.client import create_client


st.set_page_config(
    page_title="Explorer - DataVault",
    page_icon="📊",
    layout="wide",
)

init_session_state()
render_sidebar()


st.markdown("## 📊 Data Explorer")

# Check if connected
if not st.session_state.get("tables"):
    st.warning("⚠️ No data loaded. Upload a CSV file or load sample data from the sidebar.")
    st.stop()

# Initialize client if needed
if st.session_state.mcp_client is None:
    st.session_state.mcp_client = create_client()

# Table selector
table_names = [t["name"] for t in st.session_state.tables]
selected_table = st.selectbox("Select Table", table_names)

if selected_table:
    # Load full data for profiling
    result = st.session_state.mcp_client.execute_sql(f"SELECT * FROM {selected_table} LIMIT 10000")
    
    if result.success and result.data.get("rows"):
        df = pd.DataFrame(result.data["rows"])
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Schema", "📊 Profile", "🏥 Quality", "🔗 Correlations"])
        
        with tab1:
            st.markdown("### 📋 Table Schema")
            
            schema_result = st.session_state.mcp_client.describe_table(selected_table)
            
            if schema_result.success:
                schema = schema_result.data
                
                # Overview metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", f"{schema.get('row_count', len(df)):,}")
                with col2:
                    st.metric("Columns", len(schema.get("columns", [])))
                with col3:
                    numeric = sum(1 for c in schema.get("columns", []) 
                                 if any(t in c["type"].upper() for t in ["INT", "FLOAT", "DOUBLE", "DECIMAL"]))
                    st.metric("Numeric", numeric)
                with col4:
                    text = len(schema.get("columns", [])) - numeric
                    st.metric("Text/Other", text)
                
                st.divider()
                
                # Schema table
                schema_data = []
                for i, col in enumerate(schema.get("columns", [])):
                    col_type = col["type"]
                    if any(t in col_type.upper() for t in ["INT", "FLOAT", "DOUBLE", "DECIMAL"]):
                        type_icon = "🔢"
                    elif "VARCHAR" in col_type.upper() or "STRING" in col_type.upper():
                        type_icon = "📝"
                    elif "DATE" in col_type.upper() or "TIME" in col_type.upper():
                        type_icon = "📅"
                    elif "BOOL" in col_type.upper():
                        type_icon = "✅"
                    else:
                        type_icon = "📦"
                    
                    schema_data.append({
                        "#": i + 1,
                        "Column": col["name"],
                        "Type": f"{type_icon} {col['type']}",
                        "Nullable": "✓" if col.get("nullable", True) else "",
                        "Unique": df[col["name"]].nunique() if col["name"] in df.columns else "-",
                        "Nulls": df[col["name"]].isna().sum() if col["name"] in df.columns else "-",
                    })
                
                st.dataframe(schema_data, use_container_width=True, hide_index=True)
                
                st.divider()
                
                # Sample data
                st.markdown("### 👀 Sample Data")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Download schema
                st.download_button(
                    "📥 Download Schema as CSV",
                    pd.DataFrame(schema_data).to_csv(index=False),
                    f"{selected_table}_schema.csv",
                    "text/csv",
                )
        
        with tab2:
            render_data_profile(df)
            
            st.divider()
            
            # Column deep-dive
            st.markdown("### 🔬 Column Deep-Dive")
            selected_col = st.selectbox("Select column to analyze", df.columns.tolist())
            
            if selected_col:
                render_column_stats(df, selected_col)
        
        with tab3:
            render_quality_report(df)
            
            st.divider()
            
            # Data cleaning suggestions
            st.markdown("### 🧹 Cleaning Suggestions")
            
            suggestions = []
            
            # Check for missing values
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                suggestions.append(f"**Fill or drop nulls** in: {', '.join(missing_cols[:5])}")
            
            # Check for potential duplicates
            if df.duplicated().sum() > 0:
                suggestions.append(f"**Remove {df.duplicated().sum()} duplicate rows**")
            
            # Check for high cardinality
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() / len(df) > 0.9:
                    suggestions.append(f"**Consider dropping** high-cardinality column '{col}'")
            
            # Check for all-null columns
            all_null_cols = df.columns[df.isnull().all()].tolist()
            if all_null_cols:
                suggestions.append(f"**Drop empty columns**: {', '.join(all_null_cols)}")
            
            if suggestions:
                for suggestion in suggestions[:5]:
                    st.markdown(f"- {suggestion}")
            else:
                st.success("✅ Data looks clean! No immediate suggestions.")
        
        with tab4:
            st.markdown("### 🔗 Correlation Analysis")
            
            numeric_df = df.select_dtypes(include=['number'])
            
            if len(numeric_df.columns) >= 2:
                fig = create_heatmap(df, f"Correlation Matrix - {selected_table}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Top correlations
                st.markdown("#### Top Correlations")
                corr_matrix = numeric_df.corr()
                
                # Get pairs
                pairs = []
                for i, col1 in enumerate(corr_matrix.columns):
                    for j, col2 in enumerate(corr_matrix.columns):
                        if i < j:
                            pairs.append({
                                "Column 1": col1,
                                "Column 2": col2,
                                "Correlation": round(corr_matrix.iloc[i, j], 3)
                            })
                
                pairs_df = pd.DataFrame(pairs).sort_values("Correlation", key=abs, ascending=False)
                st.dataframe(pairs_df.head(10), use_container_width=True, hide_index=True)
            else:
                st.info("Need at least 2 numeric columns for correlation analysis.")
    else:
        st.error(f"Could not load data: {result.error if not result.success else 'No data'}")
