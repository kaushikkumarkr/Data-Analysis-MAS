"""Dashboard Page - Enhanced interactive visualizations."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.components.sidebar import render_sidebar
from ui.utils.session import init_session_state
from ui.components.charts import (
    create_bar_chart, create_line_chart, create_pie_chart,
    create_scatter_chart, create_histogram, create_box_plot,
    create_area_chart, auto_chart, DARK_THEME
)
from src.mcp.client import create_client


st.set_page_config(
    page_title="Dashboard - DataVault",
    page_icon="📈",
    layout="wide",
)

init_session_state()
render_sidebar()


st.markdown("## 📈 Dashboard")

# Check if connected
if not st.session_state.get("tables"):
    st.warning("⚠️ No data loaded. Upload a CSV file or load sample data from the sidebar.")
    st.stop()

# Initialize client if needed
if st.session_state.mcp_client is None:
    st.session_state.mcp_client = create_client()

# Tabs for different dashboard modes
tab1, tab2 = st.tabs(["🎨 Chart Builder", "⚡ Auto Dashboard"])

with tab1:
    # Chart builder
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        table_names = [t["name"] for t in st.session_state.tables]
        selected_table = st.selectbox("Table", table_names)
    
    with col2:
        chart_type = st.selectbox(
            "Chart Type",
            ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", 
             "Histogram", "Box Plot", "Area Chart"],
        )
    
    with col3:
        aggregation = st.selectbox(
            "Aggregation",
            ["Count", "Sum", "Average", "Min", "Max"],
        )
    
    if selected_table:
        # Get schema
        schema_result = st.session_state.mcp_client.describe_table(selected_table)
        
        if schema_result.success:
            columns = schema_result.data.get("columns", [])
            col_names = [c["name"] for c in columns]
            numeric_cols = [c["name"] for c in columns 
                           if any(t in c["type"].upper() for t in ["INT", "FLOAT", "DOUBLE", "DECIMAL"])]
            categorical_cols = [c["name"] for c in columns if c["name"] not in numeric_cols]
            
            st.divider()
            
            # Chart configuration based on type
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if chart_type in ["Bar Chart", "Line Chart", "Area Chart"]:
                    x_col = st.selectbox("X Axis (Category)", categorical_cols if categorical_cols else col_names, key="x_axis")
                elif chart_type == "Pie Chart":
                    x_col = st.selectbox("Category", categorical_cols if categorical_cols else col_names, key="pie_cat")
                elif chart_type in ["Scatter Plot"]:
                    x_col = st.selectbox("X Axis", numeric_cols if numeric_cols else col_names, key="scatter_x")
                elif chart_type in ["Histogram", "Box Plot"]:
                    x_col = st.selectbox("Column", numeric_cols if numeric_cols else col_names, key="hist_col")
            
            with col2:
                if chart_type in ["Scatter Plot"]:
                    y_options = [c for c in numeric_cols if c != x_col] if len(numeric_cols) > 1 else col_names
                    y_col = st.selectbox("Y Axis", y_options, key="scatter_y")
                elif chart_type in ["Bar Chart", "Line Chart", "Pie Chart", "Area Chart"]:
                    if aggregation != "Count" and numeric_cols:
                        y_col = st.selectbox("Value Column", numeric_cols, key="value_col")
                    else:
                        y_col = None
                elif chart_type == "Box Plot":
                    y_col = st.selectbox("Group By (optional)", ["None"] + categorical_cols, key="box_group")
                    y_col = None if y_col == "None" else y_col
                else:
                    y_col = None
            
            with col3:
                color_by = st.selectbox("Color By (optional)", ["None"] + categorical_cols, key="color_by")
                color_by = None if color_by == "None" else color_by
            
            with col4:
                limit = st.slider("Limit", 5, 100, 20)
            
            # Generate chart
            if st.button("🎨 Generate Chart", type="primary", use_container_width=True):
                with st.spinner("Generating..."):
                    try:
                        # Build SQL
                        if chart_type in ["Bar Chart", "Line Chart", "Pie Chart", "Area Chart"]:
                            if aggregation == "Count":
                                if color_by:
                                    sql = f"SELECT {x_col}, {color_by}, COUNT(*) as value FROM {selected_table} GROUP BY {x_col}, {color_by} ORDER BY value DESC LIMIT {limit}"
                                else:
                                    sql = f"SELECT {x_col}, COUNT(*) as value FROM {selected_table} GROUP BY {x_col} ORDER BY value DESC LIMIT {limit}"
                            else:
                                agg_func = {"Sum": "SUM", "Average": "AVG", "Min": "MIN", "Max": "MAX"}[aggregation]
                                if color_by:
                                    sql = f"SELECT {x_col}, {color_by}, {agg_func}({y_col}) as value FROM {selected_table} GROUP BY {x_col}, {color_by} ORDER BY value DESC LIMIT {limit}"
                                else:
                                    sql = f"SELECT {x_col}, {agg_func}({y_col}) as value FROM {selected_table} GROUP BY {x_col} ORDER BY value DESC LIMIT {limit}"
                        elif chart_type == "Scatter Plot":
                            if color_by:
                                sql = f"SELECT {x_col}, {y_col}, {color_by} FROM {selected_table} LIMIT {limit * 50}"
                            else:
                                sql = f"SELECT {x_col}, {y_col} FROM {selected_table} LIMIT {limit * 50}"
                        elif chart_type == "Histogram":
                            sql = f"SELECT {x_col} FROM {selected_table} WHERE {x_col} IS NOT NULL LIMIT {limit * 50}"
                        elif chart_type == "Box Plot":
                            if y_col:
                                sql = f"SELECT {x_col}, {y_col} FROM {selected_table} WHERE {x_col} IS NOT NULL LIMIT {limit * 50}"
                            else:
                                sql = f"SELECT {x_col} FROM {selected_table} WHERE {x_col} IS NOT NULL LIMIT {limit * 50}"
                        
                        result = st.session_state.mcp_client.execute_sql(sql)
                        
                        if result.success and result.data.get("rows"):
                            df = pd.DataFrame(result.data["rows"])
                            
                            # Create chart
                            if chart_type == "Bar Chart":
                                fig = create_bar_chart(df, x_col, "value", f"{aggregation} by {x_col}", color_by)
                            elif chart_type == "Line Chart":
                                fig = create_line_chart(df, x_col, "value", f"{aggregation} by {x_col}", color_by)
                            elif chart_type == "Pie Chart":
                                fig = create_pie_chart(df, x_col, "value", f"{aggregation} by {x_col}")
                            elif chart_type == "Scatter Plot":
                                fig = create_scatter_chart(df, x_col, y_col, f"{x_col} vs {y_col}", color_by)
                            elif chart_type == "Histogram":
                                fig = create_histogram(df, x_col, f"Distribution of {x_col}")
                            elif chart_type == "Box Plot":
                                fig = create_box_plot(df, x_col, y_col, f"Distribution of {x_col}")
                            elif chart_type == "Area Chart":
                                fig = create_area_chart(df, x_col, "value", f"{aggregation} by {x_col}", color_by)
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show SQL and data
                            col1, col2 = st.columns(2)
                            with col1:
                                with st.expander("📝 SQL"):
                                    st.code(sql, language="sql")
                            with col2:
                                with st.expander("📊 Data"):
                                    st.dataframe(df, use_container_width=True)
                        else:
                            st.error(f"Query failed: {result.error if not result.success else 'No data'}")
                    except Exception as e:
                        st.error(f"Error: {e}")

with tab2:
    # Auto Dashboard
    st.markdown("### ⚡ Auto-Generated Dashboard")
    st.caption("Automatically generate visualizations based on your data")
    
    table_names = [t["name"] for t in st.session_state.tables]
    auto_table = st.selectbox("Select table for auto-dashboard", table_names, key="auto_table")
    
    if st.button("⚡ Generate Dashboard", type="primary", use_container_width=True):
        with st.spinner("Analyzing data and generating charts..."):
            # Load data
            result = st.session_state.mcp_client.execute_sql(f"SELECT * FROM {auto_table} LIMIT 1000")
            
            if result.success and result.data.get("rows"):
                df = pd.DataFrame(result.data["rows"])
                
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                st.success(f"✅ Analyzed {len(df)} rows, {len(numeric_cols)} numeric cols, {len(categorical_cols)} categorical cols")
                
                st.divider()
                
                # KPI row
                if numeric_cols:
                    st.markdown("### 📊 Key Metrics")
                    metric_cols = st.columns(min(4, len(numeric_cols)))
                    for i, col in enumerate(numeric_cols[:4]):
                        with metric_cols[i]:
                            st.metric(col, f"{df[col].sum():,.0f}", f"Avg: {df[col].mean():,.1f}")
                
                st.divider()
                
                # Charts row 1
                col1, col2 = st.columns(2)
                
                with col1:
                    if categorical_cols and numeric_cols:
                        st.markdown(f"### {categorical_cols[0]} Distribution")
                        agg_data = df.groupby(categorical_cols[0])[numeric_cols[0]].sum().reset_index()
                        agg_data.columns = [categorical_cols[0], 'value']
                        fig = create_bar_chart(agg_data.head(10), categorical_cols[0], 'value')
                        st.plotly_chart(fig, use_container_width=True)
                    elif numeric_cols:
                        st.markdown(f"### {numeric_cols[0]} Distribution")
                        fig = create_histogram(df, numeric_cols[0])
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if categorical_cols:
                        st.markdown(f"### {categorical_cols[0]} Breakdown")
                        counts = df[categorical_cols[0]].value_counts().reset_index()
                        counts.columns = [categorical_cols[0], 'count']
                        fig = create_pie_chart(counts.head(8), categorical_cols[0], 'count')
                        st.plotly_chart(fig, use_container_width=True)
                    elif len(numeric_cols) >= 2:
                        st.markdown(f"### {numeric_cols[0]} vs {numeric_cols[1]}")
                        fig = create_scatter_chart(df, numeric_cols[0], numeric_cols[1])
                        st.plotly_chart(fig, use_container_width=True)
                
                # Charts row 2
                if len(numeric_cols) >= 2 or len(categorical_cols) >= 2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if len(numeric_cols) >= 2:
                            st.markdown(f"### {numeric_cols[0]} vs {numeric_cols[1]}")
                            color_col = categorical_cols[0] if categorical_cols else None
                            fig = create_scatter_chart(df, numeric_cols[0], numeric_cols[1], color=color_col)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if len(categorical_cols) >= 2 and numeric_cols:
                            st.markdown(f"### {categorical_cols[1]} by {categorical_cols[0]}")
                            agg_data = df.groupby([categorical_cols[0], categorical_cols[1]])[numeric_cols[0]].sum().reset_index()
                            agg_data.columns = [categorical_cols[0], categorical_cols[1], 'value']
                            fig = px.bar(agg_data.head(20), x=categorical_cols[0], y='value', color=categorical_cols[1], template="plotly_dark")
                            fig.update_layout(**DARK_THEME)
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Could not load data: {result.error if not result.success else 'No data'}")
