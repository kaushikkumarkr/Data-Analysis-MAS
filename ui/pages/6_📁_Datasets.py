"""Example Datasets Page - Load sample and public datasets."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.components.sidebar import render_sidebar
from ui.utils.session import init_session_state, add_table
from src.mcp.client import create_client


st.set_page_config(
    page_title="Datasets - DataVault",
    page_icon="📁",
    layout="wide",
)

init_session_state()
render_sidebar()


st.markdown("## 📁 Example Datasets")
st.caption("Load sample data or connect to public datasets")

# Initialize client if needed
if st.session_state.mcp_client is None:
    st.session_state.mcp_client = create_client()

# Tabs for different dataset sources
tab1, tab2, tab3 = st.tabs(["📊 Built-in Samples", "🌐 Public Datasets", "🔗 Load from URL"])

with tab1:
    st.markdown("### Built-in Sample Datasets")
    st.markdown("Pre-loaded datasets for quick testing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Sales Data")
        st.caption("50 rows • E-commerce transactions")
        st.markdown("""
        - Transaction details
        - Products & categories
        - Payment methods
        - Regional breakdown
        """)
        if st.button("Load Sales", key="load_sales", use_container_width=True):
            sample_path = Path(__file__).parent.parent.parent / "data" / "sample" / "sales_data.csv"
            result = st.session_state.mcp_client.load_dataset(str(sample_path), "sales")
            if result.success:
                schema = st.session_state.mcp_client.describe_table("sales")
                add_table("sales", result.data["rows_loaded"], len(schema.data.get("columns", [])))
                st.success("✅ Loaded!")
                st.rerun()
    
    with col2:
        st.markdown("#### 👥 Customers")
        st.caption("20 rows • Customer profiles")
        st.markdown("""
        - Contact info
        - Lifetime value
        - Order history
        - Segments
        """)
        if st.button("Load Customers", key="load_customers", use_container_width=True):
            sample_path = Path(__file__).parent.parent.parent / "data" / "sample" / "customers.csv"
            result = st.session_state.mcp_client.load_dataset(str(sample_path), "customers")
            if result.success:
                schema = st.session_state.mcp_client.describe_table("customers")
                add_table("customers", result.data["rows_loaded"], len(schema.data.get("columns", [])))
                st.success("✅ Loaded!")
                st.rerun()
    
    with col3:
        st.markdown("#### 📦 Products")
        st.caption("25 rows • Product catalog")
        st.markdown("""
        - Categories
        - Pricing & costs
        - Inventory
        - Ratings
        """)
        if st.button("Load Products", key="load_products", use_container_width=True):
            sample_path = Path(__file__).parent.parent.parent / "data" / "sample" / "products.csv"
            result = st.session_state.mcp_client.load_dataset(str(sample_path), "products")
            if result.success:
                schema = st.session_state.mcp_client.describe_table("products")
                add_table("products", result.data["rows_loaded"], len(schema.data.get("columns", [])))
                st.success("✅ Loaded!")
                st.rerun()
    
    st.divider()
    
    if st.button("📥 Load All Samples", type="primary", use_container_width=True):
        with st.spinner("Loading all datasets..."):
            datasets = [
                ("sales_data.csv", "sales"),
                ("customers.csv", "customers"),
                ("products.csv", "products"),
            ]
            for filename, table_name in datasets:
                sample_path = Path(__file__).parent.parent.parent / "data" / "sample" / filename
                result = st.session_state.mcp_client.load_dataset(str(sample_path), table_name)
                if result.success:
                    schema = st.session_state.mcp_client.describe_table(table_name)
                    add_table(table_name, result.data["rows_loaded"], len(schema.data.get("columns", [])))
            st.success("✅ All datasets loaded!")
            st.rerun()

with tab2:
    st.markdown("### Public Datasets")
    st.markdown("Load data from well-known public sources via DuckDB's httpfs extension")
    
    # Public dataset URLs
    public_datasets = {
        "🗽 NYC Taxi Sample": {
            "url": "https://github.com/cwida/duckdb-data/releases/download/v1.0/taxi_2019_04.parquet",
            "description": "NYC taxi trip data (April 2019 sample)",
            "table_name": "nyc_taxi",
            "type": "parquet",
        },
        "🌡️ Weather Stations": {
            "url": "https://raw.githubusercontent.com/datasets/global-temp/master/data/monthly.csv",
            "description": "Global temperature anomalies by month",
            "table_name": "global_temp",
            "type": "csv",
        },
        "🏠 California Housing": {
            "url": "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv",
            "description": "California housing prices dataset",
            "table_name": "housing",
            "type": "csv",
        },
        "🍷 Wine Quality": {
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "description": "Red wine quality dataset (UCI ML)",
            "table_name": "wine_quality",
            "type": "csv",
            "delimiter": ";",
        },
    }
    
    for name, info in public_datasets.items():
        with st.expander(name):
            st.markdown(info["description"])
            st.code(info["url"], language=None)
            
            if st.button(f"Load {info['table_name']}", key=f"load_{info['table_name']}"):
                with st.spinner(f"Loading {info['table_name']}..."):
                    try:
                        if info["type"] == "parquet":
                            sql = f"CREATE OR REPLACE TABLE {info['table_name']} AS SELECT * FROM read_parquet('{info['url']}') LIMIT 10000"
                        else:
                            delimiter = info.get("delimiter", ",")
                            sql = f"CREATE OR REPLACE TABLE {info['table_name']} AS SELECT * FROM read_csv_auto('{info['url']}', delim='{delimiter}') LIMIT 10000"
                        
                        result = st.session_state.mcp_client.execute_sql(sql)
                        
                        if result.success:
                            # Get row count
                            count_result = st.session_state.mcp_client.execute_sql(f"SELECT COUNT(*) as cnt FROM {info['table_name']}")
                            row_count = count_result.data["rows"][0]["cnt"] if count_result.success else 0
                            
                            schema = st.session_state.mcp_client.describe_table(info["table_name"])
                            col_count = len(schema.data.get("columns", [])) if schema.success else 0
                            
                            add_table(info["table_name"], row_count, col_count)
                            st.success(f"✅ Loaded {row_count:,} rows!")
                            st.rerun()
                        else:
                            st.error(f"Failed: {result.error}")
                    except Exception as e:
                        st.error(f"Error: {e}")

with tab3:
    st.markdown("### Load from URL")
    st.markdown("Enter a URL to a CSV or Parquet file")
    
    url = st.text_input(
        "Dataset URL",
        placeholder="https://example.com/data.csv",
    )
    
    col1, col2 = st.columns(2)
    with col1:
        table_name = st.text_input("Table Name", value="custom_data")
    with col2:
        file_type = st.selectbox("File Type", ["csv", "parquet"])
    
    if st.button("📥 Load from URL", type="primary", disabled=not url):
        with st.spinner("Loading from URL..."):
            try:
                if file_type == "parquet":
                    sql = f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{url}')"
                else:
                    sql = f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{url}')"
                
                result = st.session_state.mcp_client.execute_sql(sql)
                
                if result.success:
                    count_result = st.session_state.mcp_client.execute_sql(f"SELECT COUNT(*) as cnt FROM {table_name}")
                    row_count = count_result.data["rows"][0]["cnt"] if count_result.success else 0
                    
                    schema = st.session_state.mcp_client.describe_table(table_name)
                    col_count = len(schema.data.get("columns", [])) if schema.success else 0
                    
                    add_table(table_name, row_count, col_count)
                    st.success(f"✅ Loaded {row_count:,} rows from URL!")
                    st.rerun()
                else:
                    st.error(f"Failed: {result.error}")
            except Exception as e:
                st.error(f"Error: {e}")
