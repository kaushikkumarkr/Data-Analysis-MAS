"""Data profiling and quality report components."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Any


def render_data_profile(df: pd.DataFrame) -> None:
    """Render comprehensive data profiling section."""
    st.markdown("### 📊 Data Profile")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Memory", f"{memory_mb:.2f} MB")
    with col4:
        duplicates = df.duplicated().sum()
        st.metric("Duplicates", f"{duplicates:,}")
    
    st.divider()
    
    # Column types breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Column Types")
        type_counts = df.dtypes.astype(str).value_counts()
        type_df = pd.DataFrame({
            "Type": type_counts.index,
            "Count": type_counts.values
        })
        
        fig = px.pie(
            type_df, 
            names="Type", 
            values="Count",
            hole=0.4,
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="#0F172A",
            plot_bgcolor="#1E293B",
            font=dict(color="#F8FAFC"),
            height=250,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Missing Values")
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing": missing.values,
            "Percent": (missing.values / len(df) * 100).round(1)
        }).sort_values("Missing", ascending=False)
        
        if missing_df["Missing"].sum() > 0:
            fig = px.bar(
                missing_df[missing_df["Missing"] > 0].head(10),
                x="Column",
                y="Missing",
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="#0F172A",
                plot_bgcolor="#1E293B",
                font=dict(color="#F8FAFC"),
                height=250,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values!")


def render_column_stats(df: pd.DataFrame, column: str) -> None:
    """Render detailed statistics for a column."""
    col_data = df[column]
    col_type = col_data.dtype
    
    st.markdown(f"### 📈 Column: `{column}`")
    st.caption(f"Type: {col_type}")
    
    # Basic stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Non-null", f"{col_data.notna().sum():,}")
    with col2:
        st.metric("Null", f"{col_data.isna().sum():,}")
    with col3:
        st.metric("Unique", f"{col_data.nunique():,}")
    with col4:
        pct_unique = (col_data.nunique() / len(col_data) * 100)
        st.metric("Unique %", f"{pct_unique:.1f}%")
    
    st.divider()
    
    # Type-specific analysis
    if pd.api.types.is_numeric_dtype(col_type):
        _render_numeric_stats(col_data)
    else:
        _render_categorical_stats(col_data)


def _render_numeric_stats(col_data: pd.Series) -> None:
    """Render numeric column statistics."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Distribution")
        fig = px.histogram(
            col_data.dropna(),
            nbins=30,
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="#0F172A",
            plot_bgcolor="#1E293B",
            font=dict(color="#F8FAFC"),
            showlegend=False,
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        fig.update_traces(marker_color="#7C3AED")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Statistics")
        stats = col_data.describe()
        stats_df = pd.DataFrame({
            "Metric": ["Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
            "Value": [
                f"{stats['mean']:.2f}",
                f"{stats['std']:.2f}",
                f"{stats['min']:.2f}",
                f"{stats['25%']:.2f}",
                f"{stats['50%']:.2f}",
                f"{stats['75%']:.2f}",
                f"{stats['max']:.2f}",
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Box plot
        fig = px.box(col_data.dropna(), template="plotly_dark")
        fig.update_layout(
            paper_bgcolor="#0F172A",
            plot_bgcolor="#1E293B",
            font=dict(color="#F8FAFC"),
            height=150,
            margin=dict(l=20, r=20, t=10, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_categorical_stats(col_data: pd.Series) -> None:
    """Render categorical column statistics."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Value Distribution")
        value_counts = col_data.value_counts().head(15)
        fig = px.bar(
            x=value_counts.values,
            y=value_counts.index,
            orientation='h',
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="#0F172A",
            plot_bgcolor="#1E293B",
            font=dict(color="#F8FAFC"),
            showlegend=False,
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Count",
            yaxis_title="",
        )
        fig.update_traces(marker_color="#7C3AED")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Top Values")
        top_values = col_data.value_counts().head(10).reset_index()
        top_values.columns = ["Value", "Count"]
        top_values["Percent"] = (top_values["Count"] / len(col_data) * 100).round(1).astype(str) + "%"
        st.dataframe(top_values, use_container_width=True, hide_index=True)


def render_quality_report(df: pd.DataFrame) -> dict:
    """Render data quality report and return quality score."""
    st.markdown("### 🏥 Data Quality Report")
    
    issues = []
    score = 100
    
    # Check for missing values
    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    if missing_pct > 0:
        issues.append(f"⚠️ {missing_pct:.1f}% missing values")
        score -= min(30, missing_pct)
    
    # Check for duplicates
    dup_pct = df.duplicated().sum() / len(df) * 100
    if dup_pct > 0:
        issues.append(f"⚠️ {dup_pct:.1f}% duplicate rows")
        score -= min(20, dup_pct)
    
    # Check for high cardinality columns
    for col in df.select_dtypes(include=['object']).columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.9:
            issues.append(f"⚠️ High cardinality in '{col}' ({df[col].nunique()} unique)")
            score -= 5
    
    # Check for potential ID columns (all unique)
    for col in df.columns:
        if df[col].nunique() == len(df) and len(df) > 10:
            issues.append(f"ℹ️ '{col}' appears to be an identifier (all unique)")
    
    score = max(0, score)
    
    # Display score
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if score >= 80:
            st.success(f"## 🟢 {score:.0f}/100")
            st.caption("Excellent quality")
        elif score >= 60:
            st.warning(f"## 🟡 {score:.0f}/100")
            st.caption("Good quality")
        else:
            st.error(f"## 🔴 {score:.0f}/100")
            st.caption("Needs attention")
    
    with col2:
        if issues:
            for issue in issues[:5]:
                st.markdown(f"- {issue}")
        else:
            st.success("✅ No quality issues detected!")
    
    return {"score": score, "issues": issues}
