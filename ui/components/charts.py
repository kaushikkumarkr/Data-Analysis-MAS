"""Chart generation components."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional


# Dark theme for Plotly
DARK_THEME = {
    "paper_bgcolor": "#0F172A",
    "plot_bgcolor": "#1E293B",
    "font": {"color": "#F8FAFC"},
    "colorway": ["#7C3AED", "#EC4899", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#14B8A6"],
}


def create_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: Optional[str] = None,
    horizontal: bool = False,
) -> go.Figure:
    """Create a styled bar chart."""
    if horizontal:
        fig = px.bar(df, y=x, x=y, title=title, color=color, orientation="h", template="plotly_dark")
    else:
        fig = px.bar(df, x=x, y=y, title=title, color=color, template="plotly_dark")
    
    fig.update_layout(**DARK_THEME)
    fig.update_traces(marker_line_width=0)
    return fig


def create_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: Optional[str] = None,
) -> go.Figure:
    """Create a styled line chart."""
    fig = px.line(df, x=x, y=y, title=title, color=color, template="plotly_dark", markers=True)
    fig.update_layout(**DARK_THEME)
    return fig


def create_pie_chart(
    df: pd.DataFrame,
    names: str,
    values: str,
    title: str = "",
) -> go.Figure:
    """Create a styled pie chart."""
    fig = px.pie(df, names=names, values=values, title=title, template="plotly_dark", hole=0.4)
    fig.update_layout(**DARK_THEME)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def create_scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: Optional[str] = None,
    size: Optional[str] = None,
) -> go.Figure:
    """Create a styled scatter chart."""
    fig = px.scatter(df, x=x, y=y, title=title, color=color, size=size, template="plotly_dark")
    fig.update_layout(**DARK_THEME)
    return fig


def create_histogram(
    df: pd.DataFrame,
    x: str,
    title: str = "",
    nbins: int = 30,
) -> go.Figure:
    """Create a styled histogram."""
    fig = px.histogram(df, x=x, title=title, template="plotly_dark", nbins=nbins)
    fig.update_layout(**DARK_THEME)
    fig.update_traces(marker_line_width=0)
    return fig


def create_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
) -> go.Figure:
    """Create a correlation heatmap."""
    # Get numeric columns only
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
    ))
    
    fig.update_layout(
        title=title,
        **DARK_THEME,
    )
    return fig


def create_box_plot(
    df: pd.DataFrame,
    y: str,
    x: Optional[str] = None,
    title: str = "",
) -> go.Figure:
    """Create a styled box plot."""
    fig = px.box(df, x=x, y=y, title=title, template="plotly_dark")
    fig.update_layout(**DARK_THEME)
    return fig


def create_area_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: Optional[str] = None,
) -> go.Figure:
    """Create a styled area chart."""
    fig = px.area(df, x=x, y=y, title=title, color=color, template="plotly_dark")
    fig.update_layout(**DARK_THEME)
    return fig


def create_treemap(
    df: pd.DataFrame,
    path: list[str],
    values: str,
    title: str = "",
) -> go.Figure:
    """Create a styled treemap."""
    fig = px.treemap(df, path=path, values=values, title=title, template="plotly_dark")
    fig.update_layout(**DARK_THEME)
    return fig


def auto_chart(
    df: pd.DataFrame,
    title: str = "Auto-Generated Chart",
) -> go.Figure:
    """Automatically generate the best chart based on data types."""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(numeric_cols) >= 2:
        # Scatter plot for 2+ numeric columns
        return create_scatter_chart(df, numeric_cols[0], numeric_cols[1], title)
    elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
        # Bar chart for 1 numeric + 1 categorical
        return create_bar_chart(df, categorical_cols[0], numeric_cols[0], title)
    elif len(numeric_cols) == 1:
        # Histogram for single numeric
        return create_histogram(df, numeric_cols[0], title)
    elif len(categorical_cols) >= 1:
        # Value counts as bar chart
        value_counts = df[categorical_cols[0]].value_counts().reset_index()
        value_counts.columns = [categorical_cols[0], 'count']
        return create_bar_chart(value_counts, categorical_cols[0], 'count', title)
    else:
        # Empty chart
        fig = go.Figure()
        fig.update_layout(title="No suitable data for visualization", **DARK_THEME)
        return fig
