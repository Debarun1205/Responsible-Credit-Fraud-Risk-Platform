"""
Visual, non-technical-friendly overview dashboard. Pure pandas + Plotly —
no LLM/Claude calls anywhere in this file, so it works with zero API cost
and no ANTHROPIC_API_KEY at all.

Meant to be the "explain what's going on at a glance" tab: someone with no
ML background should be able to look at these charts and understand the
shape of the data and the target outcome, without reading any code.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_overview_metrics(df: pd.DataFrame) -> None:
    """Big, simple summary numbers at the top — the "at a glance" row."""
    n_rows, n_cols = df.shape
    n_numeric = df.select_dtypes(include="number").shape[1]
    n_categorical = n_cols - n_numeric
    missing_pct = (df.isna().sum().sum() / (n_rows * n_cols) * 100) if n_rows and n_cols else 0

    cols = st.columns(4)
    cols[0].metric("Rows", f"{n_rows:,}")
    cols[1].metric("Columns", f"{n_cols:,}")
    cols[2].metric("Numeric vs. categorical", f"{n_numeric} / {n_categorical}")
    cols[3].metric("Missing data", f"{missing_pct:.1f}%")


def render_target_balance(df: pd.DataFrame, target_col: str) -> None:
    """
    Donut chart of the target column's distribution — the single most
    important thing for a non-technical viewer to see: "what are we
    predicting, and how balanced is it?"
    """
    st.markdown(f"#### What are we predicting? — `{target_col}`")
    counts = df[target_col].value_counts()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=[str(v) for v in counts.index],
                values=counts.values,
                hole=0.55,
                marker=dict(colors=px.colors.qualitative.Set2),
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    minority_pct = counts.min() / counts.sum() * 100
    if minority_pct < 10:
        st.caption(
            f"⚠️ This is an imbalanced dataset — the smaller outcome is only {minority_pct:.1f}% of rows. "
            "This is common and expected for things like fraud, but it means plain accuracy would be "
            "misleading; look at precision/recall instead."
        )


def render_numeric_distributions(df: pd.DataFrame, numeric_cols: list[str], max_charts: int = 6) -> None:
    """Histogram grid for numeric columns, in plain everyday language."""
    st.markdown("#### How the numbers are spread out")
    if not numeric_cols:
        st.write("No numeric columns to show.")
        return

    shown = numeric_cols[:max_charts]
    if len(numeric_cols) > max_charts:
        st.caption(f"Showing the first {max_charts} of {len(numeric_cols)} numeric columns.")

    cols = st.columns(2)
    for i, col_name in enumerate(shown):
        fig = px.histogram(df, x=col_name, nbins=30, color_discrete_sequence=["#6C63FF"])
        fig.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=260, title=col_name, showlegend=False)
        with cols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)


def render_categorical_breakdown(df: pd.DataFrame, categorical_cols: list[str], max_charts: int = 6, max_categories: int = 8) -> None:
    """Bar charts for categorical columns, capped so rare-category noise doesn't clutter it."""
    st.markdown("#### Category breakdowns")
    if not categorical_cols:
        st.write("No categorical columns to show.")
        return

    shown = categorical_cols[:max_charts]
    if len(categorical_cols) > max_charts:
        st.caption(f"Showing the first {max_charts} of {len(categorical_cols)} categorical columns.")

    cols = st.columns(2)
    for i, col_name in enumerate(shown):
        counts = df[col_name].value_counts().head(max_categories)
        fig = px.bar(x=counts.index.astype(str), y=counts.values, color_discrete_sequence=["#00B4D8"])
        fig.update_layout(
            margin=dict(t=30, b=10, l=10, r=10), height=260, title=col_name,
            xaxis_title=None, yaxis_title="count",
        )
        with cols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)


def render_correlation_heatmap(df: pd.DataFrame) -> None:
    """A single heatmap showing which numeric columns move together — explained in plain terms."""
    st.markdown("#### Which numbers move together")
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        st.write("Not enough numeric columns for a correlation heatmap.")
        return

    corr = numeric_df.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=min(120 + 40 * len(corr), 600))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Closer to 1 (dark blue) means two columns tend to rise and fall together. "
        "Closer to -1 (dark red) means as one goes up, the other tends to go down. "
        "Near 0 means they're mostly unrelated."
    )


def render_full_dashboard(df: pd.DataFrame, target_col: str | None = None) -> None:
    """
    Renders the entire visual dashboard in one call: metrics, target
    balance (if a target column is given), numeric distributions,
    categorical breakdowns, and a correlation heatmap. No LLM calls.
    """
    render_overview_metrics(df)
    st.divider()

    if target_col and target_col in df.columns:
        render_target_balance(df, target_col)
        st.divider()

    feature_df = df.drop(columns=[target_col]) if target_col and target_col in df.columns else df
    numeric_cols = feature_df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    render_numeric_distributions(feature_df, numeric_cols)
    st.divider()
    render_categorical_breakdown(feature_df, categorical_cols)
    st.divider()
    render_correlation_heatmap(feature_df)
