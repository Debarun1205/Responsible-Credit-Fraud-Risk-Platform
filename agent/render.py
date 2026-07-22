"""
Renders agent/profiler.py output as Streamlit tables and charts instead of
raw JSON. Used specifically for the "no ANTHROPIC_API_KEY set" fallback path
in app.py, so the EDA tab looks like a dashboard even with zero LLM calls.

No Claude/API usage anywhere in this file — it's pure pandas + Streamlit
rendering of the same numbers agent/profiler.py already computes.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from agent.profiler import (
    categorical_summary,
    correlation_matrix,
    missingness_report,
    numeric_summary,
)


def render_missingness(df: pd.DataFrame) -> None:
    st.markdown("#### Missing values")
    report = missingness_report(df)
    if not report:
        st.write("No missing values found.")
        return
    missing_series = pd.Series({col: v["missing_pct"] for col, v in report.items()}, name="missing %")
    st.bar_chart(missing_series)
    st.dataframe(
        pd.DataFrame(report).T.rename(columns={"missing_count": "count", "missing_pct": "missing %"}),
        use_container_width=True,
    )


def render_numeric_summary(df: pd.DataFrame) -> None:
    st.markdown("#### Numeric column summary")
    summary = numeric_summary(df)
    if not summary:
        st.write("No numeric columns found.")
        return
    st.dataframe(pd.DataFrame(summary), use_container_width=True)


def render_categorical_summary(df: pd.DataFrame, max_categories: int = 10) -> None:
    st.markdown("#### Categorical column breakdown")
    summary = categorical_summary(df, max_categories=max_categories)
    if not summary:
        st.write("No categorical columns found.")
        return

    cols = st.columns(2)
    for i, (col_name, counts) in enumerate(summary.items()):
        with cols[i % 2]:
            st.caption(col_name)
            st.bar_chart(pd.Series(counts, name="count"))


def render_correlations(df: pd.DataFrame, threshold: float = 0.5) -> None:
    st.markdown(f"#### Correlated numeric pairs (|r| ≥ {threshold})")
    pairs = correlation_matrix(df, threshold=threshold)
    if not pairs:
        st.write("No numeric column pairs exceeded the correlation threshold.")
        return
    st.dataframe(pd.DataFrame(pairs), use_container_width=True)


def render_full_profile(df: pd.DataFrame, correlation_threshold: float = 0.5) -> None:
    """Runs and renders every profiling section in order, as a mini dashboard."""
    st.info(
        "No ANTHROPIC_API_KEY set, so this shows the profiling results directly "
        "(no LLM call). Add a key in Settings → Secrets to also get an agent-written "
        "plain-English summary of these numbers."
    )
    render_missingness(df)
    render_numeric_summary(df)
    render_categorical_summary(df)
    render_correlations(df, threshold=correlation_threshold)
