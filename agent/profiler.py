"""
Deterministic profiling functions the EDA agent can call as tools.

These are plain pandas — no LLM here. The "AI" part is claude_agent.py
deciding which of these to call and in what order; this module just does
the actual computation reliably.
"""

from __future__ import annotations

import pandas as pd


def missingness_report(df: pd.DataFrame) -> dict:
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    return {
        col: {"missing_count": int(missing[col]), "missing_pct": float(pct[col])}
        for col in df.columns
        if missing[col] > 0
    }


def numeric_summary(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include="number")
    return numeric_df.describe().round(3).to_dict()


def categorical_summary(df: pd.DataFrame, max_categories: int = 10) -> dict:
    cat_df = df.select_dtypes(include=["object", "category"])
    return {
        col: cat_df[col].value_counts().head(max_categories).to_dict()
        for col in cat_df.columns
    }


def correlation_matrix(df: pd.DataFrame, threshold: float = 0.5) -> list[dict]:
    """Returns only pairs above |threshold| to keep output readable."""
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if abs(value) >= threshold:
                pairs.append({"col_a": cols[i], "col_b": cols[j], "correlation": round(float(value), 3)})
    return sorted(pairs, key=lambda p: abs(p["correlation"]), reverse=True)


PROFILER_TOOLS = {
    "missingness_report": missingness_report,
    "numeric_summary": numeric_summary,
    "categorical_summary": categorical_summary,
    "correlation_matrix": correlation_matrix,
}


if __name__ == "__main__":
    import os

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "samples", "credit_risk_sample.csv"))
    print("Missingness:", missingness_report(df))
    print("Correlations (|r|>=0.3):", correlation_matrix(df, threshold=0.3))
