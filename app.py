"""
Unified dashboard: upload -> EDA agent -> credit risk / fraud scoring ->
fairness audit, all as tabs of one Streamlit app.

Every tab accepts an uploaded CSV of ANY shape. Rather than assuming fixed
column names, the credit risk and fairness tabs let the user pick which
column is the target, which columns are categorical/numeric features, and
which text columns (if any) should go through the LLM feature extractor.
Sensible defaults are pre-selected when the column names match the sample
dataset, but nothing is hardcoded to require them.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from agent.claude_agent import run as run_agent
from agent.dashboard import render_full_dashboard
from agent.render import render_full_profile
from credit_risk.features import build_features_generic, build_target_generic
from credit_risk.llm_features import add_llm_features_generic
from fairness.audit import generate_report
from fraud.model import train_and_eval as train_fraud_model
from shared import llm_client

st.set_page_config(page_title="Responsible Credit & Fraud Risk Platform", layout="wide")

# --- Sidebar: about the creator ---
with st.sidebar:
    st.markdown("### About this project")
    st.markdown(
        "Built by **Debarun Banerjee**, B.Tech CSE (AI & ML) student "
        "at Narula Institute of Technology.\n\n"
        "[LinkedIn](https://www.linkedin.com/in/debarun-banerjee-b8524a37b) · "
        "[Portfolio](https://debarun.base44.app)"
    )
    st.markdown("---")
    st.caption(
        "This app spans agentic AI (LLM-driven feature extraction and EDA), "
        "machine learning (credit risk and fraud classifiers), and data "
        "science (fairness auditing). See `docs/domain_mapping.md` in the "
        "repo for the full breakdown."
    )

st.title("Responsible Credit & Fraud Risk Platform")

if not llm_client.is_available():
    st.warning(
        "No ANTHROPIC_API_KEY set — LLM Features aren't used(due to lack of tokens and their high cost)"
        
    )


# --- Shared helpers ---


def load_dataset(uploaded_file, default_path: str) -> tuple[pd.DataFrame, str]:
    """Returns (dataframe, source_note). No schema assumptions — any CSV works here."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source_note = f"Using uploaded file: **{uploaded_file.name}** ({len(df)} rows, {len(df.columns)} columns)"
    else:
        df = pd.read_csv(default_path)
        source_note = f"No file uploaded — using default sample: `{default_path}` ({len(df)} rows)"
    return df, source_note


def column_picker(df: pd.DataFrame, context: str, default_target: str | None = None) -> dict:
    """
    Renders the target/feature/text-column selection UI for an arbitrary
    dataframe and returns the choices as a dict. `context` must be unique
    per call site so widget keys don't collide across tabs.
    """
    columns = list(df.columns)

    target_default_idx = columns.index(default_target) if default_target in columns else 0
    target_col = st.selectbox("Target column (what you're predicting)", columns, index=target_default_idx, key=f"target_{context}")

    unique_values = df[target_col].dropna().unique().tolist()
    positive_value = st.selectbox(
        "Positive / 'bad outcome' value to predict",
        unique_values,
        index=0,
        key=f"positive_{context}",
        help="The class the model treats as the outcome of interest, e.g. 'Charged Off' or 1 for fraud.",
    )

    remaining_cols = [c for c in columns if c != target_col]
    default_categorical = [c for c in remaining_cols if not pd.api.types.is_numeric_dtype(df[c])]
    default_numeric = [c for c in remaining_cols if pd.api.types.is_numeric_dtype(df[c])]

    with st.expander("Advanced: adjust feature columns"):
        categorical_cols = st.multiselect(
            "Categorical columns (one-hot encoded)", remaining_cols, default=default_categorical, key=f"cat_{context}"
        )
        numeric_cols = st.multiselect(
            "Numeric columns (used as-is)", remaining_cols, default=default_numeric, key=f"num_{context}"
        )
        text_cols = st.multiselect(
            "Text columns to send through LLM feature extraction (optional)",
            categorical_cols,
            default=[c for c in categorical_cols if c in ("emp_title", "purpose")],
            key=f"text_{context}",
            help="These get passed to Claude for structured signal extraction. Leave empty to skip the LLM step entirely.",
        )

    return {
        "target_col": target_col,
        "positive_value": positive_value,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "text_cols": text_cols,
    }


def show_schema_help(sample_path: str, context: str) -> None:
    """Shows the sample file's columns/dtypes/example values and a download button, for reference."""
    with st.expander("📋 See an example file"):
        sample_df = pd.read_csv(sample_path)
        example_row = sample_df.iloc[0]
        st.dataframe(
            pd.DataFrame(
                {
                    "column": sample_df.columns,
                    "dtype": [str(sample_df[c].dtype) for c in sample_df.columns],
                    "example value": [example_row[c] for c in sample_df.columns],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Any CSV works here — this is just one example. Pick your own target/feature columns below.")
        with open(sample_path, "rb") as f:
            st.download_button("⬇ Download this example CSV", f, file_name=os.path.basename(sample_path), mime="text/csv", key=f"dl_{context}")


# --- Tabs ---

tab_dashboard, tab_eda, tab_credit, tab_fraud, tab_fairness = st.tabs(
    ["📊 Dashboard", "1. EDA agent", "2. Credit risk", "3. Fraud detection", "4. Fairness audit"]
)

with tab_dashboard:
    st.subheader("Visual overview — no ML background needed")
    st.caption(
        "Upload any CSV to see its shape, target balance, distributions, and correlations "
        "as charts. This tab never calls Claude/the LLM — it's pure chart generation, free to run."
    )
    dashboard_upload = st.file_uploader("CSV file", type="csv", key="dashboard_upload")
    dashboard_df, dashboard_source_note = load_dataset(dashboard_upload, "data/samples/credit_risk_sample.csv")
    st.caption(dashboard_source_note)

    dashboard_columns = list(dashboard_df.columns)
    dashboard_target_default = (
        dashboard_columns.index("loan_status")
        if "loan_status" in dashboard_columns
        else (dashboard_columns.index("Class") if "Class" in dashboard_columns else 0)
    )
    dashboard_target_col = st.selectbox(
        "Which column is the outcome you care about? (optional — pick 'None' to skip)",
        ["(none)"] + dashboard_columns,
        index=dashboard_target_default + 1,
        key="dashboard_target",
    )

    render_full_dashboard(
        dashboard_df,
        target_col=None if dashboard_target_col == "(none)" else dashboard_target_col,
    )

with tab_eda:
    st.subheader("Upload a dataset to auto-profile")
    uploaded = st.file_uploader("CSV file", type="csv", key="eda_upload")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button("Run EDA agent"):
            if llm_client.is_available():
                with st.spinner("Agent is planning and running its analysis..."):
                    summary = run_agent(df)
                st.markdown(summary)
            else:
                render_full_profile(df)

with tab_credit:
    st.subheader("Credit risk scoring")
    st.caption("Works with any CSV — pick your target column and features below.")
    show_schema_help("data/samples/credit_risk_sample.csv", context="credit")

    credit_upload = st.file_uploader("CSV file", type="csv", key="credit_upload")
    raw, source_note = load_dataset(credit_upload, "data/samples/credit_risk_sample.csv")
    st.caption(source_note)

    choices = column_picker(raw, context="credit", default_target="loan_status")

    if st.button("Train credit risk models"):
        y = build_target_generic(raw, choices["target_col"], choices["positive_value"])

        if y.nunique() < 2:
            st.error(
                f"'{choices['target_col']}' only has one value present after filtering. "
                "A classifier needs both outcomes present — check your column and value selections above."
            )
            st.stop()

        X_structured = build_features_generic(raw, choices["categorical_cols"], choices["numeric_cols"])
        if X_structured.shape[1] == 0:
            st.error("No feature columns selected. Pick at least one categorical or numeric column above.")
            st.stop()

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        with st.spinner("Training baseline model..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X_structured, y, test_size=0.25, random_state=42, stratify=y
            )
            baseline = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
            baseline.fit(X_train, y_train)
            baseline_auc = roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])

        with st.spinner("Extracting LLM features and training augmented model..."):
            X_llm = add_llm_features_generic(raw, choices["text_cols"])
            X_combined = pd.concat([X_structured.reset_index(drop=True), X_llm.reset_index(drop=True)], axis=1)
            X_train2, X_test2, y_train2, y_test2 = train_test_split(
                X_combined, y, test_size=0.25, random_state=42, stratify=y
            )
            augmented = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
            augmented.fit(X_train2, y_train2)
            augmented_auc = roc_auc_score(y_test2, augmented.predict_proba(X_test2)[:, 1])

        col1, col2 = st.columns(2)
        col1.metric("Baseline ROC-AUC", f"{baseline_auc:.4f}")
        delta = None if not choices["text_cols"] else f"{augmented_auc - baseline_auc:+.4f}"
        col2.metric("LLM-augmented ROC-AUC", f"{augmented_auc:.4f}", delta=delta)
        if not choices["text_cols"]:
            st.caption("No text columns selected for LLM extraction, so this matches the baseline.")

with tab_fraud:
    st.subheader("Fraud / anomaly detection")
    st.caption("Works with any CSV — pick your target column below.")
    show_schema_help("data/samples/fraud_sample.csv", context="fraud")

    fraud_upload = st.file_uploader("CSV file", type="csv", key="fraud_upload")
    raw_fraud, source_note_fraud = load_dataset(fraud_upload, "data/samples/fraud_sample.csv")
    st.caption(source_note_fraud)

    fraud_columns = list(raw_fraud.columns)
    fraud_target_default = fraud_columns.index("Class") if "Class" in fraud_columns else 0
    fraud_target_col = st.selectbox("Target column (fraud label)", fraud_columns, index=fraud_target_default, key="fraud_target")

    numeric_feature_cols = [
        c for c in fraud_columns if c != fraud_target_col and pd.api.types.is_numeric_dtype(raw_fraud[c])
    ]
    amount_default = "Amount" if "Amount" in numeric_feature_cols else (numeric_feature_cols[0] if numeric_feature_cols else None)
    amount_col = st.selectbox(
        "Column to reference in flag explanations (e.g. transaction amount)",
        numeric_feature_cols,
        index=numeric_feature_cols.index(amount_default) if amount_default in numeric_feature_cols else 0,
        key="fraud_amount_col",
    ) if numeric_feature_cols else None

    if st.button("Train fraud model"):
        y = raw_fraud[fraud_target_col]
        X = raw_fraud[numeric_feature_cols]

        if y.nunique() < 2:
            st.error(f"'{fraud_target_col}' only has one value present. A classifier needs both outcomes to learn anything.")
            st.stop()
        if X.shape[1] == 0:
            st.error("No numeric feature columns found besides the target.")
            st.stop()

        with st.spinner("Training fraud model..."):
            clf = train_fraud_model(X, y)
        st.success("Model trained — see terminal/logs for precision, recall, ROC-AUC.")

        if amount_col:
            st.write("Sample flag explanation:")
            from fraud.explain import explain_flag, top_contributing_features

            row = X.iloc[0]
            top_feats = top_contributing_features(clf, row)
            st.json(top_feats)
            st.write(explain_flag(row[amount_col], top_feats))

with tab_fairness:
    st.subheader("Fairness audit")
    st.caption(
        "Upload any dataset, pick your target column, and pick a column to audit for "
        "subgroup fairness (a real demographic attribute if you have one, or a documented proxy)."
    )
    show_schema_help("data/samples/credit_risk_sample.csv", context="fairness")

    fairness_upload = st.file_uploader("CSV file", type="csv", key="fairness_upload")
    raw_fair, source_note_fair = load_dataset(fairness_upload, "data/samples/credit_risk_sample.csv")
    st.caption(source_note_fair)

    fair_choices = column_picker(raw_fair, context="fairness", default_target="loan_status")

    fair_columns = list(raw_fair.columns)
    default_group_col = "home_ownership" if "home_ownership" in fair_columns else fair_columns[0]
    group_col = st.selectbox(
        "Column to audit for fairness (protected attribute or proxy)",
        fair_columns,
        index=fair_columns.index(default_group_col),
        key="fairness_group_col",
    )

    if st.button("Run fairness audit"):
        y_true = build_target_generic(raw_fair, fair_choices["target_col"], fair_choices["positive_value"])

        if y_true.nunique() < 2:
            st.error(f"'{fair_choices['target_col']}' only has one value present after filtering.")
            st.stop()

        X_structured = build_features_generic(raw_fair, fair_choices["categorical_cols"], fair_choices["numeric_cols"])
        X_llm = add_llm_features_generic(raw_fair, fair_choices["text_cols"])
        X = pd.concat([X_structured.reset_index(drop=True), X_llm.reset_index(drop=True)], axis=1)

        if X.shape[1] == 0:
            st.error("No feature columns selected. Pick at least one categorical or numeric column above.")
            st.stop()

        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
        clf.fit(X, y_true)
        y_pred = pd.Series(clf.predict(X), index=raw_fair.index)

        report = generate_report(y_true, y_pred, raw_fair[group_col], group_label=group_col)
        st.markdown(report)
